#!/usr/bin/env python3
import sys
import argparse
import threading
import time
import os
import json
import queue

import pygame
import numpy as np
import cv2
import pyrealsense2 as rs
import serial
from dorna2 import Dorna

# ─── Trigger → Serial (Arduino) configuration ───────────────────────────
SERIAL_PORT    = '/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44236313735351100201-if00'
BAUDRATE       = 115200
MAX_RATE       = 800     # steps/sec at full trigger
DEADZONE       = 0.02
RIGHT_AXIS     = 5       # Xbox right trigger axis (UNCHANGED index)
LEFT_AXIS      = 2       # Xbox left  trigger axis (UNCHANGED index)
RESET_BUTTON   = 2

# ── Controller axis mapping (tweak if your driver maps differently) ─────
LS_X_AXIS = 0
LS_Y_AXIS = 1
RS_X_AXIS = 3
RS_Y_AXIS = 4
INVERT_LS_X = False
INVERT_LS_Y = False
INVERT_RS_X = False   # OS/driver-level inversion if needed
INVERT_RS_Y = False

# Optional: sign for D‑pad Up/Down movement (+1 normal, −1 to flip)
DPAD_Z_SIGN = 1

def _axis(joy, idx, invert=False):
    try:
        v = float(joy.get_axis(idx))
    except Exception:
        v = 0.0
    return -v if invert else v

# Face buttons (typical on Linux via pygame)
A_BUTTON       = 0  # advance from __midway
B_BUTTON       = 1  # cancel prompt → free control at __midway
X_BUTTON       = 2  # save current as Reload + create Reload__midway

# ─── Click safety for UI ─────────────────────────────────────────────────
AXIS_GUARD_DELTA  = 0.20
AXIS_GUARD_WINDOW = 0.15

# ─── Persistence ─────────────────────────────────────────────────────────
POSES_PATH     = "poses.json"
SETTINGS_PATH  = "settings.json"
MIDWAY_SUFFIX  = "__midway"    # helper pose suffix used for all final poses

DEFAULT_TOOL_LZ     = 205.0
DEFAULT_APPROACH_MM = 25.0      # pure tool-Z distance Midway <-> Final

# ─── Poses (Reload updated as requested earlier) ─────────────────────────
DEFAULT_POSES = {
    "Reload":  {"j0": 7.71, "j1": 80.86, "j2": -100.00, "j3": -0.07, "j4": -70.60, "j5": 6.35},
    "Default": {"j0": 39.24,"j1": 27.00, "j2": -100.44, "j3": 6.35,  "j4": 65.19,  "j5": -2.70},
}

# ─── Helpers ─────────────────────────────────────────────────────────────
def midway_name(name: str) -> str:
    return f"{name}{MIDWAY_SUFFIX}"

def load_poses(path=POSES_PATH):
    try:
        data = json.load(open(path)) if os.path.exists(path) else {}
        for name, pose in DEFAULT_POSES.items():
            data.setdefault(name, pose)
        # coerce to float
        for name, pose in list(data.items()):
            if not isinstance(pose, dict): continue
            for k in ["j0","j1","j2","j3","j4","j5"]:
                if k in pose: data[name][k] = float(pose[k])
        return data
    except Exception as e:
        print(f"⚠️ Using built-in poses (could not load {path}: {e})")
        return {k: v.copy() for k, v in DEFAULT_POSES.items()}

def save_poses(poses, path=POSES_PATH):
    try:
        with open(path, "w") as f: json.dump(poses, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save poses to {path}: {e}")

def load_settings(path=SETTINGS_PATH):
    try:
        data = json.load(open(path)) if os.path.exists(path) else {}
        tool_lz     = float(data.get("tool_lz", DEFAULT_TOOL_LZ))
        approach_mm = float(data.get("approach_mm", DEFAULT_APPROACH_MM))
        return {"tool_lz": tool_lz, "approach_mm": approach_mm}
    except Exception as e:
        print(f"⚠️ Using default settings (could not load {path}: {e})")
        return {"tool_lz": DEFAULT_TOOL_LZ, "approach_mm": DEFAULT_APPROACH_MM}

def save_settings(settings, path=SETTINGS_PATH):
    try:
        with open(path, "w") as f: json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save settings to {path}: {e}")

def normalize(v):  # normalize [-1..1] to [0..1] when needed
    return (v + 1.0) / 2.0 if v < -0.2 or v > 1.0 else v

def wrap_text(text: str, font: pygame.font.Font, max_width: int):
    words = text.split(' ')
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= max_width:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

# ─── Args ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Joystick control for Dorna + RealSense D405 + Stepper trigger bridge")
    p.add_argument("--host", "-H", default="10.42.0.11", help="IP address of your Dorna controller")
    p.add_argument("--port", "-P", type=int, default=443, help="Port for Dorna control (usually 443)")
    return p.parse_args()

# ─── Math helpers ────────────────────────────────────────────────────────
def axis_angle_to_R(a, b, c):
    vec = np.radians([a, b, c]); th = np.linalg.norm(vec)
    if th < 1e-6: return np.eye(3)
    u = vec / th; ux, uy, uz = u
    K = np.array([[  0, -uz,  uy],[ uz,   0, -ux],[-uy,  ux,   0]])
    return np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*(K @ K)

def R_to_axis_angle(R):
    tr = np.trace(R)
    th = np.arccos(max(min((tr - 1)/2, 1), -1))
    if abs(th) < 1e-6: return (0.0, 0.0, 0.0)
    ux = (R[2,1] - R[1,2])/(2*np.sin(th))
    uy = (R[0,2] - R[2,0])/(2*np.sin(th))
    uz = (R[1,0] - R[0,1])/(2*np.sin(th))
    return tuple(np.degrees([ux*th, uy*th, uz*th]))

# ─── Shared state ───────────────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.lx = self.ly = self.rx = self.ry = 0.0
        self.hx = self.hy = 0
        self.lb = self.rb = False
        self.levels = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0]
        self.idx = self.levels.index(1.0)
        self.pitch = 0.0
        self.j5    = 0.0
        self.poses = load_poses()
        # ensure defaults present
        for k, v in DEFAULT_POSES.items():
            self.poses.setdefault(k, v.copy())
        self.settings = load_settings()
        self.tool_lz = float(self.settings.get("tool_lz", DEFAULT_TOOL_LZ))
        self.approach_mm = float(self.settings.get("approach_mm", DEFAULT_APPROACH_MM))
        # Midway confirm gate
        self.await_confirm = False
        self.await_target  = None   # final pose name when prompting
        self.confirm_msg   = ""
        # New pose naming
        self.naming_active = False
        self.naming_text   = ""

# ─── Robot control thread (200 Hz) ───────────────────────────────────────
class RobotThread(threading.Thread):
    def __init__(self, state: SharedState, host: str, port: int):
        super().__init__(daemon=True)
        self.state      = state
        self.host       = host
        self.port       = port
        self.stop_event = threading.Event()
        self.cmd_q      = queue.Queue()
        self.robot      = None

        # Base and orientation
        self.x0 = self.y0 = self.z0 = 0.0
        self.R  = np.eye(3)
        self.j5v = 0.0

        # Routing / suppression
        self.current_named     = None
        self.skip_manual_until = 0.0

        # Tuning
        self.BSX, self.BSJ5 = 5.0, 5.0
        self.BSB, self.BSC  = 0.5, 0.5
        self.BSH            = 5.0
        self.VT, self.VR    = 10.0, 5.0
        self.VR_POSE        = self.VR * 2.0
        self.DZ             = 0.1

    def enqueue(self, cmd: dict): self.cmd_q.put(cmd)

    # Pose/orientation refresh for UI indicators
    def _refresh_from_robot(self):
        pr = self.robot.get_all_pose()[:6]
        self.x0, self.y0, self.z0, a, b, c = pr
        self.R = axis_angle_to_R(a, b, c)
        tz = self.R[:,2]
        pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
        with self.state.lock: self.state.pitch = pitch

    def _queue_jmove_to_pose(self, pose: dict):
        go = {"cmd":"jmove","rel":0,"vel":self.VR_POSE}; go.update(pose)
        self.robot.play_dict(go)
        self.skip_manual_until = max(self.skip_manual_until, time.time() + 0.35)
        time.sleep(0.05)
        try:
            self._refresh_from_robot()
            if "j5" in pose:
                with self.state.lock: self.state.j5 = float(pose["j5"])
                self.j5v = float(pose["j5"])
        except Exception:
            pass

    def _tool_move_along_tz(self, dist_mm: float, cont: int = 0):
        tz = self.R[:,2]
        dx, dy, dz = tz[0]*dist_mm, tz[1]*dist_mm, tz[2]*dist_mm
        self.robot.play_dict({"cmd":"lmove","rel":1,"x":dx,"y":dy,"z":dz,"vel":self.VT,"cont":cont})
        self.x0 += dx; self.y0 += dy; self.z0 += dz

    def _try_get_current_joints(self):
        candidates = ["get_all_joint", "get_joint", "get_joints", "joints", "get_pos"]
        joints = None
        for name in candidates:
            try:
                if hasattr(self.robot, name):
                    res = getattr(self.robot, name)()
                    if isinstance(res, dict):
                        vals = [res.get(k) for k in ["j0","j1","j2","j3","j4","j5"]]
                        if all(v is not None for v in vals): joints = vals; break
                        try: joints = [res[i] for i in range(6)]; break
                        except Exception: pass
                    elif isinstance(res, (list, tuple)) and len(res) >= 6:
                        joints = list(res[:6]); break
            except Exception:
                continue
        if joints is None:
            print("⚠️ Could not fetch joint angles; save operation skipped.")
            return None
        return {"j0": float(joints[0]), "j1": float(joints[1]), "j2": float(joints[2]),
                "j3": float(joints[3]), "j4": float(joints[4]), "j5": float(joints[5])}

    def _save_current_pose_with_midway(self, name: str, approach_mm: float):
        # Save current joints as final pose <name>
        final_joints = self._try_get_current_joints()
        if final_joints is None: return False
        with self.state.lock:
            self.state.poses[name] = final_joints
        save_poses(self.state.poses)
        print(f"[Robot] Saved current as '{name}' (final).")

        # Create <name>__midway via pure tool-Z retract; stay safe
        try:
            self._tool_move_along_tz(-approach_mm, cont=0)
            time.sleep(0.05)
            mid_joints = self._try_get_current_joints()
            if mid_joints is not None:
                with self.state.lock:
                    self.state.poses[midway_name(name)] = mid_joints
                save_poses(self.state.poses)
                print(f"[Robot] Created '{midway_name(name)}' via −{approach_mm} mm tool-Z.")
            else:
                print("⚠️ Could not capture midway joints.")
        finally:
            # Return to final pose safely via +approach
            self._tool_move_along_tz(+approach_mm, cont=0)
            self.current_named = name
            print(f"[Robot] Returned to '{name}'.")
        return True

    def _ensure_current_midway(self, approach_mm: float):
        """
        Ensure we are at a safe __midway before moving away from current final.
        EXCEPTION: If current is 'Default', do nothing (no retract required).
        """
        if not self.current_named: return
        name = self.current_named
        if name == "Default":  # Exception: never retract when leaving Default
            return
        if name.endswith(MIDWAY_SUFFIX): return
        cur_mid = midway_name(name)
        with self.state.lock:
            poses = self.state.poses
        if cur_mid in poses:
            self._tool_move_along_tz(-approach_mm, cont=0)
            self.current_named = cur_mid
            print(f"[Robot] Retracted to existing '{cur_mid}'.")
            return
        print(f"[Robot] '{cur_mid}' missing; creating via −{approach_mm} mm tool-Z…")
        self._tool_move_along_tz(-approach_mm, cont=0)
        time.sleep(0.05)
        mid_joints = self._try_get_current_joints()
        if mid_joints is not None:
            with self.state.lock:
                poses[cur_mid] = mid_joints
            save_poses(poses)
            print(f"[Robot] Saved new '{cur_mid}'.")
        self.current_named = cur_mid

    def _advance_if_midway(self, approach_mm: float):
        """Advance along tool-Z from the current __midway to its final, if applicable."""
        if self.current_named and str(self.current_named).endswith(MIDWAY_SUFFIX):
            final = self.current_named[: -len(MIDWAY_SUFFIX)]
            try:
                self._tool_move_along_tz(+approach_mm, cont=0)
                self.current_named = final
                print(f"[Robot] Advanced +{approach_mm} mm to '{final}'.")
            except Exception as e:
                print(f"⚠️ Advance from '{self.current_named}' failed: {e}")

    def run(self):
        robot = Dorna(); self.robot = robot
        # Connect with retries
        for attempt in range(5):
            if robot.connect(host=self.host, port=self.port): break
            print(f"⚠️  Connection attempt {attempt+1}/5 to {self.host}:{self.port} failed. Retrying…")
            time.sleep(2)
        else:
            print(f"❌ All connection attempts to {self.host}:{self.port} failed."); sys.exit(1)

        robot.set_motor(1)

        # Apply tool length
        with self.state.lock: tool_lz = float(self.state.tool_lz)
        robot.play_dict(cmd={"cmd":"tool","id":1,"lz":tool_lz})

        # Home to Default — and STOP (no auto-trips)
        try:
            with self.state.lock: default_pose = self.state.poses["Default"].copy()
            self._queue_jmove_to_pose(default_pose)
            self.current_named = "Default"
            print("[Robot] Ready at Default.")
        except Exception as e:
            print(f"⚠️ Could not home to Default pose: {e}")
            try: self._refresh_from_robot()
            except Exception: pass

        # Initialize j5 UI value
        with self.state.lock:
            if "Default" in self.state.poses:
                self.j5v = self.state.poses["Default"].get("j5", 0.0)
                self.state.j5 = self.j5v

        # Run loop
        next_t = time.time()
        while not self.stop_event.is_set():
            now = time.time()
            if now < next_t: time.sleep(next_t - now)
            next_t += 1/200

            # Command handling
            try:
                while True:
                    cmd = self.cmd_q.get_nowait()
                    typ = cmd.get("type")

                    if typ == "goto":
                        target = cmd.get("pose_name")
                        with self.state.lock:
                            poses = self.state.poses.copy()
                            awaiting    = self.state.await_confirm
                            approach_mm = float(self.state.approach_mm)
                        if awaiting:
                            print("[Robot] Awaiting confirmation; goto ignored.")
                            continue

                        # MOVING TO DEFAULT: always retract current (if non-Default), then go to Default (no prompt)
                        if target == "Default":
                            self._ensure_current_midway(approach_mm)
                            try:
                                self._queue_jmove_to_pose(poses["Default"])
                                self.current_named = "Default"
                                print("[Robot] Moved to Default (safe retract first if needed).")
                            except Exception as e:
                                print(f"⚠️ Failed to go to Default: {e}")
                            continue

                        # 1) Ensure CURRENT is at its __midway (skip if current is Default)
                        self._ensure_current_midway(approach_mm)

                        # 2) Require TARGET__midway to exist; otherwise refuse (safety)
                        target_mid = midway_name(target)
                        if target_mid not in poses:
                            print(f"⚠️ Refusing move: '{target_mid}' not found. Jog to '{target}' and press X (or Add) to teach it.")
                            continue

                        # 3) Move to TARGET__midway, then wait for A/B at the midway
                        try:
                            self._queue_jmove_to_pose(poses[target_mid])
                            self.current_named = target_mid
                            with self.state.lock:
                                self.state.await_confirm = True
                                self.state.await_target  = target
                                self.state.confirm_msg   = f"Paused at '{target_mid}'. Press A to advance +{approach_mm} mm, or B for free control."
                            print(f"[Robot] Reached '{target_mid}'. Awaiting A (advance) or B (free control).")
                        except Exception as e:
                            print(f"⚠️ Failed to go to '{target_mid}': {e}")

                    elif typ == "confirm_or_advance":
                        with self.state.lock:
                            approach_mm = float(self.state.approach_mm)
                            waiting = self.state.await_confirm
                            tgt     = self.state.await_target
                            if waiting and tgt:
                                self.state.await_confirm = False
                                self.state.await_target  = None
                                self.state.confirm_msg   = ""
                                do_prompt_advance = True
                            else:
                                do_prompt_advance = False
                        if do_prompt_advance:
                            try:
                                self._tool_move_along_tz(+approach_mm, cont=0)
                                self.current_named = tgt
                                print(f"[Robot] Advanced +{approach_mm} mm to '{tgt}'.")
                            except Exception as e:
                                print(f"⚠️ Advance to '{tgt}' failed: {e}")
                        else:
                            self._advance_if_midway(approach_mm)

                    elif typ == "midway_free_control":
                        with self.state.lock:
                            if self.state.await_confirm:
                                self.state.await_confirm = False
                                self.state.await_target  = None
                                self.state.confirm_msg   = ""
                                print("[Robot] Midway prompt dismissed; free control enabled at __midway.")

                    elif typ == "stop_motors":
                        try: robot.set_motor(0); print("[Robot] Motors stopped.")
                        except Exception as e: print(f"⚠️ stop_motors failed: {e}")

                    elif typ == "kill_motors":
                        try: robot.set_motor(0); print("[Robot] Motors killed.")
                        except Exception as e: print(f"⚠️ kill_motors failed: {e}")

                    elif typ == "set_tool_lz":
                        try:
                            lz = float(cmd.get("lz"))
                            robot.play_dict(cmd={"cmd":"tool","id":1,"lz":lz})
                            with self.state.lock:
                                self.state.tool_lz = lz
                                self.state.settings["tool_lz"] = lz
                            save_settings(self.state.settings)
                            self._refresh_from_robot()
                            print(f"[Robot] Tool length set to {lz} mm")
                        except Exception as e:
                            print(f"⚠️ set_tool_lz failed: {e}")

                    elif typ == "set_approach_mm":
                        try:
                            d = float(cmd.get("dist"))
                            with self.state.lock:
                                self.state.approach_mm = d
                                self.state.settings["approach_mm"] = d
                            save_settings(self.state.settings)
                            print(f"[Robot] Approach distance set to {d} mm")
                        except Exception as e:
                            print(f"⚠️ set_approach_mm failed: {e}")

                    elif typ == "save_reload_with_midway":
                        with self.state.lock: approach_mm = float(self.state.approach_mm)
                        ok = self._save_current_pose_with_midway("Reload", approach_mm)
                        if ok: save_poses(self.state.poses)

                    elif typ == "save_named_with_midway":
                        name = cmd.get("name", "").strip()
                        if not name:
                            print("⚠️ Empty name; skipping save.")
                            continue
                        if name.endswith(MIDWAY_SUFFIX):
                            print(f"⚠️ Name cannot end with '{MIDWAY_SUFFIX}'.")
                            continue
                        if name in ["Default"]:
                            print("⚠️ 'Default' is reserved; choose another.")
                            continue
                        with self.state.lock: approach_mm = float(self.state.approach_mm)
                        ok = self._save_current_pose_with_midway(name, approach_mm)
                        if ok: save_poses(self.state.poses)

            except queue.Empty:
                pass

            # Manual jog (disabled only during confirmation)
            with self.state.lock:
                lx, ly = self.state.lx, self.state.ly
                rx, ry = self.state.rx, self.state.ry
                hx, hy = self.state.hx, self.state.hy
                sens   = self.state.levels[self.state.idx]
                waiting = self.state.await_confirm

            sx, sj5, sb, sc, sh = (5.0*sens, 5.0*sens, 0.5*sens, 0.5*sens, 5.0*sens)
            manual_enabled = (time.time() >= self.skip_manual_until) and (not waiting)

            # LEFT stick: tool-Z / J5 (dominant axis)
            if manual_enabled and abs(ly) > self.DZ:
                if abs(ly) >= abs(lx):
                    d = -ly * sx
                    tz = self.R[:,2]
                    self.robot.play_dict({"cmd":"lmove","rel":1,"x":tz[0]*d,"y":tz[1]*d,"z":tz[2]*d,"vel":self.VT,"cont":1})
                    self.x0 += tz[0]*d; self.y0 += tz[1]*d; self.z0 += tz[2]*d
            if manual_enabled and abs(lx) > self.DZ and abs(lx) > abs(ly):
                delta = lx * sj5
                self.j5v += delta
                self.R = self.R @ axis_angle_to_R(0, 0, delta)
                self.robot.play_dict({"cmd":"jmove","rel":1,"j5":delta,"vel":self.VR})
                with self.state.lock: self.state.j5 = self.j5v

            # RIGHT stick: yaw (world Z), pitch (local Y)
            moved = False
            if manual_enabled:
                at_default = (self.current_named == "Default")
                # RS L/R: swap ONLY at Default
                rx_eff = -rx if at_default else rx
                # RS U/D: unchanged everywhere
                ry_eff = ry

                if abs(rx_eff) > self.DZ:
                    self.R = axis_angle_to_R(0, 0, -rx_eff*sc) @ self.R
                    moved = True
                if abs(ry_eff) > self.DZ:
                    self.R = self.R @ axis_angle_to_R(0, ry_eff*sb, 0)
                    moved = True

            if moved:
                a1,b1,c1 = R_to_axis_angle(self.R)
                self.robot.play_dict({"cmd":"lmove","rel":0,"x":self.x0,"y":self.y0,"z":self.z0,"a":a1,"b":b1,"c":c1,"vel":self.VR})
                tz = self.R[:,2]
                pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
                with self.state.lock: self.state.pitch = pitch

            # D‑pad mapping:
            #   Up/Down:
            #     - At Default  → WORLD Z
            #     - Else        → TOOL "up" = whichever of local X or Y has larger |z|
            #   Left/Right = strafe ⟂ heading projection (XY);
            #     NOW inverted in ALL poses per your request.
            if manual_enabled:
                tz = self.R[:,2]
                tx = self.R[:,0]
                ty = self.R[:,1]
                proj = np.array([tz[0], tz[1], 0.0]); n = np.linalg.norm(proj)
                at_default = (self.current_named == "Default")

                # D‑pad Up/Down
                if hy != 0:
                    step = float(hy) * sh * DPAD_Z_SIGN
                    if at_default:
                        self.robot.play_dict({"cmd":"lmove","rel":1,"x":0.0,"y":0.0,"z":step,"vel":self.VT,"cont":1})
                        self.z0 += step
                    else:
                        up_axis = tx if abs(tx[2]) >= abs(ty[2]) else ty
                        dx, dy, dz = up_axis[0]*step, up_axis[1]*step, up_axis[2]*step
                        self.robot.play_dict({"cmd":"lmove","rel":1,"x":dx,"y":dy,"z":dz,"vel":self.VT,"cont":1})
                        self.x0 += dx; self.y0 += dy; self.z0 += dz

                # D‑pad Left/Right (now inverted in ALL poses)
                hx_eff = -hx
                if hx_eff != 0:
                    if n > 1e-3:
                        perp = np.array([-proj[1], proj[0], 0.0]) / n * hx_eff
                    else:
                        perp = np.array([0.0, -hx_eff, 0.0])
                    dx, dy = perp[0]*sh, perp[1]*sh
                    self.robot.play_dict({"cmd":"lmove","rel":1,"x":dx,"y":dy,"z":0.0,"vel":self.VT,"cont":1})
                    self.x0 += dx; self.y0 += dy

        # Shutdown
        self.robot.set_motor(0)
        self.robot.close()

    def stop(self): self.stop_event.set()

# ─── Main HUD + Side UI + Trigger Bridge (60 FPS) ─────────────────────────
def main():
    args = parse_args()

    # Serial bridge (stepper trigger)
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        time.sleep(2)
        print(f"[Bridge] Serial open {SERIAL_PORT} @ {BAUDRATE}")
    except Exception as e:
        print(f"⚠️ Could not open serial port: {e}", file=sys.stderr)
        ser = None
    last_rate = None

    pygame.init(); pygame.font.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("❌ No joystick detected."); return
    joy = pygame.joystick.Joystick(0); joy.init()

    # RealSense D405
    pipeline = rs.pipeline(); config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        pipeline.start(config)
    except Exception as e:
        print("❌ Failed to start RealSense pipeline:", e, file=sys.stderr); return

    # Geometry
    try:
        init_frames = pipeline.wait_for_frames(timeout_ms=1000)
        init_color  = init_frames.get_color_frame()
        if not init_color: raise RuntimeError("Empty initial frame")
    except Exception as e:
        print("❌ Couldn't initialize RealSense:", e, file=sys.stderr)
        pipeline.stop(); return

    frame = np.asanyarray(init_color.get_data())
    h, w   = frame.shape [:2]
    ui_w   = int(w * 0.25)

    screen = pygame.display.set_mode((w + ui_w, h))
    pygame.display.set_caption("Robotic Injector [SERI]")
    clock = pygame.time.Clock()

    # Fonts / layout
    label_font       = pygame.font.SysFont("Consolas", 14, bold=True)
    value_font       = pygame.font.SysFont("Consolas", 12)
    placeholder_font = pygame.font.SysFont("Consolas", 12)
    big_font         = pygame.font.SysFont("Consolas", 24, bold=True)
    m           = 10
    panel_size  = int(h * 0.12)
    diag_gap    = int(w * 0.06)
    padding     = 4
    label_h     = label_font.get_height()
    value_h     = value_font.get_height()
    H_panel     = panel_size + padding + label_h + padding + value_h + padding
    btn_w   = ui_w - 2*m
    btn_gap = 8

    # Shared state & robot
    state = SharedState()
    with state.lock:
        state.poses["Reload"] = DEFAULT_POSES["Reload"].copy()
    save_poses(state.poses); save_settings(state.settings)
    rt = RobotThread(state, host=args.host, port=args.port); rt.start()

    # Inline editors
    toollen_input_active  = False
    toollen_text          = f"{state.tool_lz:.1f}"
    approach_input_active = False
    approach_text         = f"{state.approach_mm:.1f}"
    naming_input_active   = False
    naming_text           = ""

    # Button edges / sensitivity edges
    prev_a = prev_b = prev_x = False
    prev_lb = prev_rb = False
    prev_idx = -1

    # Clickables and guard
    button_rects = []
    button_items = []
    last_sig_axis_time = 0.0
    prev_lt = joy.get_axis(LEFT_AXIS)
    prev_rt = joy.get_axis(RIGHT_AXIS)

    def visible_pose_names(poses: dict):
        return [name for name in poses.keys() if not name.endswith(MIDWAY_SUFFIX)]

    def make_buttons(current_tool_lz, current_approach, poses):
        items = [
            {"label": f"Tool length (mm): {current_tool_lz:.1f}  [Edit]", "kind":"edit_tool"},
            {"label": f"Approach distance (mm): {current_approach:.1f}  [Edit]", "kind":"edit_approach"},
            {"label": "Go: Default", "kind":"goto", "pose":"Default"},
            {"label": "Go: Reload",  "kind":"goto", "pose":"Reload"},
            {"label": "Record current as new pose [Add]", "kind":"add_pose"},
            {"label": "Save current as Reload + auto Midway  (X button)", "kind":"save_reload_now"},
            {"label": "— Saved poses —", "kind":"header"},
        ]
        for name in sorted(visible_pose_names(poses)):
            if name in ["Default", "Reload"]:
                continue
            items.append({"label": f"Go: {name}", "kind":"goto", "pose": name})
        items.append({"label": "Stop Motors and Engage Brakes", "kind":"stop"})
        items.append({"label": "Kill all Motors",               "kind":"kill"})
        return items

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                rt.stop(); rt.join()
                pipeline.stop()
                if ser: ser.close()
                pygame.quit(); return

            # Editors
            if toollen_input_active and ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    try:
                        new_lz = float(toollen_text.strip())
                        with state.lock:
                            state.tool_lz = new_lz
                            state.settings["tool_lz"] = new_lz
                        save_settings(state.settings)
                        rt.enqueue({"type":"set_tool_lz","lz":new_lz})
                        try: joy.rumble(0.5, 0.5, 120)
                        except Exception: pass
                    except ValueError:
                        print("⚠️ Invalid tool length.")
                    toollen_input_active = False
                elif ev.key == pygame.K_ESCAPE:
                    toollen_input_active = False
                elif ev.key == pygame.K_BACKSPACE:
                    toollen_text = toollen_text[:-1]
                else:
                    ch = ev.unicode
                    if ch in "0123456789.-": toollen_text += ch

            if approach_input_active and ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    try:
                        new_d = float(approach_text.strip())
                        with state.lock:
                            state.approach_mm = new_d
                            state.settings["approach_mm"] = new_d
                        save_settings(state.settings)
                        rt.enqueue({"type":"set_approach_mm","dist":new_d})
                        try: joy.rumble(0.5, 0.5, 120)
                        except Exception: pass
                    except ValueError:
                        print("⚠️ Invalid approach distance.")
                    approach_input_active = False
                elif ev.key == pygame.K_ESCAPE:
                    approach_input_active = False
                elif ev.key == pygame.K_BACKSPACE:
                    approach_text = approach_text[:-1]
                else:
                    ch = ev.unicode
                    if ch in "0123456789.-": approach_text += ch

            if naming_input_active and ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    name = naming_text.strip()
                    naming_input_active = False
                    if name:
                        rt.enqueue({"type":"save_named_with_midway","name": name})
                        try: joy.rumble(0.6, 0.6, 140)
                        except Exception: pass
                elif ev.key == pygame.K_ESCAPE:
                    naming_input_active = False
                elif ev.key == pygame.K_BACKSPACE:
                    naming_text = naming_text[:-1]
                else:
                    ch = ev.unicode
                    if ch and (ch.isalnum() or ch in "_- "):
                        naming_text += ch

        # Read joystick to state (and use mapping) — ONLY in main thread
        raw_hx, raw_hy = joy.get_hat(0)
        lb, rb = joy.get_button(4), joy.get_button(5)
        with state.lock:
            state.lx, state.ly = _axis(joy, LS_X_AXIS, INVERT_LS_X), _axis(joy, LS_Y_AXIS, INVERT_LS_Y)
            state.rx, state.ry = _axis(joy, RS_X_AXIS, INVERT_RS_X), _axis(joy, RS_Y_AXIS, INVERT_RS_Y)
            # D‑pad: no global L/R swap; inversion is handled in robot thread (now for ALL poses)
            state.hx, state.hy = raw_hx, raw_hy
            state.lb, state.rb = lb, rb
            waiting = state.await_confirm
            confirm_msg = state.confirm_msg

        # Button edges (A/B/X)
        a_now = bool(joy.get_button(A_BUTTON))
        if a_now and not prev_a:
            rt.enqueue({"type":"confirm_or_advance"})
            try: joy.rumble(0.7, 0.7, 160)
            except Exception: pass
        prev_a = a_now

        b_now = bool(joy.get_button(B_BUTTON))
        if b_now and not prev_b:
            rt.enqueue({"type":"midway_free_control"})
            try: joy.rumble(0.3, 0.3, 120)
            except Exception: pass
        prev_b = b_now

        x_now = bool(joy.get_button(X_BUTTON))
        if x_now and not prev_x:
            rt.enqueue({"type":"save_reload_with_midway"})
            try: joy.rumble(0.55, 0.55, 140)
            except Exception: pass
        prev_x = x_now

        # Trigger axis guard source (for UI click suppression)
        cur_lt = joy.get_axis(LEFT_AXIS); cur_rt = joy.get_axis(RIGHT_AXIS)
        if abs(cur_lt - prev_lt) > AXIS_GUARD_DELTA or abs(cur_rt - prev_rt) > AXIS_GUARD_DELTA:
            last_sig_axis_time = time.time()
        prev_lt, prev_rt = cur_lt, cur_rt

        # Sensitivity via LB/RB
        if lb and not prev_lb and state.idx < len(state.levels)-1: state.idx += 1
        if rb and not prev_rb and state.idx > 0:                    state.idx -= 1
        if (lb and not prev_lb) or (rb and not prev_rb):
            try: joy.rumble(0.3, 0.3, 200)
            except Exception: pass
        prev_lb, prev_rb = lb, rb
        prev_idx = state.idx

        # Arduino trigger bridge (swapped semantics)
        if ser:
            rt_val = normalize(joy.get_axis(RIGHT_AXIS)); lt_val = normalize(joy.get_axis(LEFT_AXIS))
            if lt_val > DEADZONE:   rate = int(lt_val * MAX_RATE)
            elif rt_val > DEADZONE: rate = -int(rt_val * MAX_RATE)
            else:                   rate = 0
            if rate != last_rate:
                ser.write(f"V{rate}\n".encode()); last_rate = rate
                if ser.in_waiting: ser.read(ser.in_waiting)
            if joy.get_button(RESET_BUTTON):
                ser.write(b"R\n")
                if ser.in_waiting: ser.read(ser.in_waiting)
                last_rate = 0; time.sleep(0.1)

        # Camera frame
        try:
            frames = pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            frame = np.asanyarray(color_frame.get_data())
        except RuntimeError:
            continue

        # Draw feed
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        screen.blit(surf, (0, 0))

        # HUD reticle
        cx, cy = w//2, h//2; col = (0, 255, 127)
        pygame.draw.circle(screen, col, (cx, cy), 50, 2)
        for dx, dy in [(-70,0),(-20,0),(20,0),(70,0),(0,-70),(0,-20),(0,20),(0,70)]:
            ex, ey = cx+dx, cy+dy
            ix = cx + (np.sign(dx)*20 if dx else 0)
            iy = cy + (np.sign(dy)*20 if dy else 0)
            pygame.draw.line(screen, col, (ex, ey), (ix, iy), 2)

        # Panels
        with state.lock:
            p    = state.pitch
            j5v  = state.j5
            sens = state.levels[state.idx]
            tool_lz = state.tool_lz
            approach_mm = state.approach_mm
            poses_snapshot = state.poses.copy()

        # Pitch panel
        y1 = h - H_panel - m; x1 = m
        panel_size  = int(h * 0.12)
        diag_gap    = int(w * 0.06)
        padding     = 4
        label_font       = pygame.font.SysFont("Consolas", 14, bold=True)
        value_font       = pygame.font.SysFont("Consolas", 12)
        placeholder_font = pygame.font.SysFont("Consolas", 12)
        big_font         = pygame.font.SysFont("Consolas", 24, bold=True)
        label_h     = label_font.get_height()
        value_h     = value_font.get_height()
        H_panel     = panel_size + padding + label_h + padding + value_h + padding
        btn_w   = int(w * 0.25) - 2*m
        btn_gap = 8

        pan1 = pygame.Surface((panel_size, panel_size), pygame.SRCALPHA); pan1.fill((10,10,30,180))
        pivot = (5, panel_size//2); r_len = panel_size - pivot[0] - 5
        ang = np.radians(-p)
        ex = pivot[0] + r_len*np.cos(ang); ey = pivot[1] + r_len*np.sin(ang)
        pygame.draw.line(pan1, (0,255,127), pivot, (ex, ey), 2); pygame.draw.circle(pan1, (0,255,127), (int(ex), int(ey)), 4)
        screen.blit(pan1, (x1, y1))
        screen.blit(label_font.render("Pitch", True, (0,255,127)), (x1, y1 + panel_size + padding))
        screen.blit(value_font.render(f"{p:+.5f}°", True, (0,255,127)),
                    (x1, y1 + panel_size + padding + label_h + padding))

        # Angle (j5) panel
        x2 = x1 + panel_size + diag_gap; y2 = y1
        pan2 = pygame.Surface((panel_size, panel_size), pygame.SRCALPHA); pan2.fill((10,10,30,180))
        center = (panel_size//2, panel_size//2); radius = panel_size//2 - 5
        pygame.draw.circle(pan2, (0,255,127), center, radius, 2)
        ang2 = np.radians(j5v - 90)
        ex2 = center[0] + radius*np.cos(ang2); ey2 = center[1] + radius*np.sin(ang2)
        pygame.draw.line(pan2, (0,255,127), center, (ex2, ey2), 2)
        pygame.draw.polygon(pan2, (0,255,127), [
            (ex2 + 4*np.cos(ang2),    ey2 + 4*np.sin(ang2)),
            (ex2 + 4*np.cos(ang2+0.3),ey2 + 4*np.sin(ang2+0.3)),
            (ex2 + 4*np.cos(ang2-0.3),ey2 + 4*np.sin(ang2-0.3))
        ])
        screen.blit(pan2, (x2, y2))
        screen.blit(label_font.render("Angle", True, (0,255,127)), (x2, y2 + panel_size + padding))
        screen.blit(value_font.render(f"{-j5v:+.5f}°", True, (0,255,127)),
                    (x2, y2 + panel_size + padding + label_h + padding))

        # Settings panel
        texts = [
            value_font.render(f"Sensitivity ({sens*100:.0f}%)", True, (200,230,255)),
            value_font.render(f"Move step:  ±{(5.0*sens):.5f} mm", True, (0,255,127)),
            value_font.render(f"Angle step: ±{(5.0*sens):.5f}°", True, (0,255,127)),
            value_font.render(f"Pitch step: ±{(0.5*sens):.5f}°", True, (0,255,127)),
            value_font.render(f"Approach: {approach_mm:.2f} mm", True, (200,255,200)),
        ]
        sb_w = max(t.get_width() for t in texts) + 2*padding
        sb_h = sum(t.get_height() + padding for t in texts) + padding
        x3 = w - sb_w - m; y3 = m
        sbpan = pygame.Surface((sb_w, sb_h), pygame.SRCALPHA); sbpan.fill((30,30,10,180))
        y_off = padding
        for t in texts:
            sbpan.blit(t, (padding, y_off)); y_off += t.get_height() + padding
        screen.blit(sbpan, (x3, y3))

        # Side UI background
        ui_w = int(w * 0.25)
        pygame.draw.rect(screen, (20,20,20), (w, 0, ui_w, h))

        # Buttons (aligned clickables)
        buttons = make_buttons(tool_lz, approach_mm, poses_snapshot)
        button_rects = []
        button_items = []
        y_off_btn = m
        for b in buttons:
            if b["kind"] == "header":
                surf_txt = label_font.render(b["label"], True, (200,200,200))
                screen.blit(surf_txt, (w + m, y_off_btn))
                y_off_btn += surf_txt.get_height() + btn_gap
                continue
            label = b["label"]
            lines = wrap_text(label, placeholder_font, btn_w - 10)
            bh    = len(lines)*(placeholder_font.get_height()+2) + 10
            rect  = pygame.Rect(w + m, y_off_btn, btn_w, bh)
            pygame.draw.rect(screen, (40,40,60), rect)
            pygame.draw.rect(screen, (0,255,127), rect, 2)
            ly2 = y_off_btn + 5
            for ln in lines:
                surf_txt = placeholder_font.render(ln, True, (200,200,200))
                tx = w + m + (btn_w - surf_txt.get_width())//2
                screen.blit(surf_txt, (tx, ly2))
                ly2 += placeholder_font.get_height() + 2
            y_off_btn += bh + btn_gap
            button_rects.append(rect)
            button_items.append(b)

        # Handle mouse clicks now that rects exist
        if pygame.mouse.get_pressed()[0]:
            if (time.time() - last_sig_axis_time) >= AXIS_GUARD_WINDOW:
                mx, my = pygame.mouse.get_pos()
                for i, rect in enumerate(button_rects):
                    if rect.collidepoint(mx, my):
                        btn = button_items[i]
                        kind = btn.get("kind")
                        if kind == "edit_tool":
                            toollen_input_active = True; toollen_text = f"{state.tool_lz:.1f}"
                        elif kind == "edit_approach":
                            approach_input_active = True; approach_text = f"{state.approach_mm:.1f}"
                        elif kind == "goto":
                            rt.enqueue({"type":"goto","pose_name":btn["pose"]})
                            try: joy.rumble(0.4, 0.4, 120)
                            except Exception: pass
                        elif kind == "add_pose":
                            naming_input_active = True; naming_text = ""
                        elif kind == "save_reload_now":
                            rt.enqueue({"type":"save_reload_with_midway"})
                        elif kind == "stop":
                            rt.enqueue({"type":"stop_motors"})
                        elif kind == "kill":
                            rt.enqueue({"type":"kill_motors"})

        # Inline editors
        if toollen_input_active:
            box_w = btn_w; box_h = 60; box_x = w + m; box_y = y_off_btn + 10
            pygame.draw.rect(screen, (50,50,20), (box_x, box_y, box_w, box_h))
            pygame.draw.rect(screen, (0,255,127), (box_x, box_y, box_w, box_h), 2)
            prompt = placeholder_font.render("Enter tool length (mm), Enter=Apply, Esc=Cancel", True, (220,220,220))
            text_surf = placeholder_font.render(toollen_text, True, (255,255,255))
            screen.blit(prompt, (box_x + 6, box_y + 6)); screen.blit(text_surf, (box_x + 6, box_y + 30))

        if approach_input_active:
            box_w = btn_w; box_h = 60; box_x = w + m; box_y = y_off_btn + 10
            pygame.draw.rect(screen, (20,50,20), (box_x, box_y, box_w, box_h))
            pygame.draw.rect(screen, (0,255,127), (box_x, box_y, box_w, box_h), 2)
            prompt = placeholder_font.render("Enter approach distance (mm), Enter=Apply, Esc=Cancel", True, (220,220,220))
            text_surf = placeholder_font.render(approach_text, True, (255,255,255))
            screen.blit(prompt, (box_x + 6, box_y + 6)); screen.blit(text_surf, (box_x, box_y + 30))

        if naming_input_active:
            box_w = btn_w; box_h = 60; box_x = w + m; box_y = y_off_btn + 10
            pygame.draw.rect(screen, (35,35,60), (box_x, box_y, box_w, box_h))
            pygame.draw.rect(screen, (0,255,127), (box_x, box_y, box_w, box_h), 2)
            prompt = placeholder_font.render("Enter new pose name, Enter=Save, Esc=Cancel", True, (230,230,230))
            text_surf = placeholder_font.render(naming_text or "", True, (255,255,255))
            screen.blit(prompt, (box_x + 6, box_y + 6)); screen.blit(text_surf, (box_x + 6, box_y + 30))

        # Confirmation overlay (text only)
        with state.lock:
            waiting = state.await_confirm
            confirm_msg = state.confirm_msg
        if waiting:
            overlay = pygame.Surface((w + ui_w, h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            txt1 = big_font.render(confirm_msg or "Paused at Midway. A: advance  B: free control", True, (255,255,255))
            screen.blit(overlay, (0,0))
            screen.blit(txt1, ((w + ui_w - txt1.get_width())//2, h//2 - 24))

        # Frame
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
