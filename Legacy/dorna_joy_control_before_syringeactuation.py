#!/usr/bin/env python3
import sys
import argparse
import threading
import time
import os
import json
import queue
import re, glob
from dataclasses import dataclass

import pygame
import numpy as np
import cv2
# Quiet down OpenCV warnings (keep errors)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    try:
        cv2.utils.logging.setLogLevel(2)  # Fallback: ERROR
    except Exception:
        pass

import pyrealsense2 as rs
import serial
from dorna2 import Dorna

# ─────────────────────────────────────────────────────────────────────────────
#                              UVC CAMERA SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

def _read_text(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return ""

def _v4l_name_for_node(node):
    base = os.path.basename(node)  # videoN
    return _read_text(f"/sys/class/video4linux/{base}/name") or "unknown"

def _is_realsense_name(name):
    n = name.lower()
    return ("realsense" in n) or ("depth" in n) or ("infrared" in n)

def _usb_info_from_node(node: str) -> dict:
    """
    Return {'busnum','devpath','speed'} by walking sysfs
    from /sys/class/video4linux/videoN/device up to the USB device node.
    """
    base = os.path.basename(node)
    cur = os.path.realpath(f"/sys/class/video4linux/{base}/device")
    for _ in range(16):
        if os.path.exists(os.path.join(cur, "busnum")):
            return {
                "busnum": _read_text(os.path.join(cur, "busnum")) or "",
                "devpath": _read_text(os.path.join(cur, "devpath")) or "",
                "speed": _read_text(os.path.join(cur, "speed")) or "",
            }
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return {"busnum":"", "devpath":"", "speed":""}

def discover_uvc_index0(limit=4) -> list:
    """
    Discover non-RealSense by-path *video-index0* cameras.
    Return list of dicts: {'bypath','node','name','busnum','devpath'}.
    """
    found = []
    links = sorted(glob.glob("/dev/v4l/by-path/*video-index0"))
    for link in links:
        node = os.path.realpath(link)
        if not node.startswith("/dev/video"):
            continue
        name = _v4l_name_for_node(node)
        if _is_realsense_name(name):
            continue
        info = _usb_info_from_node(node)
        found.append({
            "bypath": link,
            "node": node,
            "name": name,
            "busnum": info["busnum"],
            "devpath": info["devpath"],
        })
    return found[:limit]

def find_uvc_devices(limit=2):
    """
    Prefer stable /dev/v4l/by-path links (deduplicated across usb-/usbv2-/usbv3-),
    pick index0 when available (else index1), and exclude RealSense.
    Returns up to 'limit' device paths.
    """
    links = glob.glob("/dev/v4l/by-path/*video-index[01]")

    def canon_base(p):  # collapse usbv2-/usbv3- to usb-
        b = re.sub(r"-video-index[01]$", "", p)
        return b.replace("/usbv2-", "/usb-").replace("/usbv3-", "/usb-")

    def variant_rank(p):  # prefer usb- over usbv2-/usbv3-
        if "/usb-" in p and "/usbv" not in p: return 0
        if "/usbv2-" in p: return 1
        if "/usbv3-" in p: return 2
        return 3

    buckets = {}  # base -> {"index0":[paths], "index1":[paths]}
    for link in links:
        base = canon_base(link)
        b = buckets.setdefault(base, {"index0": [], "index1": []})
        (b["index0"] if link.endswith("video-index0") else b["index1"]).append(link)

    out = []
    for base, b in buckets.items():
        i0 = sorted(b["index0"], key=variant_rank)
        i1 = sorted(b["index1"], key=variant_rank)
        pick = (i0[0] if i0 else (i1[0] if i1 else None))
        if not pick:
            continue
        node = os.path.realpath(pick)
        name = _v4l_name_for_node(node)
        if _is_realsense_name(name):
            continue
        out.append(pick)
        if len(out) >= limit:
            break
    return out

class UvcThread(threading.Thread):
    """
    Background frame grabber for a UVC camera using OpenCV.

    Resilient open:
      • Tries backends: ANY → V4L2   (ANY uses the string by-path, V4L2 uses numeric /dev/videoN)
      • Tries FourCCs:  YUYV → MJPG → (driver default)
      • Tries FPS:      requested → 15 → 10

    Converts to BGR and (optionally) rotates 180° (we set rotate180=False in this app
    and instead do all orientation in the GUI layer).
    """
    def __init__(self, device, width=640, height=480, fps=30, rotate180=False,
                 backend="v4l2", name="UVC", try_index1_fallback=False):
        super().__init__(daemon=True)
        self.device = str(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.rotate180 = bool(rotate180)
        self.backend = backend
        self.name = name
        self.try_index1_fallback = bool(try_index1_fallback)

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._opened = False
        self._status = "Initializing…"

    # ---- helpers for robust open ----
    def _decode_fourcc(self, v):
        try:
            v = int(v)
            return "".join([chr((v >> (8*i)) & 0xFF) for i in range(4)])
        except Exception:
            return "----"

    def _dev_arg_for_backend(self, dev_path, backend):
        """
        For V4L2, pass numeric /dev/videoN index to avoid 'by name' quirk.
        For ANY, keep the original string path (by-path).
        """
        if backend == "v4l2" and isinstance(dev_path, str):
            real = os.path.realpath(dev_path)
            m = re.match(r"^/dev/video(\d+)$", real)
            if m:
                return int(m.group(1))
        return dev_path  # keep string for CAP_ANY

    def _open_resilient_once(self, dev_path, width, height, fps, backend, fourcc, warmup_reads=30):
        api = cv2.CAP_V4L2 if backend == "v4l2" else cv2.CAP_ANY
        dev_arg = self._dev_arg_for_backend(dev_path, backend)
        cap = cv2.VideoCapture(dev_arg, api)
        if not cap.isOpened():
            return None, None

        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_CONVERT_RGB,  1)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        ok, frame = False, None
        for _ in range(warmup_reads):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.01)

        if not ok or frame is None:
            cap.release()
            return None, None

        # negotiated props
        w_eff = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or frame.shape[1]
        h_eff = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
        fourcc_eff = self._decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)) or (fourcc or "?")
        fps_eff = int(cap.get(cv2.CAP_PROP_FPS) or fps)
        meta = {"fourcc": fourcc_eff, "fps": fps_eff, "w": int(w_eff), "h": int(h_eff), "backend": backend}
        return cap, (frame, meta)

    def _open_resilient(self, dev_path):
        # Always try generic first, then V4L2 (reduces V4L2 warnings on some builds)
        backends = ["any", "v4l2"]
        fourccs  = ["YUYV", "MJPG", None]
        fps_try  = [self.fps] + [f for f in (15, 10) if f != self.fps]

        for b in backends:
            for fc in fourccs:
                for f in fps_try:
                    cap, fm = self._open_resilient_once(dev_path, self.width, self.height, f, b, fc)
                    if cap is not None:
                        frame, meta = fm
                        return cap, frame, meta
        return None, None, None

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def status(self):
        return self._status

    def stop(self):
        self._stop.set()

    def run(self):
        # Try provided device; optionally try sibling video-index1
        paths_to_try = [self.device]
        if self.try_index1_fallback and isinstance(self.device, str) and self.device.endswith("video-index0"):
            alt = re.sub(r"video-index0$", "video-index1", self.device)
            if os.path.exists(alt):
                paths_to_try.append(alt)

        cap = None
        chosen_path = None
        warm_frame = None
        meta = None

        try:
            for path in paths_to_try:
                cap, warm_frame, meta = self._open_resilient(path)
                if cap is not None:
                    chosen_path = path
                    break

            if cap is None:
                self._status = (f"{self.name}: no frames on " +
                                ", ".join(os.path.basename(p) for p in paths_to_try) +
                                " (ensure uvcvideo quirks=0 and device not in use)")
                return

            self._opened = True
            if self.rotate180 and warm_frame is not None:
                warm_frame = cv2.rotate(warm_frame, cv2.ROTATE_180)
            with self._lock:
                self._frame = warm_frame

            self._status = (f"{self.name}: {os.path.basename(chosen_path)} "
                            f"{meta['w']}x{meta['h']}@{meta['fps']} {meta['fourcc']} via {meta['backend'].upper()}")

            idle_sleep = 0.001
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                if self.rotate180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                with self._lock:
                    self._frame = frame
                time.sleep(idle_sleep)

        finally:
            if cap is not None:
                try: cap.release()
                except Exception: pass
            self._opened = False

# ─────────────────────────────────────────────────────────────────────────────
#                           REALSENSE CAMERA SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

class RealSenseThread(threading.Thread):
    """
    Background RealSense frame grabber. Tries COLOR first; if not available,
    falls back to INFRARED, then DEPTH+COLORIZER. Exposes latest() as BGR image.
    """
    def __init__(self, width=640, height=480, fps=30):
        super().__init__(daemon=True)
        self.req_w = int(width)
        self.req_h = int(height)
        self.req_fps = int(fps)

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._status = "Initializing…"

        self.pipeline = None
        self.mode = None            # "color" | "infrared" | "depth"
        self.colorizer = None

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def status(self):
        return self._status

    def stop(self):
        self._stop.set()

    def _start_with(self, kind):
        """
        kind: "color" | "infrared" | "depth"
        Returns True on success.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        try:
            if kind == "color":
                config.enable_stream(rs.stream.color, self.req_w, self.req_h, rs.format.bgr8, self.req_fps)
            elif kind == "infrared":
                config.enable_stream(rs.stream.infrared, self.req_w, self.req_h, rs.format.y8, self.req_fps)
            elif kind == "depth":
                config.enable_stream(rs.stream.depth, self.req_w, self.req_h, rs.format.z16, self.req_fps)
                self.colorizer = rs.colorizer()
            else:
                return False

            self.pipeline.start(config)
            self.mode = kind
            self._status = f"RealSense: {kind} {self.req_w}x{self.req_h}@{self.req_fps}"
            # Warm-up
            self.pipeline.wait_for_frames()
            return True
        except Exception as e:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self._status = f"RealSense: {kind} start failed: {e}"
            return False

    def run(self):
        # Try in order: color → infrared → depth
        if not self._start_with("color"):
            if not self._start_with("infrared"):
                if not self._start_with("depth"):
                    self._status = "RealSense: no usable stream"
                    return

        try:
            while not self._stop.is_set():
                frames = self.pipeline.poll_for_frames()
                if not frames:
                    try:
                        frames = self.pipeline.wait_for_frames(timeout_ms=200)
                    except Exception:
                        continue

                img_bgr = None
                if self.mode == "color":
                    cf = frames.get_color_frame()
                    if cf:
                        img_bgr = np.asanyarray(cf.get_data())
                elif self.mode == "infrared":
                    ir = frames.get_infrared_frame()
                    if ir:
                        y = np.asanyarray(ir.get_data())
                        img_bgr = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
                elif self.mode == "depth":
                    df = frames.get_depth_frame()
                    if df and self.colorizer:
                        c = self.colorizer.process(df)
                        img_bgr = np.asanyarray(c.get_data())
                        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

                if img_bgr is not None:
                    with self._lock:
                        self._frame = img_bgr
                else:
                    time.sleep(0.005)

        finally:
            try:
                if self.pipeline:
                    self.pipeline.stop()
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────────────────────
#                 ROBOT + UI (Asymmetric TL+TR; Equal bottom UVC) + ROUTINE
# ─────────────────────────────────────────────────────────────────────────────

# ─── Trigger → Serial (Arduino) configuration ───────────────────────────
SERIAL_PORT    = '/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44236313735351100201-if00'
BAUDRATE       = 115200
MAX_RATE       = 1600    # allow faster ceiling for wash
DEADZONE       = 0.02
RIGHT_AXIS     = 5       # Xbox right trigger axis
LEFT_AXIS      = 2       # Xbox left  trigger axis
RESET_BUTTON   = 2

# ── Controller axis mapping (tweak if your driver maps differently) ─────
LS_X_AXIS = 0
LS_Y_AXIS = 1
RS_X_AXIS = 3
RS_Y_AXIS = 4
INVERT_LS_X = False
INVERT_LS_Y = False
INVERT_RS_X = False
INVERT_RS_Y = False

# Optional: sign for D-pad Up/Down movement (+1 normal, −1 to flip)
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
MIDWAY_SUFFIX  = "__midway"

DEFAULT_TOOL_LZ     = 205.0
DEFAULT_APPROACH_MM = 25.0

DEFAULT_POSES = {
    "Reload":  {"j0": 7.71, "j1": 80.86, "j2": -100.00, "j3": -0.07, "j4": -70.60, "j5": 6.35},
    "Default": {"j0": 39.24,"j1": 27.00, "j2": -100.44, "j3": 6.35,  "j4": 65.19,  "j5": -2.70},
}
RESERVED_POSES = {"Default", "Reload"}  # built-in reserved; non-toggleable

def midway_name(name: str) -> str:
    return f"{name}{MIDWAY_SUFFIX}"

def load_poses(path=POSES_PATH):
    try:
        data = json.load(open(path)) if os.path.exists(path) else {}
        for name, pose in DEFAULT_POSES.items():
            data.setdefault(name, pose)
        for name, pose in list(data.items()):
            if not isinstance(pose, dict): continue
            for k in ["j0","j1","j2","j3","j4","j5"]:
                if k in pose: data[name][k] = float(pose[k])
        return data
    except Exception as e:
        print(f"⚠️ Using built-in poses (could not load {path}: {e})")
        return {k: v.copy() for k, v in DEFAULT_POSES.items()}

def load_settings(path=SETTINGS_PATH):
    try:
        data = json.load(open(path)) if os.path.exists(path) else {}
        tool_lz     = float(data.get("tool_lz", DEFAULT_TOOL_LZ))
        approach_mm = float(data.get("approach_mm", DEFAULT_APPROACH_MM))
        end_kw = data.get("endstop_keywords", ["ENDSTOP", "LIMIT", "LIM", "HIT", "END"])
        end_timeout_ms    = int(data.get("endstop_timeout_ms", 0))
        plunger_fwd_rate  = int(data.get("plunger_fwd_rate", MAX_RATE))
        plunger_bwd_rate  = int(data.get("plunger_bwd_rate", MAX_RATE))
        # Default wash wait time now 1.0 s
        pose_final_wait_s = float(data.get("pose_final_wait_s", 1.0))
        plunger_dir_sign  = int(data.get("plunger_dir_sign", 1))
        swap_endstops     = bool(data.get("swap_endstops", False))

        # Which endstop is considered "expel" (final plunger position after WASH).
        expel_endstop = str(data.get("expel_endstop", "")).upper()
        if not expel_endstop:
            # For your rig, BWD is the "fully ejected" side.
            expel_endstop = "BWD"
        if expel_endstop not in ("FWD", "BWD"):
            expel_endstop = "BWD"

        # User-reserved poses (in addition to built-ins like Default/Reload)
        reserved_poses = data.get("reserved_poses", [])
        if not isinstance(reserved_poses, list):
            reserved_poses = []
        reserved_poses = [str(n) for n in reserved_poses]

        # Per-UVC orientation (0,1,2,3) * 90 degrees, + flips
        uvc1_rot     = int(data.get("uvc1_rot", 2))   # default 180° to match legacy
        uvc1_flip_h  = bool(data.get("uvc1_flip_h", False))
        uvc1_flip_v  = bool(data.get("uvc1_flip_v", False))
        uvc2_rot     = int(data.get("uvc2_rot", 2))
        uvc2_flip_h  = bool(data.get("uvc2_flip_h", False))
        uvc2_flip_v  = bool(data.get("uvc2_flip_v", False))

        return {
            "tool_lz": tool_lz,
            "approach_mm": approach_mm,
            "endstop_keywords": end_kw,
            "endstop_timeout_ms": end_timeout_ms,
            "plunger_fwd_rate": plunger_fwd_rate,
            "plunger_bwd_rate": plunger_bwd_rate,
            "pose_final_wait_s": pose_final_wait_s,
            "plunger_dir_sign": plunger_dir_sign,
            "swap_endstops": swap_endstops,
            "expel_endstop": expel_endstop,
            "reserved_poses": reserved_poses,
            "uvc1_rot": uvc1_rot,
            "uvc1_flip_h": uvc1_flip_h,
            "uvc1_flip_v": uvc1_flip_v,
            "uvc2_rot": uvc2_rot,
            "uvc2_flip_h": uvc2_flip_h,
            "uvc2_flip_v": uvc2_flip_v,
        }
    except Exception as e:
        print(f"⚠️ Using default settings (could not load {path}: {e})")
        return {
            "tool_lz": DEFAULT_TOOL_LZ,
            "approach_mm": DEFAULT_APPROACH_MM,
            "endstop_keywords": ["ENDSTOP", "LIMIT", "LIM", "HIT", "END"],
            "endstop_timeout_ms": 0,
            "plunger_fwd_rate": MAX_RATE,
            "plunger_bwd_rate": MAX_RATE,
            "pose_final_wait_s": 1.0,   # default wash wait time
            "plunger_dir_sign": 1,
            "swap_endstops": False,
            "expel_endstop": "BWD",
            "reserved_poses": [],
            "uvc1_rot": 2,
            "uvc1_flip_h": False,
            "uvc1_flip_v": False,
            "uvc2_rot": 2,
            "uvc2_flip_h": False,
            "uvc2_flip_v": False,
        }

def save_settings(settings, path=SETTINGS_PATH):
    try:
        with open(path, "w") as f: json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save settings to {path}: {e}")

def save_poses(poses, path=POSES_PATH):
    try:
        with open(path, "w") as f: json.dump(poses, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save poses to {path}: {e}")

def normalize(v):
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

# ─── Routine system ──────────────────────────────────────────────────────
ROUTINE_SAMPLE = """\
# Example routine
# Keywords:
#   POSE <name>           → move to <name>__midway and wait for ADVANCE step
#   ADVANCE_WAIT          → user presses A to advance to final pose; B = free control at __midway
#   ADVANCE_AUTO          → auto advance from midway to final pose
#   WASH N                → repeat plunger BACKWARD→endstop then FORWARD→endstop N times
# Notes:
#   • Plunger NEVER actuates until the tool is at the final pose and settled (≥0.35 s).
#   • After cycles, the plunger finishes at your configured **Final expel endstop**
#     from settings.json / UI (default: BWD on this build).
#
POSE Reload
ADVANCE_WAIT   # press A to enter Reload (final); B keeps you at __midway, routine will wait for final before WASH
WASH 3
"""

# Routine persistence
ROUTINE_PATH = "routine.txt"

def load_routine_text(path=ROUTINE_PATH):
    """Load routine script from disk; fall back to ROUTINE_SAMPLE on first run."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"⚠️ Could not load routine from {path}: {e}")
    return ROUTINE_SAMPLE.strip()

def save_routine_text(text: str, path=ROUTINE_PATH):
    """Persist the current routine script to disk."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"⚠️ Could not save routine to {path}: {e}")

@dataclass
class EditorState:
    text: str
    caret: int = 0      # cursor index in text
    scroll: int = 0     # top line index for view

class RoutineExecutor(threading.Thread):
    """
    Executes scripted steps with strict final‑pose gating before any plunger motion.
    """
    def __init__(self, state, robot_thread, ser, script_text,
                 endstop_keywords, endstop_timeout_ms,
                 plunger_rate_fwd, plunger_rate_bwd,
                 pose_final_wait_s=0.0,
                 expel_endstop="FWD"):
        super().__init__(daemon=True)
        self.state = state
        self.rt = robot_thread
        self.ser = ser
        self.text = script_text
        self.stop_event = threading.Event()
        self.phase = "Idle"
        self.rate_fwd = int(max(0, plunger_rate_fwd))
        self.rate_bwd = int(max(0, plunger_rate_bwd))
        self.endstop_keywords = [str(k).upper() for k in (endstop_keywords or [])]
        self.endstop_timeout_ms = int(endstop_timeout_ms)
        self.pose_final_wait_s = float(pose_final_wait_s)
        self.expel_endstop = "FWD" if str(expel_endstop).upper() not in ("FWD","BWD") else str(expel_endstop).upper()

        self._last_endstop_state = {"L": None, "R": None}

    # ── HUD helper
    def _set_status_line(self, msg):
        print(f"[Routine] {msg}")
        with self.state.lock:
            self.state.routine_status_text = msg

    def stop(self): self.stop_event.set()

    # ---- Serial helpers --------------------------------------------------
    def _serial_readlines_nonblock(self, timeout_s=0.05, debug_prefix=None):
        lines = []
        if not self.ser:
            return lines
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                if self.ser.in_waiting:
                    raw = self.ser.read(self.ser.in_waiting).decode(errors="ignore")
                    for ln in raw.splitlines():
                        s = ln.strip()
                        if not s:
                            continue
                        lines.append(s)
                        if debug_prefix:
                            print(f"[Routine][SER][{debug_prefix}] {s}")
                        with self.state.lock:
                            self.state.last_endstop_line = s
                else:
                    time.sleep(0.005)
            except Exception:
                break
        return lines

    def _endstop_hit_in_lines(self, lines, prev_state, direction=None):
        """Detect an endstop activation; tolerant across formats."""
        with self.state.lock:
            swap_lr = bool(self.state.settings.get("swap_endstops", False))

        def maybe_transition(left, right, src_line):
            prev_L = prev_state.get("L"); prev_R = prev_state.get("R")
            if prev_L is None and prev_R is None:
                prev_state["L"] = left; prev_state["R"] = right
                if direction == "BWD" and left:
                    self._set_status_line(f"Endstop active at start (L): {src_line}")
                    return True
                if direction == "FWD" and right:
                    self._set_status_line(f"Endstop active at start (R): {src_line}")
                    return True
                return False

            changed_L = (prev_L in (False, 0)) and left
            changed_R = (prev_R in (False, 0)) and right
            prev_state["L"] = left; prev_state["R"] = right

            if direction == "BWD" and changed_L: self._set_status_line(f"Endstop ACT (L): {src_line}"); return True
            if direction == "FWD" and changed_R: self._set_status_line(f"Endstop ACT (R): {src_line}"); return True
            if direction is None and (changed_L or changed_R):
                which = []
                if changed_L: which.append("L")
                if changed_R: which.append("R")
                self._set_status_line(f"Endstop ACT ({'/'.join(which)}): {src_line}")
                return True
            return False

        for ln in lines:
            up = ln.upper()

            # ACT L/R
            if "ACT" in up and "L=" in up and "R=" in up:
                m = re.search(r"ACT[^|]*\bL\s*=\s*([01]).*\bR\s*=\s*([01])", up)
                if m:
                    left  = (m.group(1) == "1")
                    right = (m.group(2) == "1")
                    if swap_lr: left, right = right, left
                    if maybe_transition(left, right, ln): return True

            # EL/ER
            if "EL=" in up and "ER=" in up:
                mL = re.search(r'\bEL\s*=\s*([01])', up)
                mR = re.search(r'\bER\s*=\s*([01])', up)
                if mL and mR:
                    left  = (mL.group(1) == "1")
                    right = (mR.group(1) == "1")
                    if swap_lr: left, right = right, left
                    if maybe_transition(left, right, ln): return True

            # keywords
            for k in self.endstop_keywords:
                k = (k or "").upper().strip()
                if k and k in up:
                    self._set_status_line(f"Endstop keyword '{k}' seen in: {ln}")
                    return True

        return False

    def _plunger_velocity(self, rate):
        if not self.ser: return
        try:
            self.ser.write(f"V{int(rate)}\n".encode())
            with self.state.lock:
                self.state.last_plunger_note = f"V{int(rate)} sent"
        except Exception:
            pass

    def _plunger_stop(self):
        if not self.ser: return
        try:
            self.ser.write(b"V0\n")
            with self.state.lock:
                self.state.last_plunger_note = "V0 (stop) sent"
        except Exception:
            pass

    def _plunger_to_end(self, direction: str) -> bool:
        """Drive to the mechanical endstop in 'FWD' or 'BWD' (strictly after final‑pose gating)."""
        direction = direction.upper()
        sign = int(self.state.settings.get("plunger_dir_sign", 1))
        rate = (self.rate_fwd if direction == "FWD" else -self.rate_bwd) * sign
        if rate == 0:
            self._set_status_line(f"⚠️ Plunger rate {direction} is 0; skipping.")
            return False

        self._set_status_line(f"Plunger {direction} → endstop @ {rate} steps/s")
        self._plunger_velocity(rate)

        # Clear any stale serial
        try:
            if self.ser and self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
        except Exception:
            pass

        t0 = time.time()
        deadline = None if self.endstop_timeout_ms <= 0 else (t0 + self.endstop_timeout_ms/1000.0)
        hit = False

        prev_state = {"L": None, "R": None}
        self._last_endstop_state = prev_state

        try:
            while not self.stop_event.is_set():
                try:
                    if self.ser: self.ser.write(b"E\n")
                except Exception:
                    pass

                lines = self._serial_readlines_nonblock(0.05, debug_prefix=direction)
                if self._endstop_hit_in_lines(lines, prev_state, direction=direction):
                    hit = True
                    break

                if deadline is not None and time.time() >= deadline:
                    break
                time.sleep(0.02)
        finally:
            self._plunger_stop()

        if not hit:
            if deadline is not None:
                self._set_status_line(
                    f"⚠️ Endstop not detected before timeout ({self.endstop_timeout_ms} ms)."
                )
            else:
                self._set_status_line(
                    "⚠️ Endstop not detected (no timeout). Check Arduino endstop output."
                )
        return hit

    # ---- Final-pose gating ------------------------------------------------
    def _wait_until_final_pose(self, expected_final_name=None, timeout=None):
        """
        Block until:
          • not at __midway and no confirm prompt
          • AND any robot motion window has elapsed (pose_motion_until)
          • THEN dwell ≥ max(0.35 s, pose_final_wait_s)
        """
        min_settle = 0.35   # guaranteed dwell
        self._set_status_line(f"Waiting for final pose '{expected_final_name or '<any>'}' before plunger…")
        t0 = time.time()
        while not self.stop_event.is_set():
            with self.state.lock:
                waiting = bool(self.state.await_confirm)
                cur     = self.state.current_pose_name or ""
                at_mid  = cur.endswith(MIDWAY_SUFFIX)
                motion_until = float(getattr(self.state, "pose_motion_until", 0.0))
            ready_pose = (not waiting) and (not at_mid) and (not expected_final_name or cur == expected_final_name)
            after_motion = (time.time() >= motion_until)
            if ready_pose and after_motion:
                break
            if timeout is not None and (time.time() - t0) > timeout:
                self._set_status_line("⚠️ Timed out waiting for final pose; continuing to wait anyway.")
            time.sleep(0.02)

        with self.state.lock:
            custom = float(self.state.settings.get("pose_final_wait_s", self.pose_final_wait_s))
        dwell = max(min_settle, custom)
        if dwell > 0:
            self._set_status_line(f"Settling at final pose for {dwell:.2f}s…")
            t1 = time.time()
            while not self.stop_event.is_set() and (time.time() - t1) < dwell:
                time.sleep(0.02)

    # ---- Script parsing ---------------------------------------------------
    def _parse(self):
        steps = []
        for raw in self.text.splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line: continue
            m = re.match(r'^(POSE|GO)\s+(.+)$', line, re.I)
            if m: steps.append(("pose", m.group(2).strip())); continue
            if re.match(r'^(ADVANCE_WAIT|WAIT_ADVANCE)$', line, re.I): steps.append(("advance_wait", None)); continue
            if re.match(r'^(ADVANCE|ADVANCE_AUTO|GO_ADVANCE)$', line, re.I): steps.append(("advance_auto", None)); continue
            m = re.match(r'^WASH\s+(\d+)$', line, re.I)
            if m: steps.append(("wash", int(m.group(1)))); continue
            m = re.match(r'^PLUNGER\s+(FWD|BWD)$', line, re.I)
            if m: steps.append(("plunger_end", m.group(1).upper())); continue
            steps.append(("echo", f"Unrecognized: {raw.strip()}"))
        return steps

    # ---- Main run ---------------------------------------------------------
    def run(self):
        self.phase = "RUNNING"
        with self.state.lock: self.state.routine_phase = self.phase
        steps = self._parse()
        last_pose = None

        for typ, arg in steps:
            if self.stop_event.is_set():
                break

            if typ == "echo":
                self._set_status_line(arg)
                continue

            if typ == "pose":
                name = arg
                last_pose = name
                self._set_status_line(f"POSE {name}")
                self.rt.enqueue({"type":"goto","pose_name": name})
                # Wait until the midway prompt is actually visible
                t0 = time.time()
                while not self.stop_event.is_set() and (time.time() - t0) < 20:
                    with self.state.lock:
                        if self.state.await_confirm:
                            break
                    time.sleep(0.02)

            elif typ == "advance_wait":
                self._set_status_line("ADVANCE_WAIT (A = advance; B = free control at __midway)")
                # Wait indefinitely for the prompt to be cleared by user
                while not self.stop_event.is_set():
                    with self.state.lock:
                        if not self.state.await_confirm:
                            break
                    time.sleep(0.02)
                # Require *final* pose before continuing
                self._wait_until_final_pose(last_pose)

            elif typ == "advance_auto":
                self._set_status_line("ADVANCE_AUTO")
                self.rt.enqueue({"type":"confirm_or_advance"})
                # Require final pose + settle
                self._wait_until_final_pose(last_pose)

            elif typ == "wash":
                n = int(arg)
                # STRICT final‑pose gating (never auto‑advance inside wash)
                self._wait_until_final_pose(last_pose)

                self._set_status_line(f"WASH x{n} cycles (BWD→END then FWD→END; finish at {self.expel_endstop})")
                for i in range(n):
                    if self.stop_event.is_set(): break
                    self._set_status_line(f"wash {i+1}/{n}: BWD→END")
                    self._plunger_to_end("BWD")
                    if self.stop_event.is_set(): break
                    self._set_status_line(f"wash {i+1}/{n}: FWD→END")
                    self._plunger_to_end("FWD")

                # Final finishing pass to configured "expel" endstop
                self._set_status_line(f"Final stroke to {self.expel_endstop} endstop (fully expelled)")
                self._plunger_to_end(self.expel_endstop)
                time.sleep(0.2)

            elif typ == "plunger_end":
                dirn = arg
                # STRICT final‑pose gating before any standalone plunger motion
                self._wait_until_final_pose(last_pose)
                self._set_status_line(f"PLUNGER {dirn} → endstop")
                self._plunger_to_end(dirn)
                time.sleep(0.1)

            time.sleep(0.03)

        self.phase = "DONE"
        with self.state.lock: self.state.routine_phase = self.phase
        self._set_status_line(f"Routine DONE (ended at {self.expel_endstop} endstop)")

# ─── Args ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Joystick control for Dorna + RealSense + UVC microscopes (asymmetric quadrants UI + routine)")
    p.add_argument("--host", "-H", default="10.42.0.11", help="IP address of your Dorna controller")
    p.add_argument("--port", "-P", type=int, default=443, help="Port for Dorna control (usually 443)")

    # UVC options
    p.add_argument("--uvc1", default="", help="UVC camera #1 device path (default: auto-detect)")
    p.add_argument("--uvc2", default="", help="UVC camera #2 device path (default: auto-detect 2nd, else placeholder)")
    p.add_argument("--uvc-fps", type=int, default=30, help="FPS for UVCs")
    p.add_argument("--uvc-rotate", action="store_true", default=True, help="(legacy) rotate UVC cameras 180° (now overridden by GUI settings)")
    p.add_argument("--uvc-no-rotate", dest="uvc_rotate", action="store_false", help="Disable legacy 180° rotation (GUI rotation still applies)")
    p.add_argument("--uvc-try-index1", action="store_true",
                   help="Also try the sibling video-index1 node if index0 yields no frames (default: off)")

    # Windowing
    p.add_argument("--fullscreen", action="store_true", help="Start in fullscreen (toggle with F11)")
    p.add_argument("--ui-min-width", type=int, default=320, help="Minimum UI panel width (pixels)")
    p.add_argument("--ui-frac", type=float, default=0.28, help="UI width fraction of window (0..1). Left column gets the rest.")
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
        for k, v in DEFAULT_POSES.items():
            self.poses.setdefault(k, v.copy())
        self.settings = load_settings()
        self.tool_lz = float(self.settings.get("tool_lz", DEFAULT_TOOL_LZ))
        self.approach_mm = float(self.settings.get("approach_mm", DEFAULT_APPROACH_MM))
        # Midway confirm gate
        self.await_confirm = False
        self.await_target  = None
        self.confirm_msg   = ""
        # Current pose tracker ('' | '<name>' | '<name>__midway')
        self.current_pose_name = ""
        # Routine HUD fields
        self.routine_phase       = "Idle"
        self.routine_status_text = "Idle"
        self.last_endstop_line   = ""
        self.last_plunger_note   = ""
        # motion gating for plunger
        self.pose_motion_until   = 0.0

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

        self.x0 = self.y0 = self.z0 = 0.0
        self.R  = np.eye(3)
        self.j5v = 0.0

        self.current_named     = None
        self.skip_manual_until = 0.0

        self.BSX, self.BSJ5 = 5.0, 5.0
        self.BSB, self.BSC  = 0.5, 0.5
        self.BSH            = 5.0
        self.VT, self.VR    = 10.0, 5.0
        self.VR_POSE        = self.VR * 2.0
        self.DZ             = 0.1

    def _set_current_named(self, name: str | None):
        self.current_named = name
        with self.state.lock:
            self.state.current_pose_name = name or ""

    def enqueue(self, cmd: dict): self.cmd_q.put(cmd)

    def _refresh_from_robot(self):
        pr = self.robot.get_all_pose()[:6]
        self.x0, self.y0, self.z0, a, b, c = pr
        self.R = axis_angle_to_R(a, b, c)
        tz = self.R[:,2]
        pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
        with self.state.lock: self.state.pitch = pitch

    # helper to hold off plunger during robot moves
    def _mark_motion_for(self, seconds: float):
        with self.state.lock:
            self.state.pose_motion_until = max(self.state.pose_motion_until, time.time() + max(0.0, seconds))

    def _queue_jmove_to_pose(self, pose: dict):
        go = {"cmd":"jmove","rel":0,"vel":self.VR_POSE}; go.update(pose)
        self.robot.play_dict(go)
        self._mark_motion_for(0.6)  # conservative window after a pose jump
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
        # mark motion window based on commanded displacement
        seconds = abs(dist_mm) / max(1e-6, self.VT) + 0.15
        self._mark_motion_for(seconds)

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
            self._set_current_named(name)
            print(f"[Robot] Returned to '{name}'.")
        return True

    def _ensure_current_midway(self, approach_mm: float):
        """
        Ensure we are at a safe __midway before moving away from current final.
        EXCEPTION: If current is 'Default', do nothing (no retract required).
        """
        if not self.current_named: return
        name = self.current_named
        if name == "Default": return
        if name.endswith(MIDWAY_SUFFIX): return
        cur_mid = midway_name(name)
        with self.state.lock:
            poses = self.state.poses
        if cur_mid in poses:
            self._tool_move_along_tz(-approach_mm, cont=0)
            self._set_current_named(cur_mid)
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
        self._set_current_named(cur_mid)

    def _advance_if_midway(self, approach_mm: float):
        """Advance along tool-Z from the current __midway to its final, if applicable."""
        if self.current_named and str(self.current_named).endswith(MIDWAY_SUFFIX):
            final = self.current_named[: -len(MIDWAY_SUFFIX)]
            try:
                self._tool_move_along_tz(+approach_mm, cont=0)
                self._set_current_named(final)
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

        # Home to Default — and STOP
        try:
            with self.state.lock: default_pose = self.state.poses["Default"].copy()
            self._queue_jmove_to_pose(default_pose)
            self._set_current_named("Default")
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

                        if target == "Default":
                            self._ensure_current_midway(approach_mm)
                            try:
                                self._queue_jmove_to_pose(poses["Default"])
                                self._set_current_named("Default")
                                print("[Robot] Moved to Default (safe retract first if needed).")
                            except Exception as e:
                                print(f"⚠️ Failed to go to Default: {e}")
                            continue

                        # 1) Ensure CURRENT is at its __midway (skip if Default)
                        self._ensure_current_midway(approach_mm)

                        # 2) Require TARGET__midway to exist; otherwise refuse
                        target_mid = midway_name(target)
                        if target_mid not in poses:
                            print(f"⚠️ Refusing move: '{target_mid}' not found. Jog to '{target}' and press X to teach it.")
                            continue

                        # 3) Move to TARGET__midway, then wait for A/B at the midway
                        try:
                            self._queue_jmove_to_pose(poses[target_mid])
                            self._set_current_named(target_mid)
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
                                # mark predictive window; _tool_move_along_tz will also extend
                                self._mark_motion_for(approach_mm / max(1e-6, self.VT) + 0.2)
                                self._tool_move_along_tz(+approach_mm, cont=0)
                                self._set_current_named(tgt)
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
                        try: self.robot.set_motor(0); print("[Robot] Motors stopped.")
                        except Exception as e: print(f"⚠️ stop_motors failed: {e}")

                    elif typ == "kill_motors":
                        try: self.robot.set_motor(0); print("[Robot] Motors killed.")
                        except Exception as e: print(f"⚠️ kill_motors failed: {e}")

                    elif typ == "set_tool_lz":
                        try:
                            lz = float(cmd.get("lz"))
                            self.robot.play_dict(cmd={"cmd":"tool","id":1,"lz":lz})
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

                    elif typ == "set_speed_fwd":
                        try:
                            v = int(cmd.get("rate"))
                            with self.state.lock:
                                self.state.settings["plunger_fwd_rate"] = v
                            save_settings(self.state.settings)
                            print(f"[Routine] Forward plunger speed set to {v} steps/s")
                        except Exception as e:
                            print(f"⚠️ set_speed_fwd failed: {e}")

                    elif typ == "set_speed_bwd":
                        try:
                            v = int(cmd.get("rate"))
                            with self.state.lock:
                                self.state.settings["plunger_bwd_rate"] = v
                            save_settings(self.state.settings)
                            print(f"[Routine] Reverse plunger speed set to {v} steps/s")
                        except Exception as e:
                            print(f"⚠️ set_speed_bwd failed: {e}")

                    elif typ == "set_endstop_timeout":
                        try:
                            ms = int(cmd.get("ms"))
                            with self.state.lock:
                                self.state.settings["endstop_timeout_ms"] = ms
                            save_settings(self.state.settings)
                            print(f"[Routine] Failsafe timeout set to {ms} ms")
                        except Exception as e:
                            print(f"⚠️ set_endstop_timeout failed: {e}")

                    elif typ == "set_pose_wait":
                        try:
                            s = float(cmd.get("s"))
                            with self.state.lock:
                                self.state.settings["pose_final_wait_s"] = s
                            save_settings(self.state.settings)
                            print(f"[Routine] Wash wait time set to {s} s (≤0 = none; min enforced = 0.35 s)")
                        except Exception as e:
                            print(f"⚠️ set_pose_wait failed: {e}")

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

                    elif typ == "rename_pose":
                        old = (cmd.get("old") or "").strip()
                        new = (cmd.get("new") or "").strip()
                        if not old or not new:
                            print("⚠️ rename_pose: empty name(s)")
                            continue
                        if new.endswith(MIDWAY_SUFFIX):
                            print(f"⚠️ rename_pose: '{MIDWAY_SUFFIX}' suffix is reserved.")
                            continue
                        if new in ["Default", "Reload"]:
                            print("⚠️ rename_pose: target name is reserved.")
                            continue
                        with self.state.lock:
                            poses = self.state.poses
                            reserved_list = list(self.state.settings.get("reserved_poses", []))
                            if old not in poses:
                                print(f"⚠️ rename_pose: '{old}' not found.")
                                continue
                            if old in RESERVED_POSES or old in reserved_list:
                                print(f"⚠️ rename_pose: '{old}' is reserved; cannot rename.")
                                continue
                            if new in poses:
                                print(f"⚠️ rename_pose: '{new}' already exists.")
                                continue
                            poses[new] = poses.pop(old)
                            old_mid = midway_name(old)
                            new_mid = midway_name(new)
                            if old_mid in poses:
                                poses[new_mid] = poses.pop(old_mid)
                            save_poses(poses)
                            if self.current_named == old:
                                self._set_current_named(new)
                            elif self.current_named == old_mid:
                                self._set_current_named(new_mid)
                        print(f"[Robot] Renamed pose '{old}' → '{new}' (with midway if present).")

                    elif typ == "delete_pose":
                        name = (cmd.get("pose") or "").strip()
                        if not name:
                            print("⚠️ delete_pose: empty name")
                            continue
                        with self.state.lock:
                            poses = self.state.poses
                            reserved_list = list(self.state.settings.get("reserved_poses", []))
                            if name in RESERVED_POSES or name in reserved_list:
                                print("⚠️ delete_pose: reserved pose; cannot delete.")
                                continue
                            removed_any = False
                            if name in poses:
                                poses.pop(name); removed_any = True
                            mid = midway_name(name)
                            if mid in poses:
                                poses.pop(mid); removed_any = True
                            save_poses(poses)
                            if self.current_named in (name, mid):
                                self._set_current_named(None)
                        if removed_any:
                            print(f"[Robot] Deleted pose '{name}' (and midway if present).")
                        else:
                            print(f"⚠️ delete_pose: '{name}' not found.")

                    elif typ == "overwrite_pose_with_midway":
                        name = (cmd.get("pose") or "").strip()
                        if not name:
                            print("⚠️ overwrite_pose: empty name"); continue
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
                rx_eff = -rx if at_default else rx
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

            # D-pad mapping (unchanged)
            if manual_enabled:
                tz = self.R[:,2]
                tx = self.R[:,0]
                ty = self.R[:,1]
                proj = np.array([tz[0], tz[1], 0.0]); n = np.linalg.norm(proj)
                at_default = (self.current_named == "Default")

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
        try: self.robot.set_motor(0)
        except Exception: pass
        try: self.robot.close()
        except Exception: pass

    def stop(self): self.stop_event.set()

# ─── Layout helpers ──────────────────────────────────────────────────────
def compute_layout(screen_w, screen_h, ui_frac=0.28, ui_min_w=320):
    """
    Two rows, two columns at top:
      Top-left  = RealSense (width = screen_w - ui_w)
      Top-right = UI panel  (width = ui_w)

    Bottom row (UVC panes) ignores top split and uses EQUAL widths:
      Bottom-left  = left half of screen width (UVC#2)
      Bottom-right = right half of screen width (UVC#1)
    """
    half_h = screen_h // 2
    ui_w = max(ui_min_w, int(screen_w * ui_frac))
    left_w = max(1, screen_w - ui_w)

    tl = pygame.Rect(0,        0,      left_w, half_h)                # RealSense
    tr = pygame.Rect(left_w,   0,      ui_w,   half_h)                # UI

    bottom_h = screen_h - half_h
    half_w = screen_w // 2
    bl = pygame.Rect(0,          half_h, half_w,           bottom_h)  # UVC left (UVC#2)
    br = pygame.Rect(half_w,     half_h, screen_w-half_w,  bottom_h)  # UVC right (UVC#1)
    return tl, tr, bl, br

def blit_frame_fit(screen, frame_bgr, rect):
    x, y, w, h = rect
    pygame.draw.rect(screen, (0,0,0), rect)
    if frame_bgr is None:
        return False
    fh, fw = frame_bgr.shape[:2]
    if fh <= 0 or fw <= 0: return False
    dest_aspect = w / float(h)
    src_aspect  = fw / float(fh)
    if dest_aspect >= src_aspect:
        new_h = h
        new_w = int(h * src_aspect)
    else:
        new_w = w
        new_h = int(w / src_aspect)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
    if (fw, fh) != (new_w, new_h):
        surf = pygame.transform.smoothscale(surf, (new_w, new_h))
    screen.blit(surf, (x + (w - new_w)//2, y + (h - new_h)//2))
    return True

def blit_frame_cover(screen, frame_bgr, rect):
    x, y, w, h = rect
    pygame.draw.rect(screen, (25,25,25), rect)
    if frame_bgr is None:
        return False
    fh, fw = frame_bgr.shape[:2]
    if fh <= 0 or fw <= 0:
        return False
    scale = max(w / float(fw), h / float(fh))
    new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
    if (new_w, new_h) != (fw, fh):
        base = pygame.transform.smoothscale(base, (new_w, new_h))
    crop_x = max(0, (new_w - w) // 2)
    crop_y = max(0, (new_h - h) // 2)
    area = pygame.Rect(crop_x, crop_y, w, h)
    screen.blit(base, (x, y), area)
    return True

def draw_placeholder(screen, rect, title="Camera not connected", value_font=None, big_font=None):
    x, y, w, h = rect
    pygame.draw.rect(screen, (25,25,25), rect)
    bf = big_font or pygame.font.SysFont("Consolas", 24, bold=True)
    vf = value_font or pygame.font.SysFont("Consolas", 12)
    msg = bf.render(title, True, (200, 200, 200))
    sub = vf.render("Connect a device or pass --uvc1/--uvc2", True, (180, 180, 180))
    screen.blit(msg, (x + (w - msg.get_width())//2, y + (h//2) - 18))
    screen.blit(sub, (x + (w - sub.get_width())//2, y + (h//2) + 10))

# Transform UVC frames according to rotation / flips
def transform_uvc_frame(frame_bgr, rot90=0, flip_h=False, flip_v=False):
    if frame_bgr is None:
        return None
    img = frame_bgr
    k = int(rot90) % 4
    if k == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif k == 2:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif k == 3:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # flips
    if flip_h and flip_v:
        img = cv2.flip(img, -1)
    else:
        if flip_h:
            img = cv2.flip(img, 1)
        if flip_v:
            img = cv2.flip(img, 0)
    return img

def draw_uvc_controls(screen, rect, rot90, flip_h, flip_v, font):
    """
    Draw small overlay buttons for a single UVC pane.
    Returns list of (rect, op) with op in {"rot", "flip_h", "flip_v"}.
    """
    buttons = []
    pad = 8
    btn_w = 28
    btn_h = 20
    gap = 4
    x = rect.x + pad
    y = rect.y + pad

    # Rotation 90°
    r_rect = pygame.Rect(x, y, btn_w, btn_h)
    pygame.draw.rect(screen, (0,0,0), r_rect)
    pygame.draw.rect(screen, (0,255,127), r_rect, 1)
    label = font.render("⟳", True, (0,255,127))
    screen.blit(label, (r_rect.x + (btn_w - label.get_width())//2,
                        r_rect.y + (btn_h - label.get_height())//2))
    buttons.append((r_rect, "rot"))

    # H flip
    x += btn_w + gap
    h_rect = pygame.Rect(x, y, btn_w, btn_h)
    col_fill = (40,40,40) if not flip_h else (0,60,60)
    pygame.draw.rect(screen, col_fill, h_rect)
    pygame.draw.rect(screen, (0,255,127), h_rect, 1)
    label = font.render("H", True, (0,255,127))
    screen.blit(label, (h_rect.x + (btn_w - label.get_width())//2,
                        h_rect.y + (btn_h - label.get_height())//2))
    buttons.append((h_rect, "flip_h"))

    # V flip
    x += btn_w + gap
    v_rect = pygame.Rect(x, y, btn_w, btn_h)
    col_fill = (40,40,40) if not flip_v else (0,60,60)
    pygame.draw.rect(screen, col_fill, v_rect)
    pygame.draw.rect(screen, (0,255,127), v_rect, 1)
    label = font.render("V", True, (0,255,127))
    screen.blit(label, (v_rect.x + (btn_w - label.get_width())//2,
                        v_rect.y + (btn_h - label.get_height())//2))
    buttons.append((v_rect, "flip_v"))

    # Show rotation degrees
    deg = (int(rot90) % 4) * 90
    info = font.render(f"{deg}°", True, (200,200,200))
    screen.blit(info, (v_rect.right + 4, v_rect.y + (btn_h - info.get_height())//2))

    return buttons

# ─── Popup / Input helpers (for pose edits and inline editors) ───────────
def wrap_lines(font: pygame.font.Font, text: str, max_width: int):
    return wrap_text(text, font, max_width)

def draw_center_popup_vertical(screen, title_lines, button_items, width=680, btn_width=560,
                               line_h=26, btn_pad=10, gap=10):
    """
    Draws a centered modal with title_lines and a VERTICAL list of buttons.
    Each button text is wrapped to 'btn_width - 2*btn_pad'.
    button_items = list of tuples: (button_label_str, button_id)
    Returns: (dict {id: rect}, panel_rect)
    """
    sw, sh = screen.get_size()
    font = pygame.font.SysFont("Consolas", 16)

    title_area_w = width - 32
    title_lines_wrapped = []
    for ln in title_lines:
        title_lines_wrapped += wrap_lines(font, ln, title_area_w)

    btn_rects_rel = []
    total_btn_h = 0
    for label, bid in button_items:
        lines = wrap_lines(font, label, btn_width - 2*btn_pad)
        btn_h = 2*btn_pad + len(lines) * (font.get_height() + 4)
        total_btn_h += btn_h + gap
        btn_rects_rel.append((bid, pygame.Rect(0, 0, btn_width, btn_h), lines))
    if button_items:
        total_btn_h -= gap

    text_h = len(title_lines_wrapped) * line_h + 16
    panel_w = max(width, btn_width + 40)
    panel_h = 24 + text_h + 12 + total_btn_h + 20
    panel_x = (sw - panel_w) // 2
    panel_y = (sh - panel_h) // 2
    panel = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA); overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    cur_y = panel_y + 16
    for ln in title_lines_wrapped:
        t = font.render(ln, True, (230,230,230))
        screen.blit(t, (panel_x + 16, cur_y))
        cur_y += line_h

    cur_y += 6

    rects = {}
    bx = panel_x + (panel_w - btn_width) // 2
    for (bid, rrel, lines) in btn_rects_rel:
        rect = pygame.Rect(bx, cur_y, rrel.w, rrel.h)
        pygame.draw.rect(screen, (50,50,80), rect)
        pygame.draw.rect(screen, (0,255,127), rect, 2)

        ty = rect.y + 10
        for ln in lines:
            t = font.render(ln, True, (230,230,230))
            screen.blit(t, (rect.x + 10, ty))
            ty += font.get_height() + 4

        rects[bid] = rect
        cur_y += rrel.h + gap

    return rects, panel

def draw_edit_menu(screen, pose_name,
                   disable_rename_delete=False,
                   reserved=False,
                   can_toggle_reserved=True):
    """
    Edit menu for a single pose.

    disable_rename_delete: True for built‑in reserved poses (Default/Reload).
    reserved:              True if this pose is currently reserved (built‑in or user).
    can_toggle_reserved:   False for built‑in reserved poses (no unreserve).
    """
    lines = [f"Edit pose: {pose_name}"]
    buttons = []

    if disable_rename_delete:
        buttons.append(("Rename (disabled for reserved pose)", "disabled_rename"))
        buttons.append(("Delete (disabled for reserved pose)", "disabled_delete"))
    else:
        buttons.append(("Rename", "rename"))
        buttons.append(("Delete", "delete"))

    # Allow marking/unmarking as reserved for non built‑in poses
    if can_toggle_reserved:
        if reserved:
            buttons.append(("Un‑reserve this pose (make it editable/deletable again)", "unreserve"))
        else:
            buttons.append(("Mark this pose as RESERVED (protect from rename/delete)", "reserve"))

    buttons.append(("Overwrite with CURRENT robot position (also updates the __midway via tool-Z retract/approach)", "overwrite"))
    buttons.append(("Cancel", "cancel"))
    return draw_center_popup_vertical(screen, lines, buttons, width=700, btn_width=600)

def draw_confirm_delete(screen, name):
    lines = [f"Delete pose '{name}'?",
             "This will also remove its __midway (if present). This cannot be undone."]
    buttons = [("Yes, delete pose", "yes"), ("No, keep it", "no")]
    return draw_center_popup_vertical(screen, lines, buttons, width=700, btn_width=560)

def draw_confirm_overwrite(screen, name):
    lines = [f"Overwrite '{name}' with CURRENT robot position?",
             "A __midway will be (re)generated via tool-Z retract/approach to maintain safe approach."]
    buttons = [("Yes, overwrite pose with current (+ Midway)", "yes"),
               ("No, cancel", "no")]
    return draw_center_popup_vertical(screen, lines, buttons, width=740, btn_width=600)

def draw_center_input_box(screen, title, value_text):
    sw, sh = screen.get_size()
    font = pygame.font.SysFont("Consolas", 16)

    panel_w = 720
    panel_h = 180
    x = (sw - panel_w)//2; y = (sh - panel_h)//2
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA); overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    title_lines = wrap_lines(font, title, panel_w - 32)
    ty = y + 16
    for ln in title_lines:
        t = font.render(ln, True, (230,230,230))
        screen.blit(t, (x + 16, ty))
        ty += font.get_height() + 6

    box = pygame.Rect(x + 16, y + panel_h - 70, panel_w - 32, 42)
    pygame.draw.rect(screen, (50,50,80), box)
    pygame.draw.rect(screen, (0,255,127), box, 2)
    txt = font.render(value_text, True, (255,255,255))
    screen.blit(txt, (box.x + 8, box.y + (box.h - txt.get_height())//2))
    tips = pygame.font.SysFont("Consolas", 14).render("Enter=Apply   Esc=Cancel", True, (200,200,200))
    screen.blit(tips, (x + panel_w - tips.get_width() - 16, y + panel_h - tips.get_height() - 10))

# ─── Routine Editor (multiline with caret + Ln/Col) ──────────────────────
def draw_routine_editor(screen, editor: EditorState, title="Edit Routine (Ctrl+Enter=Apply, Esc=Close)",
                        height_lines=12, width_chars=56):
    sw, sh = screen.get_size()
    font = pygame.font.SysFont("Consolas", 16)
    char_w, char_h = font.size("M")[0], font.get_height()

    panel_w = int(width_chars * char_w + 40)
    panel_h = int(height_lines * char_h + 120)
    x = (sw - panel_w)//2; y = (sh - panel_h)//2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA); overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    title_lines = wrap_text(title, font, panel_w - 32)
    ty = y + 12
    for ln in title_lines:
        t = font.render(ln, True, (230,230,230))
        screen.blit(t, (x + 12, ty))
        ty += char_h + 2

    box = pygame.Rect(x + 12, y + 50, panel_w - 24, height_lines*char_h + 8)
    pygame.draw.rect(screen, (50,50,80), box)
    pygame.draw.rect(screen, (0,255,127), box, 2)

    text = editor.text
    caret = max(0, min(len(text), editor.caret))
    pre = text[:caret]
    line_idx = pre.count("\n")
    col_idx = len(pre.split("\n")[-1])

    lines = text.split("\n")
    total_lines = len(lines)
    if line_idx < editor.scroll:
        editor.scroll = line_idx
    if line_idx >= editor.scroll + height_lines:
        editor.scroll = line_idx - height_lines + 1

    view_lines = lines[editor.scroll: editor.scroll + height_lines]
    for i, ln in enumerate(view_lines):
        t = font.render(ln, True, (255,255,255))
        screen.blit(t, (box.x + 6, box.y + 4 + i*char_h))

    caret_y_in_view = line_idx - editor.scroll
    caret_x_px = box.x + 6 + col_idx * char_w
    caret_y_px = box.y + 4 + caret_y_in_view * char_h

    if 0 <= caret_y_in_view < height_lines:
        if (pygame.time.get_ticks() // 500) % 2 == 0:
            pygame.draw.rect(screen, (255,255,255), (caret_x_px, caret_y_px, max(2, char_w//9), char_h), 0)

    status_text = f"Ln {line_idx+1}, Col {col_idx+1}    ({total_lines} lines)"
    st = font.render(status_text, True, (200,200,200))
    screen.blit(st, (x + panel_w - st.get_width() - 12, y + panel_h - st.get_height() - 8))

    return panel, box, (char_w, char_h)

def editor_handle_key(editor: EditorState, ev, width_chars=56):
    text = editor.text
    caret = editor.caret
    changed = False

    if ev.key == pygame.K_LEFT:
        if caret > 0: editor.caret -= 1
    elif ev.key == pygame.K_RIGHT:
        if caret < len(text): editor.caret += 1
    elif ev.key == pygame.K_UP:
        pre = text[:caret]
        col = len(pre.split("\n")[-1])
        prev_nl = text.rfind("\n", 0, caret-1)
        if prev_nl != -1:
            prev_prev_nl = text.rfind("\n", 0, prev_nl)
            start = 0 if prev_prev_nl == -1 else prev_prev_nl + 1
            line_len = prev_nl - start
            editor.caret = start + min(col, line_len)
    elif ev.key == pygame.K_DOWN:
        next_nl = text.find("\n", caret)
        if next_nl != -1:
            next_next_nl = text.find("\n", next_nl+1)
            col = len(text[:caret].split("\n")[-1])
            start = next_nl + 1
            line_len = (next_next_nl if next_next_nl!=-1 else len(text)) - start
            editor.caret = start + min(col, line_len)
    elif ev.key == pygame.K_HOME:
        nl = text.rfind("\n", 0, caret)
        editor.caret = nl+1 if nl!=-1 else 0
    elif ev.key == pygame.K_END:
        nl = text.find("\n", caret)
        editor.caret = nl if nl!=-1 else len(text)
    elif ev.key == pygame.K_BACKSPACE:
        if caret > 0:
            editor.text = text[:caret-1] + text[caret:]
            editor.caret -= 1
            changed = True
    elif ev.key == pygame.K_DELETE:
        if caret < len(text):
            editor.text = text[:caret] + text[caret+1:]
            changed = True
    elif ev.key == pygame.K_RETURN:
        editor.text = text[:caret] + "\n" + text[caret:]
        editor.caret += 1
        changed = True
    else:
        ch = ev.unicode
        if ch and ch.isprintable():
            editor.text = text[:caret] + ch + text[caret:]
            editor.caret += len(ch)
            changed = True

    return changed

# ─── Main HUD + Side UI + Trigger Bridge (60 FPS) ─────────────────────────
def main():
    args = parse_args()

    # Serial bridge (stepper trigger)
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.05)
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

    # RealSense in a background thread
    rs_thread = RealSenseThread(width=640, height=480, fps=30)
    rs_thread.start()
    t0 = time.time()
    while rs_thread.latest() is None and time.time()-t0 < 2.0:
        time.sleep(0.01)
    print("[RS] ", rs_thread.status())

    # UVC auto-pick
    if args.uvc1 or args.uvc2:
        uvc1_path = args.uvc1 or ""
        uvc2_path = args.uvc2 or ""
    else:
        cands = discover_uvc_index0(limit=4)
        uvc1_path = ""
        uvc2_path = ""
        if len(cands) >= 2:
            by_bus = {}
            for c in cands:
                by_bus.setdefault(c["busnum"], []).append(c)
            if len(by_bus) >= 2:
                buses = list(by_bus.keys())
                pickA = by_bus[buses[0]][0]
                pickB = by_bus[buses[1]][0]
            else:
                pickA, pickB = cands[0], cands[1]
            uvc1_path = pickA["bypath"]
            uvc2_path = pickB["bypath"]
            print("[UVC auto-pick]")
            print(f"  Cam1: {pickA['bypath']} -> {pickA['node']}  {pickA['name']}  bus={pickA['busnum']} devpath={pickA['devpath']}")
            print(f"  Cam2: {pickB['bypath']} -> {pickB['node']}  {pickB['name']}  bus={pickB['busnum']} devpath={pickB['devpath']}")
        else:
            autodetected = find_uvc_devices(limit=2)
            uvc1_path = (autodetected[0] if len(autodetected) >= 1 else "")
            uvc2_path = (autodetected[1] if len(autodetected) >= 2 else "")

    # Start UVC threads (no built-in 180° rotation; GUI handles orientation)
    uvc_threads = []
    if uvc1_path:
        uvc1 = UvcThread(uvc1_path, 640, 480, args.uvc_fps, False,
                         name="UVC#1", try_index1_fallback=args.uvc_try_index1)
        uvc1.start()
        uvc_threads.append(uvc1); print(f"[UVC#1] {uvc1_path}")
    else:
        uvc1 = None; print("[UVC#1] Not found (placeholder)")

    if uvc2_path:
        uvc2 = UvcThread(uvc2_path, 640, 480, args.uvc_fps, False,
                         name="UVC#2", try_index1_fallback=args.uvc_try_index1)
        uvc2.start()
        uvc_threads.append(uvc2); print(f"[UVC#2] {uvc2_path}")
    else:
        uvc2 = None; print("[UVC#2] Not found (placeholder)")

    # FPS fallback if needed
    def _status(s): return s.status() if s else ""
    if (args.uvc_fps > 15) and (uvc1 or uvc2):
        time.sleep(2.0)
        s1 = _status(uvc1); s2 = _status(uvc2)
        if ("no frames" in s1.lower()) or ("no frames" in s2.lower()):
            print("[UVC] Fallback: restarting both UVC threads at 15 fps…")
            for t in (uvc1, uvc2):
                if t: t.stop()
            for t in (uvc1, uvc2):
                if t: t.join()
            uvc_threads.clear()
            if uvc1_path:
                uvc1 = UvcThread(uvc1_path, 640, 480, 15, False,
                                 name="UVC#1", try_index1_fallback=args.uvc_try_index1)
                uvc1.start(); uvc_threads.append(uvc1)
            if uvc2_path:
                uvc2 = UvcThread(uvc2_path, 640, 480, 15, False,
                                 name="UVC#2", try_index1_fallback=args.uvc_try_index1)
                uvc2.start(); uvc_threads.append(uvc2)

    # Window / screen
    flags = 0
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
        screen = pygame.display.set_mode((0, 0), flags)
    else:
        screen = pygame.display.set_mode((1280, 960), flags)
    pygame.display.set_caption("Robotic Injector [SERI] — RS+UVC (routine)")
    clock = pygame.time.Clock()

    # Fonts
    label_font       = pygame.font.SysFont("Consolas", 14, bold=True)
    value_font       = pygame.font.SysFont("Consolas", 12)
    placeholder_font = pygame.font.SysFont("Consolas", 12)
    big_font         = pygame.font.SysFont("Consolas", 24, bold=True)
    small_font       = pygame.font.SysFont("Consolas", 12)

    # Shared state & robot
    state = SharedState()
    with state.lock:
        state.poses["Reload"] = DEFAULT_POSES["Reload"].copy()
    save_poses(state.poses); save_settings(state.settings)
    rt = RobotThread(state, host=args.host, port=args.port); rt.start()

    # Routine editor / executor state (persistent)
    initial_routine_text = load_routine_text()
    routine_editor = EditorState(text=initial_routine_text,
                                 caret=len(initial_routine_text))
    routine_edit_active = False
    routine_executor = None

    # Inline editors and confirmations
    toollen_input_active  = False
    approach_input_active = False
    naming_input_active   = False
    rename_input_active   = False
    delete_confirm_active = False
    overwrite_confirm_active = False
    edit_menu_active      = False
    edit_menu_target      = ""

    # Routine setting editors
    fwd_rate_input_active = False
    bwd_rate_input_active = False
    endstop_timeout_input_active = False
    pose_wait_input_active = False

    toollen_text          = f"{state.tool_lz:.1f}"
    approach_text         = f"{state.approach_mm:.1f}"
    naming_text           = ""
    rename_text           = ""
    rename_old_name       = ""
    delete_target_name    = ""
    overwrite_target_name = ""

    fwd_rate_text         = str(state.settings.get("plunger_fwd_rate", MAX_RATE))
    bwd_rate_text         = str(state.settings.get("plunger_bwd_rate", MAX_RATE))
    endstop_timeout_text  = str(state.settings.get("endstop_timeout_ms", 0))
    pose_wait_text        = f"{state.settings.get('pose_final_wait_s', 1.0):.1f}"

    # Button edges / sensitivity edges
    prev_a = prev_b = prev_x = False
    prev_lb = prev_rb = False
    prev_idx = -1

    # Click guard
    last_sig_axis_time = 0.0
    prev_lt = joy.get_axis(LEFT_AXIS)
    prev_rt = joy.get_axis(RIGHT_AXIS)
    click_pos = None

    # Scrollable UI panel state (top-right)
    ui_scroll = 0
    ui_content_height = 0  # updated each frame

    def visible_pose_names(poses: dict):
        return [name for name in poses.keys() if not name.endswith(MIDWAY_SUFFIX)]

    def sorted_pose_names(poses: dict):
        names = visible_pose_names(poses)
        def key(n):
            if n == "Default": return (0, "")
            if n == "Reload":  return (1, "")
            return (2, n.lower())
        return sorted(names, key=key)

    def make_buttons(current_tool_lz, current_approach, poses, settings):
        items = [
            {"label": f"Tool length (mm): {current_tool_lz:.1f}  [Edit]", "kind": "edit_tool"},
            {"label": f"Approach distance (mm): {current_approach:.1f}  [Edit]", "kind": "edit_approach"},

            {"label": "— Routine —", "kind": "header"},
            {"label": "Edit Routine Script", "kind": "routine_edit"},
            {"label": "Run Routine", "kind": "routine_run"},
            {"label": "Stop Routine", "kind": "routine_stop"},
            {"label": f"Wash FWD speed (steps/s): {settings.get('plunger_fwd_rate', MAX_RATE)}  [Edit]", "kind": "edit_fwd_rate"},
            {"label": f"Wash BWD speed (steps/s): {settings.get('plunger_bwd_rate', MAX_RATE)}  [Edit]", "kind": "edit_bwd_rate"},
            {"label": f"Failsafe timeout (ms) [0 = infinite]: {settings.get('endstop_timeout_ms', 0)}  [Edit]", "kind": "edit_endstop_timeout"},
            {"label": f"Wash wait time (s) [0 = none]: {settings.get('pose_final_wait_s', 1.0):.1f}  [Edit]", "kind": "edit_pose_wait"},
            {"label": f"Final expel endstop: {settings.get('expel_endstop', 'BWD')}  [Toggle]", "kind": "toggle_expel_endstop"},
            {"label": f"Swap endstops L/R: {'On' if settings.get('swap_endstops', False) else 'Off'}  [Toggle]", "kind": "toggle_swap_endstops"},

            {"label": "— Saved poses —", "kind": "header"},
            {"label": "New saved pose from CURRENT robot position", "kind": "new_pose"},
        ]
        for name in sorted_pose_names(poses):
            items.append({"label": f"Go: {name}", "kind": "pose_row", "pose": name})
        return items

    running = True
    while running:
        click_pos = None

        # Events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break

            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                flags ^= pygame.FULLSCREEN
                screen = pygame.display.set_mode((0, 0) if (flags & pygame.FULLSCREEN) else (1280, 960), flags)

            # UI scroll (mouse wheel)
            if ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                _, tr_rect, _, _ = compute_layout(*pygame.display.get_surface().get_size(), args.ui_frac, args.ui_min_width)
                if tr_rect.collidepoint(mx, my):
                    ui_scroll -= ev.y * 40
                    ui_scroll = max(0, min(ui_scroll, max(0, ui_content_height - tr_rect.h + 12)))

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if (time.time() - last_sig_axis_time) >= AXIS_GUARD_WINDOW:
                    click_pos = ev.pos

            # Tool length editor
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

            # Approach editor
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

            # Routine editor keys
            if routine_edit_active and ev.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if ev.key == pygame.K_RETURN and (mods & pygame.KMOD_CTRL):
                    # Apply & close, and persist the script
                    save_routine_text(routine_editor.text)
                    routine_edit_active = False
                elif ev.key == pygame.K_ESCAPE:
                    routine_edit_active = False
                else:
                    if editor_handle_key(routine_editor, ev):
                        pass

            # Add pose editor
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

            # Rename editor
            if rename_input_active and ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    new_name = rename_text.strip()
                    rename_input_active = False
                    if new_name:
                        rt.enqueue({"type":"rename_pose","old": rename_old_name, "new": new_name})
                        try: joy.rumble(0.6, 0.6, 140)
                        except Exception: pass
                elif ev.key == pygame.K_ESCAPE:
                    rename_input_active = False
                elif ev.key == pygame.K_BACKSPACE:
                    rename_text = rename_text[:-1]
                else:
                    ch = ev.unicode
                    if ch and (ch.isalnum() or ch in "_- "):
                        rename_text += ch

            # Routine settings editors
            if fwd_rate_input_active and ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    try:
                        val = int(float(fwd_rate_text.strip()))
                        rt.enqueue({"type":"set_speed_fwd","rate": val})
                        with state.lock:
                            state.settings["plunger_fwd_rate"] = val
                        save_settings(state.settings)
                    except ValueError:
                        print("⚠️ Invalid forward speed.")
                    fwd_rate_input_active = False
                elif ev.key == pygame.K_ESCAPE:
                    fwd_rate_input_active = False
                elif ev.key == pygame.K_BACKSPACE:
                    fwd_rate_text = fwd_rate_text[:-1]
                else:
                    ch = ev.unicode
                    if ch in "0123456789.-": fwd_rate_text += ch

            if bwd_rate_input_active and ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    try:
                        val = int(float(bwd_rate_text.strip()))
                        rt.enqueue({"type":"set_speed_bwd","rate": val})
                        with state.lock:
                            state.settings["plunger_bwd_rate"] = val
                        save_settings(state.settings)
                    except ValueError:
                        print("⚠️ Invalid reverse speed.")
                    bwd_rate_input_active = False
                elif ev.key == pygame.K_ESCAPE:
                    bwd_rate_input_active = False
                elif ev.key == pygame.K_BACKSPACE:
                    bwd_rate_text = bwd_rate_text[:-1]
                else:
                    ch = ev.unicode
                    if ch in "0123456789.-": bwd_rate_text += ch

            if endstop_timeout_input_active and ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    try:
                        ms = int(float(endstop_timeout_text.strip()))
                        rt.enqueue({"type":"set_endstop_timeout","ms": ms})
                        with state.lock:
                            state.settings["endstop_timeout_ms"] = ms
                        save_settings(state.settings)
                    except ValueError:
                        print("⚠️ Invalid timeout (ms).")
                    endstop_timeout_input_active = False
                elif ev.key == pygame.K_ESCAPE:
                    endstop_timeout_input_active = False
                elif ev.key == pygame.K_BACKSPACE:
                    endstop_timeout_text = endstop_timeout_text[:-1]
                else:
                    ch = ev.unicode
                    if ch in "0123456789.-": endstop_timeout_text += ch

            if pose_wait_input_active and ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    try:
                        s = float(pose_wait_text.strip())
                        rt.enqueue({"type":"set_pose_wait","s": s})
                        with state.lock:
                            state.settings["pose_final_wait_s"] = s
                        save_settings(state.settings)
                    except ValueError:
                        print("⚠️ Invalid wait seconds.")
                    pose_wait_input_active = False
                elif ev.key == pygame.K_ESCAPE:
                    pose_wait_input_active = False
                elif ev.key == pygame.K_BACKSPACE:
                    pose_wait_text = pose_wait_text[:-1]
                else:
                    ch = ev.unicode
                    if ch in "0123456789.-": pose_wait_text += ch

        if not running:
            break

        # Read joystick to state
        raw_hx, raw_hy = joy.get_hat(0)
        lb, rb = joy.get_button(4), joy.get_button(5)
        with state.lock:
            state.lx, state.ly = _axis(joy, LS_X_AXIS, INVERT_LS_X), _axis(joy, LS_Y_AXIS, INVERT_LS_Y)
            state.rx, state.ry = _axis(joy, RS_X_AXIS, INVERT_RS_X), _axis(joy, RS_Y_AXIS, INVERT_RS_Y)
            state.hx, state.hy = raw_hx, raw_hy
            state.lb, state.rb = lb, rb
            waiting = state.await_confirm
            confirm_msg = state.confirm_msg
            cur_pose_name = state.current_pose_name or ""
            at_midway_now = cur_pose_name.endswith(MIDWAY_SUFFIX)
            motion_until = float(state.pose_motion_until)

        # A/B/X edges
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

        # Axis guard
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

        # Arduino trigger bridge — STRICTLY GATED
        if ser:
            plunger_bridge_allowed = (not waiting) and (not at_midway_now) and (time.time() >= motion_until)
            if plunger_bridge_allowed:
                sign = int(state.settings.get("plunger_dir_sign", 1))
                rt_val = normalize(joy.get_axis(RIGHT_AXIS)); lt_val = normalize(joy.get_axis(LEFT_AXIS))
                if lt_val > DEADZONE:   rate = int(lt_val * MAX_RATE) * sign
                elif rt_val > DEADZONE: rate = -int(rt_val * MAX_RATE) * sign
                else:                   rate = 0
                if rate != last_rate:
                    ser.write(f"V{rate}\n".encode()); last_rate = rate
                    if ser.in_waiting: ser.read(ser.in_waiting)
            else:
                if last_rate not in (0, None):
                    ser.write(b"V0\n"); last_rate = 0
                    if ser.in_waiting: ser.read(ser.in_waiting)
            if joy.get_button(RESET_BUTTON):
                ser.write(b"R\n")
                if ser.in_waiting: ser.read(ser.in_waiting)
                last_rate = 0; time.sleep(0.1)

        # ─────────────────────────  DRAWING  ─────────────────────────
        screen_w, screen_h = pygame.display.get_surface().get_size()
        tl_rect, tr_rect, bl_rect, br_rect = compute_layout(screen_w, screen_h, args.ui_frac, args.ui_min_width)

        # RealSense top-left
        rs_frame = rs_thread.latest()
        blit_frame_cover(screen, rs_frame, tl_rect)

        # HUD basics
        tl_w, tl_h = tl_rect.w, tl_rect.h
        m = max(8, int(min(tl_w, tl_h) * 0.012))
        panel_size  = int(tl_h * 0.22)
        diag_gap    = int(tl_w * 0.06)
        padding     = 4

        with state.lock:
            p    = state.pitch
            j5v  = state.j5
            sens = state.levels[state.idx]
            tool_lz = state.tool_lz
            approach_mm = state.approach_mm
            poses_snapshot = state.poses.copy()
            settings_snapshot = state.settings.copy()
            routine_phase = state.routine_phase
            routine_status_text = state.routine_status_text
            last_plunger_note = state.last_plunger_note or "—"
            # UVC orientation
            u1_rot = int(state.settings.get("uvc1_rot", 2))
            u1_hf  = bool(state.settings.get("uvc1_flip_h", False))
            u1_vf  = bool(state.settings.get("uvc1_flip_v", False))
            u2_rot = int(state.settings.get("uvc2_rot", 2))
            u2_hf  = bool(state.settings.get("uvc2_flip_h", False))
            u2_vf  = bool(state.settings.get("uvc2_flip_v", False))

        # Reticle
        cx, cy = tl_rect.x + tl_w//2, tl_rect.y + tl_h//2
        col = (0, 255, 127)
        pygame.draw.circle(screen, col, (cx, cy), int(min(tl_w, tl_h)*0.09), 2)

        # Pitch panel
        x1 = tl_rect.x + m
        y1 = tl_rect.y + tl_h - (panel_size + padding + 20 + padding + 16 + padding) - m
        pan1 = pygame.Surface((panel_size, panel_size), pygame.SRCALPHA); pan1.fill((10,10,30,180))
        pivot = (5, panel_size//2); r_len = panel_size - pivot[0] - 5
        ang = np.radians(-p)
        ex = pivot[0] + r_len*np.cos(ang); ey = pivot[1] + r_len*np.sin(ang)
        pygame.draw.line(pan1, (0,255,127), pivot, (ex, ey), 2); pygame.draw.circle(pan1, (0,255,127), (int(ex), int(ey)), 4)
        screen.blit(pan1, (x1, y1))
        lf = pygame.font.SysFont("Consolas", 14, bold=True); vf = pygame.font.SysFont("Consolas", 12)
        screen.blit(lf.render("Pitch", True, (0,255,127)), (x1, y1 + panel_size + padding))
        screen.blit(vf.render(f"{p:+.5f}°", True, (0,255,127)),
                    (x1, y1 + panel_size + padding + 16 + padding))

        # Angle panel
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
        screen.blit(lf.render("Angle", True, (0,255,127)), (x2, y2 + panel_size + padding))
        screen.blit(vf.render(f"{-j5v:+.5f}°", True, (0,255,127)),
                    (x2, y2 + panel_size + padding + 16 + padding))

        # Status box (Endstop line removed per request)
        texts = [
            vf.render(f"Sensitivity ({sens*100:.0f}%)", True, (200,230,255)),
            vf.render(f"Move step:  ±{(5.0*sens):.5f} mm", True, (0,255,127)),
            vf.render(f"Angle step: ±{(5.0*sens):.5f}°", True, (0,255,127)),
            vf.render(f"Approach: {approach_mm:.2f} mm", True, (200,255,200)),
            vf.render(f"Routine phase: {routine_phase}", True, (220,220,220)),
            vf.render(f"Status: {routine_status_text}", True, (220,220,220)),
            vf.render(f"Plunger: {last_plunger_note}", True, (220,220,220)),
        ]
        sb_w = max(t.get_width() for t in texts) + 8
        sb_h = sum(t.get_height() + 4 for t in texts) + 4
        x3 = tl_rect.x + tl_w - sb_w - m; y3 = tl_rect.y + m
        sbpan = pygame.Surface((sb_w, sb_h), pygame.SRCALPHA); sbpan.fill((30,30,10,180))
        y_off_s = 4
        for t in texts:
            sbpan.blit(t, (4, y_off_s)); y_off_s += t.get_height() + 4
        screen.blit(sbpan, (x3, y3))

        # UI panel (top-right) — SCROLLABLE
        pygame.draw.rect(screen, (20,20,20), tr_rect)
        m_ui    = max(8, int(tr_rect.w*0.02))
        btn_w   = tr_rect.w - 2*m_ui
        btn_gap = 8
        buttons = make_buttons(tool_lz, approach_mm, poses_snapshot, settings_snapshot)

        prev_clip = screen.get_clip()
        screen.set_clip(tr_rect)

        ui_clickables = []
        y_off_btn = tr_rect.y + m_ui - ui_scroll
        total_h = 0

        for b in buttons:
            kind = b["kind"]
            if kind == "header":
                surf_txt = label_font.render(b["label"], True, (200,200,200))
                screen.blit(surf_txt, (tr_rect.x + m_ui, y_off_btn))
                bh = surf_txt.get_height()
                total_h += bh + btn_gap
                y_off_btn += bh + btn_gap
                continue

            if kind == "pose_row":
                name = b["pose"]
                edit_btn_w = min(90, max(70, int(btn_w * 0.28)))
                go_w = btn_w - edit_btn_w - 6
                go_rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, go_w, 34)
                pygame.draw.rect(screen, (40,40,60), go_rect)
                pygame.draw.rect(screen, (0,255,127), go_rect, 2)
                go_txt = placeholder_font.render(f"Go: {name}", True, (220,220,230))
                screen.blit(go_txt, (go_rect.x + 8, go_rect.y + (go_rect.h - go_txt.get_height())//2))
                ui_clickables.append((go_rect, {"kind":"goto", "pose":name}))

                edit_rect = pygame.Rect(go_rect.right + 6, y_off_btn, edit_btn_w, 34)
                pygame.draw.rect(screen, (55,55,85), edit_rect)
                pygame.draw.rect(screen, (0,255,127), edit_rect, 2)
                etxt = placeholder_font.render("Edit", True, (230,230,230))
                screen.blit(etxt, (edit_rect.x + (edit_rect.w - etxt.get_width())//2,
                                   edit_rect.y + (edit_rect.h - etxt.get_height())//2))
                ui_clickables.append((edit_rect, {"kind":"edit_menu", "pose":name}))

                y_off_btn += 34 + btn_gap
                total_h += 34 + btn_gap
                continue

            label = b["label"]
            lines = wrap_text(label, placeholder_font, btn_w - 10)
            bh    = len(lines)*(placeholder_font.get_height()+2) + 10
            rect  = pygame.Rect(tr_rect.x + m_ui, y_off_btn, btn_w, bh)
            pygame.draw.rect(screen, (40,40,60), rect)
            pygame.draw.rect(screen, (0,255,127), rect, 2)
            ly2 = y_off_btn + 5
            for ln in lines:
                surf_txt = placeholder_font.render(ln, True, (200,200,200))
                tx = tr_rect.x + m_ui + (btn_w - surf_txt.get_width())//2
                screen.blit(surf_txt, (tx, ly2))
                ly2 += placeholder_font.get_height() + 2
            ui_clickables.append((rect, b))
            y_off_btn += bh + btn_gap
            total_h   += bh + btn_gap

        ui_content_height = total_h + m_ui

        # Scrollbar
        if ui_content_height > tr_rect.h:
            view_h = tr_rect.h - 2*m_ui
            content_h = max(1, ui_content_height - 2*m_ui)
            bar_h = max(24, int(view_h * (view_h / content_h)))
            max_scroll = max(1, ui_content_height - tr_rect.h + 12)
            bar_y = tr_rect.y + m_ui + int((ui_scroll / max_scroll) * (view_h - bar_h))
            bar_x = tr_rect.right - 10
            pygame.draw.rect(screen, (60,60,60), (bar_x, tr_rect.y + m_ui, 6, view_h))
            pygame.draw.rect(screen, (0,255,127), (bar_x, bar_y, 6, bar_h))

        screen.set_clip(prev_clip)

        # Bottom row: left=UVC#2, right=UVC#1  (with per-camera transforms)
        uvc_clickables = []

        f2_raw = uvc2.latest() if uvc2 else None
        f2 = transform_uvc_frame(f2_raw, u2_rot, u2_hf, u2_vf) if f2_raw is not None else None
        if not blit_frame_fit(screen, f2, bl_rect):
            draw_placeholder(screen, bl_rect,
                             title=(uvc2.status() if uvc2 else "UVC#2 not connected"),
                             value_font=value_font, big_font=big_font)
        # UVC#2 controls
        for r, op in draw_uvc_controls(screen, bl_rect, u2_rot, u2_hf, u2_vf, small_font):
            uvc_clickables.append((r, {"cam": 2, "op": op}))

        f1_raw = uvc1.latest() if uvc1 else None
        f1 = transform_uvc_frame(f1_raw, u1_rot, u1_hf, u1_vf) if f1_raw is not None else None
        if not blit_frame_fit(screen, f1, br_rect):
            draw_placeholder(screen, br_rect,
                             title=(uvc1.status() if uvc1 else "UVC#1 not connected"),
                             value_font=value_font, big_font=big_font)
        # UVC#1 controls
        for r, op in draw_uvc_controls(screen, br_rect, u1_rot, u1_hf, u1_vf, small_font):
            uvc_clickables.append((r, {"cam": 1, "op": op}))

        # Handle UI clicks (if no modals)
        modals_open = (edit_menu_active or delete_confirm_active or overwrite_confirm_active
                       or toollen_input_active or approach_input_active
                       or naming_input_active or rename_input_active
                       or fwd_rate_input_active or bwd_rate_input_active
                       or endstop_timeout_input_active or pose_wait_input_active
                       or routine_edit_active)
        if click_pos is not None and not modals_open:
            mx, my = click_pos

            # 1) UVC controls (rotate / flip) first
            handled = False
            for rect, act in uvc_clickables:
                if rect.collidepoint(mx, my):
                    cam = int(act["cam"])
                    op  = act["op"]
                    key_prefix = f"uvc{cam}_"
                    rot_key = f"{key_prefix}rot"
                    hf_key  = f"{key_prefix}flip_h"
                    vf_key  = f"{key_prefix}flip_v"
                    with state.lock:
                        rot = int(state.settings.get(rot_key, 2))
                        hf  = bool(state.settings.get(hf_key, False))
                        vf  = bool(state.settings.get(vf_key, False))
                        if op == "rot":
                            rot = (rot + 1) % 4
                            state.settings[rot_key] = rot
                        elif op == "flip_h":
                            hf = not hf
                            state.settings[hf_key] = hf
                        elif op == "flip_v":
                            vf = not vf
                            state.settings[vf_key] = vf
                    save_settings(state.settings)
                    handled = True
                    break

            if handled:
                click_pos = None
            else:
                # 2) Side panel UI clicks
                _, tr_rect, _, _ = compute_layout(*pygame.display.get_surface().get_size(), args.ui_frac, args.ui_min_width)
                if tr_rect.collidepoint(mx, my):
                    for rect, act in ui_clickables:
                        if rect.collidepoint(mx, my):
                            kind = act.get("kind")
                            if kind == "edit_tool":
                                toollen_input_active = True; toollen_text = f"{state.tool_lz:.1f}"
                            elif kind == "edit_approach":
                                approach_input_active = True; approach_text = f"{state.approach_mm:.1f}"
                            elif kind == "goto":
                                rt.enqueue({"type":"goto","pose_name":act["pose"]})
                                try: joy.rumble(0.4, 0.4, 120)
                                except Exception: pass
                            elif kind == "edit_menu":
                                edit_menu_active = True
                                edit_menu_target = act["pose"]
                                toollen_input_active = approach_input_active = False
                                naming_input_active = rename_input_active = False
                                click_pos = None
                            elif kind == "routine_edit":
                                routine_edit_active = True
                            elif kind == "routine_run":
                                # Persist current script before running
                                save_routine_text(routine_editor.text)
                                # stop previous
                                if routine_executor and routine_executor.is_alive():
                                    routine_executor.stop()
                                    # no join here to avoid blocking GUI
                                with state.lock:
                                    kw = state.settings.get(
                                        "endstop_keywords",
                                        ["ENDSTOP","LIMIT","LIM","HIT","END"],
                                    )
                                    tms = state.settings.get("endstop_timeout_ms", 0)
                                    vf_rate = state.settings.get("plunger_fwd_rate", MAX_RATE)
                                    vb_rate = state.settings.get("plunger_bwd_rate", MAX_RATE)
                                    wait_s = state.settings.get("pose_final_wait_s", 1.0)
                                    expel = state.settings.get("expel_endstop", "BWD")
                                routine_executor = RoutineExecutor(
                                    state, rt, ser, routine_editor.text,
                                    endstop_keywords=kw,
                                    endstop_timeout_ms=tms,
                                    plunger_rate_fwd=vf_rate, plunger_rate_bwd=vb_rate,
                                    pose_final_wait_s=wait_s,
                                    expel_endstop=expel
                                )
                                routine_executor.start()
                            elif kind == "routine_stop":
                                # Stop executor + stop motion and plunger (safety)
                                if routine_executor and routine_executor.is_alive():
                                    routine_executor.stop()
                                rt.enqueue({"type":"stop_motors"})
                                if ser:
                                    try:
                                        ser.write(b"V0\n")
                                        if ser.in_waiting: ser.read(ser.in_waiting)
                                    except Exception:
                                        pass
                                with state.lock:
                                    state.routine_phase = "Stopped"
                                    state.routine_status_text = "Routine stopped by user"
                                click_pos = None
                            elif kind == "edit_fwd_rate":
                                fwd_rate_input_active = True
                                fwd_rate_text = str(state.settings.get("plunger_fwd_rate", MAX_RATE))
                            elif kind == "edit_bwd_rate":
                                bwd_rate_input_active = True
                                bwd_rate_text = str(state.settings.get("plunger_bwd_rate", MAX_RATE))
                            elif kind == "edit_endstop_timeout":
                                endstop_timeout_input_active = True
                                endstop_timeout_text = str(state.settings.get("endstop_timeout_ms", 0))
                            elif kind == "edit_pose_wait":
                                pose_wait_input_active = True
                                pose_wait_text = f"{state.settings.get('pose_final_wait_s', 1.0):.1f}"
                            elif kind == "toggle_expel_endstop":
                                with state.lock:
                                    cur = str(state.settings.get("expel_endstop", "BWD")).upper()
                                    state.settings["expel_endstop"] = "BWD" if cur == "FWD" else "FWD"
                                save_settings(state.settings)
                            elif kind == "toggle_swap_endstops":
                                with state.lock:
                                    cur = bool(state.settings.get("swap_endstops", False))
                                    state.settings["swap_endstops"] = not cur
                                save_settings(state.settings)
                            elif kind == "new_pose":
                                naming_input_active = True
                                naming_text = ""
                            break

        # Inline editors as centered modals
        if toollen_input_active:
            draw_center_input_box(screen, "Enter tool length (mm)", toollen_text)
        if approach_input_active:
            draw_center_input_box(screen, "Enter approach distance (mm)", approach_text)
        if naming_input_active:
            draw_center_input_box(screen, "Enter NEW pose name", naming_text or "")
        if rename_input_active:
            draw_center_input_box(screen, f"Rename '{rename_old_name}' →", rename_text or "")

        if fwd_rate_input_active:
            draw_center_input_box(screen, "Wash FORWARD plunger speed (steps/s)", fwd_rate_text)
        if bwd_rate_input_active:
            draw_center_input_box(screen, "Wash REVERSE plunger speed (steps/s)", bwd_rate_text)
        if endstop_timeout_input_active:
            draw_center_input_box(screen, "Failsafe timeout (ms) — 0 = infinite", endstop_timeout_text)
        if pose_wait_input_active:
            draw_center_input_box(screen, "Wash wait time (seconds) — 0 = none", pose_wait_text)

        # Edit popup
        if edit_menu_active:
            with state.lock:
                settings_snapshot = state.settings.copy()
            built_in_reserved = (edit_menu_target in RESERVED_POSES)
            user_reserved = edit_menu_target in settings_snapshot.get("reserved_poses", [])
            rects, panel_rect = draw_edit_menu(
                screen,
                edit_menu_target,
                disable_rename_delete=built_in_reserved or user_reserved,
                reserved=(built_in_reserved or user_reserved),
                can_toggle_reserved=True,  # allow un-reserve for user; built-ins stay logically reserved anyway
            )

            if click_pos is not None:
                mx, my = click_pos
                if not panel_rect.collidepoint(mx, my):
                    edit_menu_active = False
                    click_pos = None
                else:
                    for aid, r in rects.items():
                        if r.collidepoint(mx, my):
                            if aid == "rename" and not (built_in_reserved or user_reserved):
                                rename_input_active = True
                                rename_old_name = edit_menu_target
                                rename_text = rename_old_name
                                edit_menu_active = False
                                click_pos = None
                            elif aid == "delete" and not (built_in_reserved or user_reserved):
                                delete_confirm_active = True
                                delete_target_name = edit_menu_target
                                edit_menu_active = False
                                click_pos = None
                            elif aid == "overwrite":
                                overwrite_confirm_active = True
                                overwrite_target_name = edit_menu_target
                                edit_menu_active = False
                                click_pos = None
                            elif aid == "reserve":
                                with state.lock:
                                    current = list(state.settings.get("reserved_poses", []))
                                    if edit_menu_target not in current:
                                        current.append(edit_menu_target)
                                    state.settings["reserved_poses"] = current
                                save_settings(state.settings)
                                edit_menu_active = False
                                click_pos = None
                            elif aid == "unreserve":
                                with state.lock:
                                    current = list(state.settings.get("reserved_poses", []))
                                    if edit_menu_target in current:
                                        current.remove(edit_menu_target)
                                    state.settings["reserved_poses"] = current
                                save_settings(state.settings)
                                edit_menu_active = False
                                click_pos = None
                            elif aid in ("cancel", "disabled_rename", "disabled_delete"):
                                edit_menu_active = False
                                click_pos = None
                            break

        if delete_confirm_active:
            rects, panel = draw_confirm_delete(screen, delete_target_name)
            if click_pos is not None:
                mx, my = click_pos
                if not panel.collidepoint(mx, my):
                    delete_confirm_active = False
                    click_pos = None
                elif rects["yes"].collidepoint(click_pos):
                    rt.enqueue({"type":"delete_pose","pose": delete_target_name})
                    delete_confirm_active = False
                    click_pos = None
                elif rects["no"].collidepoint(click_pos):
                    delete_confirm_active = False
                    click_pos = None

        if overwrite_confirm_active:
            rects, panel = draw_confirm_overwrite(screen, overwrite_target_name)
            if click_pos is not None:
                mx, my = click_pos
                if not panel.collidepoint(mx, my):
                    overwrite_confirm_active = False
                    click_pos = None
                elif rects["yes"].collidepoint(click_pos):
                    rt.enqueue({"type":"overwrite_pose_with_midway","pose": overwrite_target_name})
                    overwrite_confirm_active = False
                    click_pos = None
                elif rects["no"].collidepoint(click_pos):
                    overwrite_confirm_active = False
                    click_pos = None

        # Robot midway confirmation overlay
        with state.lock:
            waiting = state.await_confirm
            confirm_msg = state.confirm_msg
        if waiting:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            txt1 = big_font.render(confirm_msg or "Paused at Midway. A: advance  B: free control", True, (255,255,255))
            screen.blit(overlay, (0,0))
            screen.blit(txt1, ((screen_w - txt1.get_width())//2, screen_h//2 - 24))

        # Routine editor overlay
        if routine_edit_active:
            draw_routine_editor(screen, routine_editor)

        pygame.display.flip()
        clock.tick(60)

    # Shutdown
    try:
        rt.stop(); rt.join()
    except Exception:
        pass
    try:
        rs_thread.stop(); rs_thread.join()
    except Exception:
        pass
    for t in uvc_threads:
        try: t.stop(); t.join()
        except Exception: pass
    if ser:
        try: ser.close()
        except Exception: pass
    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
