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
import math
import pygame
import numpy as np
import pygame.surfarray
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

# Where to place step recordings
DATA_ROOT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "RobotInjectionData")

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
    links = glob.glob("/dev/v4l/by-path/*video-index[01]")

    def canon_base(p):
        b = re.sub(r"-video-index[01]$", "", p)
        return b.replace("/usbv2-", "/usb-").replace("/usbv3-", "/usb-")

    def variant_rank(p):
        if "/usb-" in p and "/usbv" not in p: return 0
        if "/usbv2-" in p: return 1
        if "/usbv3-" in p: return 2
        return 3

    buckets = {}
    for link in links:
        base = canon_base(link)
        b = buckets.setdefault(base, {"index0": [], "index1": []})
        if link.endswith("video-index0"):
            b["index0"].append(link)
        else:
            b["index1"].append(link)

    out = []
    for base, b in buckets.items():
        i0 = sorted(b["index0"], key=variant_rank)
        i1 = sorted(b["index1"], key=variant_rank)
        pick = i0[0] if i0 else (i1[0] if i1 else None)
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
    UVC capture with simple auto‑reopen on repeated frame failures to mitigate
    V4L2 select() timeouts / stuck cameras.
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

    def _decode_fourcc(self, v):
        try:
            v = int(v)
            return "".join([chr((v >> (8*i)) & 0xFF) for i in range(4)])
        except Exception:
            return "----"

    def _dev_arg_for_backend(self, dev_path, backend):
        if backend == "v4l2" and isinstance(dev_path, str):
            real = os.path.realpath(dev_path)
            m = re.match(r"^/dev/video(\d+)$", real)
            if m:
                return int(m.group(1))
        return dev_path

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

        w_eff = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or frame.shape[1]
        h_eff = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
        fourcc_eff = self._decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)) or (fourcc or "?")
        fps_eff = int(cap.get(cv2.CAP_PROP_FPS) or fps)
        meta = {"fourcc": fourcc_eff, "fps": fps_eff, "w": int(w_eff), "h": int(h_eff), "backend": backend}
        return cap, (frame, meta)

    def _open_resilient(self, dev_path):
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
            fail_count = 0
            max_fail_before_reopen = 300   # ~0.3s at 1 kHz loop

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail_count += 1
                    if fail_count >= max_fail_before_reopen:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        self._status = f"{self.name}: lost frames, reopening…"
                        time.sleep(0.2)
                        cap, warm_frame, meta = self._open_resilient(chosen_path)
                        if cap is None:
                            self._status = f"{self.name}: reopen failed (still no frames)"
                            time.sleep(0.5)
                            continue
                        fail_count = 0
                        if self.rotate180 and warm_frame is not None:
                            warm_frame = cv2.rotate(warm_frame, cv2.ROTATE_180)
                        with self._lock:
                            self._frame = warm_frame
                        self._status = (f"{self.name}: reopened {os.path.basename(chosen_path)} "
                                        f"{meta['w']}x{meta['h']}@{meta['fps']} {meta['fourcc']} via {meta['backend'].upper()}")
                    else:
                        time.sleep(0.01)
                    continue

                fail_count = 0
                if self.rotate180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                with self._lock:
                    self._frame = frame
                time.sleep(idle_sleep)

        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._opened = False

# ─────────────────────────────────────────────────────────────────────────────
#                           REALSENSE CAMERA SUPPORT
# ─────────────────────────────────────────────────────────────────────────────

class RealSenseThread(threading.Thread):
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
        self.mode = None
        self.colorizer = None

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def status(self):
        return self._status

    def stop(self):
        self._stop.set()

    def _start_with(self, kind):
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
#                 ROBOT + UI + ROUTINE + SYRINGE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

SERIAL_PORT    = '/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44236313735351100201-if00'
BAUDRATE       = 115200
MAX_RATE       = 1600
DEADZONE       = 0.02
RIGHT_AXIS     = 5
LEFT_AXIS      = 2
RESET_BUTTON   = 2

LS_X_AXIS = 0
LS_Y_AXIS = 1
RS_X_AXIS = 3
RS_Y_AXIS = 4
INVERT_LS_X = False
INVERT_LS_Y = False
INVERT_RS_X = False
INVERT_RS_Y = False

DPAD_Z_SIGN = 1

def _axis(joy, idx, invert=False):
    try:
        v = float(joy.get_axis(idx))
    except Exception:
        v = 0.0
    return -v if invert else v

A_BUTTON       = 0
B_BUTTON       = 1
X_BUTTON       = 2
Y_BUTTON       = 3

AXIS_GUARD_DELTA  = 0.20
AXIS_GUARD_WINDOW = 0.15

POSES_PATH     = "poses.json"
SETTINGS_PATH  = "settings.json"
MIDWAY_SUFFIX  = "__midway"

DEFAULT_TOOL_LZ     = 205.0
DEFAULT_APPROACH_MM = 25.0

DEFAULT_POSES = {
    "Reload":  {"j0": 7.71, "j1": 80.86, "j2": -100.00, "j3": -0.07, "j4": -70.60, "j5": 6.35},
    "Default": {"j0": 39.24,"j1": 27.00, "j2": -100.44, "j3": 6.35,  "j4": 65.19,  "j5": -2.70},
}
RESERVED_POSES = {"Default", "Reload"}

def midway_name(name: str) -> str:
    return f"{name}{MIDWAY_SUFFIX}"

def load_poses(path=POSES_PATH):
    try:
        data = json.load(open(path)) if os.path.exists(path) else {}
        for name, pose in DEFAULT_POSES.items():
            data.setdefault(name, pose)
        for name, pose in list(data.items()):
            if not isinstance(pose, dict):
                continue
            for k in ["j0","j1","j2","j3","j4","j5"]:
                if k in pose:
                    data[name][k] = float(pose[k])
        return data
    except Exception as e:
        print(f"⚠️ Using built-in poses (could not load {path}: {e})")
        return {k: v.copy() for k, v in DEFAULT_POSES.items()}

def load_settings(path=SETTINGS_PATH):
    try:
        data = json.load(open(path)) if os.path.exists(path) else {}

        tool_lz     = float(data.get("tool_lz", DEFAULT_TOOL_LZ))
        approach_mm = float(data.get("approach_mm", DEFAULT_APPROACH_MM))

        end_kw = data.get("endstop_keywords",
                          ["ENDSTOP", "LIMIT", "LIM", "HIT", "END"])
        end_timeout_ms    = int(data.get("endstop_timeout_ms", 0))
        plunger_fwd_rate  = int(data.get("plunger_fwd_rate", MAX_RATE))
        plunger_bwd_rate  = int(data.get("plunger_bwd_rate", MAX_RATE))
        pose_final_wait_s = float(data.get("pose_final_wait_s", 1.0))
        plunger_dir_sign  = int(data.get("plunger_dir_sign", 1))
        swap_endstops     = bool(data.get("swap_endstops", False))

        expel_endstop = str(data.get("expel_endstop", "")).upper()
        if not expel_endstop:
            expel_endstop = "BWD"
        if expel_endstop not in ("FWD", "BWD"):
            expel_endstop = "BWD"

        reserved_poses = data.get("reserved_poses", [])
        if not isinstance(reserved_poses, list):
            reserved_poses = []
        reserved_poses = [str(n) for n in reserved_poses]

        # UVC transforms
        uvc1_rot     = int(data.get("uvc1_rot", 2))
        uvc1_flip_h  = bool(data.get("uvc1_flip_h", False))
        uvc1_flip_v  = bool(data.get("uvc1_flip_v", False))
        uvc2_rot     = int(data.get("uvc2_rot", 2))
        uvc2_flip_h  = bool(data.get("uvc2_flip_h", False))
        uvc2_flip_v  = bool(data.get("uvc2_flip_v", False))

        # Syringe + injection
        syringe_volume_ul        = float(data.get("syringe_volume_ul", 10.0))
        syringe_rotations_total  = float(data.get("syringe_rotations_total", 60.0))
        syringe_step_ul          = float(data.get("syringe_step_ul", 1.0))
        syringe_step_rotations   = float(data.get("syringe_step_rotations", 6.0))
        syringe_remaining_ul     = float(
            data.get("syringe_remaining_ul", syringe_volume_ul)
        )
        syringe_full_travel_time_s = float(
            data.get("syringe_full_travel_time_s", 0.0)
        )
        syringe_step_time_s      = float(data.get("syringe_step_time_s", 0.0))
        if (
            syringe_step_time_s <= 0
            and syringe_volume_ul > 0
            and syringe_step_ul > 0
            and syringe_full_travel_time_s > 0
        ):
            syringe_step_time_s = (
                syringe_full_travel_time_s * (syringe_step_ul / syringe_volume_ul)
            )

        injection_active         = bool(data.get("injection_active", False))
        injection_study          = data.get("injection_study", "")
        injection_subject        = data.get("injection_subject", "")
        injection_material       = data.get("injection_material", "")
        injection_date           = data.get("injection_date", "")

        # NEW: distance settings for injection retract / advance
        injection_retract_mm = float(data.get("injection_retract_mm", 10.0))
        injection_advance_mm = float(data.get("injection_advance_mm", 0.0))
        if injection_advance_mm > injection_retract_mm:
            injection_advance_mm = injection_retract_mm

        # NEW: J5 sawing parameters
        saw_amp_deg = float(data.get("saw_amp_deg", 3.0))   # ±deg
        saw_freq_hz = float(data.get("saw_freq_hz", 1.0))   # Hz

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
            "syringe_volume_ul": syringe_volume_ul,
            "syringe_rotations_total": syringe_rotations_total,
            "syringe_step_ul": syringe_step_ul,
            "syringe_step_rotations": syringe_step_rotations,
            "syringe_remaining_ul": syringe_remaining_ul,
            "syringe_full_travel_time_s": syringe_full_travel_time_s,
            "syringe_step_time_s": syringe_step_time_s,
            "injection_active": injection_active,
            "injection_study": injection_study,
            "injection_subject": injection_subject,
            "injection_material": injection_material,
            "injection_date": injection_date,
            "injection_retract_mm": injection_retract_mm,
            "injection_advance_mm": injection_advance_mm,
            "saw_amp_deg": saw_amp_deg,
            "saw_freq_hz": saw_freq_hz,
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
            "pose_final_wait_s": 1.0,
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
            "syringe_volume_ul": 10.0,
            "syringe_rotations_total": 60.0,
            "syringe_step_ul": 1.0,
            "syringe_step_rotations": 6.0,
            "syringe_remaining_ul": 10.0,
            "syringe_full_travel_time_s": 0.0,
            "syringe_step_time_s": 0.0,
            "injection_active": False,
            "injection_study": "",
            "injection_subject": "",
            "injection_material": "",
            "injection_date": "",
            "injection_retract_mm": 10.0,
            "injection_advance_mm": 0.0,
            "saw_amp_deg": 3.0,
            "saw_freq_hz": 1.0,
        }


def save_settings(settings, path=SETTINGS_PATH):
    try:
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save settings to {path}: {e}")

def save_poses(poses, path=POSES_PATH):
    try:
        with open(path, "w") as f:
            json.dump(poses, f, indent=2)
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
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

# ─── Routine system ──────────────────────────────────────────────────────

ROUTINE_SAMPLE = """\
# Example routine
# Keywords:
#   POSE <name>
#   ADVANCE_WAIT
#   ADVANCE_AUTO
#   WASH N
POSE Reload
ADVANCE_WAIT
WASH 3
"""

ROUTINE_PATH = "routine.txt"

def load_routine_text(path=ROUTINE_PATH):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"⚠️ Could not load routine from {path}: {e}")
    return ROUTINE_SAMPLE.strip()

def save_routine_text(text: str, path=ROUTINE_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"⚠️ Could not save routine to {path}: {e}")

@dataclass
class EditorState:
    text: str
    caret: int = 0
    scroll: int = 0

# ─────────────────────────────────────────────────────────────────────────────
#  Shared endstop detection used by RoutineExecutor and SyringeCalibrator
# ─────────────────────────────────────────────────────────────────────────────

def check_endstop_hit(lines, prev_state, direction=None,
                      swap_lr=False, keywords=None, log_func=None):
    """
    Parse Arduino lines and detect a *transition* on L/R, or keywords.

    prev_state: dict with "L","R" last bool values; updated in-place.
    direction: "FWD", "BWD" or None for any.
    swap_lr: swap L/R if wiring reversed.
    keywords: list of strings; if any appears in a line, that's also a hit.
    log_func: optional callback(str) for logging/status.
    """
    if keywords is None:
        keywords = []
    kw_up = []
    for k in keywords:
        k = (k or "").upper().strip()
        if k:
            kw_up.append(k)

    def log(msg):
        if log_func:
            log_func(msg)
        else:
            print("[Endstop]", msg)

    def maybe_transition(left, right, src_line):
        prev_L = prev_state.get("L")
        prev_R = prev_state.get("R")
        if prev_L is None and prev_R is None:
            prev_state["L"] = left
            prev_state["R"] = right
            if direction == "BWD" and left:
                log(f"Endstop active at start (L): {src_line}")
                return True
            if direction == "FWD" and right:
                log(f"Endstop active at start (R): {src_line}")
                return True
            return False

        changed_L = (prev_L in (False, 0)) and left
        changed_R = (prev_R in (False, 0)) and right
        prev_state["L"] = left
        prev_state["R"] = right

        if direction == "BWD" and changed_L:
            log(f"Endstop ACT (L): {src_line}")
            return True
        if direction == "FWD" and changed_R:
            log(f"Endstop ACT (R): {src_line}")
            return True
        if direction is None and (changed_L or changed_R):
            which = []
            if changed_L:
                which.append("L")
            if changed_R:
                which.append("R")
            log(f"Endstop ACT ({'/'.join(which)}): {src_line}")
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
                if swap_lr:
                    left, right = right, left
                if maybe_transition(left, right, ln):
                    return True

        # EL/ER
        if "EL=" in up and "ER=" in up:
            mL = re.search(r'\bEL\s*=\s*([01])', up)
            mR = re.search(r'\bER\s*=\s*([01])', up)
            if mL and mR:
                left  = (mL.group(1) == "1")
                right = (mR.group(1) == "1")
                if swap_lr:
                    left, right = right, left
                if maybe_transition(left, right, ln):
                    return True

        for k in kw_up:
            if k in up:
                log(f"Endstop keyword '{k}' seen in: {ln}")
                return True

    return False

# ─────────────────────────────────────────────────────────────────────────────
#                               RoutineExecutor
# ─────────────────────────────────────────────────────────────────────────────

class RoutineExecutor(threading.Thread):
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

    def _set_status_line(self, msg):
        print(f"[Routine] {msg}")
        with self.state.lock:
            self.state.routine_status_text = msg

    def stop(self):
        self.stop_event.set()

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
        with self.state.lock:
            swap_lr = bool(self.state.settings.get("swap_endstops", False))
        return check_endstop_hit(
            lines, prev_state, direction,
            swap_lr=swap_lr,
            keywords=self.endstop_keywords,
            log_func=self._set_status_line,
        )

    def _plunger_velocity(self, rate):
        if not self.ser:
            return
        try:
            self.ser.write(f"V{int(rate)}\n".encode())
            with self.state.lock:
                self.state.last_plunger_note = f"V{int(rate)} sent"
        except Exception:
            pass

    def _plunger_stop(self):
        if not self.ser:
            return
        try:
            self.ser.write(b"V0\n")
            with self.state.lock:
                self.state.last_plunger_note = "V0 (stop) sent"
        except Exception:
            pass

    def _plunger_to_end(self, direction: str) -> bool:
        direction = direction.upper()
        sign = int(self.state.settings.get("plunger_dir_sign", 1))
        rate = (self.rate_fwd if direction == "FWD" else -self.rate_bwd) * sign
        if rate == 0:
            self._set_status_line(f"⚠️ Plunger rate {direction} is 0; skipping.")
            return False

        self._set_status_line(f"Plunger {direction} → endstop @ {rate} steps/s")
        self._plunger_velocity(rate)

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
                    if self.ser:
                        self.ser.write(b"E\n")
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

    def _wait_until_final_pose(self, expected_final_name=None, timeout=None):
        min_settle = 0.35
        self._set_status_line(f"Waiting for final pose '{expected_final_name or '<any>'}' before plunger…")
        t0 = time.time()
        while not self.stop_event.is_set():
            with self.state.lock:
                waiting = bool(self.state.await_confirm)
                cur     = self.state.current_pose_name or ""
                at_mid  = cur.endswith(MIDWAY_SUFFIX)
                motion_until = float(self.state.pose_motion_until)
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

    def _parse(self):
        steps = []
        for raw in self.text.splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            m = re.match(r'^(POSE|GO)\s+(.+)$', line, re.I)
            if m:
                steps.append(("pose", m.group(2).strip()))
                continue
            if re.match(r'^(ADVANCE_WAIT|WAIT_ADVANCE)$', line, re.I):
                steps.append(("advance_wait", None))
                continue
            if re.match(r'^(ADVANCE|ADVANCE_AUTO|GO_ADVANCE)$', line, re.I):
                steps.append(("advance_auto", None))
                continue
            m = re.match(r'^WASH\s+(\d+)$', line, re.I)
            if m:
                steps.append(("wash", int(m.group(1))))
                continue
            m = re.match(r'^PLUNGER\s+(FWD|BWD)$', line, re.I)
            if m:
                steps.append(("plunger_end", m.group(1).upper()))
                continue
            steps.append(("echo", f"Unrecognized: {raw.strip()}"))
        return steps

    def run(self):
        self.phase = "RUNNING"
        with self.state.lock:
            self.state.routine_phase = self.phase
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
                t0 = time.time()
                while not self.stop_event.is_set() and (time.time() - t0) < 20:
                    with self.state.lock:
                        if self.state.await_confirm:
                            break
                    time.sleep(0.02)

            elif typ == "advance_wait":
                self._set_status_line("ADVANCE_WAIT (A = advance; B = free control at __midway)")
                while not self.stop_event.is_set():
                    with self.state.lock:
                        if not self.state.await_confirm:
                            break
                    time.sleep(0.02)
                self._wait_until_final_pose(last_pose)

            elif typ == "advance_auto":
                self._set_status_line("ADVANCE_AUTO")
                self.rt.enqueue({"type":"confirm_or_advance"})
                self._wait_until_final_pose(last_pose)

            elif typ == "wash":
                n = int(arg)
                self._wait_until_final_pose(last_pose)

                self._set_status_line(f"WASH x{n} cycles (BWD→END then FWD→END; finish at {self.expel_endstop})")
                for i in range(n):
                    if self.stop_event.is_set():
                        break
                    self._set_status_line(f"wash {i+1}/{n}: BWD→END")
                    self._plunger_to_end("BWD")
                    if self.stop_event.is_set():
                        break
                    self._set_status_line(f"wash {i+1}/{n}: FWD→END")
                    self._plunger_to_end("FWD")

                self._set_status_line(f"Final stroke to {self.expel_endstop} endstop (fully expelled)")
                self._plunger_to_end(self.expel_endstop)
                time.sleep(0.2)

            elif typ == "plunger_end":
                dirn = arg
                self._wait_until_final_pose(last_pose)
                self._set_status_line(f"PLUNGER {dirn} → endstop")
                self._plunger_to_end(dirn)
                time.sleep(0.1)

            time.sleep(0.03)

        self.phase = "DONE"
        with self.state.lock:
            self.state.routine_phase = self.phase
        self._set_status_line(f"Routine DONE (ended at {self.expel_endstop} endstop)")

# ─────────────────────────────────────────────────────────────────────────────
#                         SyringeCalibrator thread (updated)
# ─────────────────────────────────────────────────────────────────────────────

class SyringeCalibrator(threading.Thread):
    """
    Robot-driven syringe calibration (mechanical only):

      1) Detect which endstop (if any) is currently active.
      2) If in the middle, move to one endstop to establish a reference.
      3) Move fully to the opposite endstop and back, timing each stroke.
      4) Store:
           • syringe_full_travel_time_s
           • syringe_rotations_total   (rate-units × seconds)

    Volume and step size (µL) are now applied *after* calibration, using
    the UI fields; they are not required to run calibration.
    """

    def __init__(self, state, ser, volume_ul, step_ul):
        super().__init__(daemon=True)
        self.state = state
        self.ser = ser
        self.volume_ul = float(volume_ul)
        self.step_ul = float(step_ul)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def _readlines(self, timeout_s=0.05):
        out = []
        if not self.ser:
            return out
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                if self.ser.in_waiting:
                    raw = self.ser.read(self.ser.in_waiting).decode(errors="ignore")
                    for ln in raw.splitlines():
                        s = ln.strip()
                        if s:
                            out.append(s)
                else:
                    time.sleep(0.005)
            except Exception:
                break
        return out

    def _flush_and_stop(self):
        if not self.ser:
            return
        try:
            self.ser.write(b"V0\n")
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
        except Exception:
            pass

    def _query_endstop_state(self, swap_lr, log_func=None):
        if not self.ser:
            return None
        try:
            self.ser.write(b"E\n")
        except Exception:
            return None

        lines = self._readlines(0.1)
        prev = {"L": None, "R": None}

        check_endstop_hit(
            lines,
            prev_state=prev,
            direction=None,
            swap_lr=swap_lr,
            keywords=[],
            log_func=log_func,
        )

        L = bool(prev.get("L"))
        R = bool(prev.get("R"))
        if L and not R:
            return "BWD"
        if R and not L:
            return "FWD"
        if L and R:
            return "BWD"
        return None

    def _measure_travel_to(
        self,
        direction,
        plunger_fwd_rate,
        plunger_bwd_rate,
        dir_sign,
        swap_lr,
        keywords,
        timeout_s,
        log,
    ):
        if not self.ser:
            return None

        direction = direction.upper()
        if direction == "FWD":
            rate = abs(plunger_fwd_rate) * dir_sign
        else:
            rate = -abs(plunger_bwd_rate) * dir_sign

        if rate == 0:
            log(f"Calibration: rate for {direction} is 0; aborting this move.")
            return None

        self._flush_and_stop()
        log(f"Calibration: moving {direction} until endstop… (V{rate})")

        try:
            self.ser.write(f"V{rate}\n".encode())
        except Exception as e:
            log(f"Calibration: cannot send velocity {rate}: {e}")
            return None

        prev_state = {"L": None, "R": None}
        t_start = time.time()
        hit = False

        try:
            while not self.stop_event.is_set():
                try:
                    self.ser.write(b"E\n")
                except Exception:
                    pass

                lines = self._readlines(0.05)
                if check_endstop_hit(
                    lines,
                    prev_state=prev_state,
                    direction=direction,
                    swap_lr=swap_lr,
                    keywords=keywords,
                    log_func=log,
                ):
                    hit = True
                    break

                if time.time() - t_start > timeout_s:
                    break

            t_end = time.time()
        finally:
            self._flush_and_stop()

        if not hit:
            log(
                f"Calibration: no {direction} endstop detected within "
                f"{timeout_s:.1f}s."
            )
            return None

        duration = max(0.0, t_end - t_start)
        log(f"Calibration: hit {direction} endstop after {duration:.3f}s.")
        return duration

    def run(self):
        try:
            with self.state.lock:
                self.state.syringe_calibration_running = True
                self.state.calibration_status = "Calibrating syringe…"
                settings = self.state.settings.copy()

            if not self.ser:
                with self.state.lock:
                    self.state.calibration_status = (
                        "Calibration failed: no serial connection."
                    )
                return

            keywords = settings.get(
                "endstop_keywords", ["ENDSTOP", "LIMIT", "LIM", "HIT", "END"]
            )
            swap_lr = bool(settings.get("swap_endstops", False))
            endstop_timeout_ms = int(settings.get("endstop_timeout_ms", 0))
            plunger_fwd_rate = int(
                settings.get("plunger_fwd_rate", MAX_RATE) or MAX_RATE
            )
            plunger_bwd_rate = int(
                settings.get("plunger_bwd_rate", MAX_RATE) or MAX_RATE
            )
            dir_sign = int(settings.get("plunger_dir_sign", 1) or 1)

            timeout_s = (
                endstop_timeout_ms / 1000.0 if endstop_timeout_ms > 0 else 30.0
            )

            def log(msg):
                print("[SyringeCalib]", msg)
                with self.state.lock:
                    self.state.calibration_status = msg

            self._flush_and_stop()

            log("Step 1: detect current endstop state…")
            state0 = self._query_endstop_state(swap_lr, log_func=None)
            log(f"Initial endstop state: {state0 or 'middle'}")

            if state0 is None:
                log("No endstop active; homing towards BWD first.")
                t_mid = self._measure_travel_to(
                    "BWD",
                    plunger_fwd_rate,
                    plunger_bwd_rate,
                    dir_sign,
                    swap_lr,
                    keywords,
                    timeout_s,
                    log,
                )
                if t_mid is None:
                    log(
                        "Calibration failed: could not reach BWD endstop "
                        "for reference."
                    )
                    return
                state0 = "BWD"
                log("Reference position established at BWD.")

            if state0 == "BWD":
                first = "FWD"
                second = "BWD"
            else:
                first = "BWD"
                second = "FWD"

            t1 = self._measure_travel_to(
                first,
                plunger_fwd_rate,
                plunger_bwd_rate,
                dir_sign,
                swap_lr,
                keywords,
                timeout_s,
                log,
            )
            if t1 is None:
                log(
                    f"Calibration failed: could not move from {state0} "
                    f"to {first}."
                )
                return

            t2 = self._measure_travel_to(
                second,
                plunger_fwd_rate,
                plunger_bwd_rate,
                dir_sign,
                swap_lr,
                keywords,
                timeout_s,
                log,
            )

            if t2 is not None:
                full_time = 0.5 * (t1 + t2)
                log(
                    f"Full stroke time (avg of both directions) "
                    f"= {full_time:.3f}s."
                )
            else:
                full_time = t1
                log(
                    f"Full stroke time (single direction) "
                    f"= {full_time:.3f}s."
                )

            if full_time <= 0:
                log("Calibration failed: measured stroke time is zero.")
                return

            rotations_total = abs(plunger_fwd_rate) * full_time

            with self.state.lock:
                s = self.state.settings
                s["syringe_full_travel_time_s"] = full_time
                s["syringe_rotations_total"] = rotations_total
                s["injection_active"] = False
                self.state.calibration_status = (
                    f"Stroke calibrated: {full_time:.2f}s, "
                    f"~{rotations_total:.1f} distance-units. "
                    "Now enter volume + step and apply."
                )

            save_settings(self.state.settings)
            print(
                "[SyringeCalib] stroke="
                f"{full_time:.3f}s, rot_total={rotations_total:.1f}"
            )

        finally:
            with self.state.lock:
                self.state.syringe_calibration_running = False
            self._flush_and_stop()

# ─────────────────────────────────────────────────────────────────────────────
#                          Step recorder (per injection)
# ─────────────────────────────────────────────────────────────────────────────
class StepRecorder:
    def __init__(
        self,
        root_dir,
        meta,
        step_index,
        record_telemetry=True,
        record_video=True,
        record_ui=False,
    ):
        self.root_dir = root_dir
        self.meta = meta
        self.step_index = step_index
        self.record_telemetry = record_telemetry
        self.record_video = record_video
        self.record_ui = record_ui

        self.step_dir = None
        self.telemetry_file = None
        self.frames_log_file = None
        self.video_writers = {}
        self.frame_counts = {}

        self.t0 = time.time()
        self.active = False
        self._setup()

    def _safe(self, s: str) -> str:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        return "".join(c if c in allowed else "_" for c in (s or ""))[:40]

    def _setup(self):
        os.makedirs(self.root_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        study   = self._safe(self.meta.get("study", ""))
        subject = self._safe(self.meta.get("subject", ""))
        material= self._safe(self.meta.get("material", ""))
        date_s  = self._safe(self.meta.get("date", ""))

        base = f"{ts}_step{self.step_index:02d}"
        parts = [p for p in (study, subject, material, date_s) if p]
        if parts:
            base += "_" + "_".join(parts)

        self.step_dir = os.path.join(self.root_dir, base)
        suffix = 0
        while os.path.exists(self.step_dir):
            print(f"[Recorder] WARNING: '{self.step_dir}' exists; appending suffix.")
            suffix += 1
            self.step_dir = os.path.join(self.root_dir, f"{base}_dup{suffix:02d}")
        os.makedirs(self.step_dir, exist_ok=True)

        # Telemetry CSV
        if self.record_telemetry:
            path = os.path.join(self.step_dir, "telemetry.csv")
            self.telemetry_file = open(path, "w", encoding="utf-8")
            self.telemetry_file.write(
                "t,j0,j1,j2,j3,j4,j5,"
                "lx,ly,rx,ry,hx,hy,"
                "lt,rt,lb,rb,a,b,x,y,"
                "distance,step_volume\n"
            )

        # Frames index + video writers (for RS / UVC / UI)
        if self.record_video or self.record_ui:
            path = os.path.join(self.step_dir, "frames.csv")
            self.frames_log_file = open(path, "w", encoding="utf-8")
            self.frames_log_file.write("t,camera,frame_index\n")
            self.video_writers = {"rs": None, "uvc1": None, "uvc2": None, "ui": None}
            self.frame_counts = {"rs": 0, "uvc1": 0, "uvc2": 0, "ui": 0}
        else:
            self.video_writers = {}
            self.frame_counts = {}

        self.active = True

    def log_telemetry(self, t_rel, joints, joystick, extra):
        if not self.active or not self.record_telemetry or not self.telemetry_file:
            return
        (lx, ly, rx, ry, hx, hy,
         lt, rt, lb, rb, a, b, x, y) = joystick
        j = joints or {}
        row = [
            f"{t_rel:.6f}",
            *(f"{j.get(k,'')}" for k in ("j0","j1","j2","j3","j4","j5")),
            f"{lx:.5f}", f"{ly:.5f}", f"{rx:.5f}", f"{ry:.5f}",
            str(hx), str(hy),
            f"{lt:.5f}", f"{rt:.5f}",
            str(int(lb)), str(int(rb)),
            str(int(a)), str(int(b)), str(int(x)), str(int(y)),
            f"{extra.get('distance',0.0):.6f}",
            f"{extra.get('step_volume',0.0):.6f}",
        ]
        self.telemetry_file.write(",".join(row) + "\n")

    def _ensure_writer(self, cam_key, frame):
        if cam_key not in self.video_writers:
            return
        if self.video_writers[cam_key] is not None:
            return
        if frame is None:
            return
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        filename = os.path.join(self.step_dir, f"{cam_key}.mp4")
        self.video_writers[cam_key] = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))

    def write_frame(self, t_rel, cam_key, frame):
        if not self.active or frame is None:
            return
        # Only write if that type is enabled
        if cam_key == "ui" and not self.record_ui:
            return
        if cam_key != "ui" and not self.record_video:
            return
        if cam_key not in self.video_writers:
            return

        self._ensure_writer(cam_key, frame)
        writer = self.video_writers.get(cam_key)
        if writer is None:
            return
        writer.write(frame)
        idx = self.frame_counts[cam_key]
        self.frame_counts[cam_key] = idx + 1
        if self.frames_log_file:
            self.frames_log_file.write(f"{t_rel:.6f},{cam_key},{idx}\n")

    def write_ui_frame(self, t_rel, surface):
        """Capture full UI surface and write as 'ui.mp4'."""
        if not self.active or not self.record_ui or surface is None:
            return
        # Pygame surface -> RGB ndarray (w,h,3) -> (h,w,3) -> BGR
        arr = pygame.surfarray.array3d(surface)  # (w,h,3)
        frame_rgb = np.transpose(arr, (1, 0, 2))  # (h,w,3)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.write_frame(t_rel, "ui", frame_bgr)

    def stop(self):
        if not self.active:
            return
        self.active = False
        try:
            if self.telemetry_file:
                self.telemetry_file.flush()
                self.telemetry_file.close()
        except Exception:
            pass
        try:
            if self.frames_log_file:
                self.frames_log_file.flush()
                self.frames_log_file.close()
        except Exception:
            pass
        for w in (self.video_writers or {}).values():
            if w is not None:
                try:
                    w.release()
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
#                         Robot + UI scaffolding
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Joystick control for Dorna + RealSense + UVC microscopes (routine + injection)")
    p.add_argument("--host", "-H", default="10.42.0.11", help="IP address of your Dorna controller")
    p.add_argument("--port", "-P", type=int, default=443, help="Port for Dorna control (usually 443)")

    p.add_argument("--uvc1", default="", help="UVC camera #1 device path (default: auto-detect)")
    p.add_argument("--uvc2", default="", help="UVC camera #2 device path (default: auto-detect)")
    p.add_argument("--uvc-fps", type=int, default=30, help="FPS for UVCs")
    p.add_argument("--uvc-rotate", action="store_true", default=True, help="(legacy) rotate UVC cameras 180° (now overridden by GUI settings)")
    p.add_argument("--uvc-no-rotate", dest="uvc_rotate", action="store_false", help="Disable legacy 180° rotation (GUI rotation still applies)")
    p.add_argument("--uvc-try-index1", action="store_true", help="Also try the sibling video-index1 node if index0 yields no frames")

    p.add_argument("--fullscreen", action="store_true", help="Start in fullscreen (toggle with F11)")
    p.add_argument("--ui-min-width", type=int, default=320, help="Minimum UI panel width (pixels)")
    p.add_argument("--ui-frac", type=float, default=0.28, help="UI width fraction of window (0..1)")
    return p.parse_args()

def axis_angle_to_R(a, b, c):
    vec = np.radians([a, b, c]); th = np.linalg.norm(vec)
    if th < 1e-6:
        return np.eye(3)
    u = vec / th
    ux, uy, uz = u
    K = np.array([[  0, -uz,  uy],
                  [ uz,   0, -ux],
                  [-uy,  ux,   0]])
    return np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*(K @ K)

def R_to_axis_angle(R):
    tr = np.trace(R)
    th = np.arccos(max(min((tr - 1)/2, 1), -1))
    if abs(th) < 1e-6:
        return (0.0, 0.0, 0.0)
    ux = (R[2,1] - R[1,2])/(2*np.sin(th))
    uy = (R[0,2] - R[2,0])/(2*np.sin(th))
    uz = (R[1,0] - R[0,1])/(2*np.sin(th))
    return tuple(np.degrees([ux*th, uy*th, uz*th]))

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        # Joystick state
        self.lx = self.ly = self.rx = self.ry = 0.0
        self.hx = self.hy = 0
        self.lb = self.rb = False

        # Sensitivity levels (LB/RB to change)
        self.levels = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0]
        self.idx = self.levels.index(1.0)  # 100%

        # Tool attitude
        self.pitch = 0.0
        self.j5    = 0.0

        # Poses + settings
        self.poses = load_poses()
        for k, v in DEFAULT_POSES.items():
            self.poses.setdefault(k, v.copy())
        self.settings = load_settings()

        self.tool_lz     = float(self.settings.get("tool_lz", DEFAULT_TOOL_LZ))
        self.approach_mm = float(self.settings.get("approach_mm", DEFAULT_APPROACH_MM))

        # Routine / pose state
        self.await_confirm = False
        self.await_target  = None
        self.confirm_msg   = ""
        self.current_pose_name = ""
        self.routine_phase       = "Idle"
        self.routine_status_text = "Idle"
        self.last_endstop_line   = ""
        self.last_plunger_note   = ""
        self.pose_motion_until   = 0.0

        # Syringe calibration status
        self.calibration_status  = "Not calibrated"
        self.syringe_calibration_running = False

        # Last robot joints snapshot (for logging)
        self.last_joints = None

        # Injection logging toggles
        self.record_telemetry = False
        self.record_video     = False
        self.record_ui        = False  # NEW: record full UI overlay
        self.current_injection_step = 0

        # NEW: sawing parameters/state
        self.saw_enabled = False
        self.saw_amp_deg = float(self.settings.get("saw_amp_deg", 3.0))
        self.saw_freq_hz = float(self.settings.get("saw_freq_hz", 1.0))

        # NEW: retract / advance distances
        self.injection_retract_mm = float(
            self.settings.get("injection_retract_mm", 10.0)
        )
        self.injection_advance_mm = float(
            self.settings.get("injection_advance_mm", 0.0)
        )
        if self.injection_advance_mm > self.injection_retract_mm:
            self.injection_advance_mm = self.injection_retract_mm


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

        # Sawing motion (J5 oscillation) state
        self.saw_prev_offset = 0.0
        self.saw_phase = 0.0
        self.saw_last_t = time.time()

    def _set_current_named(self, name: str | None):
        self.current_named = name
        with self.state.lock:
            self.state.current_pose_name = name or ""

    def enqueue(self, cmd: dict):
        self.cmd_q.put(cmd)

    def _refresh_from_robot(self):
        pr = self.robot.get_all_pose()[:6]
        self.x0, self.y0, self.z0, a, b, c = pr
        self.R = axis_angle_to_R(a, b, c)
        tz = self.R[:,2]
        pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
        with self.state.lock:
            self.state.pitch = pitch

    def _mark_motion_for(self, seconds: float):
        with self.state.lock:
            self.state.pose_motion_until = max(
                self.state.pose_motion_until,
                time.time() + max(0.0, seconds)
            )

    def _queue_jmove_to_pose(self, pose: dict):
        go = {"cmd":"jmove","rel":0,"vel":self.VR_POSE}
        go.update(pose)
        self.robot.play_dict(go)
        self._mark_motion_for(0.6)
        self.skip_manual_until = max(self.skip_manual_until, time.time() + 0.35)
        time.sleep(0.05)
        try:
            self._refresh_from_robot()
            if "j5" in pose:
                with self.state.lock:
                    self.state.j5 = float(pose["j5"])
                self.j5v = float(pose["j5"])
        except Exception:
            pass

    def _tool_move_along_tz(self, dist_mm: float, cont: int = 0):
        tz = self.R[:,2]
        dx, dy, dz = tz[0]*dist_mm, tz[1]*dist_mm, tz[2]*dist_mm
        self.robot.play_dict({"cmd":"lmove","rel":1,"x":dx,"y":dy,"z":dz,"vel":self.VT,"cont":cont})
        self.x0 += dx
        self.y0 += dy
        self.z0 += dz
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
                        if all(v is not None for v in vals):
                            joints = vals
                            break
                        try:
                            joints = [res[i] for i in range(6)]
                            break
                        except Exception:
                            pass
                    elif isinstance(res, (list, tuple)) and len(res) >= 6:
                        joints = list(res[:6])
                        break
            except Exception:
                continue
        if joints is None:
            return None
        return {"j0": float(joints[0]), "j1": float(joints[1]), "j2": float(joints[2]),
                "j3": float(joints[3]), "j4": float(joints[4]), "j5": float(joints[5])}

    def _save_current_pose_with_midway(self, name: str, approach_mm: float):
        final_joints = self._try_get_current_joints()
        if final_joints is None:
            print("⚠️ Could not fetch joint angles; save operation skipped.")
            return False
        with self.state.lock:
            self.state.poses[name] = final_joints
        save_poses(self.state.poses)
        print(f"[Robot] Saved current as '{name}' (final).")

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
            self._tool_move_along_tz(+approach_mm, cont=0)
            self._set_current_named(name)
            print(f"[Robot] Returned to '{name}'.")
        return True

    def _ensure_current_midway(self, approach_mm: float):
        if not self.current_named:
            return
        name = self.current_named
        if name == "Default":
            return
        if name.endswith(MIDWAY_SUFFIX):
            return
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
        if self.current_named and str(self.current_named).endswith(MIDWAY_SUFFIX):
            final = self.current_named[: -len(MIDWAY_SUFFIX)]
            try:
                self._tool_move_along_tz(+approach_mm, cont=0)
                self._set_current_named(final)
                print(f"[Robot] Advanced +{approach_mm} mm to '{final}'.")
            except Exception as e:
                print(f"⚠️ Advance from '{self.current_named}' failed: {e}")

    def run(self):
        robot = Dorna()
        self.robot = robot
        for attempt in range(5):
            if robot.connect(host=self.host, port=self.port):
                break
            print(f"⚠️  Connection attempt {attempt+1}/5 to {self.host}:{self.port} failed. Retrying…")
            time.sleep(2)
        else:
            print(f"❌ All connection attempts to {self.host}:{self.port} failed.")
            sys.exit(1)

        robot.set_motor(1)

        with self.state.lock:
            tool_lz = float(self.state.tool_lz)
        robot.play_dict(cmd={"cmd":"tool","id":1,"lz":tool_lz})

        try:
            with self.state.lock:
                default_pose = self.state.poses["Default"].copy()
            self._queue_jmove_to_pose(default_pose)
            self._set_current_named("Default")
            print("[Robot] Ready at Default.")
        except Exception as e:
            print(f"⚠️ Could not home to Default pose: {e}")
            try:
                self._refresh_from_robot()
            except Exception:
                pass

        with self.state.lock:
            if "Default" in self.state.poses:
                self.j5v = self.state.poses["Default"].get("j5", 0.0)
                self.state.j5 = self.j5v

        next_t = time.time()
        joint_poll_time = 0.0

        while not self.stop_event.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
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

                        self._ensure_current_midway(approach_mm)

                        target_mid = midway_name(target)
                        if target_mid not in poses:
                            print(f"⚠️ Refusing move: '{target_mid}' not found. Jog to '{target}' and press X to teach it (when injection inactive).")
                            continue

                        try:
                            self._queue_jmove_to_pose(poses[target_mid])
                            self._set_current_named(target_mid)
                            with self.state.lock:
                                self.state.await_confirm = True
                                self.state.await_target  = target
                                self.state.confirm_msg   = (
                                    f"Paused at '{target_mid}'. Press A to advance +{approach_mm} mm, or B for free control."
                                )
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
                        try:
                            self.robot.play_dict({"cmd": "halt"})
                            print("[Robot] Motion halted.")
                        except Exception as e:
                            print(f"⚠️ halt failed: {e}")

                    elif typ == "kill_motors":
                        try:
                            self.robot.set_motor(0)
                            print("[Robot] Motors killed.")
                        except Exception as e:
                            print(f"⚠️ kill_motors failed: {e}")

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
                            s_val = float(cmd.get("s"))
                            with self.state.lock:
                                self.state.settings["pose_final_wait_s"] = s_val
                            save_settings(self.state.settings)
                            print(f"[Routine] Wash wait time set to {s_val} s (≤0 = none; min enforced = 0.35 s)")
                        except Exception as e:
                            print(f"⚠️ set_pose_wait failed: {e}")

                    elif typ == "save_reload_with_midway":
                        with self.state.lock:
                            approach_mm = float(self.state.approach_mm)
                        ok = self._save_current_pose_with_midway("Reload", approach_mm)
                        if ok:
                            save_poses(self.state.poses)

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
                        with self.state.lock:
                            approach_mm = float(self.state.approach_mm)
                        ok = self._save_current_pose_with_midway(name, approach_mm)
                        if ok:
                            save_poses(self.state.poses)

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
                                poses.pop(name)
                                removed_any = True
                            mid = midway_name(name)
                            if mid in poses:
                                poses.pop(mid)
                                removed_any = True
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
                            print("⚠️ overwrite_pose: empty name")
                            continue
                        with self.state.lock:
                            approach_mm = float(self.state.approach_mm)
                        ok = self._save_current_pose_with_midway(name, approach_mm)
                        if ok:
                            save_poses(self.state.poses)

                    # NEW: injection retract / advance commands used by UI
                    elif typ == "inject_retract":
                        try:
                            dist = float(cmd.get("dist", 0.0) or 0.0)
                        except Exception:
                            dist = 0.0
                        if dist <= 0:
                            print("⚠️ inject_retract: non-positive distance; ignoring.")
                            continue
                        try:
                            self._tool_move_along_tz(-dist, cont=0)
                            print(f"[Robot] Injection retract {dist:.3f} mm along tool-Z.")
                        except Exception as e:
                            print(f"⚠️ inject_retract failed: {e}")

                    elif typ == "inject_advance":
                        try:
                            dist = float(cmd.get("dist", 0.0) or 0.0)
                        except Exception:
                            dist = 0.0
                        if dist <= 0:
                            print("⚠️ inject_advance: non-positive distance; ignoring.")
                            continue
                        try:
                            self._tool_move_along_tz(+dist, cont=0)
                            print(f"[Robot] Injection advance {dist:.3f} mm along tool-Z.")
                        except Exception as e:
                            print(f"⚠️ inject_advance failed: {e}")

            except queue.Empty:
                pass

            with self.state.lock:
                lx, ly = self.state.lx, self.state.ly
                rx, ry = self.state.rx, self.state.ry
                hx, hy = self.state.hx, self.state.hy
                sens   = self.state.levels[self.state.idx]
                waiting = self.state.await_confirm

            sx, sj5, sb, sc, sh = (5.0*sens, 5.0*sens, 0.5*sens, 0.5*sens, 5.0*sens)
            manual_enabled = (time.time() >= self.skip_manual_until) and (not waiting)

            if manual_enabled and abs(ly) > self.DZ:
                if abs(ly) >= abs(lx):
                    d = -ly * sx
                    tz = self.R[:,2]
                    self.robot.play_dict({"cmd":"lmove","rel":1,
                                          "x":tz[0]*d,"y":tz[1]*d,"z":tz[2]*d,
                                          "vel":self.VT,"cont":1})
                    self.x0 += tz[0]*d
                    self.y0 += tz[1]*d
                    self.z0 += tz[2]*d

            if manual_enabled and abs(lx) > self.DZ and abs(lx) > abs(ly):
                delta = lx * sj5
                self.j5v += delta
                self.R = self.R @ axis_angle_to_R(0, 0, delta)
                self.robot.play_dict({"cmd":"jmove","rel":1,"j5":delta,"vel":self.VR})
                with self.state.lock:
                    self.state.j5 = self.j5v

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
                self.robot.play_dict({"cmd":"lmove","rel":0,
                                      "x":self.x0,"y":self.y0,"z":self.z0,
                                      "a":a1,"b":b1,"c":c1,"vel":self.VR})
                tz = self.R[:,2]
                pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
                with self.state.lock:
                    self.state.pitch = pitch

            if manual_enabled:
                tz = self.R[:,2]
                tx = self.R[:,0]
                ty = self.R[:,1]
                proj = np.array([tz[0], tz[1], 0.0])
                n = np.linalg.norm(proj)
                at_default = (self.current_named == "Default")

                if hy != 0:
                    step = float(hy) * sh * DPAD_Z_SIGN
                    if at_default:
                        self.robot.play_dict({"cmd":"lmove","rel":1,
                                              "x":0.0,"y":0.0,"z":step,
                                              "vel":self.VT,"cont":1})
                        self.z0 += step
                    else:
                        up_axis = tx if abs(tx[2]) >= abs(ty[2]) else ty
                        dx, dy, dz = up_axis[0]*step, up_axis[1]*step, up_axis[2]*step
                        self.robot.play_dict({"cmd":"lmove","rel":1,
                                              "x":dx,"y":dy,"z":dz,
                                              "vel":self.VT,"cont":1})
                        self.x0 += dx
                        self.y0 += dy
                        self.z0 += dz

                hx_eff = -hx
                if hx_eff != 0:
                    if n > 1e-3:
                        perp = np.array([-proj[1], proj[0], 0.0]) / n * hx_eff
                    else:
                        perp = np.array([0.0, -hx_eff, 0.0])
                    dx, dy = perp[0]*sh, perp[1]*sh
                    self.robot.play_dict({"cmd":"lmove","rel":1,
                                          "x":dx,"y":dy,"z":0.0,
                                          "vel":self.VT,"cont":1})
                    self.x0 += dx
                    self.y0 += dy

            # Sawing motion around J5 during injection (if enabled)
            with self.state.lock:
                saw_on = bool(self.state.saw_enabled)
                inj_active_flag = bool(self.state.settings.get("injection_active", False))
                # CHANGED: use saw_amp_deg / saw_freq_hz keys consistent with load_settings
                saw_amp_deg = float(self.state.settings.get("saw_amp_deg", self.state.saw_amp_deg) or 0.0)
                saw_freq_hz = float(self.state.settings.get("saw_freq_hz", self.state.saw_freq_hz) or 0.0)
            if manual_enabled and saw_on and inj_active_flag and saw_amp_deg > 0 and saw_freq_hz > 0:
                now_t = time.time()
                dt_saw = now_t - self.saw_last_t
                if dt_saw < 0:
                    dt_saw = 0
                self.saw_last_t = now_t
                self.saw_phase += 2.0 * np.pi * saw_freq_hz * dt_saw
                new_offset = saw_amp_deg * np.sin(self.saw_phase)
                delta = new_offset - self.saw_prev_offset
                if abs(delta) > 1e-4:
                    self.saw_prev_offset = new_offset
                    self.j5v += delta
                    self.R = self.R @ axis_angle_to_R(0, 0, delta)
                    try:
                        self.robot.play_dict({"cmd": "jmove", "rel": 1, "j5": delta, "vel": self.VR})
                    except Exception:
                        pass
                    with self.state.lock:
                        self.state.j5 = self.j5v
            else:
                # Reset so next enable starts relative to current pose
                self.saw_prev_offset = 0.0
                self.saw_phase = 0.0
                self.saw_last_t = time.time()

            # Periodic joint snapshot for logging
            if time.time() - joint_poll_time >= 0.05:
                joints = self._try_get_current_joints()
                with self.state.lock:
                    self.state.last_joints = joints
                joint_poll_time = time.time()

        try:
            self.robot.set_motor(0)
        except Exception:
            pass
        try:
            self.robot.close()
        except Exception:
            pass

    def stop(self):
        self.stop_event.set()

# ─────────────────────────────────────────────────────────────────────────────
#                                UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_layout(screen_w, screen_h, ui_frac=0.28, ui_min_w=320):
    half_h = screen_h // 2
    ui_w = max(ui_min_w, int(screen_w * ui_frac))
    left_w = max(1, screen_w - ui_w)

    tl = pygame.Rect(0,        0,      left_w, half_h)
    tr = pygame.Rect(left_w,   0,      ui_w,   half_h)

    bottom_h = screen_h - half_h
    half_w = screen_w // 2
    bl = pygame.Rect(0,          half_h, half_w,           bottom_h)
    br = pygame.Rect(half_w,     half_h, screen_w-half_w,  bottom_h)
    return tl, tr, bl, br

def blit_frame_fit(screen, frame_bgr, rect):
    x, y, w, h = rect
    pygame.draw.rect(screen, (0,0,0), rect)
    if frame_bgr is None:
        return False
    fh, fw = frame_bgr.shape[:2]
    if fh <= 0 or fw <= 0:
        return False
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
    if flip_h and flip_v:
        img = cv2.flip(img, -1)
    else:
        if flip_h:
            img = cv2.flip(img, 1)
        if flip_v:
            img = cv2.flip(img, 0)
    return img

def draw_uvc_controls(screen, rect, rot90, flip_h, flip_v, font):
    buttons = []
    pad = 8
    btn_w = 28
    btn_h = 20
    gap = 4
    x = rect.x + pad
    y = rect.y + pad

    r_rect = pygame.Rect(x, y, btn_w, btn_h)
    pygame.draw.rect(screen, (0,0,0), r_rect)
    pygame.draw.rect(screen, (0,255,127), r_rect, 1)
    label = font.render("⟳", True, (0,255,127))
    screen.blit(label, (r_rect.x + (btn_w - label.get_width())//2,
                        r_rect.y + (btn_h - label.get_height())//2))
    buttons.append((r_rect, "rot"))

    x += btn_w + gap
    h_rect = pygame.Rect(x, y, btn_w, btn_h)
    col_fill = (40,40,40) if not flip_h else (0,60,60)
    pygame.draw.rect(screen, col_fill, h_rect)
    pygame.draw.rect(screen, (0,255,127), h_rect, 1)
    label = font.render("H", True, (0,255,127))
    screen.blit(label, (h_rect.x + (btn_w - label.get_width())//2,
                        h_rect.y + (btn_h - label.get_height())//2))
    buttons.append((h_rect, "flip_h"))

    x += btn_w + gap
    v_rect = pygame.Rect(x, y, btn_w, btn_h)
    col_fill = (40,40,40) if not flip_v else (0,60,60)
    pygame.draw.rect(screen, col_fill, v_rect)
    pygame.draw.rect(screen, (0,255,127), v_rect, 1)
    label = font.render("V", True, (0,255,127))
    screen.blit(label, (v_rect.x + (btn_w - label.get_width())//2,
                        v_rect.y + (btn_h - label.get_height())//2))
    buttons.append((v_rect, "flip_v"))

    deg = (int(rot90) % 4) * 90
    info = font.render(f"{deg}°", True, (200,200,200))
    screen.blit(info, (v_rect.right + 4, v_rect.y + (btn_h - info.get_height())//2))

    return buttons

def wrap_lines(font: pygame.font.Font, text: str, max_width: int):
    return wrap_text(text, font, max_width)

def draw_center_popup_vertical(screen, title_lines, button_items, width=680, btn_width=560,
                               line_h=26, btn_pad=10, gap=10):
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

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
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
    lines = [f"Edit pose: {pose_name}"]
    buttons = []

    if disable_rename_delete:
        buttons.append(("Rename (disabled for reserved pose)", "disabled_rename"))
        buttons.append(("Delete (disabled for reserved pose)", "disabled_delete"))
    else:
        buttons.append(("Rename", "rename"))
        buttons.append(("Delete", "delete"))

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
    x = (sw - panel_w)//2
    y = (sh - panel_h)//2
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
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

def draw_routine_editor(screen, editor: EditorState, title="Edit Routine (Ctrl+Enter=Apply, Esc=Close)",
                        height_lines=12, width_chars=56):
    sw, sh = screen.get_size()
    font = pygame.font.SysFont("Consolas", 16)
    char_w, char_h = font.size("M")[0], font.get_height()

    panel_w = int(width_chars * char_w + 40)
    panel_h = int(height_lines * char_h + 120)
    x = (sw - panel_w)//2
    y = (sh - panel_h)//2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
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
            pygame.draw.rect(screen, (255,255,255),
                             (caret_x_px, caret_y_px, max(2, char_w//9), char_h), 0)

    status_text = f"Ln {line_idx+1}, Col {col_idx+1}    ({total_lines} lines)"
    st = font.render(status_text, True, (200,200,200))
    screen.blit(st, (x + panel_w - st.get_width() - 12, y + panel_h - st.get_height() - 8))

    return panel, box, (char_w, char_h)

def editor_handle_key(editor: EditorState, ev, width_chars=56):
    text = editor.text
    caret = editor.caret
    changed = False

    if ev.key == pygame.K_LEFT:
        if caret > 0:
            editor.caret -= 1
    elif ev.key == pygame.K_RIGHT:
        if caret < len(text):
            editor.caret += 1
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

def draw_syringe_calibration_modal(screen, vol_text, step_text, status_text, active_field):
    sw, sh = screen.get_size()
    font = pygame.font.SysFont("Consolas", 16)
    small = pygame.font.SysFont("Consolas", 14)

    panel_w = 560
    panel_h = 260
    x = (sw - panel_w)//2
    y = (sh - panel_h)//2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    title = font.render("Syringe Calibration", True, (230,230,230))
    screen.blit(title, (x + 16, y + 14))

    label_color = (210,210,210)
    box_color   = (50,50,80)
    active_col  = (0,255,127)
    idle_col    = (0,150,90)

    y1 = y + 50
    label1 = font.render("Total syringe volume (µL):", True, label_color)
    screen.blit(label1, (x + 16, y1))
    box1 = pygame.Rect(x + 16 + label1.get_width() + 8, y1 - 4, panel_w - label1.get_width() - 40, 30)
    pygame.draw.rect(screen, box_color, box1)
    pygame.draw.rect(screen, active_col if active_field=="vol" else idle_col, box1, 2)
    t1 = font.render(vol_text or "", True, (255,255,255))
    screen.blit(t1, (box1.x + 6, box1.y + 5))

    y2 = y1 + 50
    label2 = font.render("Step size (µL per injection):", True, label_color)
    screen.blit(label2, (x + 16, y2))
    box2 = pygame.Rect(x + 16 + label2.get_width() + 8, y2 - 4, panel_w - label2.get_width() - 40, 30)
    pygame.draw.rect(screen, box_color, box2)
    pygame.draw.rect(screen, active_col if active_field=="step" else idle_col, box2, 2)
    t2 = font.render(step_text or "", True, (255,255,255))
    screen.blit(t2, (box2.x + 6, box2.y + 5))

    btn_w = 150
    btn_h = 32
    spacing = 10
    total_btn_w = btn_w * 3 + spacing * 2
    bx = x + panel_w - (total_btn_w + 16)
    by = y + panel_h - btn_h - 16

    btn_cal = pygame.Rect(bx, by, btn_w, btn_h)
    btn_apply = pygame.Rect(bx + btn_w + spacing, by, btn_w, btn_h)
    btn_cancel = pygame.Rect(bx + (btn_w + spacing) * 2, by, btn_w, btn_h)

    for rect, text in [
        (btn_cal, "Calibrate stroke"),
        (btn_apply, "Apply volume/step"),
        (btn_cancel, "Close"),
    ]:
        pygame.draw.rect(screen, (50,50,80), rect)
        pygame.draw.rect(screen, (0,255,127), rect, 2)
        tt = font.render(text, True, (230,230,230))
        screen.blit(tt, (rect.x + (rect.w - tt.get_width())//2,
                         rect.y + (rect.h - tt.get_height())//2))

    if status_text:
        max_w = panel_w - 32
        lines = wrap_lines(small, f"Status: {status_text}", max_w)
        sy = y + panel_h - btn_h - 40
        for ln in lines[-2:]:
            stxt = small.render(ln, True, (210,210,210))
            screen.blit(stxt, (x + 16, sy))
            sy += stxt.get_height() + 2

    return {"vol": box1, "step": box2}, {
        "calibrate": btn_cal,
        "apply": btn_apply,
        "cancel": btn_cancel,
    }, panel

def draw_injection_info_modal(
    screen,
    study_text,
    material_text,
    date_text,
    active_field,
    record_telemetry,
    record_video,
    record_ui,
):
    """
    Start Injection popup: Study, Injection Material, Date
    plus checkboxes for telemetry, video, and UI overlay recording.
    """
    sw, sh = screen.get_size()
    font  = pygame.font.SysFont("Consolas", 16)
    small = pygame.font.SysFont("Consolas", 14)

    panel_w = 680
    panel_h = 380
    x = (sw - panel_w)//2
    y = (sh - panel_h)//2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    title = font.render("Start Injection — metadata & logging", True, (230,230,230))
    screen.blit(title, (x + 16, y + 14))

    label_color = (210,210,210)
    box_color   = (50,50,80)
    active_col  = (0,255,127)
    idle_col    = (0,150,90)

    def field(ypos, text, label, key):
        lbl = font.render(label, True, label_color)
        screen.blit(lbl, (x + 16, ypos))
        box = pygame.Rect(
            x + 16 + lbl.get_width() + 8,
            ypos - 4,
            panel_w - lbl.get_width() - 40,
            30,
        )
        pygame.draw.rect(screen, box_color, box)
        pygame.draw.rect(
            screen,
            active_col if active_field == key else idle_col,
            box,
            2,
        )
        tt = font.render(text or "", True, (255,255,255))
        screen.blit(tt, (box.x + 6, box.y + 5))
        return box

    y1 = y + 60
    box_study   = field(y1, study_text,        "Study:",              "study")
    y2 = y1 + 42
    box_mat     = field(y2, material_text,     "Injection Material:", "material")
    y3 = y2 + 42
    box_date    = field(y3, date_text,         "Date:",               "date")

    cb_size = 18
    cb_y1 = y3 + 40

    # Telemetry
    cb1 = pygame.Rect(x + 20, cb_y1, cb_size, cb_size)
    pygame.draw.rect(screen, (50,50,80), cb1)
    pygame.draw.rect(screen, (0,255,127), cb1, 2)
    if record_telemetry:
        pygame.draw.line(screen, (0,255,127),
                         (cb1.x+3, cb1.y+cb_size//2),
                         (cb1.x+cb_size//2, cb1.y+cb_size-3), 2)
        pygame.draw.line(screen, (0,255,127),
                         (cb1.x+cb_size//2, cb1.y+cb_size-3),
                         (cb1.x+cb_size-3, cb1.y+3), 2)
    lbl1 = font.render("Record robot joints + controller inputs", True, label_color)
    screen.blit(lbl1, (cb1.right + 8, cb1.y + (cb_size - lbl1.get_height())//2))

    # Video
    cb_y2 = cb_y1 + 28
    cb2 = pygame.Rect(x + 20, cb_y2, cb_size, cb_size)
    pygame.draw.rect(screen, (50,50,80), cb2)
    pygame.draw.rect(screen, (0,255,127), cb2, 2)
    if record_video:
        pygame.draw.line(screen, (0,255,127),
                         (cb2.x+3, cb2.y+cb_size//2),
                         (cb2.x+cb_size//2, cb2.y+cb_size-3), 2)
        pygame.draw.line(screen, (0,255,127),
                         (cb2.x+cb_size//2, cb2.y+cb_size-3),
                         (cb2.x+cb_size-3, cb2.y+3), 2)
    lbl2 = font.render("Record camera video (RealSense + both UVC feeds)", True, label_color)
    screen.blit(lbl2, (cb2.right + 8, cb2.y + (cb_size - lbl2.get_height())//2))

    # UI overlay
    cb_y3 = cb_y2 + 28
    cb3 = pygame.Rect(x + 20, cb_y3, cb_size, cb_size)
    pygame.draw.rect(screen, (50,50,80), cb3)
    pygame.draw.rect(screen, (0,255,127), cb3, 2)
    if record_ui:
        pygame.draw.line(screen, (0,255,127),
                         (cb3.x+3, cb3.y+cb_size//2),
                         (cb3.x+cb_size//2, cb3.y+cb_size-3), 2)
        pygame.draw.line(screen, (0,255,127),
                         (cb3.x+cb_size//2, cb3.y+cb_size-3),
                         (cb3.x+cb_size-3, cb3.y+3), 2)
    lbl3 = font.render("Record full UI overlay per step", True, label_color)
    screen.blit(lbl3, (cb3.right + 8, cb3.y + (cb_size - lbl3.get_height())//2))

    # Buttons
    btn_w = 180
    btn_h = 32
    bx = x + panel_w - (btn_w*2 + 16 + 10)
    by = y + panel_h - btn_h - 16
    btn_start  = pygame.Rect(bx,               by, btn_w, btn_h)
    btn_cancel = pygame.Rect(bx + btn_w + 10,  by, btn_w, btn_h)

    for rect, text in [(btn_start, "Confirm & Activate"), (btn_cancel, "Cancel")]:
        pygame.draw.rect(screen, (50,50,80), rect)
        pygame.draw.rect(screen, (0,255,127), rect, 2)
        tt = font.render(text, True, (230,230,230))
        screen.blit(
            tt,
            (rect.x + (rect.w - tt.get_width())//2,
             rect.y + (rect.h - tt.get_height())//2),
        )

    return (
        {"study": box_study, "material": box_mat, "date": box_date},
        {"start": btn_start, "cancel": btn_cancel},
        {"telemetry": cb1, "video": cb2, "ui": cb3},
        panel,
    )

def draw_subject_modal(screen, subject_text, step_index, allow_advance):
    """
    Popup to capture Subject/Animal for this injection step.

    step_index: 1-based index of the upcoming step.
    allow_advance: if True, show "Confirm & Advance" button.
    """
    sw, sh = screen.get_size()
    font  = pygame.font.SysFont("Consolas", 16)
    small = pygame.font.SysFont("Consolas", 14)

    panel_w = 620
    panel_h = 240
    x = (sw - panel_w)//2
    y = (sh - panel_h)//2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,190))
    screen.blit(overlay, (0,0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    title = font.render(
        f"Subject / Animal for injection step {step_index}",
        True, (230,230,230),
    )
    screen.blit(title, (x + 16, y + 14))

    info_lines = [
        "Each syringe step can have its own Subject/Animal ID.",
        "This value will be stored with the per-step recording.",
    ]
    ty = y + 44
    for ln in info_lines:
        t = small.render(ln, True, (210,210,210))
        screen.blit(t, (x + 16, ty))
        ty += t.get_height() + 2

    label = font.render("Subject / Animal ID:", True, (210,210,210))
    screen.blit(label, (x + 16, ty + 6))

    box = pygame.Rect(x + 16, ty + 30, panel_w - 32, 32)
    pygame.draw.rect(screen, (50,50,80), box)
    pygame.draw.rect(screen, (0,255,127), box, 2)
    txt = font.render(subject_text or "", True, (255,255,255))
    screen.blit(txt, (box.x + 8, box.y + (box.h - txt.get_height())//2))

    # Buttons
    btn_w = 170
    btn_h = 32
    gap = 12
    n_btn = 3 if allow_advance else 2
    total_w = n_btn*btn_w + (n_btn-1)*gap
    bx = x + (panel_w - total_w)//2
    by = y + panel_h - btn_h - 18

    btn_confirm = pygame.Rect(bx, by, btn_w, btn_h)
    pygame.draw.rect(screen, (50,80,50), btn_confirm)
    pygame.draw.rect(screen, (0,255,127), btn_confirm, 2)
    t1 = font.render("Confirm", True, (230,230,230))
    screen.blit(t1, (btn_confirm.x + (btn_w - t1.get_width())//2,
                     btn_confirm.y + (btn_h - t1.get_height())//2))

    btn_confirm_adv = None
    if allow_advance:
        bx2 = bx + btn_w + gap
        btn_confirm_adv = pygame.Rect(bx2, by, btn_w, btn_h)
        pygame.draw.rect(screen, (50,60,80), btn_confirm_adv)
        pygame.draw.rect(screen, (0,255,200), btn_confirm_adv, 2)
        t2 = font.render("Confirm & Advance", True, (230,230,230))
        screen.blit(t2, (btn_confirm_adv.x + (btn_w - t2.get_width())//2,
                         btn_confirm_adv.y + (btn_h - t2.get_height())//2))
        bx_cancel = bx2 + btn_w + gap
    else:
        bx_cancel = bx + btn_w + gap

    btn_cancel = pygame.Rect(bx_cancel, by, btn_w, btn_h)
    pygame.draw.rect(screen, (80,50,50), btn_cancel)
    pygame.draw.rect(screen, (255,80,80), btn_cancel, 2)
    t3 = font.render("Cancel", True, (230,230,230))
    screen.blit(t3, (btn_cancel.x + (btn_w - t3.get_width())//2,
                     btn_cancel.y + (btn_h - t3.get_height())//2))

    # Controller hint
    hint = "A = Confirm, B = Cancel"
    if allow_advance:
        hint += ", X = Confirm & Advance"
    hh = small.render(hint, True, (200,200,200))
    screen.blit(hh, (x + 16, y + panel_h - hh.get_height() - 6))

    return box, {
        "confirm": btn_confirm,
        "confirm_adv": btn_confirm_adv,
        "cancel": btn_cancel,
    }, panel


# ─────────────────────────────────────────────────────────────────────────────
#                                   main()
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ─────────────────────────────────────────────────────────────
    # Serial → Arduino plunger bridge
    # ─────────────────────────────────────────────────────────────
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.05)
        time.sleep(2)
        print(f"[Bridge] Serial open {SERIAL_PORT} @ {BAUDRATE}")
    except Exception as e:
        print(f"⚠️ Could not open serial port: {e}", file=sys.stderr)
        ser = None
    last_rate = None  # last plunger rate we sent

    # ─────────────────────────────────────────────────────────────
    # Pygame + joystick
    # ─────────────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("❌ No joystick detected.")
        return
    joy = pygame.joystick.Joystick(0)
    joy.init()

    # ─────────────────────────────────────────────────────────────
    # RealSense
    # ─────────────────────────────────────────────────────────────
    rs_thread = RealSenseThread(width=640, height=480, fps=30)
    rs_thread.start()
    t0 = time.time()
    while rs_thread.latest() is None and time.time() - t0 < 2.0:
        time.sleep(0.01)
    print("[RS] ", rs_thread.status())

    # ─────────────────────────────────────────────────────────────
    # UVC detection + startup
    # ─────────────────────────────────────────────────────────────
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

    uvc_threads = []
    if uvc1_path:
        uvc1 = UvcThread(
            uvc1_path, 640, 480, args.uvc_fps, False,
            name="UVC#1", try_index1_fallback=args.uvc_try_index1
        )
        uvc1.start()
        uvc_threads.append(uvc1)
        print(f"[UVC#1] {uvc1_path}")
    else:
        uvc1 = None
        print("[UVC#1] Not found (placeholder)")

    if uvc2_path:
        uvc2 = UvcThread(
            uvc2_path, 640, 480, args.uvc_fps, False,
            name="UVC#2", try_index1_fallback=args.uvc_try_index1
        )
        uvc2.start()
        uvc_threads.append(uvc2)
        print(f"[UVC#2] {uvc2_path}")
    else:
        uvc2 = None
        print("[UVC#2] Not found (placeholder)")

    def _status(s):
        return s.status() if s else ""

    # Simple FPS fallback if UVC is unhappy at high FPS
    if (args.uvc_fps > 15) and (uvc1 or uvc2):
        time.sleep(2.0)
        s1 = _status(uvc1)
        s2 = _status(uvc2)
        if ("no frames" in s1.lower()) or ("no frames" in s2.lower()):
            print("[UVC] Fallback: restarting both UVC threads at 15 fps…")
            for t in (uvc1, uvc2):
                if t:
                    t.stop()
            for t in (uvc1, uvc2):
                if t:
                    t.join()
            uvc_threads.clear()
            if uvc1_path:
                uvc1 = UvcThread(
                    uvc1_path, 640, 480, 15, False,
                    name="UVC#1", try_index1_fallback=args.uvc_try_index1
                )
                uvc1.start()
                uvc_threads.append(uvc1)
            if uvc2_path:
                uvc2 = UvcThread(
                    uvc2_path, 640, 480, 15, False,
                    name="UVC#2", try_index1_fallback=args.uvc_try_index1
                )
                uvc2.start()
                uvc_threads.append(uvc2)

    # ─────────────────────────────────────────────────────────────
    # Window / fonts
    # ─────────────────────────────────────────────────────────────
    flags = 0
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
        screen = pygame.display.set_mode((0, 0), flags)
    else:
        screen = pygame.display.set_mode((1280, 960), flags)
    pygame.display.set_caption(
        "Robotic Injector — RS+UVC (routine + syringe calibration + logging)"
    )
    clock = pygame.time.Clock()

    label_font       = pygame.font.SysFont("Consolas", 14, bold=True)
    value_font       = pygame.font.SysFont("Consolas", 12)
    placeholder_font = pygame.font.SysFont("Consolas", 12)
    big_font         = pygame.font.SysFont("Consolas", 24, bold=True)
    small_font       = pygame.font.SysFont("Consolas", 12)

    # ─────────────────────────────────────────────────────────────
    # Shared state + Robot thread
    # ─────────────────────────────────────────────────────────────
    state = SharedState()
    with state.lock:
        state.poses["Reload"] = DEFAULT_POSES["Reload"].copy()
    save_poses(state.poses)
    save_settings(state.settings)

    rt = RobotThread(state, host=args.host, port=args.port)
    rt.start()

    # ─────────────────────────────────────────────────────────────
    # Routine editor / executor
    # ─────────────────────────────────────────────────────────────
    initial_routine_text = load_routine_text()
    routine_editor = EditorState(text=initial_routine_text,
                                 caret=len(initial_routine_text))
    routine_edit_active = False
    routine_executor = None

    # ─────────────────────────────────────────────────────────────
    # UI modal flags / text fields
    # ─────────────────────────────────────────────────────────────
    toollen_input_active  = False
    approach_input_active = False
    naming_input_active   = False
    rename_input_active   = False
    delete_confirm_active = False
    overwrite_confirm_active = False
    edit_menu_active      = False
    edit_menu_target      = ""

    fwd_rate_input_active = False
    bwd_rate_input_active = False
    endstop_timeout_input_active = False
    pose_wait_input_active       = False

    syringe_calib_active  = False
    syringe_field_active  = "vol"

    injection_info_active  = False
    injection_field_active = "study"
    post_step_prompt_active = False
    subject_prompt_active   = False

    # text fields
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

    with state.lock:
        s_vol   = state.settings.get("syringe_volume_ul", 10.0)
        s_step  = state.settings.get("syringe_step_ul", 1.0)
        inj_active_flag = state.settings.get("injection_active", False)
        inj_study   = state.settings.get("injection_study", "")
        inj_subject = state.settings.get("injection_subject", "")
        inj_material= state.settings.get("injection_material", "")
        inj_date    = state.settings.get("injection_date", "")

    syringe_vol_text    = f"{s_vol:.3f}"
    syringe_step_ul_txt = f"{s_step:.3f}"

    injection_study_text   = inj_study
    injection_subject_text = inj_subject
    injection_material_text= inj_material
    injection_date_text    = inj_date

    injection_record_telemetry = True
    injection_record_video     = True
    injection_record_ui        = False  # NEW: record UI overlay per step

    # Injection step state
    injection_step_in_progress = False
    injection_step_done        = False
    injection_step_distance    = 0.0
    injection_step_target      = 0.0
    injection_step_volume_ul   = 0.0
    injection_last_step_small  = False
    injection_current_rate     = 0.0
    last_right_trigger_down    = False

    # Final small-step warning
    final_small_warning_active   = False
    final_small_step_volume_ul   = 0.0
    final_small_step_target      = 0.0

    # Subject popup text (per-step)
    subject_field_text = injection_subject_text or ""

    # UI scrolling on right panel
    ui_scroll = 0
    ui_content_height = 0

    syringe_calibrator   = None
    current_step_recorder = None

    # On-screen “Stop recording” button over RS view
    rec_stop_rect = None

    # ─────────────────────────────────────────────────────────────
    # Helper closures local to main()
    # ─────────────────────────────────────────────────────────────
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
        syringe_volume_ul   = settings.get("syringe_volume_ul", 10.0)
        syringe_step_ul     = settings.get("syringe_step_ul", 1.0)
        syringe_remaining_ul= settings.get("syringe_remaining_ul", syringe_volume_ul)
        if syringe_step_ul > 0:
            ammo = max(0, int(syringe_remaining_ul // syringe_step_ul))
        else:
            ammo = 0

        items = [
            {"label": f"Tool length (mm): {current_tool_lz:.1f}  [Edit]", "kind": "edit_tool"},
            {"label": f"Approach distance (mm): {current_approach:.1f}  [Edit]", "kind": "edit_approach"},

            {"label": "— Routine —", "kind": "header"},
            {"label": "Edit Routine Script", "kind": "routine_edit"},
            {"label": "Run Routine", "kind": "routine_run"},
            {"label": "Stop Routine (halt motion only)", "kind": "routine_stop"},
            {"label": f"Wash FWD speed (steps/s): {settings.get('plunger_fwd_rate', MAX_RATE)}  [Edit]", "kind": "edit_fwd_rate"},
            {"label": f"Wash BWD speed (steps/s): {settings.get('plunger_bwd_rate', MAX_RATE)}  [Edit]", "kind": "edit_bwd_rate"},
            {"label": f"Failsafe timeout (ms) [0 = infinite]: {settings.get('endstop_timeout_ms', 0)}  [Edit]", "kind": "edit_endstop_timeout"},
            {"label": f"Wash wait time (s) [0 = none]: {settings.get('pose_final_wait_s', 1.0):.1f}  [Edit]", "kind": "edit_pose_wait"},
            # NOTE: swap_endstops still supported in settings.json, but UI toggle removed

            {"label": "— Injection —", "kind": "header"},
            {"label": (
                f"Syringe calib: stroke={settings.get('syringe_full_travel_time_s', 0.0):.2f}s, "
                f"vol={syringe_volume_ul:.3f} µL, step={syringe_step_ul:.3f} µL"
            ), "kind": "syringe_calib"},
            {"label": f"Start Injection (remaining ≈ {ammo} steps)", "kind": "start_injection"},

            {"label": "— Saved poses —", "kind": "header"},
            {"label": "New saved pose from CURRENT robot position", "kind": "new_pose"},
        ]
        for name in sorted_pose_names(poses):
            items.append({"label": f"Go: {name}", "kind": "pose_row", "pose": name})
        return items

    def start_syringe_calibration():
        nonlocal syringe_calib_active, syringe_calibrator
        if ser is None:
            print("⚠️ Cannot calibrate syringe: no serial connection.")
            with state.lock:
                state.calibration_status = "Calibration failed: no serial connection."
            return
        with state.lock:
            if state.syringe_calibration_running:
                print("⚠️ Calibration already running.")
                return
            state.calibration_status = "Starting mechanical calibration…"
        syringe_calibrator = SyringeCalibrator(state, ser, 0.0, 0.0)
        syringe_calibrator.start()

    def apply_syringe_volume_step():
        nonlocal syringe_calib_active
        try:
            vol_ul = float(syringe_vol_text.strip())
            step_ul = float(syringe_step_ul_txt.strip())
            if vol_ul <= 0 or step_ul <= 0:
                raise ValueError
        except ValueError:
            print("⚠️ Invalid volume/step; must be > 0.")
            with state.lock:
                state.calibration_status = "Apply failed: enter volume & step > 0."
            return

        with state.lock:
            if state.syringe_calibration_running:
                state.calibration_status = "Apply failed: calibration still running."
                return
            s = state.settings
            full_time_local = float(s.get("syringe_full_travel_time_s", 0.0) or 0.0)
            rotations_total = float(s.get("syringe_rotations_total", 0.0) or 0.0)
            if full_time_local <= 0:
                state.calibration_status = "Apply failed: run stroke calibration first."
                return

            s["syringe_volume_ul"] = vol_ul
            s["syringe_step_ul"]   = step_ul
            step_time = full_time_local * (step_ul / vol_ul)
            step_rot  = rotations_total * (step_ul / vol_ul) if rotations_total > 0 else 0.0
            s["syringe_step_time_s"]     = step_time
            s["syringe_step_rotations"]  = step_rot
            s["syringe_remaining_ul"]    = vol_ul
            s["injection_active"]        = False
            state.calibration_status = (
                f"Volume/step applied: {step_ul:.3f} µL ⇒ "
                f"{step_time:.3f}s, {step_rot:.1f} distance-units"
            )

        save_settings(state.settings)
        syringe_calib_active = False

    def start_injection_session():
        nonlocal injection_info_active, injection_step_in_progress
        nonlocal injection_step_done, post_step_prompt_active
        nonlocal injection_last_step_small
        nonlocal injection_record_telemetry, injection_record_video, injection_record_ui
        nonlocal subject_prompt_active, subject_field_text, injection_subject_text

        with state.lock:
            s = state.settings
            s["injection_study"]    = injection_study_text
            s["injection_material"] = injection_material_text
            s["injection_date"]     = injection_date_text

            step_time = float(s.get("syringe_step_time_s", 0.0) or 0.0)
            vol_ul = float(s.get("syringe_volume_ul", 0.0) or 0.0)

            if step_time <= 0 or vol_ul <= 0:
                state.calibration_status = "Cannot start injection: syringe not calibrated."
                s["injection_active"] = False
                injection_active_now = False
                print("⚠️ Cannot start injection: syringe not calibrated.")
            else:
                s["injection_active"] = True
                s["syringe_remaining_ul"] = vol_ul  # reset to full
                state.record_telemetry = injection_record_telemetry
                state.record_video     = injection_record_video
                state.record_ui        = injection_record_ui
                state.current_injection_step = 0
                injection_last_step_small = False
                state.calibration_status = "Injection session active."
                injection_active_now = True
                print("[Injection] Session started.")

        save_settings(state.settings)
        injection_step_in_progress = False
        injection_step_done = False
        post_step_prompt_active = False
        injection_info_active = False

        if injection_active_now:
            # Immediately ask Subject/Animal for FIRST step
            with state.lock:
                prev_subj = state.settings.get("injection_subject", "")
            subject_field_text = injection_subject_text or prev_subj or ""
            subject_prompt_active = True

    # ─────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────
    running = True
    loop_prev_time = time.time()

    prev_a = prev_b = prev_x = prev_y = False
    prev_lb = prev_rb = False
    last_sig_axis_time = 0.0
    prev_lt = 0.0
    prev_rt = 0.0

    click_pos = None

    while running:
        click_pos = None

        # ───────────── Event pump ─────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break

            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                flags ^= pygame.FULLSCREEN
                screen = pygame.display.set_mode((0, 0), flags)

            if ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                _, tr_rect, _, _ = compute_layout(
                    *pygame.display.get_surface().get_size(),
                    args.ui_frac, args.ui_min_width
                )
                if tr_rect.collidepoint(mx, my):
                    ui_scroll -= ev.y * 40
                    ui_scroll = max(
                        0,
                        min(ui_scroll, max(0, ui_content_height - tr_rect.h + 12))
                    )

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if (time.time() - last_sig_axis_time) >= AXIS_GUARD_WINDOW:
                    click_pos = ev.pos

            # Text-input KEYDOWN handling
            if ev.type == pygame.KEYDOWN:
                # Tool length
                if toollen_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            new_lz = float(toollen_text.strip())
                            with state.lock:
                                state.tool_lz = new_lz
                                state.settings["tool_lz"] = new_lz
                            save_settings(state.settings)
                            rt.enqueue({"type": "set_tool_lz", "lz": new_lz})
                            try:
                                joy.rumble(0.5, 0.5, 120)
                            except Exception:
                                pass
                        except ValueError:
                            print("⚠️ Invalid tool length.")
                        toollen_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        toollen_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        toollen_text = toollen_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            toollen_text += ch
                    continue

                # Approach distance
                if approach_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            new_d = float(approach_text.strip())
                            with state.lock:
                                state.approach_mm = new_d
                                state.settings["approach_mm"] = new_d
                            save_settings(state.settings)
                            rt.enqueue({"type": "set_approach_mm", "dist": new_d})
                            try:
                                joy.rumble(0.5, 0.5, 120)
                            except Exception:
                                pass
                        except ValueError:
                            print("⚠️ Invalid approach distance.")
                        approach_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        approach_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        approach_text = approach_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            approach_text += ch
                    continue

                # Routine editor
                if routine_edit_active:
                    mods = pygame.key.get_mods()
                    if ev.key == pygame.K_RETURN and (mods & pygame.KMOD_CTRL):
                        save_routine_text(routine_editor.text)
                        routine_edit_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        routine_edit_active = False
                    else:
                        editor_handle_key(routine_editor, ev)
                    continue

                # New pose name
                if naming_input_active:
                    if ev.key == pygame.K_RETURN:
                        name = naming_text.strip()
                        naming_input_active = False
                        if name:
                            rt.enqueue({
                                "type": "save_named_with_midway",
                                "name": name
                            })
                            try:
                                joy.rumble(0.6, 0.6, 140)
                            except Exception:
                                pass
                    elif ev.key == pygame.K_ESCAPE:
                        naming_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        naming_text = naming_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and (ch.isalnum() or ch in "_- "):
                            naming_text += ch
                    continue

                # Rename pose
                if rename_input_active:
                    if ev.key == pygame.K_RETURN:
                        new_name = rename_text.strip()
                        rename_input_active = False
                        if new_name:
                            rt.enqueue({
                                "type": "rename_pose",
                                "old": rename_old_name,
                                "new": new_name
                            })
                            try:
                                joy.rumble(0.6, 0.6, 140)
                            except Exception:
                                pass
                    elif ev.key == pygame.K_ESCAPE:
                        rename_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        rename_text = rename_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and (ch.isalnum() or ch in "_- "):
                            rename_text += ch
                    continue

                # Wash forward speed
                if fwd_rate_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            val = int(float(fwd_rate_text.strip()))
                            rt.enqueue({"type": "set_speed_fwd", "rate": val})
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
                        if ch in "0123456789.-":
                            fwd_rate_text += ch
                    continue

                # Wash reverse speed
                if bwd_rate_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            val = int(float(bwd_rate_text.strip()))
                            rt.enqueue({"type": "set_speed_bwd", "rate": val})
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
                        if ch in "0123456789.-":
                            bwd_rate_text += ch
                    continue

                # Endstop timeout
                if endstop_timeout_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            ms = int(float(endstop_timeout_text.strip()))
                            rt.enqueue({"type": "set_endstop_timeout", "ms": ms})
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
                        if ch in "0123456789.-":
                            endstop_timeout_text += ch
                    continue

                # Pose final wait seconds
                if pose_wait_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            s_val = float(pose_wait_text.strip())
                            rt.enqueue({"type": "set_pose_wait", "s": s_val})
                            with state.lock:
                                state.settings["pose_final_wait_s"] = s_val
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
                        if ch in "0123456789.-":
                            pose_wait_text += ch
                    continue

                # Syringe calib modal
                if syringe_calib_active:
                    if ev.key == pygame.K_ESCAPE:
                        syringe_calib_active = False
                    elif ev.key == pygame.K_TAB:
                        syringe_field_active = (
                            "step" if syringe_field_active == "vol" else "vol"
                        )
                    elif ev.key == pygame.K_RETURN:
                        apply_syringe_volume_step()
                    elif ev.key == pygame.K_BACKSPACE:
                        if syringe_field_active == "vol":
                            syringe_vol_text = syringe_vol_text[:-1]
                        else:
                            syringe_step_ul_txt = syringe_step_ul_txt[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            if syringe_field_active == "vol":
                                syringe_vol_text += ch
                            else:
                                syringe_step_ul_txt += ch
                    continue

                # Subject popup text input (only text, not button)
                if subject_prompt_active:
                    if ev.key == pygame.K_ESCAPE:
                        subject_prompt_active = False
                    elif ev.key == pygame.K_RETURN:
                        # handled by A/B/X mapping (joystick), so Enter = Confirm
                        pass
                    elif ev.key == pygame.K_BACKSPACE:
                        subject_field_text = subject_field_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and ch.isprintable() and ch not in "\r\n\t":
                            subject_field_text += ch
                    continue

                # Start injection metadata popup
                if injection_info_active:
                    if ev.key == pygame.K_ESCAPE:
                        injection_info_active = False
                    elif ev.key == pygame.K_TAB:
                        cycle = {"study": "material", "material": "date", "date": "study"}
                        injection_field_active = cycle.get(
                            injection_field_active, "study"
                        )
                    elif ev.key == pygame.K_RETURN:
                        start_injection_session()
                    elif ev.key == pygame.K_BACKSPACE:
                        if injection_field_active == "study":
                            injection_study_text = injection_study_text[:-1]
                        elif injection_field_active == "material":
                            injection_material_text = injection_material_text[:-1]
                        elif injection_field_active == "date":
                            injection_date_text = injection_date_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and ch.isprintable() and ch not in "\r\n\t":
                            if injection_field_active == "study":
                                injection_study_text += ch
                            elif injection_field_active == "material":
                                injection_material_text += ch
                            elif injection_field_active == "date":
                                injection_date_text += ch
                    continue

        if not running:
            break

        # ───────────── Joystick axes / buttons ─────────────
        raw_hx, raw_hy = joy.get_hat(0)
        lb, rb = joy.get_button(4), joy.get_button(5)

        lx = _axis(joy, LS_X_AXIS, INVERT_LS_X)
        ly = _axis(joy, LS_Y_AXIS, INVERT_LS_Y)
        rx = _axis(joy, RS_X_AXIS, INVERT_RS_X)
        ry = _axis(joy, RS_Y_AXIS, INVERT_RS_Y)

        with state.lock:
            state.lx, state.ly = lx, ly
            state.rx, state.ry = rx, ry
            state.hx, state.hy = raw_hx, raw_hy
            state.lb, state.rb = lb, rb

        # Sensitivity change via LB/RB on press
        lb_edge = lb and not prev_lb
        rb_edge = rb and not prev_rb
        with state.lock:
            idx = state.idx
            levels = state.levels[:]
        if lb_edge:
            idx = max(0, idx - 1)
            with state.lock:
                state.idx = idx
            try:
                joy.rumble(0.2, 0.2, 120)
            except Exception:
                pass
        if rb_edge:
            idx = min(len(levels) - 1, idx + 1)
            with state.lock:
                state.idx = idx
            try:
                joy.rumble(0.5, 0.5, 140)
            except Exception:
                pass
        prev_lb, prev_rb = lb, rb

        cur_lt = joy.get_axis(LEFT_AXIS)
        cur_rt = joy.get_axis(RIGHT_AXIS)
        if abs(cur_lt - prev_lt) > AXIS_GUARD_DELTA or abs(cur_rt - prev_rt) > AXIS_GUARD_DELTA:
            last_sig_axis_time = time.time()
        prev_lt, prev_rt = cur_lt, cur_rt
        right_trigger_norm = normalize(cur_rt)
        right_down = right_trigger_norm > DEADZONE

        # Buttons (A,B,X,Y)
        a_now = bool(joy.get_button(A_BUTTON))
        b_now = bool(joy.get_button(B_BUTTON))
        x_now = bool(joy.get_button(X_BUTTON))
        y_now = bool(joy.get_button(Y_BUTTON))

        a_edge = a_now and not prev_a
        b_edge = b_now and not prev_b
        x_edge = x_now and not prev_x
        y_edge = y_now and not prev_y

        now_loop = time.time()
        dt_loop = max(0.0, now_loop - loop_prev_time)
        loop_prev_time = now_loop

        with state.lock:
            waiting = state.await_confirm
            confirm_msg = state.confirm_msg
            cur_pose_name = state.current_pose_name or ""
            at_midway_now = cur_pose_name.endswith(MIDWAY_SUFFIX)
            motion_until = float(state.pose_motion_until)

            syringe_volume_ul      = state.settings.get("syringe_volume_ul", 10.0)
            syringe_step_ul        = state.settings.get("syringe_step_ul", 1.0)
            syringe_step_time_s    = float(state.settings.get("syringe_step_time_s", 0.0) or 0.0)
            syringe_remaining_ul   = state.settings.get("syringe_remaining_ul", syringe_volume_ul)
            injection_active_flag  = bool(state.settings.get("injection_active", False))
            calibration_running_flag = bool(state.syringe_calibration_running)
            saw_enabled_flag       = bool(getattr(state, "saw_enabled", False))

        # Injection step distance integration
        if injection_active_flag and injection_step_in_progress and ser and not calibration_running_flag:
            if injection_current_rate != 0.0 and dt_loop > 0.0:
                injection_step_distance += abs(injection_current_rate) * dt_loop

            if injection_step_target > 0 and injection_step_distance >= injection_step_target:
                # stop plunger
                try:
                    ser.write(b"V0\n")
                    if ser.in_waiting:
                        ser.read(ser.in_waiting)
                except Exception:
                    pass
                last_rate = 0
                injection_current_rate = 0.0
                injection_step_in_progress = False
                injection_step_done = True
                post_step_prompt_active = True

                new_remaining = max(0.0, syringe_remaining_ul - injection_step_volume_ul)
                with state.lock:
                    s = state.settings
                    s["syringe_remaining_ul"] = new_remaining
                    step_nom = float(s.get("syringe_step_ul", 0.0) or 0.0)
                    small = (step_nom > 0 and injection_step_volume_ul < step_nom - 1e-6)
                    injection_last_step_small = small
                    if step_nom > 0:
                        steps_left = int(new_remaining // step_nom)
                    else:
                        steps_left = 0
                    if small:
                        state.calibration_status = (
                            f"Inject step complete (small step {injection_step_volume_ul:.3f} µL). "
                            f"Remaining {new_remaining:.3f} µL (≈{steps_left} step(s))."
                        )
                    else:
                        state.calibration_status = (
                            f"Inject step complete ({injection_step_volume_ul:.3f} µL). "
                            f"Remaining {new_remaining:.3f} µL (≈{steps_left} step(s))."
                        )
                save_settings(state.settings)
                print(
                    f"[Injection] Step complete: {injection_step_volume_ul:.3f} µL, "
                    f"remaining {new_remaining:.3f} µL"
                )

        # A/B/X/Y global behavior (when no modal / text input is active)
        any_text_input_active = (
            toollen_input_active or approach_input_active or naming_input_active or
            rename_input_active or fwd_rate_input_active or bwd_rate_input_active or
            endstop_timeout_input_active or pose_wait_input_active
        )
        any_modal_active = (
            syringe_calib_active or injection_info_active or subject_prompt_active or
            final_small_warning_active or post_step_prompt_active or
            routine_edit_active or edit_menu_active or
            delete_confirm_active or overwrite_confirm_active
        )

        if a_edge and not any_text_input_active and not any_modal_active:
            rt.enqueue({"type": "confirm_or_advance"})
            try:
                joy.rumble(0.7, 0.7, 160)
            except Exception:
                pass

        if b_edge and not any_text_input_active and not any_modal_active:
            rt.enqueue({"type": "midway_free_control"})
            try:
                joy.rumble(0.3, 0.3, 120)
            except Exception:
                pass

        # X button:
        #  - outside injection: NO MOVEMENT, NO SAVE
        #  - inside injection (no modal): toggle J5 sawing
        #  - inside subject popup: used separately as Confirm & Advance
        if x_edge and injection_active_flag and not any_modal_active:
            with state.lock:
                cur = bool(getattr(state, "saw_enabled", False))
                state.saw_enabled = not cur
            save_settings(state.settings)
            print(f"[Saw] {'enabled' if not cur else 'disabled'} via X")
            try:
                joy.rumble(0.8 if not cur else 0.4,
                           0.0 if not cur else 0.4, 180)
            except Exception:
                pass

        prev_a, prev_b = a_now, b_now
        prev_x, prev_y = x_now, y_now

        # Plunger bridge when NOT in injection nor calibration
        if ser:
            plunger_bridge_allowed = (
                not waiting and not at_midway_now and time.time() >= motion_until
            )
            if (
                plunger_bridge_allowed
                and not injection_active_flag
                and not calibration_running_flag
            ):
                sign = int(state.settings.get("plunger_dir_sign", 1))
                rt_val = normalize(cur_rt)
                lt_val = normalize(cur_lt)
                if lt_val > DEADZONE:
                    rate = int(lt_val * MAX_RATE) * sign
                elif rt_val > DEADZONE:
                    rate = -int(rt_val * MAX_RATE) * sign
                else:
                    rate = 0
                if rate != last_rate:
                    try:
                        ser.write(f"V{rate}\n".encode())
                        last_rate = rate
                        if ser.in_waiting:
                            ser.read(ser.in_waiting)
                    except Exception:
                        pass
            else:
                if not injection_active_flag and not calibration_running_flag:
                    if last_rate not in (0, None):
                        try:
                            ser.write(b"V0\n")
                            last_rate = 0
                            if ser.in_waiting:
                                ser.read(ser.in_waiting)
                        except Exception:
                            pass

            if joy.get_button(RESET_BUTTON):
                try:
                    ser.write(b"R\n")
                    if ser.in_waiting:
                        ser.read(ser.in_waiting)
                except Exception:
                    pass
                last_rate = 0
                time.sleep(0.1)

        # Injection-specific plunger control
        if injection_active_flag and ser and not calibration_running_flag:
            with state.lock:
                syringe_rot_total = float(
                    state.settings.get("syringe_rotations_total", 0.0) or 0.0
                )
                vol_total = float(
                    state.settings.get("syringe_volume_ul", 0.0) or 0.0
                )
                rec_tel = state.record_telemetry
                rec_vid = state.record_video
                rec_ui = state.record_ui

            if syringe_rot_total <= 0 or vol_total <= 0:
                with state.lock:
                    state.settings["injection_active"] = False
                    state.calibration_status = (
                        "Injection disabled: syringe stroke/volume not calibrated."
                    )
                save_settings(state.settings)
                injection_step_in_progress = False
                injection_step_done = False
                post_step_prompt_active = False
                final_small_warning_active = False

            elif syringe_remaining_ul <= 1e-9:
                with state.lock:
                    state.settings["injection_active"] = False
                save_settings(state.settings)
                injection_step_in_progress = False
                injection_step_done = False
                post_step_prompt_active = False
                final_small_warning_active = False

            else:
                # Start a new step (or final small step) when:
                #  no step in progress, no post-step prompt, no subject popup, no final warning
                if (
                    not injection_step_in_progress
                    and not injection_step_done
                    and not post_step_prompt_active
                    and not subject_prompt_active
                    and not final_small_warning_active
                ):
                    if right_down and not last_right_trigger_down:
                        if syringe_remaining_ul >= syringe_step_ul + 1e-6:
                            # Full-size step
                            step_vol = syringe_step_ul
                            step_ratio = step_vol / vol_total if vol_total > 0 else 0.0
                            injection_step_target    = max(0.0, syringe_rot_total * step_ratio)
                            injection_step_distance  = 0.0
                            injection_step_volume_ul = step_vol
                            injection_last_step_small = False
                            injection_step_in_progress = True
                            injection_step_done        = False

                            with state.lock:
                                state.current_injection_step += 1
                                step_index = state.current_injection_step
                                s = state.settings
                                inj_meta = {
                                    "study":   s.get("injection_study",   ""),
                                    "subject": s.get("injection_subject", ""),
                                    "material":s.get("injection_material",""),
                                    "date":    s.get("injection_date",    ""),
                                }
                            if rec_tel or rec_vid or rec_ui:
                                try:
                                    current_step_recorder = StepRecorder(
                                        DATA_ROOT_DIR, inj_meta, step_index,
                                        record_telemetry=rec_tel,
                                        record_video=rec_vid,
                                        record_ui=rec_ui,
                                    )
                                except Exception as e:
                                    print(f"[Recorder] Failed to create recorder: {e}")
                                    current_step_recorder = None

                            print(
                                f"[Injection] Step {step_index} started: {step_vol:.3f} µL "
                                f"(target distance {injection_step_target:.1f})"
                            )
                        else:
                            # Last small step: show warning overlay; actual start on A
                            final_small_step_volume_ul = syringe_remaining_ul
                            if final_small_step_volume_ul > 0.0 and vol_total > 0.0:
                                step_ratio = final_small_step_volume_ul / vol_total
                                final_small_step_target = max(
                                    0.0, syringe_rot_total * step_ratio
                                )
                            else:
                                final_small_step_target = 0.0

                            final_small_warning_active = True
                            injection_last_step_small = True
                            injection_step_in_progress = False
                            injection_step_done = False

                            if joy:
                                try:
                                    joy.rumble(1.0, 0.0, 300)
                                except Exception:
                                    pass

                            print(
                                f"[Injection] Final small step pending: "
                                f"{final_small_step_volume_ul:.3f} µL "
                                f"(distance {final_small_step_target:.1f})"
                            )

                # If a step is in progress, drive plunger based on right trigger
                if injection_step_in_progress and not post_step_prompt_active and not final_small_warning_active:
                    with state.lock:
                        plunger_fwd_rate = int(
                            state.settings.get("plunger_fwd_rate", MAX_RATE) or MAX_RATE
                        )
                        dir_sign = int(
                            state.settings.get("plunger_dir_sign", 1) or 1
                        )

                    desired_rate = 0
                    right_down = (right_trigger_norm > DEADZONE)
                    if right_down:
                        desired_rate = -int(abs(plunger_fwd_rate) * right_trigger_norm) * dir_sign

                    if desired_rate != last_rate:
                        try:
                            ser.write(f"V{desired_rate}\n".encode())
                            if ser.in_waiting:
                                ser.read(ser.in_waiting)
                        except Exception:
                            pass
                        last_rate = desired_rate
                        injection_current_rate = abs(desired_rate)

                # If no step in progress (or in modal), enforce stop
                if (
                    (not injection_step_in_progress)
                    or post_step_prompt_active
                    or final_small_warning_active
                ) and last_rate not in (0, None):
                    try:
                        ser.write(b"V0\n")
                        if ser.in_waiting:
                            ser.read(ser.in_waiting)
                    except Exception:
                        pass
                    last_rate = 0
                    injection_current_rate = 0.0

        last_right_trigger_down = right_down

        # ───────────── Draw layout ─────────────
        screen_w, screen_h = pygame.display.get_surface().get_size()
        tl_rect, tr_rect, bl_rect, br_rect = compute_layout(
            screen_w, screen_h, args.ui_frac, args.ui_min_width
        )

        rs_frame = rs_thread.latest()
        blit_frame_cover(screen, rs_frame, tl_rect)

        # Recording indicator & RS “Stop recording” button
        rec_stop_rect = None
        if current_step_recorder and current_step_recorder.active:
            blink_on = (pygame.time.get_ticks() // 400) % 2 == 0
            dot_r = 8
            dot_x = tl_rect.x + 16
            dot_y = tl_rect.y + 16

            if blink_on:
                pygame.draw.circle(screen, (255, 0, 0), (dot_x, dot_y), dot_r)

            rec_label = small_font.render("REC", True, (255, 0, 0))
            screen.blit(
                rec_label,
                (dot_x + dot_r + 6,
                 dot_y - rec_label.get_height() // 2),
            )

            btn_w = 140
            btn_h = 26
            btn_rect = pygame.Rect(
                dot_x + dot_r + 6 + rec_label.get_width() + 12,
                dot_y - btn_h // 2,
                btn_w,
                btn_h,
            )
            pygame.draw.rect(screen, (60, 60, 60), btn_rect)
            pygame.draw.rect(screen, (255, 0, 0), btn_rect, 2)
            btn_txt = small_font.render("Stop recording", True, (255, 255, 255))
            screen.blit(
                btn_txt,
                (btn_rect.x + (btn_w - btn_txt.get_width()) // 2,
                 btn_rect.y + (btn_h - btn_txt.get_height()) // 2),
            )
            rec_stop_rect = btn_rect

        # Pitch / J5 / status overlay on RS view
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
            u1_rot = int(state.settings.get("uvc1_rot", 2))
            u1_hf  = bool(state.settings.get("uvc1_flip_h", False))
            u1_vf  = bool(state.settings.get("uvc1_flip_v", False))
            u2_rot = int(state.settings.get("uvc2_rot", 2))
            u2_hf  = bool(state.settings.get("uvc2_flip_h", False))
            u2_vf  = bool(state.settings.get("uvc2_flip_v", False))
            syringe_volume_ul      = state.settings.get("syringe_volume_ul", 10.0)
            syringe_step_ul        = state.settings.get("syringe_step_ul", 1.0)
            syringe_remaining_ul   = state.settings.get("syringe_remaining_ul", syringe_volume_ul)
            injection_active_flag  = bool(state.settings.get("injection_active", False))
            calibration_status     = state.calibration_status or ""
            inj_retract_mm_val     = float(state.settings.get("injection_retract_mm", 10.0) or 0.0)

        if syringe_step_ul > 0:
            ammo_steps = max(0, int(syringe_remaining_ul // syringe_step_ul))
            remainder_ul_for_overlay = max(0.0, syringe_remaining_ul - ammo_steps * syringe_step_ul)
        else:
            ammo_steps = 0
            remainder_ul_for_overlay = 0.0

        cx, cy = tl_rect.x + tl_w // 2, tl_rect.y + tl_h // 2
        pygame.draw.circle(
            screen,
            (0, 255, 127),
            (cx, cy),
            int(min(tl_w, tl_h) * 0.09),
            2,
        )

        # Pitch dial
        x1 = tl_rect.x + m
        y1 = tl_rect.y + tl_h - (panel_size + padding + 20 + padding + 16 + padding) - m
        pan1 = pygame.Surface((panel_size, panel_size), pygame.SRCALPHA)
        pan1.fill((10, 10, 30, 180))
        pivot = (5, panel_size // 2)
        r_len = panel_size - pivot[0] - 5
        ang = np.radians(-p)
        ex = pivot[0] + r_len * np.cos(ang)
        ey = pivot[1] + r_len * np.sin(ang)
        pygame.draw.line(pan1, (0, 255, 127), pivot, (ex, ey), 2)
        pygame.draw.circle(pan1, (0, 255, 127), (int(ex), int(ey)), 4)
        screen.blit(pan1, (x1, y1))
        lf = pygame.font.SysFont("Consolas", 14, bold=True)
        vf_ = pygame.font.SysFont("Consolas", 12)
        screen.blit(lf.render("Pitch", True, (0, 255, 127)),
                    (x1, y1 + panel_size + padding))
        screen.blit(
            vf_.render(f"{p:+.5f}°", True, (0, 255, 127)),
            (x1, y1 + panel_size + padding + 16 + padding),
        )

        # J5 dial
        x2 = x1 + panel_size + diag_gap
        y2 = y1
        pan2 = pygame.Surface((panel_size, panel_size), pygame.SRCALPHA)
        pan2.fill((10, 10, 30, 180))
        center = (panel_size // 2, panel_size // 2)
        radius = panel_size // 2 - 5
        pygame.draw.circle(pan2, (0, 255, 127), center, radius, 2)
        ang2 = np.radians(j5v - 90)
        ex2 = center[0] + radius * np.cos(ang2)
        ey2 = center[1] + radius * np.sin(ang2)
        pygame.draw.line(pan2, (0, 255, 127), center, (ex2, ey2), 2)
        pygame.draw.polygon(
            pan2,
            (0, 255, 127),
            [
                (ex2 + 4 * np.cos(ang2), ey2 + 4 * np.sin(ang2)),
                (ex2 + 4 * np.cos(ang2 + 0.3), ey2 + 4 * np.sin(ang2 + 0.3)),
                (ex2 + 4 * np.cos(ang2 - 0.3), ey2 + 4 * np.sin(ang2 - 0.3)),
            ],
        )
        screen.blit(pan2, (x2, y2))
        screen.blit(lf.render("Angle", True, (0, 255, 127)),
                    (x2, y2 + panel_size + padding))
        screen.blit(
            vf_.render(f"{-j5v:+.5f}°", True, (0, 255, 127)),
            (x2, y2 + panel_size + padding + 16 + padding),
        )

        inj_label = (
            f"Injection: {'ACTIVE' if injection_active_flag else 'idle'}; "
            f"ammo={ammo_steps} steps (retract A={inj_retract_mm_val:.1f} mm)"
        )

        texts = [
            vf_.render(f"Sensitivity ({sens*100:.0f}%)", True, (200, 230, 255)),
            vf_.render(f"Move step:  ±{(5.0 * sens):.5f} mm", True, (0, 255, 127)),
            vf_.render(f"Angle step: ±{(5.0 * sens):.5f}°", True, (0, 255, 127)),
            vf_.render(f"Approach: {approach_mm:.2f} mm", True, (200, 255, 200)),
            vf_.render(f"Routine phase: {routine_phase}", True, (220, 220, 220)),
            vf_.render(f"Status: {routine_status_text}", True, (220, 220, 220)),
            vf_.render(f"Plunger: {last_plunger_note}", True, (220, 220, 220)),
            vf_.render(inj_label, True, (255, 220, 180)),
            vf_.render(f"Syringe calib: {calibration_status}", True, (200, 230, 200)),
        ]
        sb_w = max(t.get_width() for t in texts) + 8
        sb_h = sum(t.get_height() + 4 for t in texts) + 4
        x3 = tl_rect.x + tl_w - sb_w - m
        y3 = tl_rect.y + m
        sbpan = pygame.Surface((sb_w, sb_h), pygame.SRCALPHA)
        sbpan.fill((30, 30, 10, 180))
        y_off_s = 4
        for t in texts:
            sbpan.blit(t, (4, y_off_s))
            y_off_s += t.get_height() + 4
        screen.blit(sbpan, (x3, y3))

        # Ammo bar (top-right of RS)
        if syringe_step_ul > 0 and syringe_volume_ul > 0:
            ammo_float = syringe_remaining_ul / syringe_step_ul
            full_bars = int(ammo_float)
            frac_bar  = ammo_float - full_bars
            max_bars = 16
            full_bars_disp = min(full_bars, max_bars)
            frac_bar_disp = frac_bar if full_bars < max_bars else 0.0
            bar_w_pix = 10
            bar_h_pix = 20
            gap_pix   = 3
            total_bars = full_bars_disp + (1 if frac_bar_disp > 1e-3 else 0)
            panel_w_ammo = 16 + max(1, total_bars) * (bar_w_pix + gap_pix) + 90
            panel_h_ammo = bar_h_pix + 28
            ammo_x = tl_rect.right - panel_w_ammo - 8
            ammo_y = tl_rect.y + tl_h - panel_h_ammo - 8
            ammo_surf = pygame.Surface((panel_w_ammo, panel_h_ammo), pygame.SRCALPHA)
            ammo_surf.fill((0, 0, 0, 170))
            bx = 10
            by = 6
            for i in range(full_bars_disp):
                r = pygame.Rect(bx, by, bar_w_pix, bar_h_pix)
                pygame.draw.rect(ammo_surf, (0, 255, 127), r)
                bx += bar_w_pix + gap_pix
            if frac_bar_disp > 1e-3:
                h_frac = max(4, int(bar_h_pix * frac_bar_disp))
                dy = bar_h_pix - h_frac
                r = pygame.Rect(bx, by + dy, bar_w_pix, h_frac)
                pygame.draw.rect(ammo_surf, (255, 200, 120), r)
                bx += bar_w_pix + gap_pix
            ammo_txt = vf_.render(f"{ammo_float:.1f} step(s)", True, (230, 230, 230))
            ammo_surf.blit(ammo_txt, (bx + 4, by))
            vol_txt = vf_.render(
                f"{syringe_remaining_ul:.2f}/{syringe_volume_ul:.2f} µL",
                True,
                (200, 220, 255),
            )
            ammo_surf.blit(vol_txt, (10, by + bar_h_pix + 6))
            if remainder_ul_for_overlay > 1e-3:
                warn = vf_.render(
                    f"Final≈{remainder_ul_for_overlay:.2f} µL",
                    True,
                    (255, 200, 120),
                )
                ammo_surf.blit(
                    warn,
                    (panel_w_ammo - warn.get_width() - 10, by + bar_h_pix + 6),
                )
            screen.blit(ammo_surf, (ammo_x, ammo_y))

        # ── Right panel UI ──
        pygame.draw.rect(screen, (20, 20, 20), tr_rect)
        m_ui    = max(8, int(tr_rect.w * 0.02))
        btn_w   = tr_rect.w - 2 * m_ui
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
                surf_txt = label_font.render(b["label"], True, (200, 200, 200))
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
                pygame.draw.rect(screen, (40, 40, 60), go_rect)
                pygame.draw.rect(screen, (0, 255, 127), go_rect, 2)
                go_txt_lines = wrap_text(
                    f"Go: {name}", placeholder_font, go_w - 10
                )
                ly2 = go_rect.y + 4
                for ln in go_txt_lines:
                    go_txt = placeholder_font.render(ln, True, (220, 220, 230))
                    screen.blit(go_txt, (go_rect.x + 8, ly2))
                    ly2 += placeholder_font.get_height()
                ui_clickables.append((go_rect, {"kind": "goto", "pose": name}))

                edit_rect = pygame.Rect(go_rect.right + 6, y_off_btn, edit_btn_w, 34)
                pygame.draw.rect(screen, (55, 55, 85), edit_rect)
                pygame.draw.rect(screen, (0, 255, 127), edit_rect, 2)
                etxt = placeholder_font.render("Edit", True, (230, 230, 230))
                screen.blit(
                    etxt,
                    (edit_rect.x + (edit_rect.w - etxt.get_width()) // 2,
                     edit_rect.y + (edit_rect.h - etxt.get_height()) // 2),
                )
                ui_clickables.append((edit_rect, {"kind": "edit_menu", "pose": name}))

                y_off_btn += 34 + btn_gap
                total_h += 34 + btn_gap
                continue

            label = b["label"]
            lines = wrap_text(label, placeholder_font, btn_w - 10)
            bh    = len(lines) * (placeholder_font.get_height() + 2) + 10
            rect  = pygame.Rect(tr_rect.x + m_ui, y_off_btn, btn_w, bh)
            pygame.draw.rect(screen, (40, 40, 60), rect)
            pygame.draw.rect(screen, (0, 255, 127), rect, 2)
            ly2 = y_off_btn + 5
            for ln in lines:
                surf_txt = placeholder_font.render(ln, True, (200, 200, 200))
                tx = tr_rect.x + m_ui + (btn_w - surf_txt.get_width()) // 2
                screen.blit(surf_txt, (tx, ly2))
                ly2 += placeholder_font.get_height() + 2
            ui_clickables.append((rect, b))
            y_off_btn += bh + btn_gap
            total_h   += bh + btn_gap

        ui_content_height = total_h + m_ui

        # Scrollbar
        if ui_content_height > tr_rect.h:
            view_h = tr_rect.h - 2 * m_ui
            content_h = max(1, ui_content_height - 2 * m_ui)
            bar_h = max(24, int(view_h * (view_h / content_h)))
            max_scroll = max(1, ui_content_height - tr_rect.h + 12)
            bar_y = tr_rect.y + m_ui + int(
                (ui_scroll / max_scroll) * (view_h - bar_h)
            )
            bar_x = tr_rect.right - 10
            pygame.draw.rect(
                screen,
                (60, 60, 60),
                (bar_x, tr_rect.y + m_ui, 6, view_h),
            )
            pygame.draw.rect(screen, (0, 255, 127), (bar_x, bar_y, 6, bar_h))

        screen.set_clip(prev_clip)

        # ── UVC bottom views ──
        uvc_clickables = []

        f2_raw = uvc2.latest() if uvc2 else None
        f2 = transform_uvc_frame(f2_raw, u2_rot, u2_hf, u2_vf) if f2_raw is not None else None
        if not blit_frame_fit(screen, f2, bl_rect):
            draw_placeholder(
                screen, bl_rect,
                title=(uvc2.status() if uvc2 else "UVC#2 not connected"),
                value_font=value_font, big_font=big_font
            )
        for r, op in draw_uvc_controls(screen, bl_rect, u2_rot, u2_hf, u2_vf, small_font):
            uvc_clickables.append((r, {"cam": 2, "op": op}))

        f1_raw = uvc1.latest() if uvc1 else None
        f1 = transform_uvc_frame(f1_raw, u1_rot, u1_hf, u1_vf) if f1_raw is not None else None
        if not blit_frame_fit(screen, f1, br_rect):
            draw_placeholder(
                screen, br_rect,
                title=(uvc1.status() if uvc1 else "UVC#1 not connected"),
                value_font=value_font, big_font=big_font
            )
        for r, op in draw_uvc_controls(screen, br_rect, u1_rot, u1_hf, u1_vf, small_font):
            uvc_clickables.append((r, {"cam": 1, "op": op}))

        # Per-step progress bar on bottom-left UVC
        if injection_active_flag and injection_step_in_progress and injection_step_target > 0.0:
            frac = max(0.0, min(1.0, injection_step_distance / injection_step_target))
            bar_margin = 12
            bar_h = 14
            bar_w = bl_rect.w - 2 * bar_margin
            bar_x = bl_rect.x + bar_margin
            bar_y = bl_rect.bottom - bar_h - 6

            bg = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
            pygame.draw.rect(screen, (0, 0, 0), bg)
            pygame.draw.rect(screen, (70, 70, 70), bg, 1)

            fill_w = int(bar_w * frac)
            if fill_w > 0:
                pygame.draw.rect(
                    screen,
                    (0, 200, 255),
                    pygame.Rect(bar_x, bar_y, fill_w, bar_h),
                )

            txt = small_font.render(
                f"Step {state.current_injection_step}: {frac*100:4.1f}% of {injection_step_volume_ul:.3f} µL",
                True,
                (230, 230, 230),
            )
            screen.blit(txt, (bar_x, bar_y - txt.get_height() - 2))

        # Per-step logging (telemetry + video)
        if current_step_recorder and current_step_recorder.active:
            t_rel = time.time() - current_step_recorder.t0
            with state.lock:
                joints_snapshot = (
                    state.last_joints
                    if isinstance(state.last_joints, dict)
                    else (state.last_joints or {})
                )
                lx_s, ly_s = state.lx, state.ly
                rx_s, ry_s = state.rx, state.ry
                hx_s, hy_s = state.hx, state.hy
                lb_s, rb_s = state.lb, state.rb
                rec_tel = state.record_telemetry
                rec_vid = state.record_video
                rec_ui  = state.record_ui
            lt_val = normalize(cur_lt)
            rt_val_norm = normalize(cur_rt)
            joystick_tuple = (
                lx_s, ly_s, rx_s, ry_s,
                hx_s, hy_s,
                lt_val, rt_val_norm,
                lb_s, rb_s,
                a_now, b_now, x_now, y_now
            )
            if rec_tel:
                current_step_recorder.log_telemetry(
                    t_rel,
                    joints_snapshot,
                    joystick_tuple,
                    {
                        "distance": injection_step_distance,
                        "step_volume": injection_step_volume_ul,
                    },
                )
            if rec_vid:
                current_step_recorder.write_frame(t_rel, "rs", rs_frame)
                current_step_recorder.write_frame(t_rel, "uvc1", f1)
                current_step_recorder.write_frame(t_rel, "uvc2", f2)
            if rec_ui:
                current_step_recorder.write_ui_frame(t_rel, screen)

        # Stop recorder when step is fully finished and overlay dismissed
        if (
            current_step_recorder
            and not injection_step_in_progress
            and not post_step_prompt_active
            and not final_small_warning_active
            and not injection_step_done
        ):
            current_step_recorder.stop()
            current_step_recorder = None

        # ───────────── Click handling (no modals) ─────────────
        modals_open = (
            edit_menu_active or delete_confirm_active or overwrite_confirm_active
            or toollen_input_active or approach_input_active
            or naming_input_active or rename_input_active
            or fwd_rate_input_active or bwd_rate_input_active
            or endstop_timeout_input_active or pose_wait_input_active
            or routine_edit_active or syringe_calib_active
            or injection_info_active or subject_prompt_active
            or post_step_prompt_active or final_small_warning_active
        )

        if click_pos is not None and not modals_open:
            mx, my = click_pos

            # RS stop recording
            if rec_stop_rect is not None and rec_stop_rect.collidepoint(mx, my):
                if current_step_recorder:
                    current_step_recorder.stop()
                    current_step_recorder = None
                print("[Recording] Stopped manually via on-screen button.")
                click_pos = None
            else:
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
                    _, tr_rect2, _, _ = compute_layout(
                        *pygame.display.get_surface().get_size(),
                        args.ui_frac, args.ui_min_width
                    )
                    if tr_rect2.collidepoint(mx, my):
                        for rect, act in ui_clickables:
                            if rect.collidepoint(mx, my):
                                kind = act.get("kind")
                                if kind == "edit_tool":
                                    toollen_input_active = True
                                    toollen_text = f"{state.tool_lz:.1f}"
                                elif kind == "edit_approach":
                                    approach_input_active = True
                                    approach_text = f"{state.approach_mm:.1f}"
                                elif kind == "goto":
                                    rt.enqueue({"type": "goto", "pose_name": act["pose"]})
                                    try:
                                        joy.rumble(0.4, 0.4, 120)
                                    except Exception:
                                        pass
                                elif kind == "edit_menu":
                                    edit_menu_active = True
                                    edit_menu_target = act["pose"]
                                    click_pos = None
                                elif kind == "routine_edit":
                                    routine_edit_active = True
                                elif kind == "routine_run":
                                    save_routine_text(routine_editor.text)
                                    if routine_executor and routine_executor.is_alive():
                                        routine_executor.stop()
                                    with state.lock:
                                        kw = state.settings.get(
                                            "endstop_keywords",
                                            ["ENDSTOP", "LIMIT", "LIM", "HIT", "END"],
                                        )
                                        tms = state.settings.get("endstop_timeout_ms", 0)
                                        vf_rate = state.settings.get("plunger_fwd_rate", MAX_RATE)
                                        vb_rate = state.settings.get("plunger_bwd_rate", MAX_RATE)
                                        wait_s = state.settings.get("pose_final_wait_s", 1.0)
                                        expel = state.settings.get("expel_endstop", "BWD")
                                    routine_executor = RoutineExecutor(
                                        state,
                                        rt,
                                        ser,
                                        routine_editor.text,
                                        endstop_keywords=kw,
                                        endstop_timeout_ms=tms,
                                        plunger_rate_fwd=vf_rate,
                                        plunger_rate_bwd=vb_rate,
                                        pose_final_wait_s=wait_s,
                                        expel_endstop=expel,
                                    )
                                    routine_executor.start()
                                elif kind == "routine_stop":
                                    if routine_executor and routine_executor.is_alive():
                                        routine_executor.stop()
                                    rt.enqueue({"type": "stop_motors"})
                                    if ser:
                                        try:
                                            ser.write(b"V0\n")
                                            if ser.in_waiting:
                                                ser.read(ser.in_waiting)
                                        except Exception:
                                            pass
                                    with state.lock:
                                        state.routine_phase = "Stopped"
                                        state.routine_status_text = "Routine halted by user"
                                    click_pos = None
                                elif kind == "edit_fwd_rate":
                                    fwd_rate_input_active = True
                                    fwd_rate_text = str(
                                        state.settings.get("plunger_fwd_rate", MAX_RATE)
                                    )
                                elif kind == "edit_bwd_rate":
                                    bwd_rate_input_active = True
                                    bwd_rate_text = str(
                                        state.settings.get("plunger_bwd_rate", MAX_RATE)
                                    )
                                elif kind == "edit_endstop_timeout":
                                    endstop_timeout_input_active = True
                                    endstop_timeout_text = str(
                                        state.settings.get("endstop_timeout_ms", 0)
                                    )
                                elif kind == "edit_pose_wait":
                                    pose_wait_input_active = True
                                    pose_wait_text = (
                                        f"{state.settings.get('pose_final_wait_s', 1.0):.1f}"
                                    )
                                elif kind == "new_pose":
                                    naming_input_active = True
                                    naming_text = ""
                                elif kind == "syringe_calib":
                                    syringe_calib_active = True
                                    syringe_field_active = "vol"
                                    click_pos = None
                                elif kind == "start_injection":
                                    injection_info_active = True
                                    injection_field_active = "study"
                                    click_pos = None
                                break

        # Text-input popups
        if toollen_input_active:
            draw_center_input_box(
                screen, "Enter tool length (mm)", toollen_text
            )
        if approach_input_active:
            draw_center_input_box(
                screen, "Enter approach distance (mm)", approach_text
            )
        if naming_input_active:
            draw_center_input_box(
                screen, "Enter NEW pose name", naming_text or ""
            )
        if rename_input_active:
            draw_center_input_box(
                screen, f"Rename '{rename_old_name}' →", rename_text or ""
            )
        if fwd_rate_input_active:
            draw_center_input_box(
                screen,
                "Wash FORWARD plunger speed (steps/s)",
                fwd_rate_text,
            )
        if bwd_rate_input_active:
            draw_center_input_box(
                screen,
                "Wash REVERSE plunger speed (steps/s)",
                bwd_rate_text,
            )
        if endstop_timeout_input_active:
            draw_center_input_box(
                screen,
                "Failsafe timeout (ms) — 0 = infinite",
                endstop_timeout_text,
            )
        if pose_wait_input_active:
            draw_center_input_box(
                screen,
                "Wash wait time (seconds) — 0 = none",
                pose_wait_text,
            )

        # Syringe calibration modal
        if syringe_calib_active:
            with state.lock:
                status_text = state.calibration_status
            field_rects, button_rects, panel_rect = draw_syringe_calibration_modal(
                screen,
                syringe_vol_text,
                syringe_step_ul_txt,
                status_text,
                syringe_field_active,
            )
            if click_pos is not None:
                mx, my = click_pos
                if not panel_rect.collidepoint(mx, my):
                    syringe_calib_active = False
                else:
                    if field_rects["vol"].collidepoint(mx, my):
                        syringe_field_active = "vol"
                    elif field_rects["step"].collidepoint(mx, my):
                        syringe_field_active = "step"
                    elif button_rects["cancel"].collidepoint(mx, my):
                        syringe_calib_active = False
                    elif button_rects["apply"].collidepoint(mx, my):
                        apply_syringe_volume_step()
                    elif button_rects["calibrate"].collidepoint(mx, my):
                        start_syringe_calibration()
                click_pos = None

        # Injection metadata modal
        if injection_info_active:
            field_rects2, button_rects2, checkbox_rects2, panel_rect2 = (
                draw_injection_info_modal(
                    screen,
                    injection_study_text,
                    injection_material_text,
                    injection_date_text,
                    injection_field_active,
                    injection_record_telemetry,
                    injection_record_video,
                    injection_record_ui,
                )
            )
            if click_pos is not None:
                mx, my = click_pos
                if not panel_rect2.collidepoint(mx, my):
                    injection_info_active = False
                else:
                    if field_rects2["study"].collidepoint(mx, my):
                        injection_field_active = "study"
                    elif field_rects2["material"].collidepoint(mx, my):
                        injection_field_active = "material"
                    elif field_rects2["date"].collidepoint(mx, my):
                        injection_field_active = "date"
                    elif checkbox_rects2["telemetry"].collidepoint(mx, my):
                        injection_record_telemetry = not injection_record_telemetry
                    elif checkbox_rects2["video"].collidepoint(mx, my):
                        injection_record_video = not injection_record_video
                    elif checkbox_rects2["ui"].collidepoint(mx, my):
                        injection_record_ui = not injection_record_ui
                    elif button_rects2["cancel"].collidepoint(mx, my):
                        injection_info_active = False
                    elif button_rects2["start"].collidepoint(mx, my):
                        start_injection_session()
                click_pos = None

        # Subject / Animal popup (next step; includes Confirm & Advance)
        if subject_prompt_active:
            with state.lock:
                prev_step_idx = state.current_injection_step
                retract_mm = float(
                    state.settings.get("injection_retract_mm", 10.0) or 0.0
                )
                advance_mm = float(
                    state.settings.get("injection_advance_mm", 0.0) or 0.0
                )
                if advance_mm > retract_mm:
                    advance_mm = retract_mm
            step_idx = prev_step_idx + 1
            allow_advance = prev_step_idx > 0 and advance_mm > 0.0

            box_s, btns_s, panel_s = draw_subject_modal(
                screen,
                subject_field_text,
                step_idx,
                allow_advance,
            )

            choice = None

            if click_pos is not None:
                mx, my = click_pos
                if not panel_s.collidepoint(mx, my):
                    subject_prompt_active = False
                else:
                    if box_s.collidepoint(mx, my):
                        pass
                    elif btns_s["confirm"].collidepoint(mx, my):
                        choice = "confirm"
                    elif allow_advance and btns_s["confirm_adv"] and btns_s["confirm_adv"].collidepoint(mx, my):
                        choice = "confirm_adv"
                    elif btns_s["cancel"].collidepoint(mx, my):
                        choice = "cancel"
                click_pos = None

            # A / B / X inside subject popup
            if a_edge:
                choice = "confirm"
            elif b_edge:
                choice = "cancel"
            elif allow_advance and x_edge:
                choice = "confirm_adv"

            if choice in ("confirm", "confirm_adv"):
                subj = subject_field_text.strip()
                injection_subject_text = subj
                with state.lock:
                    state.settings["injection_subject"] = subj
                save_settings(state.settings)

                if choice == "confirm_adv" and advance_mm > 0.0:
                    rt.enqueue({"type": "inject_advance", "dist": advance_mm})

                subject_prompt_active = False

            elif choice == "cancel":
                subject_prompt_active = False

        # Final small-step warning overlay
        if final_small_warning_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((40, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            with state.lock:
                s = state.settings
                syringe_step_ul_overlay   = s.get("syringe_step_ul", 1.0)
                syringe_remaining_ul_overlay = s.get("syringe_remaining_ul", 0.0)

            lines = [
                "WARNING: Final injection step is smaller than configured step size.",
                f"Configured step size: {syringe_step_ul_overlay:.3f} µL",
                f"Final step volume:   {final_small_step_volume_ul:.3f} µL",
                "",
                "Press A to inject this final dose.",
                "Press B to cancel this final dose and end the injection session.",
                "Press Y to stop any ongoing recording for this step.",
            ]
            y0 = screen_h // 2 - (
                len(lines) * (big_font.get_height() + 4)
            ) // 2
            for ln in lines:
                txt = big_font.render(ln, True, (255, 80, 80))
                screen.blit(txt, ((screen_w - txt.get_width()) // 2, y0))
                y0 += txt.get_height() + 4

            if a_edge:
                with state.lock:
                    s = state.settings
                    syringe_rot_total = float(
                        s.get("syringe_rotations_total", 0.0) or 0.0
                    )
                    vol_total         = float(
                        s.get("syringe_volume_ul", 0.0) or 0.0
                    )
                    rec_tel = state.record_telemetry
                    rec_vid = state.record_video
                    rec_ui  = state.record_ui

                if syringe_rot_total <= 0 or vol_total <= 0 or final_small_step_volume_ul <= 0:
                    print("[Injection] Cannot start final small step: calibration missing.")
                    final_small_warning_active = False
                else:
                    step_ratio = final_small_step_volume_ul / vol_total
                    injection_step_target    = max(0.0, syringe_rot_total * step_ratio)
                    injection_step_distance  = 0.0
                    injection_step_volume_ul = final_small_step_volume_ul
                    injection_last_step_small = True
                    injection_step_in_progress = True
                    injection_step_done        = False
                    final_small_warning_active = False

                    with state.lock:
                        s = state.settings
                        s["injection_active"] = True
                        state.current_injection_step += 1
                        step_index = state.current_injection_step
                        inj_meta = {
                            "study":   s.get("injection_study",   ""),
                            "subject": s.get("injection_subject", ""),
                            "material":s.get("injection_material",""),
                            "date":    s.get("injection_date",    ""),
                        }

                    if rec_tel or rec_vid or rec_ui:
                        try:
                            current_step_recorder = StepRecorder(
                                DATA_ROOT_DIR, inj_meta, step_index,
                                record_telemetry=rec_tel,
                                record_video=rec_vid,
                                record_ui=rec_ui,
                            )
                        except Exception as e:
                            print(f"[Recorder] Failed to create recorder for final step: {e}")
                            current_step_recorder = None

                    print(
                        f"[Injection] Final small step {step_index} started: "
                        f"{final_small_step_volume_ul:.3f} µL "
                        f"(target distance {injection_step_target:.1f})"
                    )

            elif b_edge:
                with state.lock:
                    s = state.settings
                    s["injection_active"] = False
                    s["syringe_remaining_ul"] = 0.0
                    state.calibration_status = (
                        f"Final small step of {final_small_step_volume_ul:.3f} µL cancelled; "
                        "injection session ended."
                    )
                save_settings(state.settings)
                final_small_warning_active = False
                injection_last_step_small  = False
                print("[Injection] Final small step cancelled; remaining volume discarded.")
            elif y_edge:
                if current_step_recorder:
                    current_step_recorder.stop()
                    current_step_recorder = None
                print("[Injection] Recording stopped via Y during final-step warning.")

        # Post-step retract / skip overlay (A / B only)
        if post_step_prompt_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))

            with state.lock:
                s = state.settings
                syringe_step_ul_overlay    = s.get("syringe_step_ul", 1.0)
                syringe_remaining_ul_overlay = s.get("syringe_remaining_ul", 0.0)
                retract_mm_cfg = float(
                    s.get("injection_retract_mm", 10.0) or 0.0
                )

            steps_left = 0
            remainder_ul = syringe_remaining_ul_overlay
            if syringe_step_ul_overlay > 0:
                steps_left = int(syringe_remaining_ul_overlay // syringe_step_ul_overlay)
                remainder_ul = (
                    syringe_remaining_ul_overlay
                    - steps_left * syringe_step_ul_overlay
                )

            lines = ["Injection step complete."]
            if syringe_step_ul_overlay > 0:
                lines.append(
                    f"Remaining full steps: {steps_left} × {syringe_step_ul_overlay:.3f} µL"
                )
            lines.append(f"Remaining volume: {syringe_remaining_ul_overlay:.3f} µL")
            if remainder_ul > 1e-3 and syringe_step_ul_overlay > 0:
                lines.append(f"Final smaller step ≈ {remainder_ul:.3f} µL")
            if injection_last_step_small:
                lines.append(
                    f"Last step was smaller than configured step ({injection_step_volume_ul:.3f} µL)"
                )

            lines.append(f"Retract toolhead by {retract_mm_cfg:.1f} mm?")
            lines.append("A = retract, B = skip   (Y = stop recording)")

            color = (255, 255, 255)
            y0 = screen_h // 2 - (len(lines) * (big_font.get_height() + 4)) // 2
            for ln in lines:
                t = big_font.render(ln, True, color)
                screen.blit(t, ((screen_w - t.get_width()) // 2, y0))
                y0 += t.get_height() + 4

            if a_edge:
                if retract_mm_cfg > 0.0:
                    rt.enqueue({"type": "inject_retract", "dist": retract_mm_cfg})
                    print(f"[Injection] Toolhead retracted by {retract_mm_cfg:.2f} mm.")
                post_step_prompt_active = False
                injection_step_done = False

                with state.lock:
                    inj_active_next = bool(state.settings.get("injection_active", False))
                    vol_left = float(
                        state.settings.get("syringe_remaining_ul", 0.0) or 0.0
                    )
                    prev_subj = state.settings.get("injection_subject", "")
                if inj_active_next and vol_left > 1e-9:
                    subject_field_text = injection_subject_text or prev_subj or ""
                    subject_prompt_active = True

            elif b_edge:
                print("[Injection] Retraction skipped after injection step.")
                post_step_prompt_active = False
                injection_step_done = False

                with state.lock:
                    inj_active_next = bool(state.settings.get("injection_active", False))
                    vol_left = float(
                        state.settings.get("syringe_remaining_ul", 0.0) or 0.0
                    )
                    prev_subj = state.settings.get("injection_subject", "")
                if inj_active_next and vol_left > 1e-9:
                    subject_field_text = injection_subject_text or prev_subj or ""
                    subject_prompt_active = True

            elif y_edge:
                if current_step_recorder:
                    current_step_recorder.stop()
                    current_step_recorder = None
                print("[Injection] Recording stopped for this step via Y button.")

        # Pose edit menu
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
                can_toggle_reserved=True,
            )

            if click_pos is not None:
                mx, my = click_pos
                if not panel_rect.collidepoint(mx, my):
                    edit_menu_active = False
                else:
                    for aid, r in rects.items():
                        if r.collidepoint(mx, my):
                            if aid == "rename" and not (built_in_reserved or user_reserved):
                                rename_input_active = True
                                rename_old_name = edit_menu_target
                                rename_text = rename_old_name
                                edit_menu_active = False
                            elif aid == "delete" and not (built_in_reserved or user_reserved):
                                delete_confirm_active = True
                                delete_target_name = edit_menu_target
                                edit_menu_active = False
                            elif aid == "overwrite":
                                overwrite_confirm_active = True
                                overwrite_target_name = edit_menu_target
                                edit_menu_active = False
                            elif aid == "reserve":
                                with state.lock:
                                    current = list(
                                        state.settings.get("reserved_poses", [])
                                    )
                                    if edit_menu_target not in current:
                                        current.append(edit_menu_target)
                                    state.settings["reserved_poses"] = current
                                save_settings(state.settings)
                                edit_menu_active = False
                            elif aid == "unreserve":
                                with state.lock:
                                    current = list(
                                        state.settings.get("reserved_poses", [])
                                    )
                                    if edit_menu_target in current:
                                        current.remove(edit_menu_target)
                                    state.settings["reserved_poses"] = current
                                save_settings(state.settings)
                                edit_menu_active = False
                            elif aid in ("cancel", "disabled_rename", "disabled_delete"):
                                edit_menu_active = False
                            break
                click_pos = None

        # Delete pose confirm
        if delete_confirm_active:
            rects, panel = draw_confirm_delete(screen, delete_target_name)
            if click_pos is not None:
                mx, my = click_pos
                if not panel.collidepoint(mx, my):
                    delete_confirm_active = False
                elif rects["yes"].collidepoint(click_pos):
                    rt.enqueue({"type": "delete_pose", "pose": delete_target_name})
                    delete_confirm_active = False
                elif rects["no"].collidepoint(click_pos):
                    delete_confirm_active = False
                click_pos = None

        # Overwrite pose confirm
        if overwrite_confirm_active:
            rects, panel = draw_confirm_overwrite(screen, overwrite_target_name)
            if click_pos is not None:
                mx, my = click_pos
                if not panel.collidepoint(mx, my):
                    overwrite_confirm_active = False
                elif rects["yes"].collidepoint(click_pos):
                    rt.enqueue({
                        "type": "overwrite_pose_with_midway",
                        "pose": overwrite_target_name,
                    })
                    overwrite_confirm_active = False
                elif rects["no"].collidepoint(click_pos):
                    overwrite_confirm_active = False
                click_pos = None

        # Midway confirmation overlay
        with state.lock:
            waiting = state.await_confirm
            confirm_msg = state.confirm_msg
        if waiting:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            screen.blit(overlay, (0, 0))
            txt1_lines = wrap_text(
                confirm_msg or "Paused at Midway. A: advance  B: free control",
                big_font,
                screen_w - 40,
            )
            y0 = screen_h // 2 - (
                len(txt1_lines) * (big_font.get_height() + 4)
            ) // 2
            for ln in txt1_lines:
                txt1 = big_font.render(ln, True, (255, 255, 255))
                screen.blit(txt1, ((screen_w - txt1.get_width()) // 2, y0))
                y0 += txt1.get_height() + 4

        # Routine editor overlay
        if routine_edit_active:
            draw_routine_editor(screen, routine_editor)

        pygame.display.flip()
        clock.tick(60)

    # ─────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────
    try:
        rt.stop()
        rt.join()
    except Exception:
        pass
    try:
        rs_thread.stop()
        rs_thread.join()
    except Exception:
        pass
    for t in uvc_threads:
        try:
            t.stop()
            t.join()
        except Exception:
            pass
    if ser:
        try:
            ser.close()
        except Exception:
            pass
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
