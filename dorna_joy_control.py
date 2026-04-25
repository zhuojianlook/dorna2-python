#!/usr/bin/env python3
import sys
import argparse
import threading
import time
import os
import json
import queue
import re, glob
import shutil
import subprocess
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
DEFAULT_DORNA_HOST = "10.42.0.11"
DEFAULT_DORNA_PORT = 443
DEFAULT_UVC_FPS = 30
DEFAULT_UVC_WIDTH = 640
DEFAULT_UVC_HEIGHT = 480
DEFAULT_RS_WIDTH = 640
DEFAULT_RS_HEIGHT = 480
DEFAULT_RS_FPS = 30
_V4L2_CTL = shutil.which("v4l2-ctl")
_V4L2_CAPS_CACHE = {}

def _profile_label(width: int, height: int, fps: int, note: str = "") -> str:
    label = f"{int(width)}x{int(height)} @ {int(fps)} fps"
    if note:
        label += f" {note}"
    return label

UVC_QUALITY_PRESETS = [
    (_profile_label(640, 480, 30, "(Best UVC)"), (640, 480, 30)),
    (_profile_label(640, 480, 15), (640, 480, 15)),
    (_profile_label(320, 240, 30), (320, 240, 30)),
    (_profile_label(320, 240, 15), (320, 240, 15)),
]

RS_QUALITY_PRESETS = [
    (_profile_label(640, 480, 30, "(Default)"), (640, 480, 30)),
    (_profile_label(1280, 720, 30, "(Max color/depth)"), (1280, 720, 30)),
    (_profile_label(640, 360, 30), (640, 360, 30)),
    (_profile_label(424, 240, 30), (424, 240, 30)),
]

HALT_PID_PRESETS = [
    ("Stock (200 / 10000)", (200, 10000)),
    ("Mild (180 / 7000)", (180, 7000)),
    ("Moderate (160 / 5000)", (160, 5000)),
    ("Sensitive (140 / 3000)", (140, 3000)),
    ("Aggressive (120 / 2000)", (120, 2000)),
]

def _profile_value_map(presets):
    return {label: values for label, values in presets}

def _profile_label_for_values(width: int, height: int, fps: int, presets) -> str:
    wanted = (int(width), int(height), int(fps))
    for label, values in presets:
        if tuple(values) == wanted:
            return label
    return _profile_label(*wanted, note="(Custom)")

def _profile_values_from_label(label: str, presets, fallback):
    label = str(label or "").strip()
    by_label = _profile_value_map(presets)
    if label in by_label:
        return tuple(by_label[label])
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*@\s*(\d+)\s*fps", label, re.IGNORECASE)
    if m:
        return tuple(int(g) for g in m.groups())
    return tuple(int(v) for v in fallback)

def _profile_choice_values(presets, current_label: str) -> list:
    values = [label for label, _ in presets]
    current_label = str(current_label or "").strip()
    if current_label and current_label not in values:
        values.append(current_label)
    return values

def _halt_preset_value_map():
    return {label: values for label, values in HALT_PID_PRESETS}

def _halt_preset_label_for_values(threshold: float, duration: float) -> str:
    current = _clamp_alarm_pid(threshold, duration)
    for label, values in HALT_PID_PRESETS:
        if tuple(values) == tuple(current):
            return label
    return f"Custom ({int(current[0])} / {int(current[1])})"

def _halt_preset_choice_values(current_label: str) -> list:
    values = [label for label, _ in HALT_PID_PRESETS]
    current_label = str(current_label or "").strip()
    if current_label and current_label not in values:
        values.append(current_label)
    return values

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
                "serial": _read_text(os.path.join(cur, "serial")) or "",
                "product": _read_text(os.path.join(cur, "product")) or "",
                "manufacturer": _read_text(os.path.join(cur, "manufacturer")) or "",
                "vendor_id": _read_text(os.path.join(cur, "idVendor")) or "",
                "model_id": _read_text(os.path.join(cur, "idProduct")) or "",
            }
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return {
        "busnum":"", "devpath":"", "speed":"",
        "serial":"", "product":"", "manufacturer":"",
        "vendor_id":"", "model_id":""
    }

def _canon_usb_alias(path: str) -> str:
    return re.sub(r"-usbv[23]-", "-usb-", path)

def _usb_variant_rank(path: str) -> int:
    if "-usb-" in path and "-usbv" not in path:
        return 0
    if "-usbv2-" in path:
        return 1
    if "-usbv3-" in path:
        return 2
    return 3

def _camera_path_rank(path: str):
    if path.endswith("video-index0"):
        idx_rank = 0
    elif path.endswith("video-index1"):
        idx_rank = 1
    else:
        idx_rank = 2
    return (idx_rank, _usb_variant_rank(path), path)

def _video_node_sort_key(path: str):
    m = re.match(r"^/dev/video(\d+)$", str(path))
    return int(m.group(1)) if m else 9999

def _extract_usb_port_path(path: str) -> str:
    path = str(path or "")
    m = re.search(r"-usb(?:v[23])?-(.+)-video-index\d+$", path)
    return (m.group(1) if m else "")

def _short_serial(serial: str) -> str:
    serial = str(serial or "").strip()
    if not serial:
        return ""
    return serial.rsplit("_", 1)[-1]

def _v4l2_caps_text(node: str) -> str:
    node = os.path.realpath(str(node))
    cached = _V4L2_CAPS_CACHE.get(node)
    if cached is not None:
        return cached
    text = ""
    if _V4L2_CTL:
        try:
            cp = subprocess.run(
                [_V4L2_CTL, "-D", "-d", node],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            text = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
        except Exception:
            text = ""
    _V4L2_CAPS_CACHE[node] = text
    return text

def _v4l2_device_caps(node: str) -> list:
    caps = []
    in_device_caps = False
    for raw in _v4l2_caps_text(node).splitlines():
        line = raw.strip().lower()
        if line.startswith("device caps"):
            in_device_caps = True
            continue
        if in_device_caps:
            if raw and not raw[0].isspace():
                break
            if line:
                caps.append(line)
    return caps

def _uvc_path_capture_rank(path: str) -> int:
    node = os.path.realpath(str(path))
    caps = _v4l2_device_caps(node)
    if caps:
        if any("video capture" in cap for cap in caps):
            return 0
        if any("metadata capture" in cap for cap in caps):
            return 2
    idx = _read_text(f"/sys/class/video4linux/{os.path.basename(node)}/index")
    if idx == "0":
        return 0
    if idx.isdigit():
        return 1
    return 1

def _uvc_alias_sort_key(path: str):
    return (_uvc_path_capture_rank(path),) + _camera_path_rank(path)

def _prefer_capture_aliases(paths) -> list:
    uniq = []
    seen = set()
    for path in paths or []:
        if path and path not in seen:
            uniq.append(path)
            seen.add(path)
    return sorted(uniq, key=_uvc_alias_sort_key)

def _finalize_uvc_inventory_item(item: dict):
    aliases = _prefer_capture_aliases(item.get("aliases", []))
    if not aliases:
        return None
    primary = aliases[0]
    node = os.path.realpath(primary)
    info = _usb_info_from_node(node)
    item["aliases"] = aliases
    item["path"] = primary
    item["node"] = node
    item["name"] = _v4l_name_for_node(node)
    item["busnum"] = info["busnum"]
    item["devpath"] = info["devpath"]
    item["serial"] = info["serial"]
    item["serial_short"] = _short_serial(info["serial"])
    item["vendor_id"] = info["vendor_id"]
    item["model_id"] = info["model_id"]
    item["product"] = info["product"]
    item["manufacturer"] = info["manufacturer"]
    item["port_path"] = _extract_usb_port_path(primary) or info["devpath"]
    item["capture_rank"] = _uvc_path_capture_rank(primary)
    return item

def _find_uvc_inventory_item(device: str):
    device = str(device or "")
    real = os.path.realpath(device) if device else ""
    for item in discover_uvc_inventory(limit=64):
        aliases = item.get("aliases", [])
        if device == item.get("path") or device in aliases:
            return item
        if real and (real == os.path.realpath(item.get("path", ""))):
            return item
        for alias in aliases:
            if real and real == os.path.realpath(alias):
                return item
    return None

def _candidate_uvc_paths(device: str, include_secondary_aliases=False) -> list:
    item = _find_uvc_inventory_item(device)
    if item:
        aliases = list(item.get("aliases", []))
    else:
        aliases = [str(device)]
    if include_secondary_aliases:
        return aliases
    preferred = [p for p in aliases if _uvc_path_capture_rank(p) < 2]
    return preferred or aliases[:1]

def discover_uvc_index0(limit=4) -> list:
    found = {}
    links = sorted(glob.glob("/dev/v4l/by-path/*video-index0"))
    for link in links:
        node = os.path.realpath(link)
        if not node.startswith("/dev/video"):
            continue
        name = _v4l_name_for_node(node)
        if _is_realsense_name(name):
            continue
        info = _usb_info_from_node(node)
        item = {
            "bypath": link,
            "node": node,
            "name": name,
            "busnum": info["busnum"],
            "devpath": info["devpath"],
        }
        # Deduplicate alias paths that point to the same physical camera.
        key = (info["busnum"], info["devpath"]) if (info["busnum"] or info["devpath"]) else node
        prev = found.get(key)
        if prev is None or _usb_variant_rank(link) < _usb_variant_rank(prev["bypath"]):
            found[key] = item
    return sorted(
        found.values(),
        key=lambda item: (item["busnum"], item["devpath"], item["bypath"]),
    )[:limit]

def discover_uvc_inventory(limit=12) -> list:
    bypath_groups = {}
    covered_nodes = set()
    for link in sorted(glob.glob("/dev/v4l/by-path/*video-index[01]")):
        node = os.path.realpath(link)
        if not node.startswith("/dev/video"):
            continue
        name = _v4l_name_for_node(node)
        if _is_realsense_name(name):
            continue
        base = _canon_usb_alias(re.sub(r"-video-index[01]$", "", link))
        item = bypath_groups.setdefault(base, {
            "aliases": [],
            "source": "by-path",
        })
        item["aliases"].append(link)
        covered_nodes.add(node)

    out = []
    for base in sorted(bypath_groups):
        item = _finalize_uvc_inventory_item(bypath_groups[base])
        if item is not None:
            out.append(item)

    for node in sorted(glob.glob("/dev/video[0-9]*"), key=_video_node_sort_key):
        if node in covered_nodes:
            continue
        name = _v4l_name_for_node(node)
        if _is_realsense_name(name):
            continue
        capture_rank = _uvc_path_capture_rank(node)
        if capture_rank >= 2:
            continue
        info = _usb_info_from_node(node)
        out.append({
            "path": node,
            "node": node,
            "name": name,
            "busnum": info["busnum"],
            "devpath": info["devpath"],
            "aliases": [node],
            "source": "direct",
            "serial": info["serial"],
            "serial_short": _short_serial(info["serial"]),
            "vendor_id": info["vendor_id"],
            "model_id": info["model_id"],
            "product": info["product"],
            "manufacturer": info["manufacturer"],
            "port_path": info["devpath"],
            "capture_rank": capture_rank,
        })

    out = sorted(
        out,
        key=lambda item: (
            item.get("capture_rank", 1),
            item.get("busnum", ""),
            item.get("devpath", ""),
            item.get("path", ""),
        ),
    )
    return out[:limit]

def find_uvc_devices(limit=2):
    links = glob.glob("/dev/v4l/by-path/*video-index[01]")

    def canon_base(p):
        b = re.sub(r"-video-index[01]$", "", p)
        return _canon_usb_alias(b)

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
        i0 = sorted(b["index0"], key=_usb_variant_rank)
        i1 = sorted(b["index1"], key=_usb_variant_rank)
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

        self._stop_event = threading.Event()
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
        self._stop_event.set()

    def run(self):
        paths_to_try = _candidate_uvc_paths(self.device, include_secondary_aliases=self.try_index1_fallback)
        if self.device and self.device not in paths_to_try:
            paths_to_try.insert(0, self.device)
        paths_to_try = _prefer_capture_aliases(paths_to_try)

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

            while not self._stop_event.is_set():
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
                        reopen_paths = [chosen_path] + [p for p in paths_to_try if p != chosen_path]
                        for retry_path in reopen_paths:
                            cap, warm_frame, meta = self._open_resilient(retry_path)
                            if cap is not None:
                                chosen_path = retry_path
                                break
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

        self._stop_event = threading.Event()
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
        self._stop_event.set()

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
            while not self._stop_event.is_set():
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

def _apply_deadzone(v: float, dz: float) -> float:
    av = abs(float(v))
    if av <= dz:
        return 0.0
    return float(np.sign(v) * ((av - dz) / max(1e-9, 1.0 - dz)))

A_BUTTON       = 0
B_BUTTON       = 1
X_BUTTON       = 2
Y_BUTTON       = 3

AXIS_GUARD_DELTA  = 0.20
AXIS_GUARD_WINDOW = 0.15

POSES_PATH     = "poses.json"
SETTINGS_PATH  = "settings.json"
STARTUP_SETTINGS_PATH = ".dorna_launcher.json"
MIDWAY_SUFFIX  = "__midway"

DEFAULT_TOOL_LZ     = 205.0
DEFAULT_APPROACH_MM = 25.0
DEFAULT_TOOL_CX     = 0.0
DEFAULT_TOOL_CY     = 0.0
DEFAULT_TOOL_CENTER_RADIUS = 10.0
DEFAULT_ALARM_SENSITIVITY  = 1.0
DEFAULT_PID_THRESHOLD_MAIN = 200.0
DEFAULT_PID_DURATION_MAIN = 10000.0
DEFAULT_PID_THRESHOLD_MIN = 1.0
DEFAULT_PID_THRESHOLD_MAX = 400.0
DEFAULT_PID_DURATION_MIN = 1.0
DEFAULT_PID_DURATION_MAX = 20000.0
HALT_TUNE_MOVE_MM = 50.0
HALT_TUNE_MOVE_VEL = 10.0
HALT_TUNE_HOLD_S = 0.75
COLLISION_JOINT_AXES = ("j0", "j1", "j2", "j3", "j4", "j5")
COLLISION_TCP_AXES = ("x", "y", "z", "a", "b", "c")
DEFAULT_SELF_COLLISION = {
    "enabled": True,
    "link_radii_mm": [32.0, 35.0, 20.0, 32.0, 24.0, 18.0],
    "pair_margin_mm": 8.0,
    "min_link_gap": 3,
    "path_step_deg": 6.0,
    "base_radius_mm": 75.0,
    "base_height_mm": 230.0,
    "base_segments": [3, 4, 5],
}

DEFAULT_POSES = {
    "Reload":  {"j0": 7.71, "j1": 80.86, "j2": -100.00, "j3": -0.07, "j4": -70.60, "j5": 6.35},
    "Default": {"j0": 39.24,"j1": 27.00, "j2": -100.44, "j3": 6.35,  "j4": 65.19,  "j5": -2.70},
}
RESERVED_POSES = {"Default", "Reload"}

def midway_name(name: str) -> str:
    return f"{name}{MIDWAY_SUFFIX}"

def _normalize_axis_range(raw):
    if isinstance(raw, dict):
        if "min" in raw and "max" in raw:
            try:
                lo = float(raw["min"])
                hi = float(raw["max"])
                return (min(lo, hi), max(lo, hi))
            except Exception:
                return None
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            lo = float(raw[0])
            hi = float(raw[1])
            return (min(lo, hi), max(lo, hi))
        except Exception:
            return None
    return None

def _extract_collision_ranges(container, axes):
    if not isinstance(container, dict):
        return {}
    out = {}
    for axis in axes:
        rng = None
        if axis in container:
            rng = _normalize_axis_range(container.get(axis))
        if rng is None and f"{axis}_range" in container:
            rng = _normalize_axis_range(container.get(f"{axis}_range"))
        if rng is None and f"{axis}_min" in container and f"{axis}_max" in container:
            rng = _normalize_axis_range([container.get(f"{axis}_min"), container.get(f"{axis}_max")])
        if rng is not None:
            out[axis] = rng
    return out

def _normalize_collision_zones(raw_zones):
    zones = []
    if not isinstance(raw_zones, list):
        return zones

    for idx, raw in enumerate(raw_zones, start=1):
        if not isinstance(raw, dict):
            continue

        joint_src = raw.get("joints", raw.get("joint_ranges", raw))
        tcp_src = raw.get("tcp", raw.get("tcp_ranges", raw))
        joint_ranges = _extract_collision_ranges(joint_src, COLLISION_JOINT_AXES)
        tcp_ranges = _extract_collision_ranges(tcp_src, COLLISION_TCP_AXES)
        if not joint_ranges and not tcp_ranges:
            continue

        zones.append({
            "name": str(raw.get("name") or f"zone_{idx}"),
            "enabled": bool(raw.get("enabled", True)),
            "joint_ranges": joint_ranges,
            "tcp_ranges": tcp_ranges,
        })

    return zones

def _normalize_float_list(raw, length, fallback):
    if not isinstance(raw, (list, tuple)):
        return list(fallback)
    out = []
    for idx in range(length):
        try:
            out.append(float(raw[idx]))
        except Exception:
            out.append(float(fallback[idx]))
    return out

def _normalize_int_list(raw, fallback):
    if not isinstance(raw, (list, tuple)):
        return list(fallback)
    out = []
    for item in raw:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out or list(fallback)

def _normalize_self_collision(raw_cfg):
    base = dict(DEFAULT_SELF_COLLISION)
    if isinstance(raw_cfg, bool):
        base["enabled"] = bool(raw_cfg)
        return base
    if not isinstance(raw_cfg, dict):
        return base

    if "enabled" in raw_cfg:
        base["enabled"] = bool(raw_cfg.get("enabled"))
    base["link_radii_mm"] = _normalize_float_list(
        raw_cfg.get("link_radii_mm", raw_cfg.get("link_radii", base["link_radii_mm"])),
        6,
        base["link_radii_mm"],
    )
    try:
        base["pair_margin_mm"] = float(raw_cfg.get("pair_margin_mm", base["pair_margin_mm"]))
    except Exception:
        pass
    try:
        base["min_link_gap"] = max(2, int(raw_cfg.get("min_link_gap", base["min_link_gap"])))
    except Exception:
        pass
    try:
        base["path_step_deg"] = max(0.5, float(raw_cfg.get("path_step_deg", base["path_step_deg"])))
    except Exception:
        pass
    try:
        base["base_radius_mm"] = max(0.0, float(raw_cfg.get("base_radius_mm", base["base_radius_mm"])))
    except Exception:
        pass
    try:
        base["base_height_mm"] = max(0.0, float(raw_cfg.get("base_height_mm", base["base_height_mm"])))
    except Exception:
        pass
    base["base_segments"] = _normalize_int_list(
        raw_cfg.get("base_segments", base["base_segments"]),
        base["base_segments"],
    )
    return base

def _alarm_pid_from_sensitivity(val: float):
    """
    Map a 0..1 halt-sensitivity setting onto Dorna's threshold/duration scale.
    0.0 keeps the stock threshold/duration, 1.0 makes halting much more eager.
    """
    try:
        val = float(val)
    except Exception:
        val = DEFAULT_ALARM_SENSITIVITY
    val = max(0.0, min(1.0, val))
    threshold_scale = 1.0 - 0.70 * val
    duration_scale = 1.0 - 0.75 * val
    threshold = max(DEFAULT_PID_THRESHOLD_MIN, round(DEFAULT_PID_THRESHOLD_MAIN * threshold_scale))
    duration = max(DEFAULT_PID_DURATION_MIN, round(DEFAULT_PID_DURATION_MAIN * duration_scale))
    return val, threshold, duration

def _clamp_alarm_pid(threshold=None, duration=None):
    try:
        threshold = float(threshold)
    except Exception:
        threshold = DEFAULT_PID_THRESHOLD_MAIN
    try:
        duration = float(duration)
    except Exception:
        duration = DEFAULT_PID_DURATION_MAIN
    threshold = round(max(DEFAULT_PID_THRESHOLD_MIN, min(DEFAULT_PID_THRESHOLD_MAX, threshold)))
    duration = round(max(DEFAULT_PID_DURATION_MIN, min(DEFAULT_PID_DURATION_MAX, duration)))
    return threshold, duration

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
        tool_cx     = float(data.get("tool_cx", DEFAULT_TOOL_CX))
        tool_cy     = float(data.get("tool_cy", DEFAULT_TOOL_CY))
        tool_center_demo_radius_mm = float(
            data.get("tool_center_demo_radius_mm", DEFAULT_TOOL_CENTER_RADIUS)
        )
        legacy_alarm_sensitivity = None
        if "alarm_sensitivity" in data:
            try:
                legacy_alarm_sensitivity = float(data.get("alarm_sensitivity"))
            except Exception:
                legacy_alarm_sensitivity = DEFAULT_ALARM_SENSITIVITY
        if ("alarm_threshold" in data) or ("alarm_duration" in data):
            alarm_threshold, alarm_duration = _clamp_alarm_pid(
                data.get("alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN),
                data.get("alarm_duration", DEFAULT_PID_DURATION_MAIN),
            )
        elif legacy_alarm_sensitivity is not None:
            _, alarm_threshold, alarm_duration = _alarm_pid_from_sensitivity(legacy_alarm_sensitivity)
            alarm_threshold, alarm_duration = _clamp_alarm_pid(alarm_threshold, alarm_duration)
        else:
            alarm_threshold, alarm_duration = _clamp_alarm_pid(
                DEFAULT_PID_THRESHOLD_MAIN,
                DEFAULT_PID_DURATION_MAIN,
            )
        approach_mm = float(data.get("approach_mm", DEFAULT_APPROACH_MM))

        end_kw = data.get("endstop_keywords",
                          ["ENDSTOP", "LIMIT", "LIM", "HIT", "END"])
        end_timeout_ms    = int(data.get("endstop_timeout_ms", 0))
        plunger_fwd_rate  = int(data.get("plunger_fwd_rate", MAX_RATE))
        plunger_bwd_rate  = int(data.get("plunger_bwd_rate", MAX_RATE))
        pose_final_wait_s = float(data.get("pose_final_wait_s", 0.0))
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

        # Tool presets (name + lz + cx + cy)
        tool_presets = data.get("tool_presets", [])
        if not isinstance(tool_presets, list):
            tool_presets = []
        norm_presets = []
        for p in tool_presets:
            if not isinstance(p, dict):
                continue
            try:
                nm = str(p.get("name", "")).strip()
                lz_p = float(p.get("lz", tool_lz))
                cx_p = float(p.get("cx", tool_cx))
                cy_p = float(p.get("cy", tool_cy))
            except Exception:
                continue
            if nm:
                norm_presets.append({"name": nm, "lz": lz_p, "cx": cx_p, "cy": cy_p})

        collision_zones = _normalize_collision_zones(data.get("collision_zones", []))
        self_collision = _normalize_self_collision(data.get("self_collision", {}))

        return {
            "tool_lz": tool_lz,
            "tool_cx": tool_cx,
            "tool_cy": tool_cy,
            "tool_center_demo_radius_mm": tool_center_demo_radius_mm,
            "alarm_threshold": alarm_threshold,
            "alarm_duration": alarm_duration,
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
            "tool_presets": norm_presets,
            "collision_zones": collision_zones,
            "self_collision": self_collision,
        }
    except Exception as e:
        print(f"⚠️ Using default settings (could not load {path}: {e})")
        return {
            "tool_lz": DEFAULT_TOOL_LZ,
            "tool_cx": DEFAULT_TOOL_CX,
            "tool_cy": DEFAULT_TOOL_CY,
            "tool_center_demo_radius_mm": DEFAULT_TOOL_CENTER_RADIUS,
            "alarm_threshold": DEFAULT_PID_THRESHOLD_MAIN,
            "alarm_duration": DEFAULT_PID_DURATION_MAIN,
            "approach_mm": DEFAULT_APPROACH_MM,
            "endstop_keywords": ["ENDSTOP", "LIMIT", "LIM", "HIT", "END"],
            "endstop_timeout_ms": 0,
            "plunger_fwd_rate": MAX_RATE,
            "plunger_bwd_rate": MAX_RATE,
            "pose_final_wait_s": 0.0,
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
            "tool_presets": [],
            "collision_zones": [],
            "self_collision": dict(DEFAULT_SELF_COLLISION),
        }


def save_settings(settings, path=SETTINGS_PATH):
    try:
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save settings to {path}: {e}")

def load_startup_settings(path=STARTUP_SETTINGS_PATH):
    defaults = {
        "startup_host": DEFAULT_DORNA_HOST,
        "startup_port": DEFAULT_DORNA_PORT,
        "startup_uvc1": "",
        "startup_uvc2": "",
        "startup_uvc_width": DEFAULT_UVC_WIDTH,
        "startup_uvc_height": DEFAULT_UVC_HEIGHT,
        "startup_uvc_fps": DEFAULT_UVC_FPS,
        "startup_rs_width": DEFAULT_RS_WIDTH,
        "startup_rs_height": DEFAULT_RS_HEIGHT,
        "startup_rs_fps": DEFAULT_RS_FPS,
        "startup_uvc_try_index1": False,
        "startup_fullscreen": False,
        "startup_clear_alarm": True,
        "startup_apply_halt_settings": True,
        "startup_auto_tune_halt": False,
        "startup_alarm_threshold": DEFAULT_PID_THRESHOLD_MAIN,
        "startup_alarm_duration": DEFAULT_PID_DURATION_MAIN,
        "startup_show_launcher": True,
    }

    # Backward compatibility with any startup keys previously written to settings.json.
    try:
        legacy = load_settings()
        for key in defaults:
            if key in legacy:
                defaults[key] = legacy[key]
        if "alarm_threshold" in legacy:
            defaults["startup_alarm_threshold"] = float(
                legacy.get("alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN)
            )
        if "alarm_duration" in legacy:
            defaults["startup_alarm_duration"] = float(
                legacy.get("alarm_duration", DEFAULT_PID_DURATION_MAIN)
            )
    except Exception:
        pass

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in defaults:
                    if key in data:
                        defaults[key] = data[key]
                if (
                    "startup_alarm_sensitivity" in data
                    and "startup_alarm_threshold" not in data
                    and "startup_alarm_duration" not in data
                ):
                    _, threshold, duration = _alarm_pid_from_sensitivity(
                        data.get("startup_alarm_sensitivity", DEFAULT_ALARM_SENSITIVITY)
                    )
                    defaults["startup_alarm_threshold"] = threshold
                    defaults["startup_alarm_duration"] = duration
    except Exception as e:
        print(f"⚠️ Could not load startup settings from {path}: {e}")
    return defaults

def save_startup_settings(settings, path=STARTUP_SETTINGS_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save startup settings to {path}: {e}")

def save_startup_halt_settings(threshold, duration):
    try:
        settings = load_startup_settings()
        threshold, duration = _clamp_alarm_pid(threshold, duration)
        settings["startup_alarm_threshold"] = threshold
        settings["startup_alarm_duration"] = duration
        save_startup_settings(settings)
    except Exception as e:
        print(f"⚠️ Could not save startup halt settings: {e}")

def _format_uvc_inventory_label(item: dict) -> str:
    base = f"{item.get('name', 'Unknown')} [{os.path.basename(item.get('node', ''))}]"
    src = item.get("path", "")
    details = []
    if item.get("port_path"):
        details.append(f"port={item.get('port_path')}")
    elif item.get("busnum") or item.get("devpath"):
        details.append(f"bus={item.get('busnum', '')} devpath={item.get('devpath', '')}")
    if item.get("serial_short"):
        details.append(f"serial={item.get('serial_short')}")
    if item.get("vendor_id") and item.get("model_id"):
        details.append(f"usb={item.get('vendor_id')}:{item.get('model_id')}")
    if details:
        base += " " + " ".join(details)
    if src:
        base += f" via {src}"
    return base

def _format_uvc_selected_path(path: str) -> str:
    item = _find_uvc_inventory_item(path)
    if item is not None:
        return _format_uvc_inventory_label(item)
    return str(path or "")

def _resolve_startup_args(args, settings):
    args.host = str(args.host or settings.get("startup_host", DEFAULT_DORNA_HOST) or DEFAULT_DORNA_HOST)
    args.port = int(args.port if args.port is not None else settings.get("startup_port", DEFAULT_DORNA_PORT))
    args.uvc1 = str(args.uvc1 or settings.get("startup_uvc1", "") or "")
    args.uvc2 = str(args.uvc2 or settings.get("startup_uvc2", "") or "")
    args.uvc_width = int(args.uvc_width if args.uvc_width is not None else settings.get("startup_uvc_width", DEFAULT_UVC_WIDTH))
    args.uvc_height = int(args.uvc_height if args.uvc_height is not None else settings.get("startup_uvc_height", DEFAULT_UVC_HEIGHT))
    args.uvc_fps = int(args.uvc_fps if args.uvc_fps is not None else settings.get("startup_uvc_fps", DEFAULT_UVC_FPS))
    args.rs_width = int(args.rs_width if args.rs_width is not None else settings.get("startup_rs_width", DEFAULT_RS_WIDTH))
    args.rs_height = int(args.rs_height if args.rs_height is not None else settings.get("startup_rs_height", DEFAULT_RS_HEIGHT))
    args.rs_fps = int(args.rs_fps if args.rs_fps is not None else settings.get("startup_rs_fps", DEFAULT_RS_FPS))
    if args.uvc_try_index1 is None:
        args.uvc_try_index1 = bool(settings.get("startup_uvc_try_index1", False))
    if args.fullscreen is None:
        args.fullscreen = bool(settings.get("startup_fullscreen", False))
    if getattr(args, "clear_alarm_startup", None) is None:
        args.clear_alarm_startup = bool(settings.get("startup_clear_alarm", True))
    if getattr(args, "apply_halt_settings_startup", None) is None:
        args.apply_halt_settings_startup = bool(settings.get("startup_apply_halt_settings", True))
    if getattr(args, "auto_tune_halt_startup", None) is None:
        args.auto_tune_halt_startup = bool(settings.get("startup_auto_tune_halt", False))
    args.alarm_threshold = float(
        args.alarm_threshold
        if getattr(args, "alarm_threshold", None) is not None
        else settings.get("startup_alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN)
    )
    args.alarm_duration = float(
        args.alarm_duration
        if getattr(args, "alarm_duration", None) is not None
        else settings.get("startup_alarm_duration", DEFAULT_PID_DURATION_MAIN)
    )
    if args.launcher is None:
        args.launcher = bool(settings.get("startup_show_launcher", True))
    return args

def _persist_startup_args(settings, args):
    settings["startup_host"] = str(args.host or DEFAULT_DORNA_HOST)
    settings["startup_port"] = int(args.port or DEFAULT_DORNA_PORT)
    settings["startup_uvc1"] = str(args.uvc1 or "")
    settings["startup_uvc2"] = str(args.uvc2 or "")
    settings["startup_uvc_width"] = int(args.uvc_width or DEFAULT_UVC_WIDTH)
    settings["startup_uvc_height"] = int(args.uvc_height or DEFAULT_UVC_HEIGHT)
    settings["startup_uvc_fps"] = int(args.uvc_fps or DEFAULT_UVC_FPS)
    settings["startup_rs_width"] = int(args.rs_width or DEFAULT_RS_WIDTH)
    settings["startup_rs_height"] = int(args.rs_height or DEFAULT_RS_HEIGHT)
    settings["startup_rs_fps"] = int(args.rs_fps or DEFAULT_RS_FPS)
    settings["startup_uvc_try_index1"] = bool(args.uvc_try_index1)
    settings["startup_fullscreen"] = bool(args.fullscreen)
    settings["startup_clear_alarm"] = bool(getattr(args, "clear_alarm_startup", True))
    settings["startup_apply_halt_settings"] = bool(getattr(args, "apply_halt_settings_startup", True))
    settings["startup_auto_tune_halt"] = bool(getattr(args, "auto_tune_halt_startup", False))
    threshold, duration = _clamp_alarm_pid(
        getattr(args, "alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN),
        getattr(args, "alarm_duration", DEFAULT_PID_DURATION_MAIN),
    )
    settings["startup_alarm_threshold"] = threshold
    settings["startup_alarm_duration"] = duration
    settings["startup_show_launcher"] = bool(args.launcher)
    save_startup_settings(settings)


class _AlarmEventLatch:
    def __init__(self):
        self.lock = threading.Lock()
        self.latched = False
        self.last_alarm_msg = None

    def on_robot_event(self, msg, union=None, **kwargs):
        if not isinstance(msg, dict):
            return
        if msg.get("cmd") != "alarm":
            return
        alarm_val = msg.get("alarm")
        with self.lock:
            if alarm_val in (1, 1.0, True):
                self.latched = True
                self.last_alarm_msg = dict(msg)
            elif alarm_val in (0, 0.0, False):
                self.latched = False
                self.last_alarm_msg = dict(msg)

    def clear_local(self):
        with self.lock:
            self.latched = False
            self.last_alarm_msg = None

    def is_latched(self):
        with self.lock:
            return bool(self.latched)

    def last_message(self):
        with self.lock:
            return dict(self.last_alarm_msg) if isinstance(self.last_alarm_msg, dict) else None


def _emit_launcher_tune_progress(progress_cb, kind: str, **payload):
    if callable(progress_cb):
        try:
            progress_cb(kind, **payload)
        except Exception:
            pass


def _launcher_tune_log(progress_cb, message: str):
    print(message)
    _emit_launcher_tune_progress(progress_cb, "log", message=str(message))


def _launcher_try_get_current_joints(robot):
    candidates = ["get_all_joint", "get_joint", "get_joints", "joints", "get_pos"]
    joints = None
    for name in candidates:
        try:
            if hasattr(robot, name):
                res = getattr(robot, name)()
                if isinstance(res, dict):
                    vals = [res.get(k) for k in ["j0", "j1", "j2", "j3", "j4", "j5"]]
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
    return {
        "j0": float(joints[0]),
        "j1": float(joints[1]),
        "j2": float(joints[2]),
        "j3": float(joints[3]),
        "j4": float(joints[4]),
        "j5": float(joints[5]),
    }


def _launcher_wait_for_joint_settle(
    robot,
    progress_cb=None,
    max_wait_s: float = 4.0,
    stable_for_s: float = 1.0,
    tol_deg: float = 0.05,
):
    start = time.time()
    stable_since = None
    prev = None
    while time.time() - start < max_wait_s:
        joints = _launcher_try_get_current_joints(robot)
        if joints is None:
            time.sleep(0.05)
            continue
        cur = np.array([joints[f"j{i}"] for i in range(6)], dtype=float)
        if prev is None:
            prev = cur
            stable_since = time.time()
            time.sleep(0.05)
            continue
        max_delta = float(np.max(np.abs(cur - prev)))
        prev = cur
        if max_delta <= tol_deg:
            if stable_since is None:
                stable_since = time.time()
            if (time.time() - stable_since) >= stable_for_s:
                _launcher_tune_log(progress_cb, f"[Startup] Joints settled (max Δ={max_delta:.3f} deg).")
                return True
        else:
            stable_since = None
        time.sleep(0.05)
    _launcher_tune_log(
        progress_cb,
        f"[Startup] Joint settle timeout after {max_wait_s:.1f}s; launcher auto-tune aborted.",
    )
    return False


def _launcher_clear_alarm_latch(robot, alarm_latch, context: str, progress_cb=None):
    try:
        stat = robot.set_alarm(0)
        time.sleep(0.05)
        alarm_latch.clear_local()
        _launcher_tune_log(progress_cb, f"[Startup] Cleared controller alarm ({context}); stat={stat}")
        return True
    except Exception as e:
        _launcher_tune_log(progress_cb, f"⚠️ Could not clear controller alarm during {context}: {e}")
        return False


def _launcher_apply_alarm_pid(robot, alarm_latch, threshold: float, duration: float, progress_cb=None):
    threshold, duration = _clamp_alarm_pid(threshold, duration)
    alarm_latch.clear_local()
    try:
        robot.set_alarm(0)
        time.sleep(0.05)
    except Exception:
        pass
    for axis in range(6):
        robot.set_pid(index=axis, threshold=threshold, duration=duration)
    _emit_launcher_tune_progress(
        progress_cb,
        "values",
        threshold=int(threshold),
        duration=int(duration),
    )
    _launcher_tune_log(
        progress_cb,
        f"[Halt] Settings applied (threshold={int(threshold)}, duration={int(duration)})",
    )
    return threshold, duration


def _launcher_prepare_relaxed_alarm_pid(robot, alarm_latch):
    alarm_latch.clear_local()
    try:
        robot.set_alarm(0)
        time.sleep(0.05)
    except Exception:
        pass
    for axis in range(6):
        robot.set_pid(
            index=axis,
            threshold=DEFAULT_PID_THRESHOLD_MAX,
            duration=DEFAULT_PID_DURATION_MAX,
        )


def _launcher_hold_without_alarm(alarm_latch, hold_s: float = 0.75, poll_s: float = 0.05):
    end_t = time.time() + max(0.0, float(hold_s))
    while time.time() < end_t:
        if alarm_latch.is_latched():
            return False
        time.sleep(max(0.01, float(poll_s)))
    return not alarm_latch.is_latched()


def _build_alarm_threshold_values(limit: int):
    vals = [int(DEFAULT_PID_THRESHOLD_MIN)]
    cur = vals[0]
    limit = int(limit)
    while cur < limit:
        if cur < 10:
            step = 1
        elif cur < 40:
            step = 2
        elif cur < 100:
            step = 5
        elif cur < 200:
            step = 10
        else:
            step = 20
        cur = min(limit, cur + step)
        if cur != vals[-1]:
            vals.append(cur)
    return vals


def _build_alarm_duration_values(limit: int):
    vals = [int(DEFAULT_PID_DURATION_MIN)]
    cur = vals[0]
    limit = int(limit)
    while cur < limit:
        if cur < 10:
            step = 1
        elif cur < 50:
            step = 5
        elif cur < 100:
            step = 10
        elif cur < 250:
            step = 25
        elif cur < 500:
            step = 50
        elif cur < 1000:
            step = 100
        elif cur < 2500:
            step = 250
        elif cur < 5000:
            step = 500
        else:
            step = 1000
        cur = min(limit, cur + step)
        if cur != vals[-1]:
            vals.append(cur)
    return vals


def _launcher_test_alarm_pid_candidate(
    robot,
    alarm_latch,
    threshold: float,
    duration: float,
    progress_cb=None,
    hold_s: float = HALT_TUNE_HOLD_S,
):
    threshold, duration = _clamp_alarm_pid(threshold, duration)
    _launcher_tune_log(progress_cb, f"[HaltTune] Testing threshold={int(threshold)}, duration={int(duration)}")
    _launcher_prepare_relaxed_alarm_pid(robot, alarm_latch)
    _launcher_clear_alarm_latch(robot, alarm_latch, "before candidate setup", progress_cb=progress_cb)
    try:
        robot.set_motor(1)
    except Exception:
        pass
    poses = load_poses()
    default_pose = poses.get("Default", DEFAULT_POSES["Default"]).copy()
    go = {"cmd": "jmove", "rel": 0, "vel": 10.0}
    go.update(default_pose)
    _launcher_tune_log(progress_cb, "[HaltTune] Returning to Default pose before candidate test.")
    robot.play_dict(go)
    if not _launcher_wait_for_joint_settle(
        robot,
        progress_cb=progress_cb,
        max_wait_s=4.0,
        stable_for_s=1.0,
        tol_deg=0.05,
    ):
        _launcher_clear_alarm_latch(robot, alarm_latch, "after failed candidate setup", progress_cb=progress_cb)
        return False
    _launcher_apply_alarm_pid(robot, alarm_latch, threshold, duration, progress_cb=progress_cb)
    settled = _launcher_wait_for_joint_settle(
        robot,
        progress_cb=progress_cb,
        max_wait_s=2.0,
        stable_for_s=0.35,
        tol_deg=0.05,
    )
    stable = bool(settled) and _launcher_hold_without_alarm(alarm_latch, hold_s=hold_s, poll_s=0.05)
    if stable:
        for dist_mm, label in ((HALT_TUNE_MOVE_MM, "forward"), (-HALT_TUNE_MOVE_MM, "backward")):
            try:
                pose = robot.get_all_pose()[:6]
                x, y, z, a, b, c = [float(v) for v in pose]
                R = axis_angle_to_R(a, b, c)
                tz = R[:, 2]
                dx, dy, dz = tz[0] * dist_mm, tz[1] * dist_mm, tz[2] * dist_mm
                _launcher_tune_log(
                    progress_cb,
                    f"[HaltTune] Movement test: tool-axis {label} {abs(int(dist_mm))} mm.",
                )
                robot.play_dict({
                    "cmd": "lmove",
                    "rel": 1,
                    "x": float(dx),
                    "y": float(dy),
                    "z": float(dz),
                    "vel": HALT_TUNE_MOVE_VEL,
                    "cont": 0,
                })
                settled = _launcher_wait_for_joint_settle(
                    robot,
                    progress_cb=progress_cb,
                    max_wait_s=max(4.0, abs(dist_mm) / max(1e-6, HALT_TUNE_MOVE_VEL) + 2.0),
                    stable_for_s=0.35,
                    tol_deg=0.05,
                )
                stable = bool(settled) and _launcher_hold_without_alarm(alarm_latch, hold_s=0.35, poll_s=0.05)
            except Exception as e:
                _launcher_tune_log(progress_cb, f"⚠️ [HaltTune] Movement test failed during {label} move: {e}")
                stable = False
            if not stable:
                break
    if not stable:
        alarm_msg = alarm_latch.last_message()
        if alarm_msg:
            _launcher_tune_log(progress_cb, f"[HaltTune] Alarm during test: {alarm_msg}")
        _launcher_clear_alarm_latch(
            robot,
            alarm_latch,
            f"after testing {int(threshold)}/{int(duration)}",
            progress_cb=progress_cb,
        )
    return stable


def run_launcher_halt_autotune(host: str, port: int, threshold: float, duration: float, progress_cb=None):
    requested_threshold, requested_duration = _clamp_alarm_pid(threshold, duration)
    _launcher_tune_log(
        progress_cb,
        "[HaltTune] Auto-tuning in launcher: connect, move to Default, and search for the most sensitive stable halt settings.",
    )
    _launcher_tune_log(
        progress_cb,
        f"[HaltTune] Requested starting point threshold={int(requested_threshold)}, duration={int(requested_duration)}.",
    )
    _launcher_tune_log(
        progress_cb,
        "[HaltTune] Starting from the most sensitive pair and increasing slowly until no alarm is met "
        f"(up to threshold={int(DEFAULT_PID_THRESHOLD_MAX)}, duration={int(DEFAULT_PID_DURATION_MAX)}).",
    )

    alarm_latch = _AlarmEventLatch()
    robot = Dorna(model="dorna_ta")
    connected = False
    try:
        for attempt in range(5):
            if robot.connect(host=host, port=port):
                connected = True
                break
            _launcher_tune_log(
                progress_cb,
                f"⚠️ Connection attempt {attempt + 1}/5 to {host}:{port} failed. Retrying…",
            )
            time.sleep(2)
        if not connected:
            raise RuntimeError(f"All connection attempts to {host}:{port} failed.")

        _launcher_tune_log(
            progress_cb,
            f"[Robot] Kinematic model: {getattr(robot, 'model', 'unknown')} (n_dof={getattr(robot.kinematic, 'n_dof', '?')})",
        )
        try:
            robot.add_event(alarm_latch.on_robot_event)
        except Exception as e:
            _launcher_tune_log(progress_cb, f"⚠️ Could not register robot alarm event hook: {e}")

        _launcher_prepare_relaxed_alarm_pid(robot, alarm_latch)
        _launcher_clear_alarm_latch(robot, alarm_latch, "before motor enable", progress_cb=progress_cb)
        robot.set_motor(1)

        poses = load_poses()
        default_pose = poses.get("Default", DEFAULT_POSES["Default"]).copy()
        go = {"cmd": "jmove", "rel": 0, "vel": 10.0}
        go.update(default_pose)
        _launcher_tune_log(progress_cb, "[HaltTune] Moving to Default pose for launcher auto-tune.")
        robot.play_dict(go)
        _launcher_tune_log(progress_cb, "[Robot] Ready at Default for launcher auto-tune.")
        _launcher_clear_alarm_latch(robot, alarm_latch, "after default move", progress_cb=progress_cb)
        if not _launcher_wait_for_joint_settle(
            robot,
            progress_cb=progress_cb,
            max_wait_s=4.0,
            stable_for_s=1.0,
            tol_deg=0.05,
        ):
            return None

        tuned_threshold = None
        tuned_duration = None
        found = False

        for threshold_candidate in _build_alarm_threshold_values(int(DEFAULT_PID_THRESHOLD_MAX)):
            for duration_candidate in _build_alarm_duration_values(int(DEFAULT_PID_DURATION_MAX)):
                if _launcher_test_alarm_pid_candidate(
                    robot,
                    alarm_latch,
                    threshold_candidate,
                    duration_candidate,
                    progress_cb=progress_cb,
                    hold_s=0.75,
                ):
                    tuned_threshold = int(threshold_candidate)
                    tuned_duration = int(duration_candidate)
                    found = True
                    break
            if found:
                break

        if not found or tuned_threshold is None or tuned_duration is None:
            _launcher_tune_log(
                progress_cb,
                "⚠️ [HaltTune] No stable halt settings were found at Default pose within the allowed range.",
            )
            return None

        _launcher_apply_alarm_pid(
            robot,
            alarm_latch,
            tuned_threshold,
            tuned_duration,
            progress_cb=progress_cb,
        )
        _launcher_tune_log(
            progress_cb,
            "[HaltTune] Selected most sensitive stable pair at Default pose: "
            f"threshold={int(tuned_threshold)}, duration={int(tuned_duration)}",
        )
        return tuned_threshold, tuned_duration
    finally:
        try:
            robot.set_motor(0)
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        _launcher_tune_log(progress_cb, "[HaltTune] Launcher auto-tune worker disconnected from the robot.")

def show_startup_launcher(args):
    if not os.environ.get("DISPLAY") and sys.platform not in ("win32", "darwin"):
        print("[Launcher] DISPLAY is not set; starting without the launcher UI.")
        return args

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as e:
        print(f"[Launcher] Could not start Tk launcher: {e}")
        return args

    result = {"ok": False}
    root = tk.Tk()
    root.title("Dorna Joy Control Launcher")
    root.resizable(False, False)

    host_var = tk.StringVar(value=str(args.host or DEFAULT_DORNA_HOST))
    port_var = tk.StringVar(value=str(args.port or DEFAULT_DORNA_PORT))
    uvc1_var = tk.StringVar(value=str(args.uvc1 or ""))
    uvc2_var = tk.StringVar(value=str(args.uvc2 or ""))
    uvc_quality_var = tk.StringVar(
        value=_profile_label_for_values(
            args.uvc_width or DEFAULT_UVC_WIDTH,
            args.uvc_height or DEFAULT_UVC_HEIGHT,
            args.uvc_fps or DEFAULT_UVC_FPS,
            UVC_QUALITY_PRESETS,
        )
    )
    rs_quality_var = tk.StringVar(
        value=_profile_label_for_values(
            args.rs_width or DEFAULT_RS_WIDTH,
            args.rs_height or DEFAULT_RS_HEIGHT,
            args.rs_fps or DEFAULT_RS_FPS,
            RS_QUALITY_PRESETS,
        )
    )
    try_index1_var = tk.BooleanVar(value=bool(args.uvc_try_index1))
    fullscreen_var = tk.BooleanVar(value=bool(args.fullscreen))
    clear_alarm_var = tk.BooleanVar(value=bool(getattr(args, "clear_alarm_startup", True)))
    apply_halt_settings_var = tk.BooleanVar(value=bool(getattr(args, "apply_halt_settings_startup", True)))
    auto_tune_halt_var = tk.BooleanVar(value=False)
    alarm_threshold_var = tk.DoubleVar(
        value=float(getattr(args, "alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN))
    )
    alarm_duration_var = tk.DoubleVar(
        value=float(getattr(args, "alarm_duration", DEFAULT_PID_DURATION_MAIN))
    )
    halt_preset_var = tk.StringVar(
        value=_halt_preset_label_for_values(
            getattr(args, "alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN),
            getattr(args, "alarm_duration", DEFAULT_PID_DURATION_MAIN),
        )
    )
    launcher_var = tk.BooleanVar(value=bool(args.launcher))
    status_var = tk.StringVar(value="Detecting cameras...")
    alarm_status_var = tk.StringVar(value="")
    tune_status_var = tk.StringVar(value="Launcher auto-tune is idle.")
    option_paths = [""]
    tune_queue = queue.Queue()
    tune_state = {"running": False}

    frame = ttk.Frame(root, padding=14)
    frame.grid(row=0, column=0, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    ttk.Label(frame, text="Dorna Joy Control", font=("TkDefaultFont", 13, "bold")).grid(
        row=0, column=0, columnspan=4, sticky="w"
    )
    ttk.Label(
        frame,
        text="Pick startup options here instead of launching the robot from a long CLI command.",
        wraplength=640,
    ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 12))

    ttk.Label(frame, text="Robot host").grid(row=2, column=0, sticky="w")
    host_entry = ttk.Entry(frame, textvariable=host_var, width=26)
    host_entry.grid(row=2, column=1, sticky="we", padx=(8, 0))

    ttk.Label(frame, text="Port").grid(row=2, column=2, sticky="w", padx=(12, 0))
    port_entry = ttk.Entry(frame, textvariable=port_var, width=8)
    port_entry.grid(row=2, column=3, sticky="w", padx=(8, 0))

    ttk.Label(frame, text="UVC #1").grid(row=3, column=0, sticky="w", pady=(10, 0))
    uvc1_combo = ttk.Combobox(frame, textvariable=uvc1_var, width=80)
    uvc1_combo.grid(row=3, column=1, columnspan=3, sticky="we", padx=(8, 0), pady=(10, 0))

    ttk.Label(frame, text="UVC #2").grid(row=4, column=0, sticky="w", pady=(6, 0))
    uvc2_combo = ttk.Combobox(frame, textvariable=uvc2_var, width=80)
    uvc2_combo.grid(row=4, column=1, columnspan=3, sticky="we", padx=(8, 0), pady=(6, 0))

    ttk.Label(
        frame,
        text="Leave a UVC path blank to use auto-detect. The inventory below shows the UVC camera groups Linux currently exposes.",
        wraplength=640,
    ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))

    ttk.Label(frame, text="UVC quality").grid(row=6, column=0, sticky="w", pady=(10, 0))
    uvc_quality_combo = ttk.Combobox(
        frame,
        textvariable=uvc_quality_var,
        values=_profile_choice_values(UVC_QUALITY_PRESETS, uvc_quality_var.get()),
        width=28,
        state="readonly",
    )
    uvc_quality_combo.grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

    ttk.Label(frame, text="RealSense quality").grid(row=6, column=2, sticky="w", padx=(12, 0), pady=(10, 0))
    rs_quality_combo = ttk.Combobox(
        frame,
        textvariable=rs_quality_var,
        values=_profile_choice_values(RS_QUALITY_PRESETS, rs_quality_var.get()),
        width=28,
        state="readonly",
    )
    rs_quality_combo.grid(row=6, column=3, sticky="w", padx=(8, 0), pady=(10, 0))

    ttk.Checkbutton(frame, text="Try sibling video-index1 if index0 has no frames", variable=try_index1_var).grid(
        row=7, column=0, columnspan=4, sticky="w", pady=(8, 0)
    )
    ttk.Checkbutton(frame, text="Clear latched alarms on startup", variable=clear_alarm_var).grid(
        row=8, column=0, columnspan=4, sticky="w", pady=(4, 0)
    )
    ttk.Checkbutton(frame, text="Apply halt settings on startup", variable=apply_halt_settings_var).grid(
        row=9, column=0, columnspan=4, sticky="w", pady=(4, 0)
    )
    auto_tune_check = ttk.Checkbutton(
        frame,
        text="Auto-tune halt at Default pose before starting",
        variable=auto_tune_halt_var,
    )
    auto_tune_check.grid(
        row=10, column=0, columnspan=4, sticky="w", pady=(4, 0)
    )
    ttk.Label(frame, text="Halt threshold").grid(row=11, column=0, sticky="w", pady=(8, 0))
    threshold_scale = ttk.Scale(
        frame,
        from_=DEFAULT_PID_THRESHOLD_MIN,
        to=DEFAULT_PID_THRESHOLD_MAX,
        variable=alarm_threshold_var,
    )
    threshold_scale.grid(row=11, column=1, columnspan=3, sticky="we", padx=(8, 0), pady=(8, 0))
    ttk.Label(frame, text="Halt duration").grid(row=12, column=0, sticky="w", pady=(4, 0))
    duration_scale = ttk.Scale(
        frame,
        from_=DEFAULT_PID_DURATION_MIN,
        to=DEFAULT_PID_DURATION_MAX,
        variable=alarm_duration_var,
    )
    duration_scale.grid(row=12, column=1, columnspan=3, sticky="we", padx=(8, 0), pady=(4, 0))
    ttk.Label(frame, text="Halt preset").grid(row=13, column=0, sticky="w", pady=(4, 0))
    halt_preset_combo = ttk.Combobox(
        frame,
        textvariable=halt_preset_var,
        values=_halt_preset_choice_values(halt_preset_var.get()),
        width=28,
        state="readonly",
    )
    halt_preset_combo.grid(row=13, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=(4, 0))
    ttk.Label(frame, textvariable=alarm_status_var).grid(
        row=14, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=(2, 0)
    )
    ttk.Label(frame, text="Halt Auto-Tune Progress", font=("TkDefaultFont", 10, "bold")).grid(
        row=15, column=0, columnspan=4, sticky="w", pady=(10, 0)
    )
    tune_text = tk.Text(frame, width=92, height=8, wrap="word")
    tune_text.grid(row=16, column=0, columnspan=4, sticky="we", pady=(6, 4))
    tune_text.configure(state="disabled")
    ttk.Label(frame, textvariable=tune_status_var).grid(
        row=17, column=0, columnspan=4, sticky="w", pady=(0, 4)
    )
    ttk.Checkbutton(frame, text="Start fullscreen", variable=fullscreen_var).grid(
        row=18, column=0, columnspan=4, sticky="w", pady=(4, 0)
    )
    ttk.Checkbutton(frame, text="Show this launcher on startup", variable=launcher_var).grid(
        row=19, column=0, columnspan=4, sticky="w", pady=(4, 0)
    )

    ttk.Separator(frame).grid(row=20, column=0, columnspan=4, sticky="we", pady=10)
    ttk.Label(frame, text="Detected UVC inventory", font=("TkDefaultFont", 10, "bold")).grid(
        row=21, column=0, columnspan=4, sticky="w"
    )
    inventory_text = tk.Text(frame, width=92, height=8, wrap="word")
    inventory_text.grid(row=22, column=0, columnspan=4, sticky="we", pady=(6, 4))
    inventory_text.configure(state="disabled")
    ttk.Label(frame, textvariable=status_var, foreground="#b00020").grid(
        row=23, column=0, columnspan=4, sticky="w", pady=(0, 8)
    )

    button_bar = ttk.Frame(frame)
    button_bar.grid(row=24, column=0, columnspan=4, sticky="e", pady=(4, 0))

    def refresh_alarm_status(*_args):
        threshold, duration = _clamp_alarm_pid(
            alarm_threshold_var.get(),
            alarm_duration_var.get(),
        )
        preset_label = _halt_preset_label_for_values(threshold, duration)
        halt_preset_combo["values"] = _halt_preset_choice_values(preset_label)
        halt_preset_var.set(preset_label)
        alarm_status_var.set(
            f"default: threshold {int(DEFAULT_PID_THRESHOLD_MAIN)}, duration {int(DEFAULT_PID_DURATION_MAIN)}   "
            f"selected: threshold {int(threshold)}, duration {int(duration)}"
        )

    def apply_halt_preset(*_args):
        values = _halt_preset_value_map().get(str(halt_preset_var.get()).strip())
        if not values:
            return
        alarm_threshold_var.set(float(values[0]))
        alarm_duration_var.set(float(values[1]))
        refresh_alarm_status()

    def refresh_inventory():
        nonlocal option_paths
        inventory = discover_uvc_inventory(limit=12)
        option_paths = [""] + [item["path"] for item in inventory]
        uvc1_combo["values"] = option_paths
        uvc2_combo["values"] = option_paths
        if not uvc1_var.get():
            uvc1_combo.set("")
        if not uvc2_var.get():
            uvc2_combo.set("")
        lines = []
        if not inventory:
            lines.append("No non-RealSense UVC cameras were detected.")
        else:
            for idx, item in enumerate(inventory, start=1):
                lines.append(f"{idx}. {_format_uvc_inventory_label(item)}")
                aliases = item.get("aliases", [])
                extra_aliases = [a for a in aliases if a != item.get("path")]
                if extra_aliases:
                    lines.append("   aliases: " + ", ".join(extra_aliases))
        inventory_text.configure(state="normal")
        inventory_text.delete("1.0", "end")
        inventory_text.insert("1.0", "\n".join(lines))
        inventory_text.configure(state="disabled")
        if len(inventory) >= 2:
            status_var.set(f"{len(inventory)} UVC camera groups detected.")
        elif len(inventory) == 1:
            status_var.set("Only 1 UVC camera group was detected. You can still start with one camera.")
        else:
            status_var.set("No UVC camera groups were detected. Start only if you expect placeholders.")

    def append_tune_log(message: str, reset: bool = False):
        tune_text.configure(state="normal")
        if reset:
            tune_text.delete("1.0", "end")
        tune_text.insert("end", str(message).rstrip() + "\n")
        tune_text.see("end")
        tune_text.configure(state="disabled")

    def set_launcher_busy(is_busy: bool):
        tune_state["running"] = bool(is_busy)
        disabled = "disabled" if is_busy else None
        controls = [
            (host_entry, "normal"),
            (port_entry, "normal"),
            (uvc1_combo, "normal"),
            (uvc2_combo, "normal"),
            (uvc_quality_combo, "readonly"),
            (rs_quality_combo, "readonly"),
            (threshold_scale, "normal"),
            (duration_scale, "normal"),
            (halt_preset_combo, "readonly"),
            (refresh_button, "normal"),
            (cancel_button, "normal"),
            (autotune_button, "normal"),
            (start_button, "normal"),
            (auto_tune_check, "normal"),
        ]
        for widget, enabled_state in controls:
            try:
                widget.configure(state="disabled" if is_busy else enabled_state)
            except Exception:
                pass

    def collect_launch_settings():
        try:
            port = int(port_var.get().strip())
        except Exception:
            messagebox.showerror("Invalid port", "Port must be an integer.")
            return None
        try:
            uvc_width, uvc_height, uvc_fps = _profile_values_from_label(
                uvc_quality_var.get(), UVC_QUALITY_PRESETS,
                (DEFAULT_UVC_WIDTH, DEFAULT_UVC_HEIGHT, DEFAULT_UVC_FPS),
            )
            rs_width, rs_height, rs_fps = _profile_values_from_label(
                rs_quality_var.get(), RS_QUALITY_PRESETS,
                (DEFAULT_RS_WIDTH, DEFAULT_RS_HEIGHT, DEFAULT_RS_FPS),
            )
        except Exception:
            messagebox.showerror("Invalid quality", "Choose valid UVC and RealSense quality profiles.")
            return None

        host = host_var.get().strip() or DEFAULT_DORNA_HOST
        uvc1 = uvc1_var.get().strip()
        uvc2 = uvc2_var.get().strip()
        if uvc1 and uvc2 and os.path.realpath(uvc1) == os.path.realpath(uvc2):
            messagebox.showerror("Duplicate UVC selection", "UVC #1 and UVC #2 resolve to the same device.")
            return None

        alarm_threshold, alarm_duration = _clamp_alarm_pid(
            alarm_threshold_var.get(),
            alarm_duration_var.get(),
        )
        return {
            "host": host,
            "port": port,
            "uvc1": uvc1,
            "uvc2": uvc2,
            "uvc_width": uvc_width,
            "uvc_height": uvc_height,
            "uvc_fps": uvc_fps,
            "rs_width": rs_width,
            "rs_height": rs_height,
            "rs_fps": rs_fps,
            "uvc_try_index1": bool(try_index1_var.get()),
            "fullscreen": bool(fullscreen_var.get()),
            "clear_alarm_startup": bool(clear_alarm_var.get()),
            "apply_halt_settings_startup": bool(apply_halt_settings_var.get()),
            "auto_tune_halt_startup": bool(auto_tune_halt_var.get()),
            "alarm_threshold": alarm_threshold,
            "alarm_duration": alarm_duration,
            "launcher": bool(launcher_var.get()),
        }

    def finalize_start(config: dict):
        args.host = config["host"]
        args.port = config["port"]
        args.uvc1 = config["uvc1"]
        args.uvc2 = config["uvc2"]
        args.uvc_width = config["uvc_width"]
        args.uvc_height = config["uvc_height"]
        args.uvc_fps = config["uvc_fps"]
        args.rs_width = config["rs_width"]
        args.rs_height = config["rs_height"]
        args.rs_fps = config["rs_fps"]
        args.uvc_try_index1 = config["uvc_try_index1"]
        args.fullscreen = config["fullscreen"]
        args.clear_alarm_startup = config["clear_alarm_startup"]
        args.apply_halt_settings_startup = config["apply_halt_settings_startup"]
        args.auto_tune_halt_startup = config["auto_tune_halt_startup"]
        args.alarm_threshold = config["alarm_threshold"]
        args.alarm_duration = config["alarm_duration"]
        args.launcher = config["launcher"]
        result["ok"] = True
        root.destroy()

    def begin_auto_tune(start_after: bool):
        if tune_state["running"]:
            return
        config = collect_launch_settings()
        if config is None:
            return
        append_tune_log(
            "[Launcher] Starting auto-tune from the launcher window. The robot will move to Default before testing halt settings.",
            reset=True,
        )
        tune_status_var.set("Connecting to robot and starting launcher auto-tune…")
        set_launcher_busy(True)

        def progress_cb(kind, **payload):
            tune_queue.put({"kind": kind, **payload})

        def worker():
            try:
                tuned = run_launcher_halt_autotune(
                    config["host"],
                    config["port"],
                    config["alarm_threshold"],
                    config["alarm_duration"],
                    progress_cb=progress_cb,
                )
                tune_queue.put({
                    "kind": "done",
                    "ok": tuned is not None,
                    "tuned": tuned,
                    "config": config,
                    "start_after": bool(start_after),
                })
            except Exception as e:
                tune_queue.put({
                    "kind": "done",
                    "ok": False,
                    "error": str(e),
                    "config": config,
                    "start_after": bool(start_after),
                })

        threading.Thread(target=worker, daemon=True).start()

    def cancel():
        if tune_state["running"]:
            tune_status_var.set("Launcher auto-tune is still running. Wait for it to finish.")
            return
        root.destroy()

    def start():
        config = collect_launch_settings()
        if config is None:
            return
        if config["auto_tune_halt_startup"]:
            begin_auto_tune(start_after=True)
            return
        finalize_start(config)

    def run_auto_tune_now():
        begin_auto_tune(start_after=False)

    def poll_tune_queue():
        try:
            while True:
                item = tune_queue.get_nowait()
                kind = item.get("kind")
                if kind == "log":
                    message = str(item.get("message", "")).rstrip()
                    append_tune_log(message)
                    tune_status_var.set(message or "Launcher auto-tune is running…")
                elif kind == "values":
                    alarm_threshold_var.set(float(item.get("threshold", alarm_threshold_var.get())))
                    alarm_duration_var.set(float(item.get("duration", alarm_duration_var.get())))
                    refresh_alarm_status()
                elif kind == "done":
                    set_launcher_busy(False)
                    config = item.get("config") or {}
                    tuned = item.get("tuned")
                    if item.get("ok") and isinstance(tuned, (list, tuple)) and len(tuned) >= 2:
                        threshold, duration = _clamp_alarm_pid(tuned[0], tuned[1])
                        alarm_threshold_var.set(float(threshold))
                        alarm_duration_var.set(float(duration))
                        refresh_alarm_status()
                        tune_status_var.set(
                            f"Auto-tune complete. Selected threshold {int(threshold)}, duration {int(duration)}."
                        )
                        append_tune_log(
                            f"[Launcher] Auto-tune complete. Selected threshold={int(threshold)}, duration={int(duration)}."
                        )
                        if bool(item.get("start_after")):
                            config["alarm_threshold"] = threshold
                            config["alarm_duration"] = duration
                            config["auto_tune_halt_startup"] = False
                            finalize_start(config)
                            return
                        auto_tune_halt_var.set(False)
                    else:
                        err = str(item.get("error") or "Launcher auto-tune did not find a stable halt setting.")
                        tune_status_var.set(err)
                        append_tune_log(f"⚠️ {err}")
                        if bool(item.get("start_after")):
                            messagebox.showerror("Auto-tune failed", err)
        except queue.Empty:
            pass
        try:
            root.after(100, poll_tune_queue)
        except Exception:
            pass

    refresh_button = ttk.Button(button_bar, text="Refresh Cameras", command=refresh_inventory)
    refresh_button.grid(row=0, column=0, padx=(0, 8))
    cancel_button = ttk.Button(button_bar, text="Cancel", command=cancel)
    cancel_button.grid(row=0, column=1, padx=(0, 8))
    autotune_button = ttk.Button(button_bar, text="Auto-Tune Now", command=run_auto_tune_now)
    autotune_button.grid(row=0, column=2, padx=(0, 8))
    start_button = ttk.Button(button_bar, text="Start Control", command=start)
    start_button.grid(row=0, column=3)

    refresh_inventory()
    refresh_alarm_status()
    threshold_scale.configure(command=lambda _v: refresh_alarm_status())
    duration_scale.configure(command=lambda _v: refresh_alarm_status())
    halt_preset_combo.bind("<<ComboboxSelected>>", apply_halt_preset)
    root.after(100, poll_tune_queue)
    host_entry.focus_set()
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    if result["ok"]:
        return args
    return None

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
#   ADVANCE_CANCEL   # Skip __midway; go directly to final pose on next POSE
#   WASH N
POSE Reload
ADVANCE_WAIT
WASH 3
"""

ROUTINE_PATH = "routine.txt"
ROUTINES_PATH = "routines.json"
SETTINGS_PATH = "settings.json"

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

def load_routines(path=ROUTINES_PATH):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        print(f"⚠️ Could not load routines from {path}: {e}")

    # Fallback to legacy single routine.txt
    legacy = load_routine_text()
    return {"Default": legacy}

def save_routines(routines: dict, path=ROUTINES_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(routines, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save routines to {path}: {e}")

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
        min_settle = 0.0
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
            if re.match(r'^ADVANCE_CANCEL$', line, re.I):
                steps.append(("advance_cancel", None))
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
        skip_midway_next = False

        for typ, arg in steps:
            if self.stop_event.is_set():
                break

            if typ == "echo":
                self._set_status_line(arg)
                continue

            if typ == "pose":
                name = arg
                last_pose = name
                self._set_status_line(
                    f"POSE {name}" + (" (direct, skip __midway)" if skip_midway_next else "")
                )
                if skip_midway_next:
                    self.rt.enqueue({"type": "goto_direct", "pose_name": name})
                else:
                    self.rt.enqueue({"type": "goto", "pose_name": name})
                skip_midway_next = False
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

            elif typ == "advance_cancel":
                skip_midway_next = True
                self._set_status_line("ADVANCE_CANCEL (next POSE will skip __midway)")

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
    p.add_argument("--host", "-H", default=None, help="IP address of your Dorna controller")
    p.add_argument("--port", "-P", type=int, default=None, help="Port for Dorna control (usually 443)")

    p.add_argument("--uvc1", default=None, help="UVC camera #1 device path (default: launcher/settings/auto-detect)")
    p.add_argument("--uvc2", default=None, help="UVC camera #2 device path (default: launcher/settings/auto-detect)")
    p.add_argument("--uvc-width", type=int, default=None, help="Requested UVC width")
    p.add_argument("--uvc-height", type=int, default=None, help="Requested UVC height")
    p.add_argument("--uvc-fps", type=int, default=None, help="FPS for UVCs")
    p.add_argument("--rs-width", type=int, default=None, help="Requested RealSense width")
    p.add_argument("--rs-height", type=int, default=None, help="Requested RealSense height")
    p.add_argument("--rs-fps", type=int, default=None, help="Requested RealSense FPS")
    p.add_argument("--uvc-rotate", action="store_true", default=True, help="(legacy) rotate UVC cameras 180° (now overridden by GUI settings)")
    p.add_argument("--uvc-no-rotate", dest="uvc_rotate", action="store_false", help="Disable legacy 180° rotation (GUI rotation still applies)")
    p.add_argument("--uvc-try-index1", dest="uvc_try_index1", action="store_true", help="Also try the sibling video-index1 node if index0 yields no frames")
    p.add_argument("--no-uvc-try-index1", dest="uvc_try_index1", action="store_false", help="Do not try sibling video-index1 nodes")
    p.set_defaults(uvc_try_index1=None)
    p.add_argument("--clear-alarm-startup", dest="clear_alarm_startup", action="store_true", help="Clear any latched controller alarm during startup")
    p.add_argument("--no-clear-alarm-startup", dest="clear_alarm_startup", action="store_false", help="Do not clear latched controller alarms during startup")
    p.set_defaults(clear_alarm_startup=None)
    p.add_argument("--apply-halt-settings-startup", dest="apply_halt_settings_startup", action="store_true", help="Apply halt threshold/duration automatically after startup settles")
    p.add_argument("--no-apply-halt-settings-startup", dest="apply_halt_settings_startup", action="store_false", help="Skip applying halt threshold/duration on startup")
    p.set_defaults(apply_halt_settings_startup=None)
    p.add_argument("--auto-tune-halt-startup", dest="auto_tune_halt_startup", action="store_true", help="Auto-tune halt threshold/duration at Default pose during startup")
    p.add_argument("--no-auto-tune-halt-startup", dest="auto_tune_halt_startup", action="store_false", help="Skip halt auto-tuning during startup")
    p.set_defaults(auto_tune_halt_startup=None)
    p.add_argument("--alarm-threshold", type=float, default=None, help="Robot halt threshold")
    p.add_argument("--alarm-duration", type=float, default=None, help="Robot halt duration")

    p.add_argument("--fullscreen", dest="fullscreen", action="store_true", help="Start in fullscreen (toggle with F11)")
    p.add_argument("--windowed", dest="fullscreen", action="store_false", help="Force windowed startup")
    p.add_argument("--launcher", dest="launcher", action="store_true", help="Show the startup launcher window")
    p.add_argument("--no-launcher", dest="launcher", action="store_false", help="Skip the startup launcher window")
    p.set_defaults(fullscreen=None, launcher=None)
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

def orthonormalize_R(R):
    """Project a near-rotation matrix back onto SO(3) to prevent drift."""
    try:
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn
    except Exception:
        return R


def extend_line_to_edge(shape, line, edge):
    """Extend a line segment to a given image edge ('right' or 'bottom')."""
    h, w = shape[:2]
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    if edge == "right" and dx != 0:
        t = (w - 1 - x1) / dx
        y_at_edge = int(round(y1 + t * dy))
        return (x1, y1, w - 1, np.clip(y_at_edge, 0, h - 1))
    if edge == "bottom" and dy != 0:
        t = (h - 1 - y1) / dy
        x_at_edge = int(round(x1 + t * dx))
        return (x1, y1, np.clip(x_at_edge, 0, w - 1), h - 1)
    return (x1, y1, x2, y2)


def compute_correction(shift_left, shift_right, step_mm=0.5, thr=0.05):
    """
    Derive ΔX/ΔY based on observed shifts at +45°:
    - left>0 = needle left; left<0 = needle right
    - right>0 = needle down; right<0 = needle up
    Rules (actual center offset):
      +X: left right, right down -> move tool -X
      -X: left left,  right up   -> move tool +X
      +Y: left left,  right down -> move tool -Y
      -Y: left right, right up   -> move tool +Y
    """
    dx = dy = 0.0
    reason = "no clear shift"
    if (shift_right > thr) and (shift_left < -thr):
        dx = -step_mm
        reason = "+X -> apply ΔX=-step"
    elif (shift_right < -thr) and (shift_left > thr):
        dx = step_mm
        reason = "-X -> apply ΔX=+step"
    elif (shift_right > thr) and (shift_left > thr):
        dy = -step_mm
        reason = "+Y -> apply ΔY=-step"
    elif (shift_right < -thr) and (shift_left < -thr):
        dy = step_mm
        reason = "-Y -> apply ΔY=+step"
    return dx, dy, reason


def median_or_default(arr, default=0.0):
    if not arr:
        return default
    s = sorted(arr)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def draw_dev_plot(frame, history):
    """Overlay a tiny plot of recent ΔX/ΔY decisions on the frame."""
    if frame is None or not history:
        return
    h, w = frame.shape[:2]
    plot_w = min(200, w // 3)
    plot_h = min(100, h // 4)
    x0, y0 = 10, 10
    # Normalize history
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    max_abs = max(1e-3, max(abs(max(xs, default=0)), abs(min(xs, default=0)), abs(max(ys, default=0)), abs(min(ys, default=0))))
    pts_x = []
    pts_y = []
    n = len(history)
    for i, (dx, dy) in enumerate(history[-plot_w:]):
        px = int(x0 + i)
        py_x = int(y0 + plot_h / 2 - (dx / max_abs) * (plot_h / 2))
        py_y = int(y0 + plot_h / 2 - (dy / max_abs) * (plot_h / 2))
        pts_x.append((px, py_x))
        pts_y.append((px, py_y))
    cv2.rectangle(frame, (x0, y0), (x0 + plot_w, y0 + plot_h), (40, 40, 40), 1)
    if len(pts_x) >= 2:
        cv2.polylines(frame, [np.int32(pts_x)], False, (0, 255, 0), 1)
    if len(pts_y) >= 2:
        cv2.polylines(frame, [np.int32(pts_y)], False, (255, 0, 255), 1)
    cv2.putText(frame, "ΔX(green)/ΔY(mag)", (x0, y0 + plot_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        # Joystick state
        self.lx = self.ly = self.rx = self.ry = 0.0
        self.hx = self.hy = 0
        self.lb = self.rb = False

        # Manual speed scale levels (LB/RB to change)
        self.levels = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
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
        self.tool_cx     = float(self.settings.get("tool_cx", DEFAULT_TOOL_CX))
        self.tool_cy     = float(self.settings.get("tool_cy", DEFAULT_TOOL_CY))
        self.tool_center_demo_radius_mm = float(
            self.settings.get("tool_center_demo_radius_mm", DEFAULT_TOOL_CENTER_RADIUS)
        )
        self.alarm_threshold = float(self.settings.get("alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN))
        self.alarm_duration = float(self.settings.get("alarm_duration", DEFAULT_PID_DURATION_MAIN))
        self.alarm_armed = False
        self.tool_center_demo_mode = "circle"
        self.approach_mm = float(self.settings.get("approach_mm", DEFAULT_APPROACH_MM))
        self.tool_center_demo = False
        self.tool_center_demo_waiting = False

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

        # Tool rotation test UI state
        self.j5_test_waiting = False
        self.j5_last_shift_x = 0.0  # left cam shift (X)
        self.j5_last_shift_y = 0.0  # right cam shift (Y)
        self.j5_wait_med_left = 0.0
        self.j5_wait_med_right = 0.0
        self.j5_last_nonzero_left = 0.0   # left cam last non-zero shift
        self.j5_last_nonzero_right = 0.0  # right cam last non-zero shift
        self.j5_motion_med_left = 0.0     # median during motion (left cam)
        self.j5_motion_med_right = 0.0    # median during motion (right cam)
        self.j5_reset_history = False
        self.j5_max_shift_left = 0.0      # max |shift| during motion (left cam)
        self.j5_max_shift_right = 0.0     # max |shift| during motion (right cam)
        self.j5_start_shift_left = None   # first shift seen this sweep (left)
        self.j5_start_shift_right = None  # first shift seen this sweep (right)
        self.j5_end_shift_left = None     # last shift seen this sweep (left)
        self.j5_end_shift_right = None    # last shift seen this sweep (right)
        self.j5_step_sizes = [2.0, 1.0, 0.5, 0.1]
        self.j5_step_idx = 0
        self.j5_last_dev = None  # track last avg deviation for oscillation detection
        self.j5_dev_history = []  # list of (dx, dy) decisions
        self.j5_start_shift_left = None
        self.j5_start_shift_right = None
        self.j5_end_shift_left = None
        self.j5_end_shift_right = None
        self.j5_angle_mode = "actual"  # 'actual' or 'base'
        self.j5_base_ref = None
        self.j5_angle_set = 0.0
        self.yaw = 0.0
        self.yaw_base = None
        self.last_pose_refresh = time.time()


class RobotThread(threading.Thread):
    def __init__(
        self,
        state: SharedState,
        host: str,
        port: int,
        clear_alarm_on_launch: bool = True,
        apply_halt_settings_startup: bool = True,
        auto_tune_halt_startup: bool = False,
    ):
        super().__init__(daemon=True)
        self.state      = state
        self.host       = host
        self.port       = port
        self.clear_alarm_on_launch = bool(clear_alarm_on_launch)
        self.apply_halt_settings_startup = bool(apply_halt_settings_startup)
        self.auto_tune_halt_startup = bool(auto_tune_halt_startup)
        self.stop_event = threading.Event()
        self.cmd_q      = queue.Queue()
        self.robot      = None
        self.keep_motors_on_exit = False
        self.tool_center_demo_thread = None
        self.tool_center_demo_stop = threading.Event()
        self.tool_center_demo_resume = threading.Event()
        self.tool_center_demo_base_pose = None
        self.tool_center_demo_mode = "circle"
        self.j5_test_thread = None
        self.j5_test_stop = threading.Event()
        self.j5_test_resume_event = threading.Event()
        self.last_pose_refresh = time.time()
        self.alarm_state_lock = threading.Lock()
        self.alarm_latched = False
        self.last_alarm_msg = None

    def _on_robot_event(self, msg, union, **kwargs):
        if not isinstance(msg, dict):
            return
        if msg.get("cmd") != "alarm":
            return
        alarm_val = msg.get("alarm")
        with self.alarm_state_lock:
            if alarm_val in (1, 1.0, True):
                self.alarm_latched = True
                self.last_alarm_msg = dict(msg)
            elif alarm_val in (0, 0.0, False):
                self.alarm_latched = False
                self.last_alarm_msg = dict(msg)

    def _update_tcp_from_settings(self):
        """Sync kinematic TCP with current tool center/length settings."""
        try:
            cx = float(self.state.settings.get("tool_cx", DEFAULT_TOOL_CX))
            cy = float(self.state.settings.get("tool_cy", DEFAULT_TOOL_CY))
            lz = float(self.state.settings.get("tool_lz", DEFAULT_TOOL_LZ))
            # tcp xyzabc (mm, mm, mm, rX, rY, rZ) — here only offset, no rotation
            self.robot.kinematic.set_tcp_xyzabc([cx, cy, lz, 0.0, 0.0, 0.0])
        except Exception:
            pass

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
        self.left_stick_deadzone = 0.14
        self.left_stick_filter_alpha = 0.35
        self.left_stick_cmd_threshold = 0.01
        self.left_stick_release_snap = 0.03
        self.left_stick_axis_margin = 0.08
        self.left_stick_lx_cmd = 0.0
        self.left_stick_ly_cmd = 0.0
        self.left_stick_mode = None

        # Sawing motion (J5 oscillation) state
        self.saw_prev_offset = 0.0
        self.saw_phase = 0.0
        self.saw_last_t = time.time()
        self.live_motion_active = False
        self.live_lmove_dirty = False
        self.live_abs_pose_dirty = False
        self.live_rel_xyz_pending = np.zeros(3, dtype=float)
        self.live_rel_abc_pending = np.zeros(3, dtype=float)
        self.live_j5_pending = 0.0
        self.live_send_interval = 1.0 / 40.0
        self.live_send_interval_tool_axis = 1.0 / 80.0
        self.live_next_send_t = 0.0
        self.live_last_send_t = 0.0
        self.live_last_abs_pose = None
        self.live_linear_epsilon = 0.01
        self.live_angular_epsilon = 0.01
        self.live_j5_epsilon = 0.01
        self.live_halt_accel = 8.0
        self.live_halt_accel_translation = 12.0
        self.live_last_motion_mode = None
        self.last_collision_zone = ""
        self.last_collision_t = 0.0
        self.orient_deadzone = 0.18
        self.last_j4_poll = None
        self.last_j4_poll_t = None
        self.j4_jump_guard_deg = 1.5

    def _play_live(self, cmd: dict):
        """
        Dispatch a live/manual motion command without waiting for completion.
        Blocking on every incremental joystick move adds visible control lag.
        """
        return self.robot.play_dict(cmd, timeout=0)

    def _halt_live_motion(self):
        accel = self.live_halt_accel
        if self.live_last_motion_mode == "tool_axis_translation":
            accel = max(accel, self.live_halt_accel_translation)
        try:
            self.robot.halt(accel=accel, timeout=0)
        except Exception:
            pass

    def _apply_alarm_pid(self, threshold: float, duration: float, persist: bool = True):
        threshold, duration = _clamp_alarm_pid(threshold, duration)
        with self.alarm_state_lock:
            self.alarm_latched = False
            self.last_alarm_msg = None
        try:
            self.robot.set_alarm(0)
            time.sleep(0.05)
        except Exception:
            pass
        for axis in range(6):
            self.robot.set_pid(index=axis, threshold=threshold, duration=duration)
        with self.state.lock:
            self.state.settings["alarm_threshold"] = threshold
            self.state.settings["alarm_duration"] = duration
            self.state.alarm_threshold = threshold
            self.state.alarm_duration = duration
            self.state.alarm_armed = True
        if persist:
            save_settings(self.state.settings)
            save_startup_halt_settings(threshold, duration)
        print(
            f"[Halt] Settings applied "
            f"(threshold={int(threshold)}, duration={int(duration)})"
        )

    def _prepare_relaxed_alarm_pid(self):
        with self.alarm_state_lock:
            self.alarm_latched = False
            self.last_alarm_msg = None
        try:
            self.robot.set_alarm(0)
            time.sleep(0.05)
        except Exception:
            pass
        for axis in range(6):
            self.robot.set_pid(
                index=axis,
                threshold=DEFAULT_PID_THRESHOLD_MAX,
                duration=DEFAULT_PID_DURATION_MAX,
            )

    def _clear_alarm_latch(self, context: str = "startup"):
        try:
            stat = self.robot.set_alarm(0)
            time.sleep(0.05)
            with self.alarm_state_lock:
                self.alarm_latched = False
                self.last_alarm_msg = None
            with self.state.lock:
                self.state.alarm_armed = False
            print(f"[Startup] Cleared controller alarm ({context}); stat={stat}")
            return True
        except Exception as e:
            print(f"⚠️ Could not clear controller alarm during {context}: {e}")
            return False

    def _wait_for_joint_settle(self, max_wait_s: float = 4.0, stable_for_s: float = 1.0, tol_deg: float = 0.05):
        start = time.time()
        stable_since = None
        prev = None
        while time.time() - start < max_wait_s:
            joints = self._try_get_current_joints()
            if joints is None:
                time.sleep(0.05)
                continue
            cur = np.array([joints[f"j{i}"] for i in range(6)], dtype=float)
            if prev is None:
                prev = cur
                stable_since = time.time()
                time.sleep(0.05)
                continue
            max_delta = float(np.max(np.abs(cur - prev)))
            prev = cur
            if max_delta <= tol_deg:
                if stable_since is None:
                    stable_since = time.time()
                if (time.time() - stable_since) >= stable_for_s:
                    print(f"[Startup] Joints settled (max Δ={max_delta:.3f} deg).")
                    return True
            else:
                stable_since = None
            time.sleep(0.05)
        print(f"[Startup] Joint settle timeout after {max_wait_s:.1f}s; leaving startup alarm disabled.")
        return False

    def _apply_alarm_sensitivity(self, val: float, persist: bool = True):
        _, threshold, duration = _alarm_pid_from_sensitivity(val)
        self._apply_alarm_pid(threshold, duration, persist=persist)

    def _is_alarm_latched(self):
        with self.alarm_state_lock:
            return bool(self.alarm_latched)

    def _hold_without_alarm(self, hold_s: float = 0.75, poll_s: float = 0.05):
        end_t = time.time() + max(0.0, float(hold_s))
        while time.time() < end_t:
            if self._is_alarm_latched():
                return False
            time.sleep(max(0.01, float(poll_s)))
        return not self._is_alarm_latched()

    def _test_alarm_pid_candidate(self, threshold: float, duration: float, hold_s: float = HALT_TUNE_HOLD_S):
        threshold, duration = _clamp_alarm_pid(threshold, duration)
        print(f"[HaltTune] Testing threshold={int(threshold)}, duration={int(duration)}")
        try:
            self._prepare_relaxed_alarm_pid()
            self._clear_alarm_latch("before candidate setup")
            self.robot.set_motor(1)
            with self.state.lock:
                default_pose = self.state.poses.get("Default", {}).copy()
            if default_pose:
                print("[HaltTune] Returning to Default pose before candidate test.")
                if not self._queue_jmove_to_pose(default_pose):
                    self._clear_alarm_latch("after failed candidate setup")
                    return False
                self._set_current_named("Default")
            if not self._wait_for_joint_settle(max_wait_s=4.0, stable_for_s=1.0, tol_deg=0.05):
                self._clear_alarm_latch("after failed candidate setup")
                return False
        except Exception as e:
            print(f"⚠️ [HaltTune] Could not prepare candidate test: {e}")
            self._clear_alarm_latch("after failed candidate setup")
            return False
        self._apply_alarm_pid(threshold, duration, persist=False)
        settled = self._wait_for_joint_settle(max_wait_s=2.0, stable_for_s=0.35, tol_deg=0.05)
        stable = bool(settled) and self._hold_without_alarm(hold_s=hold_s, poll_s=0.05)
        if stable:
            for dist_mm, label in ((HALT_TUNE_MOVE_MM, "forward"), (-HALT_TUNE_MOVE_MM, "backward")):
                try:
                    print(f"[HaltTune] Movement test: tool-axis {label} {abs(int(dist_mm))} mm.")
                    if not self._tool_move_along_tz(dist_mm, cont=0):
                        stable = False
                        break
                    settled = self._wait_for_joint_settle(
                        max_wait_s=max(4.0, abs(dist_mm) / max(1e-6, self.VT) + 2.0),
                        stable_for_s=0.35,
                        tol_deg=0.05,
                    )
                    stable = bool(settled) and self._hold_without_alarm(hold_s=0.35, poll_s=0.05)
                except Exception as e:
                    print(f"⚠️ [HaltTune] Movement test failed during {label} move: {e}")
                    stable = False
                if not stable:
                    break
        if not stable:
            with self.alarm_state_lock:
                alarm_msg = dict(self.last_alarm_msg) if isinstance(self.last_alarm_msg, dict) else None
            if alarm_msg:
                print(f"[HaltTune] Alarm during test: {alarm_msg}")
            self._clear_alarm_latch(f"after testing {int(threshold)}/{int(duration)}")
        return stable

    def _auto_tune_alarm_pid(self, threshold: float, duration: float, persist: bool = True):
        requested_threshold, requested_duration = _clamp_alarm_pid(threshold, duration)
        print(
            "[HaltTune] Auto-tuning at Default pose. "
            f"Requested starting point threshold={int(requested_threshold)}, duration={int(requested_duration)}."
        )
        print(
            "[HaltTune] Starting from the most sensitive pair and increasing slowly until no alarm is met "
            f"(up to threshold={int(DEFAULT_PID_THRESHOLD_MAX)}, duration={int(DEFAULT_PID_DURATION_MAX)})."
        )

        try:
            with self.state.lock:
                default_pose = self.state.poses.get("Default", {}).copy()
            if default_pose:
                self._prepare_relaxed_alarm_pid()
                self._clear_alarm_latch("before auto-tune setup")
                self.robot.set_motor(1)
                if self.current_named != "Default":
                    if self._queue_jmove_to_pose(default_pose):
                        self._set_current_named("Default")
                self._wait_for_joint_settle(max_wait_s=4.0, stable_for_s=1.0, tol_deg=0.05)
        except Exception as e:
            print(f"⚠️ [HaltTune] Could not confirm Default pose before tuning: {e}")

        def build_threshold_values(limit: int):
            vals = [int(DEFAULT_PID_THRESHOLD_MIN)]
            cur = vals[0]
            limit = int(limit)
            while cur < limit:
                if cur < 10:
                    step = 1
                elif cur < 40:
                    step = 2
                elif cur < 100:
                    step = 5
                elif cur < 200:
                    step = 10
                else:
                    step = 20
                cur = min(limit, cur + step)
                if cur != vals[-1]:
                    vals.append(cur)
            return vals

        def build_duration_values(limit: int):
            vals = [int(DEFAULT_PID_DURATION_MIN)]
            cur = vals[0]
            limit = int(limit)
            while cur < limit:
                if cur < 10:
                    step = 1
                elif cur < 50:
                    step = 5
                elif cur < 100:
                    step = 10
                elif cur < 250:
                    step = 25
                elif cur < 500:
                    step = 50
                elif cur < 1000:
                    step = 100
                elif cur < 2500:
                    step = 250
                elif cur < 5000:
                    step = 500
                else:
                    step = 1000
                cur = min(limit, cur + step)
                if cur != vals[-1]:
                    vals.append(cur)
            return vals

        tuned_threshold = None
        tuned_duration = None
        found = False

        for threshold_candidate in build_threshold_values(int(DEFAULT_PID_THRESHOLD_MAX)):
            for duration_candidate in build_duration_values(int(DEFAULT_PID_DURATION_MAX)):
                if self._test_alarm_pid_candidate(threshold_candidate, duration_candidate, hold_s=0.75):
                    tuned_threshold = int(threshold_candidate)
                    tuned_duration = int(duration_candidate)
                    found = True
                    break
            if found:
                break

        if not found or tuned_threshold is None or tuned_duration is None:
            print("⚠️ [HaltTune] No stable halt settings were found at Default pose within the allowed range.")
            return None

        self._apply_alarm_pid(tuned_threshold, tuned_duration, persist=persist)
        print(
            "[HaltTune] Selected most sensitive stable pair at Default pose: "
            f"threshold={int(tuned_threshold)}, duration={int(tuned_duration)}"
        )
        return tuned_threshold, tuned_duration

    def _collision_print(self, zone_name: str, context: str):
        zone_name = str(zone_name or "unnamed")
        now = time.time()
        if zone_name == self.last_collision_zone and (now - self.last_collision_t) < 0.5:
            return
        self.last_collision_zone = zone_name
        self.last_collision_t = now
        print(f"[Collision] Blocked {context}: {zone_name}")

    def _self_collision_cfg(self):
        with self.state.lock:
            cfg = dict(self.state.settings.get("self_collision", {}))
        if not cfg:
            cfg = dict(DEFAULT_SELF_COLLISION)
        if "link_radii_mm" not in cfg:
            cfg["link_radii_mm"] = list(DEFAULT_SELF_COLLISION["link_radii_mm"])
        else:
            cfg["link_radii_mm"] = _normalize_float_list(
                cfg.get("link_radii_mm"),
                6,
                DEFAULT_SELF_COLLISION["link_radii_mm"],
            )
        cfg["base_segments"] = _normalize_int_list(
            cfg.get("base_segments"),
            DEFAULT_SELF_COLLISION["base_segments"],
        )
        return cfg

    @staticmethod
    def _segment_distance(p0, p1, q0, q1):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        u = p1 - p0
        v = q1 - q0
        w = p0 - q0
        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w))
        e = float(np.dot(v, w))
        D = a * c - b * b
        small = 1e-9
        sN = 0.0
        sD = D
        tN = 0.0
        tD = D

        if D < small:
            sN = 0.0
            sD = 1.0
            tN = e
            tD = c
        else:
            sN = b * e - c * d
            tN = a * e - b * d
            if sN < 0.0:
                sN = 0.0
                tN = e
                tD = c
            elif sN > sD:
                sN = sD
                tN = e + b
                tD = c

        if tN < 0.0:
            tN = 0.0
            if -d < 0.0:
                sN = 0.0
            elif -d > a:
                sN = sD
            else:
                sN = -d
                sD = a
        elif tN > tD:
            tN = tD
            if (-d + b) < 0.0:
                sN = 0.0
            elif (-d + b) > a:
                sN = sD
            else:
                sN = -d + b
                sD = a

        sc = 0.0 if abs(sN) < small else sN / sD
        tc = 0.0 if abs(tN) < small else tN / tD
        dp = w + sc * u - tc * v
        return float(np.linalg.norm(dp))

    @staticmethod
    def _segment_hits_base_cylinder(p0, p1, radius_mm: float, height_mm: float):
        if radius_mm <= 0 or height_mm <= 0:
            return False
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        for t in np.linspace(0.0, 1.0, 11):
            pt = p0 + (p1 - p0) * float(t)
            if pt[2] <= height_mm and (pt[0] * pt[0] + pt[1] * pt[1]) <= (radius_mm * radius_mm):
                return True
        return False

    def _dict_in_ranges(self, values: dict, ranges: dict) -> bool:
        for axis, bounds in (ranges or {}).items():
            if axis not in values or values[axis] is None:
                return False
            lo, hi = bounds
            val = float(values[axis])
            if val < lo or val > hi:
                return False
        return True

    def _joint_dict_to_list(self, joints: dict):
        if not isinstance(joints, dict):
            return None
        try:
            return [float(joints[f"j{i}"]) for i in range(6)]
        except Exception:
            return None

    def _tcp_pose_from_joint_dict(self, joints: dict):
        joint_list = self._joint_dict_to_list(joints)
        if joint_list is None:
            return None
        try:
            T_flange = np.array(self.robot.kinematic.t_flange_r_world(joint=joint_list))
            T_tcp = T_flange @ self.robot.kinematic.T_tcp_r_flange
            xyzabc = self.robot.kinematic.mat_to_xyzabc(T_tcp)
            return tuple(float(v) for v in xyzabc[:6])
        except Exception:
            return None

    def _solve_joints_for_tcp_pose(self, tcp_pose):
        if not tcp_pose:
            return None
        current = self._try_get_current_joints()
        current_list = self._joint_dict_to_list(current) or [0.0] * 6
        try:
            sols = self.robot.kinematic.inv(list(tcp_pose), joint_current=current_list, all_sol=True)
        except Exception:
            return None
        if sols is None:
            return None
        if isinstance(sols, np.ndarray):
            if sols.ndim == 1:
                sols_iter = [sols.tolist()]
            else:
                sols_iter = sols.tolist()
        else:
            sols_iter = list(sols)
        best = None
        best_cost = None
        for sol in sols_iter:
            try:
                sol_list = [float(v) for v in list(sol)[:6]]
            except Exception:
                continue
            if len(sol_list) < 6:
                continue
            cost = float(np.linalg.norm(np.array(sol_list) - np.array(current_list)))
            if best is None or cost < best_cost:
                best = sol_list
                best_cost = cost
        if best is None:
            return None
        return {f"j{i}": best[i] for i in range(6)}

    def _merge_joint_target(self, joint_update: dict):
        base = self._try_get_current_joints()
        if base is None:
            return None
        merged = dict(base)
        for axis in COLLISION_JOINT_AXES:
            if axis in joint_update:
                try:
                    merged[axis] = float(joint_update[axis])
                except Exception:
                    pass
        return merged

    def _link_points_from_joint_list(self, joint_list):
        if joint_list is None or len(joint_list) < 6:
            return None
        points = []
        try:
            for i in range(0, 7):
                T = np.array(self.robot.kinematic.Ti_r_world(joint=joint_list, i=i), dtype=float)
                points.append(T[:3, 3].astype(float))
        except Exception:
            return None
        return points

    def _joint_path_samples(self, start_joint, target_joint, step_deg: float):
        if start_joint is None or target_joint is None:
            return []
        start = np.asarray(start_joint, dtype=float)
        target = np.asarray(target_joint, dtype=float)
        if start.shape[0] < 6 or target.shape[0] < 6:
            return []
        max_delta = float(np.max(np.abs(target[:6] - start[:6])))
        if max_delta <= 1e-6:
            return [target[:6].tolist()]
        samples = int(np.ceil(max_delta / max(step_deg, 0.5)))
        samples = max(1, min(samples, 30))
        return [
            (start + (target - start) * (idx / samples))[:6].tolist()
            for idx in range(1, samples + 1)
        ]

    def _find_self_collision_for_joint_list(self, joint_list, cfg):
        points = self._link_points_from_joint_list(joint_list)
        if not points or len(points) < 7:
            return None

        radii = list(cfg.get("link_radii_mm", DEFAULT_SELF_COLLISION["link_radii_mm"]))
        margin = float(cfg.get("pair_margin_mm", DEFAULT_SELF_COLLISION["pair_margin_mm"]))
        min_gap = max(2, int(cfg.get("min_link_gap", DEFAULT_SELF_COLLISION["min_link_gap"])))

        for i in range(6):
            for j in range(i + min_gap, 6):
                if j >= len(radii) or i >= len(radii):
                    continue
                dist = self._segment_distance(points[i], points[i + 1], points[j], points[j + 1])
                limit = float(radii[i]) + float(radii[j]) + margin
                if dist < limit:
                    return f"self-collision link{i}-link{j}"

        base_radius = float(cfg.get("base_radius_mm", DEFAULT_SELF_COLLISION["base_radius_mm"]))
        base_height = float(cfg.get("base_height_mm", DEFAULT_SELF_COLLISION["base_height_mm"]))
        for seg_idx in cfg.get("base_segments", DEFAULT_SELF_COLLISION["base_segments"]):
            if not isinstance(seg_idx, int) or seg_idx < 0 or seg_idx >= 6:
                continue
            if self._segment_hits_base_cylinder(points[seg_idx], points[seg_idx + 1], base_radius, base_height):
                return f"self-collision base-link{seg_idx}"

        return None

    def _find_self_collision(self, tcp_pose=None, joints=None, sweep=False):
        cfg = self._self_collision_cfg()
        if not cfg.get("enabled", True):
            return None

        target_joint = None
        if isinstance(joints, dict):
            target_joint = self._joint_dict_to_list(joints)
        elif isinstance(joints, (list, tuple, np.ndarray)):
            try:
                target_joint = [float(v) for v in list(joints)[:6]]
            except Exception:
                target_joint = None
        elif tcp_pose is not None:
            solved = self._solve_joints_for_tcp_pose(tcp_pose)
            target_joint = self._joint_dict_to_list(solved)

        if target_joint is None:
            return None

        if sweep:
            current = self._joint_dict_to_list(self._try_get_current_joints())
            if current is not None:
                samples = self._joint_path_samples(
                    current,
                    target_joint,
                    float(cfg.get("path_step_deg", DEFAULT_SELF_COLLISION["path_step_deg"])),
                )
                for sample in samples:
                    hit = self._find_self_collision_for_joint_list(sample, cfg)
                    if hit:
                        return hit

        return self._find_self_collision_for_joint_list(target_joint, cfg)

    def _find_collision_zone(self, tcp_pose=None, joints=None):
        with self.state.lock:
            zones = list(self.state.settings.get("collision_zones", []))
        zones = [z for z in zones if isinstance(z, dict) and z.get("enabled", True)]
        if not zones:
            return None

        need_tcp = any(z.get("tcp_ranges") for z in zones)
        need_joints = any(z.get("joint_ranges") for z in zones)
        resolved_tcp = tuple(float(v) for v in tcp_pose[:6]) if tcp_pose is not None else None
        resolved_joints = dict(joints) if isinstance(joints, dict) else None

        if need_tcp and resolved_tcp is None and resolved_joints is not None:
            resolved_tcp = self._tcp_pose_from_joint_dict(resolved_joints)
        if need_joints and resolved_joints is None and resolved_tcp is not None:
            resolved_joints = self._solve_joints_for_tcp_pose(resolved_tcp)

        tcp_values = None
        if resolved_tcp is not None:
            tcp_values = {axis: resolved_tcp[i] for i, axis in enumerate(COLLISION_TCP_AXES)}

        for zone in zones:
            joint_ranges = zone.get("joint_ranges", {})
            tcp_ranges = zone.get("tcp_ranges", {})
            if tcp_ranges:
                if tcp_values is None or not self._dict_in_ranges(tcp_values, tcp_ranges):
                    continue
            if joint_ranges:
                if resolved_joints is None or not self._dict_in_ranges(resolved_joints, joint_ranges):
                    continue
            return str(zone.get("name") or "unnamed")
        return None

    def _guard_tcp_target(self, tcp_pose, context: str, sweep: bool = False):
        return True

    def _guard_joint_target(self, joint_target: dict, context: str, sweep: bool = False):
        return True

    def _preprocess_left_stick(self, lx: float, ly: float):
        lx = _apply_deadzone(lx, self.left_stick_deadzone)
        ly = _apply_deadzone(ly, self.left_stick_deadzone)
        if abs(lx) < self.left_stick_release_snap:
            lx = 0.0
        if abs(ly) < self.left_stick_release_snap:
            ly = 0.0
        return lx, ly

    def _shape_left_stick(self, lx: float, ly: float):

        if lx == 0.0:
            self.left_stick_lx_cmd = 0.0
        else:
            self.left_stick_lx_cmd += (
                lx - self.left_stick_lx_cmd
            ) * self.left_stick_filter_alpha

        if ly == 0.0:
            self.left_stick_ly_cmd = 0.0
        else:
            self.left_stick_ly_cmd += (
                ly - self.left_stick_ly_cmd
            ) * self.left_stick_filter_alpha

        return self.left_stick_lx_cmd, self.left_stick_ly_cmd

    def _resolve_left_stick_mode(self, lx: float, ly: float):
        ax = abs(lx)
        ay = abs(ly)
        thr = self.left_stick_cmd_threshold
        margin = self.left_stick_axis_margin

        if ax < thr and ay < thr:
            self.left_stick_mode = None
            return None

        if self.left_stick_mode == "x":
            if ax < thr:
                self.left_stick_mode = "y" if ay >= thr else None
            elif ay > ax + margin:
                self.left_stick_mode = "y"
        elif self.left_stick_mode == "y":
            if ay < thr:
                self.left_stick_mode = "x" if ax >= thr else None
            elif ax > ay + margin:
                self.left_stick_mode = "x"
        else:
            if ay >= ax + margin:
                self.left_stick_mode = "y"
            elif ax >= ay + margin:
                self.left_stick_mode = "x"
            else:
                self.left_stick_mode = "y" if ay >= ax else "x"

        return self.left_stick_mode

    def _soft_stop_live_motion(self):
        """
        Release live manual motion with a controller-side halt, then resync the
        cached pose. This stops the arm promptly on stick release without
        queuing an extra absolute lmove that can reverse the motion.
        """
        self._halt_live_motion()

        stop_pose = None
        try:
            stop_pose = self.robot.get_all_pose()[:6]
        except Exception:
            stop_pose = None

        if not stop_pose:
            return

        x, y, z, a, b, c = [float(v) for v in stop_pose]
        self.x0, self.y0, self.z0 = x, y, z
        self.R = axis_angle_to_R(a, b, c)
        self.live_last_abs_pose = (x, y, z, a, b, c)

        tz = self.R[:,2]
        pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
        yaw_deg = np.degrees(np.arctan2(tz[1], tz[0]))
        with self.state.lock:
            self.state.pitch = pitch
            self.state.yaw = yaw_deg

        joints = self._try_get_current_joints()
        if isinstance(joints, dict) and joints.get("j5") is not None:
            self.j5v = float(joints["j5"])
            with self.state.lock:
                self.state.j5 = self.j5v
        self.live_last_motion_mode = None

    def _reset_live_motion_pending(self):
        self.live_lmove_dirty = False
        self.live_abs_pose_dirty = False
        self.live_rel_xyz_pending.fill(0.0)
        self.live_rel_abc_pending.fill(0.0)
        self.live_j5_pending = 0.0
        self.live_next_send_t = 0.0
        self.live_last_send_t = 0.0
        self.live_last_abs_pose = None

    def _flush_live_motion(self, now_t: float):
        send_interval = self.live_send_interval_tool_axis if self.live_last_motion_mode == "tool_axis_translation" else self.live_send_interval
        if now_t < self.live_next_send_t:
            return False

        sent = False
        segment_dt = send_interval
        if self.live_last_send_t > 0.0:
            segment_dt = max(1e-3, now_t - self.live_last_send_t)

        if self.live_abs_pose_dirty and abs(self.live_j5_pending) > 1e-9:
            # The cached absolute TCP pose already includes this tool-axis change.
            self.live_j5_pending = 0.0

        if abs(self.live_j5_pending) > self.live_j5_epsilon:
            delta = self.live_j5_pending
            j_target = self._merge_joint_target({"j5": self.j5v})
            if j_target is not None and not self._guard_joint_target(j_target, "live J5 motion", sweep=True):
                self.live_j5_pending = 0.0
                self._reset_live_motion_pending()
                self._soft_stop_live_motion()
                return True
            self.live_j5_pending = 0.0
            self._play_live({"cmd":"jmove","rel":1,"j5":delta,"vel":self.VR,"cont":1})
            sent = True

        if self.live_abs_pose_dirty:
            a1, b1, c1 = R_to_axis_angle(self.R)
            pose_now = (float(self.x0), float(self.y0), float(self.z0), float(a1), float(b1), float(c1))
            if not self._guard_tcp_target(pose_now, "live TCP motion", sweep=True):
                self._reset_live_motion_pending()
                self._soft_stop_live_motion()
                return True
            if self.live_last_abs_pose is None:
                linear_delta = 0.0
                angular_delta = 0.0
            else:
                prev = np.array(self.live_last_abs_pose, dtype=float)
                cur = np.array(pose_now, dtype=float)
                linear_delta = float(np.linalg.norm(cur[:3] - prev[:3]))
                angular_delta = float(np.linalg.norm(cur[3:] - prev[3:]))
            if (
                linear_delta >= self.live_linear_epsilon
                or angular_delta >= self.live_angular_epsilon
                or self.live_last_abs_pose is None
            ):
                cmd_vel = max(0.2, linear_delta / segment_dt, angular_delta / segment_dt)
                self.live_rel_xyz_pending.fill(0.0)
                self.live_rel_abc_pending.fill(0.0)
                self._play_live({
                    "cmd": "lmove",
                    "rel": 0,
                    "x": pose_now[0],
                    "y": pose_now[1],
                    "z": pose_now[2],
                    "a": pose_now[3],
                    "b": pose_now[4],
                    "c": pose_now[5],
                    "vel": cmd_vel,
                    "cont": 1,
                })
                self.live_last_abs_pose = pose_now
                tz = self.R[:,2]
                pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
                yaw_deg = np.degrees(np.arctan2(tz[1], tz[0]))
                with self.state.lock:
                    self.state.pitch = pitch
                    self.state.yaw = yaw_deg
                self.live_abs_pose_dirty = False
                sent = True

        rel_xyz_norm = np.linalg.norm(self.live_rel_xyz_pending)
        rel_abc_norm = np.linalg.norm(self.live_rel_abc_pending)
        if (
            (not sent)
            and (
                rel_xyz_norm >= self.live_linear_epsilon
                or rel_abc_norm >= self.live_angular_epsilon
            )
        ):
            a1, b1, c1 = R_to_axis_angle(self.R)
            pose_now = (float(self.x0), float(self.y0), float(self.z0), float(a1), float(b1), float(c1))
            if not self._guard_tcp_target(pose_now, "live TCP motion", sweep=True):
                self._reset_live_motion_pending()
                self._soft_stop_live_motion()
                return True
            dx, dy, dz = [float(v) for v in self.live_rel_xyz_pending]
            da, db, dc = [float(v) for v in self.live_rel_abc_pending]
            self.live_rel_xyz_pending.fill(0.0)
            self.live_rel_abc_pending.fill(0.0)
            linear_speed = rel_xyz_norm / segment_dt if rel_xyz_norm > 1e-9 else 0.0
            angular_speed = rel_abc_norm / segment_dt if rel_abc_norm > 1e-9 else 0.0
            cmd_vel = max(0.2, linear_speed, angular_speed)
            self._play_live({
                "cmd": "lmove",
                "rel": 1,
                "x": dx,
                "y": dy,
                "z": dz,
                "a": da,
                "b": db,
                "c": dc,
                "vel": cmd_vel,
                "cont": 1,
            })
            tz = self.R[:,2]
            pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
            yaw_deg = np.degrees(np.arctan2(tz[1], tz[0]))
            with self.state.lock:
                self.state.pitch = pitch
                self.state.yaw = yaw_deg
            sent = True

        if sent:
            self.live_next_send_t = now_t + send_interval
            self.live_last_send_t = now_t

        return sent

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
        # Heading of tool-Z projection onto XY
        yaw_deg = np.degrees(np.arctan2(tz[1], tz[0]))
        with self.state.lock:
            self.state.pitch = pitch
            self.state.yaw = yaw_deg

    def _mark_motion_for(self, seconds: float):
        with self.state.lock:
            self.state.pose_motion_until = max(
                self.state.pose_motion_until,
                time.time() + max(0.0, seconds)
            )

    def _queue_jmove_to_pose(self, pose: dict):
        target_joints = {axis: pose[axis] for axis in COLLISION_JOINT_AXES if axis in pose}
        merged_target = self._merge_joint_target(target_joints)
        if merged_target is not None and not self._guard_joint_target(merged_target, "joint move", sweep=True):
            return False
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
        return True

    def tool_center_spin_demo(self, sweep_deg=360.0, steps=24, dwell=0.15):
        joints = self._try_get_current_joints()
        if not joints:
            print("⚠️ Tool center demo: no joint feedback available.")
            return
        base_j5 = float(joints.get("j5", 0.0) or 0.0)
        angles = np.linspace(0.0, sweep_deg, max(2, int(steps)), endpoint=True)
        print(f"[Tool] Starting center spin demo ({sweep_deg:.1f}° over {len(angles)} steps)…")
        for ang in angles:
            if self.stop_event.is_set():
                break
            pose = joints.copy()
            pose["j5"] = base_j5 + ang
            try:
                self._queue_jmove_to_pose(pose)
            except Exception as e:
                print(f"⚠️ Tool center demo step failed: {e}")
                break
            time.sleep(max(0.02, dwell))

        # Return to base orientation
        pose = joints.copy()
        pose["j5"] = base_j5
        try:
            self._queue_jmove_to_pose(pose)
        except Exception:
            pass
        print("[Tool] Center spin demo complete.")

    def _stop_tool_center_demo(self):
        self.tool_center_demo_stop.set()
        self.tool_center_demo_resume.set()
        try:
            if self.robot:
                self.robot.play_dict({"cmd": "halt"})
        except Exception:
            pass
        if self.tool_center_demo_thread and self.tool_center_demo_thread.is_alive():
            self.tool_center_demo_thread.join(timeout=1.0)
        self.tool_center_demo_thread = None
        self.tool_center_demo_stop.clear()
        with self.state.lock:
            self.state.tool_center_demo_waiting = False
            self.state.tool_center_demo = False

    def _apply_tool_center_delta(self, dx_mm, dy_mm, ref_pose=None):
        """Apply tool center delta and restore ref TCP pose."""
        try:
            pose_before = None
            try:
                pose_before = self._get_tcp_pose_from_robot()
            except Exception:
                pass
            with self.state.lock:
                cx = float(self.state.tool_cx) + dx_mm
                cy = float(self.state.tool_cy) + dy_mm
                lz = float(self.state.tool_lz)
                self.state.tool_cx = cx
                self.state.tool_cy = cy
                self.state.settings["tool_cx"] = cx
                self.state.settings["tool_cy"] = cy
            try:
                if hasattr(self.robot, "set_tool"):
                    self.robot.set_tool(lx=cx, ly=cy, lz=lz)
                else:
                    self.robot.play_dict(cmd={"cmd": "tool", "id": 1, "lx": cx, "ly": cy, "lz": lz})
            except Exception:
                self.robot.play_dict(cmd={"cmd": "tool", "id": 1, "lx": cx, "ly": cy, "lz": lz})
            self._update_tcp_from_settings()
            save_settings(self.state.settings)
            # Recenter: move new TCP back to the previous TCP world position
            if ref_pose:
                self._move_tcp_to_pose(ref_pose)
            pose_after = None
            try:
                pose_after = self._get_tcp_pose_from_robot()
            except Exception:
                pass
            if pose_before and pose_after:
                dxw = pose_after[0] - pose_before[0]
                dyw = pose_after[1] - pose_before[1]
                dzw = pose_after[2] - pose_before[2]
                print(f"[Tool] Recenter move delta (world) dx={dxw:+.3f} dy={dyw:+.3f} dz={dzw:+.3f}")
            print(f"[Tool] Applied tool center delta ΔX={dx_mm:+.3f}mm ΔY={dy_mm:+.3f}mm -> new CX={cx:.3f} CY={cy:.3f}")
        except Exception as e:
            print(f"⚠️ _apply_tool_center_delta failed: {e}")

    def _run_j5_test(self):
        """
        Tool center rotation test (auto-loop if enabled):
        - Move +45°, gather motion shifts (UI computes medians during motion)
        - Return to 0°
        - Apply a 0.5mm correction based on motion medians
        - Repeat while auto-correct toggle remains ON
        """
        print("[Tool] Starting tool center rotation test (auto-loop).")
        self.j5_test_stop.clear()
        with self.state.lock:
            self.state.j5_test_active = True
            self.state.j5_test_waiting = False
            self.state.j5_motion_med_left = 0.0
            self.state.j5_motion_med_right = 0.0
            self.state.j5_last_shift_x = 0.0
            self.state.j5_last_shift_y = 0.0
            self.state.j5_last_nonzero_left = 0.0
            self.state.j5_last_nonzero_right = 0.0
            self.state.j5_max_shift_left = 0.0
            self.state.j5_max_shift_right = 0.0
            self.state.j5_reset_history = True
            self.state.j5_start_shift_left = None
            self.state.j5_start_shift_right = None
            self.state.j5_end_shift_left = None
            self.state.j5_end_shift_right = None
            self.state.j5_dev_history = []

        base_pose = self._get_tcp_pose_from_robot()
        if not base_pose:
            with self.state.lock:
                self.state.j5_test_active = False
            print("⚠️ Tool rotation test: no pose available.")
            return

        bx, by, bz, ba, bb, bc = base_pose
        print(f"[Tool] Rotation test base pose: x={bx:.3f} y={by:.3f} z={bz:.3f} a={ba:.3f} b={bb:.3f} c={bc:.3f}")
        base_R = axis_angle_to_R(ba, bb, bc)
        ang = 45.0

        try:
            while not self.stop_event.is_set() and not self.j5_test_stop.is_set():
                # Reset per-cycle histories/medians/max
                with self.state.lock:
                    self.state.j5_reset_history = True
                    self.state.j5_motion_med_left = 0.0
                    self.state.j5_motion_med_right = 0.0
                    self.state.j5_max_shift_left = 0.0
                    self.state.j5_max_shift_right = 0.0
                    self.state.j5_last_nonzero_left = 0.0
                    self.state.j5_last_nonzero_right = 0.0
                    self.state.j5_start_shift_left = None
                    self.state.j5_start_shift_right = None
                    self.state.j5_end_shift_left = None
                    self.state.j5_end_shift_right = None

                # Move to +45° and let UI collect motion medians
                R_target = base_R @ axis_angle_to_R(0, 0, ang)
                ta, tb, tc = R_to_axis_angle(R_target)
                print(f"[Tool] Rotation test move to ang={ang:+.1f} (a={ta:.3f} b={tb:.3f} c={tc:.3f})")
                cmd = {
                    "cmd": "lmove",
                    "rel": 0,
                    "x": bx,
                    "y": by,
                    "z": bz,
                    "a": ta,
                    "b": tb,
                    "c": tc,
                    "vel": min(self.VR_POSE * 2.0, 8.0),
                    "cont": 0,
                }
                self.robot.play_dict(cmd)
                dt = abs(ang) / max(1e-3, self.VR_POSE) + 1.5
                self._mark_motion_for(dt)
                time.sleep(dt)
                # Check TCP displacement at +45°
                try:
                    pose_45 = self._get_tcp_pose_from_robot()
                    if pose_45 and base_pose:
                        dx = pose_45[0] - bx
                        dy = pose_45[1] - by
                        dz = pose_45[2] - bz
                        print(f"[Tool][Check @+45] Δtcp = ({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) mm")
                except Exception:
                    pass

                # Return to 0°
                print("[Tool] Rotation test move back to 0° for correction.")
                cmd_zero = {
                    "cmd": "lmove",
                    "rel": 0,
                    "x": bx,
                    "y": by,
                    "z": bz,
                    "a": ba,
                    "b": bb,
                    "c": bc,
                    "vel": min(self.VR_POSE * 2.0, 8.0),
                    "cont": 0,
                }
                self.robot.play_dict(cmd_zero)
                dt0 = abs(ang) / max(1e-3, self.VR_POSE) + 0.8
                self._mark_motion_for(dt0)
                time.sleep(dt0)
                try:
                    pose_back = self._get_tcp_pose_from_robot()
                    if pose_back and base_pose:
                        dx0 = pose_back[0] - bx
                        dy0 = pose_back[1] - by
                        dz0 = pose_back[2] - bz
                        print(f"[Tool][Check @0] Δtcp = ({dx0:+.3f}, {dy0:+.3f}, {dz0:+.3f}) mm")
                except Exception:
                    pass

                # Log motion medians; no automatic correction
                with self.state.lock:
                    med_left = float(getattr(self.state, "j5_motion_med_left", 0.0))
                    med_right = float(getattr(self.state, "j5_motion_med_right", 0.0))
                    last_nz_left = float(getattr(self.state, "j5_last_nonzero_left", 0.0))
                    last_nz_right = float(getattr(self.state, "j5_last_nonzero_right", 0.0))
                    max_left = float(getattr(self.state, "j5_max_shift_left", 0.0))
                    max_right = float(getattr(self.state, "j5_max_shift_right", 0.0))
                    start_left = getattr(self.state, "j5_start_shift_left", None)
                    start_right = getattr(self.state, "j5_start_shift_right", None)
                    end_left = getattr(self.state, "j5_end_shift_left", None)
                    end_right = getattr(self.state, "j5_end_shift_right", None)
                thr_dec = 0.02
                if start_left is not None and end_left is not None:
                    med_left = end_left - start_left
                if start_right is not None and end_right is not None:
                    med_right = end_right - start_right
                if abs(med_left) < thr_dec and abs(last_nz_left) >= thr_dec:
                    med_left = last_nz_left
                if abs(med_right) < thr_dec and abs(last_nz_right) >= thr_dec:
                    med_right = last_nz_right
                if abs(med_left) < thr_dec and abs(max_left) >= thr_dec:
                    med_left = max_left if med_left >= 0 else -max_left
                if abs(med_right) < thr_dec and abs(max_right) >= thr_dec:
                    med_right = max_right if med_right >= 0 else -max_right
                print(f"[Needle][Motion medians] L={med_left:+.3f}px R={med_right:+.3f}px (start L={start_left} R={start_right} nz L={last_nz_left:+.3f} R={last_nz_right:+.3f} max L={max_left:+.3f} R={max_right:+.3f})")
                with self.state.lock:
                    hist = list(getattr(self.state, "j5_dev_history", []))
                    hist.append((0.0, 0.0))
                    if len(hist) > 300:
                        hist = hist[-300:]
                    self.state.j5_dev_history = hist

                # After each calibration cycle, return to Default before next loop
                default_pose = None
                with self.state.lock:
                    default_pose = (self.state.poses.get("Default") or {}).copy()
                if default_pose:
                    try:
                        self._queue_jmove_to_pose(default_pose)
                        self._set_current_named("Default")
                        print("[Tool] Returned to Default pose after calibration.")
                    except Exception as e:
                        print(f"⚠️ Failed to return to Default pose: {e}")

                # Auto-correction removed; run a single cycle
                continue_loop = False
                if not continue_loop or self.stop_event.is_set() or self.j5_test_stop.is_set():
                    break

        except Exception as e:
            print(f"⚠️ Tool rotation test failed: {e}")
        finally:
            try:
                if self.robot:
                    self.robot.play_dict({"cmd": "halt"})
            except Exception:
                pass
            self.j5_test_stop.clear()
            with self.state.lock:
                self.state.j5_test_active = False
                self.state.j5_test_waiting = False
            print("[Tool] Tool center rotation test finished.")

    def _tool_center_demo_loop(self):
        self.tool_center_demo_resume.set()
        joints = self._try_get_current_joints()
        if not joints:
            print("⚠️ Tool center demo: no joint feedback available.")
            with self.state.lock:
                self.state.tool_center_demo = False
            return
        base_pose = joints.copy()
        base_list = [float(base_pose.get(f"j{i}", 0.0) or 0.0) for i in range(6)]
        last_sol = np.array(base_list, dtype=float)
        try:
            base_mat = self.robot.kinematic.t_flange_r_world(joint=base_list)
            tcp_mat = np.matmul(base_mat, self.robot.kinematic.T_tcp_r_flange)
        except Exception as e:
            print(f"⚠️ Tool center demo: cannot compute TCP pose ({e})")
            with self.state.lock:
                self.state.tool_center_demo = False
            return

        # Short dwell for smoother continuous path
        dwell = 0.02
        dwell_vert = 0.003  # faster updates for continuous vertical oscillation
        try:
            tilt_deg = float(self.state.settings.get("tool_center_demo_radius_mm", DEFAULT_TOOL_CENTER_RADIUS))
        except Exception:
            tilt_deg = DEFAULT_TOOL_CENTER_RADIUS
        mode = self.tool_center_demo_mode or "circle"
        print(f"[Tool] Tool-center orientation demo started (mode={mode}, fixed TCP).")

        def rot_x(deg):
            r = np.radians(deg)
            c, s = np.cos(r), np.sin(r)
            return np.array([[1, 0, 0],
                             [0, c,-s],
                             [0, s, c]])

        def rot_y(deg):
            r = np.radians(deg)
            c, s = np.cos(r), np.sin(r)
            return np.array([[ c, 0, s],
                             [ 0, 1, 0],
                             [-s, 0, c]])

        base_R_tcp = np.array(tcp_mat[:3, :3])
        base_tcp_xyz = [tcp_mat[0,3], tcp_mat[1,3], tcp_mat[2,3]]
        last_sol = np.array(base_list, dtype=float)

        def tcp_from_joints(jdict):
            if not jdict:
                return None
            jlist = [float(jdict.get(f"j{i}", 0.0) or 0.0) for i in range(6)]
            try:
                T_f = np.array(self.robot.kinematic.t_flange_r_world(joint=jlist))
                T_tcp = T_f @ self.robot.kinematic.T_tcp_r_flange
                return (float(T_tcp[0,3]), float(T_tcp[1,3]), float(T_tcp[2,3]))
            except Exception:
                return None
        def pick_tight_solution(solutions, current, target_tcp_mat, max_pos_err=0.01, max_joint_delta=None):
            """Prefer solutions that keep TCP error tiny, then stay nearest to current joints."""
            best = None
            best_cost = 1e9
            cur = np.array(current[:len(solutions[0])])
            tgt_pos = np.array([target_tcp_mat[0,3], target_tcp_mat[1,3], target_tcp_mat[2,3]])
            tgt_R   = np.array(target_tcp_mat[:3,:3])
            best_metrics = (0.0, 0.0, 0.0)
            for sol in solutions:
                s = np.array(sol[:len(cur)], dtype=float)
                try:
                    T_flange = np.array(self.robot.kinematic.t_flange_r_world(joint=s.tolist()))
                    T_tcp_cand = T_flange @ self.robot.kinematic.T_tcp_r_flange
                    pos_cand = np.array([T_tcp_cand[0,3], T_tcp_cand[1,3], T_tcp_cand[2,3]])
                    R_cand   = np.array(T_tcp_cand[:3,:3])
                    pos_err  = float(np.linalg.norm(pos_cand - tgt_pos))
                    rot_trace = np.trace(R_cand.T @ tgt_R)
                    rot_trace = max(min(rot_trace, 3.0), -1.0)
                    ang_err = float(np.degrees(np.arccos(max(min((rot_trace - 1)/2, 1), -1))))
                except Exception:
                    pos_err = 1e6
                    ang_err = 1e6
                joint_delta = s - cur
                joint_dist = float(np.linalg.norm(joint_delta))
                if max_joint_delta is not None and np.any(np.abs(joint_delta) > max_joint_delta):
                    continue
                cost = pos_err * 100.0 + ang_err * 0.5 + joint_dist * 0.2
                # Prefer only solutions with tiny TCP error; otherwise consider all but log
                if pos_err > max_pos_err:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best = s
                    best_metrics = (pos_err, ang_err, joint_dist)
            if best is None:
                # fallback to best overall if all exceeded max_pos_err
                for sol in solutions:
                    s = np.array(sol[:len(cur)], dtype=float)
                    try:
                        T_flange = np.array(self.robot.kinematic.t_flange_r_world(joint=s.tolist()))
                        T_tcp_cand = T_flange @ self.robot.kinematic.T_tcp_r_flange
                        pos_cand = np.array([T_tcp_cand[0,3], T_tcp_cand[1,3], T_tcp_cand[2,3]])
                        R_cand   = np.array(T_tcp_cand[:3,:3])
                        pos_err  = float(np.linalg.norm(pos_cand - tgt_pos))
                        rot_trace = np.trace(R_cand.T @ tgt_R)
                        rot_trace = max(min(rot_trace, 3.0), -1.0)
                        ang_err = float(np.degrees(np.arccos(max(min((rot_trace - 1)/2, 1), -1))))
                    except Exception:
                        pos_err = 1e6
                        ang_err = 1e6
                    joint_dist = float(np.linalg.norm(s - cur))
                    cost = pos_err * 100.0 + ang_err * 0.5 + joint_dist * 0.2
                    if cost < best_cost:
                        best_cost = cost
                        best = s
                        best_metrics = (pos_err, ang_err, joint_dist)
            if best is not None:
                pe, ae, jd = best_metrics
                if pe > max_pos_err or ae > 0.5:
                    print(f"[Tool][Demo] IK pick err pos={pe*1000:.2f}µm rot={ae:.3f}° joint_dist={jd:.3f}")
            return best

        steps_circle = max(240, int(360 / max(1e-3, tilt_deg if tilt_deg != 0 else 1)))
        steps_osc = 160
        steps_osc_smooth = 1440  # higher resolution for smoother oscillation
        demo_vel = min(self.VR_POSE, 8.0)
        demo_acc = 120.0
        demo_jerk = 800.0
        # Vertical-specific tuning: quieter (lower jerk/accel) but higher velocity
        demo_vel_vert = min(self.VR_POSE * 1.5, 12.0)
        demo_acc_vert = 80.0
        demo_jerk_vert = 300.0
        ang = 0.0
        total_deg = 0.0
        cycles_completed = 0
        cycle_pause = 0.35  # pause at end of each oscillation

        # Optional pre-seat: tiny oscillation to settle branch before main loop
        def _seat_motion(axis="y", eps_deg=0.2):
            try:
                if axis == "y":
                    target_R_tcp = base_R_tcp @ rot_y(eps_deg)
                else:
                    target_R_tcp = base_R_tcp @ rot_x(eps_deg)
                target_R_tcp = orthonormalize_R(target_R_tcp)
                target_tcp_mat = tcp_mat.copy()
                target_tcp_mat[:3,:3] = target_R_tcp
                target_tcp_mat[0,3], target_tcp_mat[1,3], target_tcp_mat[2,3] = base_tcp_xyz
                sols = self.robot.kinematic.inv(self.robot.kinematic.mat_to_xyzabc(target_tcp_mat),
                                                joint_current=base_list, all_sol=True)
                sol = pick_tight_solution(sols, base_list, target_tcp_mat, max_pos_err=0.01, max_joint_delta=None)
                sol_list = [float(x) for x in sol]
                pose_dict = {f"j{i}": sol_list[i] for i in range(min(6, len(sol_list)))}
                self.robot.play_dict({"cmd": "jmove", "rel": 0, "vel": demo_vel, "acc": demo_acc, "jerk": demo_jerk, "cont": 1, **pose_dict})
                time.sleep(0.05)
            except Exception:
                pass

        if mode in ("vertical", "oscillate_vertical"):
            _seat_motion(axis="y", eps_deg=0.2)
            _seat_motion(axis="y", eps_deg=-0.2)
        elif mode in ("horizontal", "oscillate_horizontal"):
            _seat_motion(axis="x", eps_deg=0.2)
            _seat_motion(axis="x", eps_deg=-0.2)
        while not self.tool_center_demo_stop.is_set() and not self.stop_event.is_set():
            if mode in ("vertical", "oscillate_vertical"):
                dx = 0.0
                dy = tilt_deg * np.sin(np.radians(ang))
                step_inc = 360.0 / steps_osc_smooth
                cycle_pause = 0.0  # run vertical oscillation continuously
                dwell = dwell_vert
                vel_use = demo_vel_vert
                acc_use = demo_acc_vert
                jerk_use = demo_jerk_vert
            elif mode in ("horizontal", "oscillate_horizontal"):
                dx = tilt_deg * np.sin(np.radians(ang))
                dy = 0.0
                step_inc = 360.0 / steps_osc
                vel_use = demo_vel
                acc_use = demo_acc
                jerk_use = demo_jerk
            else:  # circle
                dx = tilt_deg * np.cos(np.radians(ang))
                dy = tilt_deg * np.sin(np.radians(ang))
                step_inc = 360.0 / steps_circle
                vel_use = demo_vel
                acc_use = demo_acc
                jerk_use = demo_jerk

            target_R_tcp = base_R_tcp @ rot_x(dx) @ rot_y(dy)
            target_R_tcp = orthonormalize_R(target_R_tcp)
            target_tcp_mat = tcp_mat.copy()
            target_tcp_mat[:3,:3] = target_R_tcp
            target_tcp_mat[0,3], target_tcp_mat[1,3], target_tcp_mat[2,3] = base_tcp_xyz

            # Skip issuing a move when the orientation change is negligible to avoid an IK snap at start
            first_step = (total_deg == 0.0)
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                time.sleep(dwell)
                ang = (ang + step_inc) % 360.0
                total_deg += step_inc
                continue
            target_xyzabc = self.robot.kinematic.mat_to_xyzabc(target_tcp_mat)
            try:
                sols = self.robot.kinematic.inv(target_xyzabc, joint_current=base_list, all_sol=True)
                if sols is None or len(sols) == 0:
                    print("[Tool] Tool center demo: IK failed.")
                    break
                # On first step of a cycle, enforce staying on the same IK branch (tiny joint change)
                max_joint_delta = None
                if first_step:
                    max_joint_delta = np.radians(0.5)  # limit joint change per axis on first step
                sol = pick_tight_solution(sols, base_list, target_tcp_mat, max_pos_err=0.01, max_joint_delta=max_joint_delta)
                sol_list = [float(x) for x in sol]
                pose_dict = {f"j{i}": sol_list[i] for i in range(min(6, len(sol_list)))}
                cmd = {"cmd": "jmove", "rel": 0, "vel": vel_use, "acc": acc_use, "jerk": jerk_use, "cont": 1}
                cmd.update(pose_dict)
                # Log before/after TCP pose on first step to catch jump
                pose_before = None
                joints_before = None
                if first_step:
                    try:
                        pose_before = self.robot.get_all_pose()
                        joints_before = self._try_get_current_joints()
                    except Exception:
                        pose_before = None
                        joints_before = None
                self.robot.play_dict(cmd)
                self._mark_motion_for(dwell + 0.05)
                try:
                    # Prefer feedback for drift; fall back to commanded joints if missing
                    joints_fb = self._try_get_current_joints()
                    if joints_fb:
                        T_flange_exec = np.array(self.robot.kinematic.t_flange_r_world(
                            joint=[joints_fb.get(f"j{i}", 0.0) for i in range(6)]
                        ))
                    else:
                        T_flange_exec = np.array(self.robot.kinematic.t_flange_r_world(joint=sol_list))
                    T_tcp_exec = T_flange_exec @ self.robot.kinematic.T_tcp_r_flange
                    pos_exec = np.array([T_tcp_exec[0,3], T_tcp_exec[1,3], T_tcp_exec[2,3]])
                    pos_tgt = np.array([target_tcp_mat[0,3], target_tcp_mat[1,3], target_tcp_mat[2,3]])
                    pos_err = float(np.linalg.norm(pos_exec - pos_tgt))
                    print(f"[Tool][Demo] TCP drift this step (feedback): {pos_err*1000:.3f} µm")
                    if first_step:
                        try:
                            pose_after = self.robot.get_all_pose()
                            joints_after = joints_fb or self._try_get_current_joints()
                            if pose_before and pose_after and len(pose_before) >= 3 and len(pose_after) >= 3:
                                dz = pose_after[2] - pose_before[2]
                                dx = pose_after[0] - pose_before[0]
                                dy = pose_after[1] - pose_before[1]
                                print(f"[Tool][Demo] First-step TCP delta (pose): dx={dx*1000:.3f}µm dy={dy*1000:.3f}µm dz={dz*1000:.3f}µm")
                            if joints_before and joints_after:
                                tcp_b = tcp_from_joints(joints_before)
                                tcp_a = tcp_from_joints(joints_after)
                                if tcp_b and tcp_a:
                                    dx = (tcp_a[0] - tcp_b[0]) * 1000.0
                                    dy = (tcp_a[1] - tcp_b[1]) * 1000.0
                                    dz = (tcp_a[2] - tcp_b[2]) * 1000.0
                                    print(f"[Tool][Demo] First-step TCP delta (feedback IK): dx={dx:.3f}µm dy={dy:.3f}µm dz={dz:.3f}µm")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[Tool][Demo] Drift log error: {e}")
                base_list = sol_list
            except Exception as e:
                print(f"⚠️ Tool center demo move failed: {e}")
                break
            time.sleep(dwell)
            ang = (ang + step_inc) % 360.0
            total_deg += step_inc
            if total_deg >= 360.0 and cycle_pause > 0:  # completed a loop
                total_deg -= 360.0
                cycles_completed += 1
                # Pause and wait for UI resume (for manual/auto offsets)
                self._mark_motion_for(cycle_pause)
                with self.state.lock:
                    self.state.tool_center_demo_waiting = True
                self.tool_center_demo_resume.clear()
                waited = 0.0
                while not self.tool_center_demo_stop.is_set() and not self.stop_event.is_set():
                    if self.tool_center_demo_resume.wait(timeout=0.05):
                        break
                    waited += 0.05
                with self.state.lock:
                    self.state.tool_center_demo_waiting = False
        # Return to base
        try:
            pose = {f"j{i}": float(base_list[i]) for i in range(min(6, len(base_list)))}
            self._queue_jmove_to_pose(pose)
        except Exception:
            pass
        with self.state.lock:
            self.state.tool_center_demo = False
        print("[Tool] Tool-center orientation demo stopped.")

    def _tool_move_along_tz(self, dist_mm: float, cont: int = 0):
        # Refresh pose to ensure offsets are computed from the true TCP frame
        try:
            self._refresh_from_robot()
        except Exception:
            pass
        tz = self.R[:,2]
        dx, dy, dz = tz[0]*dist_mm, tz[1]*dist_mm, tz[2]*dist_mm
        a_deg, b_deg, c_deg = R_to_axis_angle(self.R)
        target_pose = (
            float(self.x0 + dx), float(self.y0 + dy), float(self.z0 + dz),
            float(a_deg), float(b_deg), float(c_deg),
        )
        if not self._guard_tcp_target(target_pose, "tool-axis move", sweep=True):
            return False
        self.robot.play_dict({"cmd":"lmove","rel":1,"x":dx,"y":dy,"z":dz,"vel":self.VT,"cont":cont})
        self.x0 += dx
        self.y0 += dy
        self.z0 += dz
        seconds = abs(dist_mm) / max(1e-6, self.VT) + 0.15
        self._mark_motion_for(seconds)
        return True

    def _capture_tcp_pose(self):
        """Return current TCP pose (world frame) as tuple (x,y,z,a,b,c)."""
        try:
            self._refresh_from_robot()
        except Exception:
            pass
        a_deg, b_deg, c_deg = R_to_axis_angle(self.R)
        return (self.x0, self.y0, self.z0, a_deg, b_deg, c_deg)

    def _get_tcp_pose_from_robot(self):
        """Query controller for absolute pose; fallback to cached pose."""
        try:
            if self.robot and hasattr(self.robot, "get_all_pose"):
                pose = self.robot.get_all_pose()
                if pose and len(pose) >= 6:
                    x, y, z, a, b, c = pose[:6]
                    self.x0, self.y0, self.z0 = float(x), float(y), float(z)
                    self.R = axis_angle_to_R(a, b, c)
                    return (self.x0, self.y0, self.z0, float(a), float(b), float(c))
        except Exception:
            pass
        return self._capture_tcp_pose()

    def _move_tcp_to_pose(self, pose):
        """Command robot to an absolute TCP pose and update cached state."""
        if not pose:
            return False
        try:
            x, y, z, a, b, c = pose
            if not self._guard_tcp_target((x, y, z, a, b, c), "TCP move", sweep=True):
                return False
            self.robot.play_dict({
                "cmd": "lmove",
                "rel": 0,
                "x": x,
                "y": y,
                "z": z,
                "a": a,
                "b": b,
                "c": c,
                "vel": self.VR,
            })
            self.x0, self.y0, self.z0 = x, y, z
            self.R = axis_angle_to_R(a, b, c)
            self._mark_motion_for(0.6)
            return True
        except Exception as e:
            print(f"⚠️ move_tcp_to_pose failed: {e}")
            return False

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

    def _save_current_pose_with_midway(self, name: str, approach_mm: float, demonstrate: bool = False):
        final_joints = self._try_get_current_joints()
        if final_joints is None:
            print("⚠️ Could not fetch joint angles; save operation skipped.")
            return False
        with self.state.lock:
            self.state.poses[name] = final_joints
        # Default: no motion; midway identical to final
        if not demonstrate:
            with self.state.lock:
                self.state.poses[midway_name(name)] = final_joints.copy()
            save_poses(self.state.poses)
            print(f"[Robot] Saved current as '{name}' (no motion; midway identical).")
            self._set_current_named(name)
            return True

        # Demonstration path: retract to midway and return, capturing midway joints
        moved_demo = False
        try:
            if not self._tool_move_along_tz(-approach_mm, cont=0):
                print(f"⚠️ Could not demonstrate midway for '{name}' because the retract move did not complete.")
                return False
            moved_demo = True
            time.sleep(0.05)
            mid_joints = self._try_get_current_joints()
            if mid_joints is None:
                print("⚠️ Could not capture midway joints; midway will mirror final.")
                mid_joints = final_joints.copy()
            with self.state.lock:
                self.state.poses[midway_name(name)] = mid_joints
            save_poses(self.state.poses)
            print(f"[Robot] Saved '{name}' with midway via −{approach_mm} mm demonstration.")
        finally:
            if moved_demo:
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
            if self._tool_move_along_tz(-approach_mm, cont=0):
                self._set_current_named(cur_mid)
                print(f"[Robot] Retracted to existing '{cur_mid}'.")
            return
        print(f"[Robot] '{cur_mid}' missing; creating via −{approach_mm} mm tool-Z…")
        if not self._tool_move_along_tz(-approach_mm, cont=0):
            return
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
                if not self._tool_move_along_tz(+approach_mm, cont=0):
                    return
                self._set_current_named(final)
                print(f"[Robot] Advanced +{approach_mm} mm to '{final}'.")
            except Exception as e:
                print(f"⚠️ Advance from '{self.current_named}' failed: {e}")

    def run(self):
        # Explicitly select the 6-DOF TA kinematics so yaw commands keep full orientation DOF
        robot = Dorna(model="dorna_ta")
        self.robot = robot
        for attempt in range(5):
            if robot.connect(host=self.host, port=self.port):
                break
            print(f"⚠️  Connection attempt {attempt+1}/5 to {self.host}:{self.port} failed. Retrying…")
            time.sleep(2)
        else:
            print(f"❌ All connection attempts to {self.host}:{self.port} failed.")
            sys.exit(1)

        print(f"[Robot] Kinematic model: {getattr(robot, 'model', 'unknown')} (n_dof={getattr(robot.kinematic, 'n_dof', '?')})")
        try:
            robot.add_event(self._on_robot_event)
        except Exception as e:
            print(f"⚠️ Could not register robot alarm event hook: {e}")
        try:
            with self.state.lock:
                startup_alarm_threshold = float(
                    self.state.settings.get("alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN)
                )
                startup_alarm_duration = float(
                    self.state.settings.get("alarm_duration", DEFAULT_PID_DURATION_MAIN)
                )
        except Exception as e:
            startup_alarm_threshold = DEFAULT_PID_THRESHOLD_MAIN
            startup_alarm_duration = DEFAULT_PID_DURATION_MAIN
            print(f"⚠️ Could not read startup halt settings: {e}")
        try:
            for axis in range(6):
                robot.set_pid(
                    index=axis,
                    threshold=DEFAULT_PID_THRESHOLD_MAIN,
                    duration=DEFAULT_PID_DURATION_MAIN,
                )
            if self.clear_alarm_on_launch:
                self._clear_alarm_latch("before motor enable")
            else:
                robot.set_alarm(0)
        except Exception as e:
            print(f"⚠️ Could not prepare disarmed stock halt settings for startup move: {e}")
        robot.set_motor(1)
        self._update_tcp_from_settings()

        with self.state.lock:
            tool_lz = float(self.state.tool_lz)
            tool_cx = float(getattr(self.state, "tool_cx", DEFAULT_TOOL_CX))
            tool_cy = float(getattr(self.state, "tool_cy", DEFAULT_TOOL_CY))
        robot.play_dict(cmd={"cmd":"tool","id":1,"lx":tool_cx,"ly":tool_cy,"lz":tool_lz})

        try:
            with self.state.lock:
                default_pose = self.state.poses["Default"].copy()
            if self._queue_jmove_to_pose(default_pose):
                self._set_current_named("Default")
                print("[Robot] Ready at Default.")
            else:
                print("⚠️ Default pose move did not complete at startup.")
        except Exception as e:
            print(f"⚠️ Could not home to Default pose: {e}")
            try:
                self._refresh_from_robot()
            except Exception:
                pass
        try:
            if self.clear_alarm_on_launch:
                self._clear_alarm_latch("after default move")
            settled = self._wait_for_joint_settle(max_wait_s=4.0, stable_for_s=1.0, tol_deg=0.05)
            if self.auto_tune_halt_startup and settled:
                self._auto_tune_alarm_pid(startup_alarm_threshold, startup_alarm_duration, persist=True)
            elif not self.apply_halt_settings_startup:
                print("[Startup] Skipped applying halt settings by launcher setting.")
            elif settled:
                try:
                    self._refresh_from_robot()
                except Exception:
                    pass
                self._apply_alarm_pid(startup_alarm_threshold, startup_alarm_duration, persist=False)
        except Exception as e:
            print(f"⚠️ Could not apply selected halt settings after startup move: {e}")

        with self.state.lock:
            if "Default" in self.state.poses:
                self.j5v = self.state.poses["Default"].get("j5", 0.0)
                self.state.j5 = self.j5v

        control_hz = 120.0
        next_t = time.time()
        joint_poll_time = 0.0
        last_control_t = next_t

        while not self.stop_event.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
                now = time.time()
            else:
                next_t = now
            loop_dt = min(0.05, max(1.0 / control_hz, now - last_control_t))
            last_control_t = now
            next_t += 1 / control_hz

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
                                if self._queue_jmove_to_pose(poses["Default"]):
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
                            if self._queue_jmove_to_pose(poses[target_mid]):
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

                    elif typ == "goto_direct":
                        target = cmd.get("pose_name")
                        with self.state.lock:
                            poses = self.state.poses.copy()
                        if not target or target not in poses:
                            print(f"⚠️ Cannot go direct: pose '{target}' not found.")
                            continue

                        # Clear any pending midway prompt before direct move
                        with self.state.lock:
                            self.state.await_confirm = False
                            self.state.await_target  = None
                            self.state.confirm_msg   = ""
                        try:
                            if self._queue_jmove_to_pose(poses[target]):
                                self._set_current_named(target)
                                print(f"[Robot] Direct move to '{target}' (midway skipped).")
                        except Exception as e:
                            print(f"⚠️ Failed direct move to '{target}': {e}")

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
                            target_pose = self._capture_tcp_pose()
                            lz = float(cmd.get("lz"))
                            with self.state.lock:
                                cx = float(getattr(self.state, "tool_cx", DEFAULT_TOOL_CX))
                                cy = float(getattr(self.state, "tool_cy", DEFAULT_TOOL_CY))
                            try:
                                if hasattr(self.robot, "set_tool"):
                                    self.robot.set_tool(lx=cx, ly=cy, lz=lz)
                                else:
                                    self.robot.play_dict(cmd={"cmd":"tool","id":1,"lx":cx,"ly":cy,"lz":lz})
                            except Exception:
                                # Fallback to play_dict if set_tool fails
                                self.robot.play_dict(cmd={"cmd":"tool","id":1,"lx":cx,"ly":cy,"lz":lz})
                            with self.state.lock:
                                self.state.tool_lz = lz
                                self.state.settings["tool_lz"] = lz
                            self._update_tcp_from_settings()
                            save_settings(self.state.settings)
                            self._refresh_from_robot()
                            self._move_tcp_to_pose(target_pose)
                            print(f"[Robot] Tool length set to {lz} mm")
                        except Exception as e:
                            print(f"⚠️ set_tool_lz failed: {e}")
                        continue

                    elif typ == "set_tool_center":
                        try:
                            target_pose = self._capture_tcp_pose()
                            cx = float(cmd.get("cx"))
                            cy = float(cmd.get("cy"))
                            with self.state.lock:
                                lz = float(self.state.tool_lz)
                                self.state.tool_cx = cx
                                self.state.tool_cy = cy
                                self.state.settings["tool_cx"] = cx
                                self.state.settings["tool_cy"] = cy
                            try:
                                if hasattr(self.robot, "set_tool"):
                                    self.robot.set_tool(lx=cx, ly=cy, lz=lz)
                                else:
                                    self.robot.play_dict(cmd={"cmd":"tool","id":1,"lx":cx,"ly":cy,"lz":lz})
                            except Exception:
                                self.robot.play_dict(cmd={"cmd":"tool","id":1,"lx":cx,"ly":cy,"lz":lz})
                            self._update_tcp_from_settings()
                            save_settings(self.state.settings)
                            self._refresh_from_robot()
                            self._move_tcp_to_pose(target_pose)
                            print(f"[Robot] Tool center set to X={cx} Y={cy}")
                        except Exception as e:
                            print(f"⚠️ set_tool_center failed: {e}")

                    elif typ == "set_pitch_deg":
                        try:
                            target_p = float(cmd.get("pitch", 0.0) or 0.0)
                        except Exception:
                            print("⚠️ set_pitch_deg: invalid pitch value.")
                            continue
                        try:
                            # Refresh pose before applying a new tilt
                            self._refresh_from_robot()
                            tz = self.R[:,2]
                            cur_pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
                            delta = target_p - cur_pitch
                            # Rotate about local Y to approach requested pitch (invert sign if needed)
                            self.R = self.R @ axis_angle_to_R(0, -delta, 0)
                            self.R = orthonormalize_R(self.R)
                            a1, b1, c1 = R_to_axis_angle(self.R)
                            if not self._move_tcp_to_pose((self.x0, self.y0, self.z0, a1, b1, c1)):
                                continue
                            tz_new = self.R[:,2]
                            new_pitch = -np.degrees(np.arcsin(np.clip(tz_new[2], -1, 1)))
                            with self.state.lock:
                                self.state.pitch = new_pitch
                        except Exception as e:
                            print(f"⚠️ set_pitch_deg failed: {e}")
                        continue

                    elif typ == "set_yaw_heading":
                        try:
                            target_yaw = float(cmd.get("yaw"))
                        except Exception:
                            print("⚠️ set_yaw_heading: invalid target.")
                            continue
                        try:
                            self._refresh_from_robot()
                            tz = self.R[:,2]
                            cur_yaw = np.degrees(np.arctan2(tz[1], tz[0]))
                            delta = target_yaw - cur_yaw
                            # Rotate about world Z to adjust heading of tool-Z projection
                            self.R = axis_angle_to_R(0, 0, delta) @ self.R
                            self.R = orthonormalize_R(self.R)
                            a1, b1, c1 = R_to_axis_angle(self.R)
                            if not self._move_tcp_to_pose((self.x0, self.y0, self.z0, a1, b1, c1)):
                                continue
                            tz_new = self.R[:,2]
                            new_yaw = np.degrees(np.arctan2(tz_new[1], tz_new[0]))
                            with self.state.lock:
                                self.state.yaw = new_yaw
                        except Exception as e:
                            print(f"⚠️ set_yaw_heading failed: {e}")
                        continue

                    elif typ == "set_j5_angle":
                        try:
                            target = float(cmd.get("j5"))
                        except Exception:
                            print("⚠️ set_j5_angle: invalid target.")
                            continue
                        try:
                            self._refresh_from_robot()
                            merged_target = self._merge_joint_target({"j5": target})
                            if merged_target is not None and not self._guard_joint_target(merged_target, "absolute J5 move", sweep=True):
                                continue
                            self.robot.play_dict({"cmd": "jmove", "rel": 0, "j5": target, "vel": self.VR_POSE})
                            self._mark_motion_for(0.6)
                            time.sleep(0.05)
                            self._refresh_from_robot()
                            with self.state.lock:
                                self.state.j5 = target
                                self.state.settings["last_j5"] = target
                            self.j5v = target
                        except Exception as e:
                            print(f"⚠️ set_j5_angle failed: {e}")
                        continue

                    elif typ == "j5_test_resume":
                        # Resume the rotation test after UI-triggered auto-correct
                        self.j5_test_resume_event.set()

                    elif typ == "save_tool_preset":
                        nm = str(cmd.get("name", "")).strip()
                        if not nm:
                            print("⚠️ save_tool_preset: empty name")
                            continue
                        with self.state.lock:
                            lz = float(self.state.tool_lz)
                            cx = float(getattr(self.state, "tool_cx", DEFAULT_TOOL_CX))
                            cy = float(getattr(self.state, "tool_cy", DEFAULT_TOOL_CY))
                            presets = list(self.state.settings.get("tool_presets", []))
                            presets = [p for p in presets if str(p.get("name", "")).strip() != nm]
                            presets.append({"name": nm, "lz": lz, "cx": cx, "cy": cy})
                            self.state.settings["tool_presets"] = presets
                        save_settings(self.state.settings)
                        print(f"[Robot] Saved tool preset '{nm}' (lz={lz}, cx={cx}, cy={cy}).")

                    elif typ == "apply_tool_preset":
                        nm = str(cmd.get("name", "")).strip()
                        if not nm:
                            continue
                        with self.state.lock:
                            presets = list(self.state.settings.get("tool_presets", []))
                        target = None
                        for p in presets:
                            if str(p.get("name", "")).strip() == nm:
                                target = p
                                break
                        if not target:
                            print(f"⚠️ Tool preset '{nm}' not found.")
                            continue
                        try:
                            lz = float(target.get("lz", self.state.tool_lz))
                            cx = float(target.get("cx", DEFAULT_TOOL_CX))
                            cy = float(target.get("cy", DEFAULT_TOOL_CY))
                        except Exception as e:
                            print(f"⚠️ Invalid preset '{nm}': {e}")
                            continue
                        target_pose = self._capture_tcp_pose()
                        try:
                            if hasattr(self.robot, "set_tool"):
                                self.robot.set_tool(lx=cx, ly=cy, lz=lz)
                            else:
                                self.robot.play_dict(cmd={"cmd":"tool","id":1,"lx":cx,"ly":cy,"lz":lz})
                        except Exception:
                            self.robot.play_dict(cmd={"cmd":"tool","id":1,"lx":cx,"ly":cy,"lz":lz})
                        with self.state.lock:
                            self.state.tool_lz = lz
                            self.state.tool_cx = cx
                            self.state.tool_cy = cy
                            self.state.settings["tool_lz"] = lz
                            self.state.settings["tool_cx"] = cx
                            self.state.settings["tool_cy"] = cy
                        self._update_tcp_from_settings()
                        save_settings(self.state.settings)
                        self._refresh_from_robot()
                        self._move_tcp_to_pose(target_pose)
                        print(f"[Robot] Applied tool preset '{nm}'.")

                    elif typ == "delete_tool_preset":
                        nm = str(cmd.get("name", "")).strip()
                        if not nm:
                            continue
                        with self.state.lock:
                            presets = list(self.state.settings.get("tool_presets", []))
                            new_presets = [p for p in presets if str(p.get("name", "")).strip() != nm]
                            self.state.settings["tool_presets"] = new_presets
                        save_settings(self.state.settings)
                        print(f"[Robot] Deleted tool preset '{nm}'.")
                        continue

                    elif typ == "tool_center_spin":
                        sweep = float(cmd.get("deg", 360.0) or 360.0)
                        steps = int(cmd.get("steps", 24) or 24)
                        dwell = float(cmd.get("dwell", 0.15) or 0.15)
                        self.tool_center_spin_demo(sweep, steps, dwell)
                        continue

                    elif typ == "tool_center_demo_toggle":
                        mode_req = cmd.get("mode")
                        current_mode = self.tool_center_demo_mode
                        should_start = True
                        running = bool(self.tool_center_demo_thread and self.tool_center_demo_thread.is_alive())
                        if running or self.state.tool_center_demo:
                            self._stop_tool_center_demo()
                            with self.state.lock:
                                self.state.tool_center_demo = False
                                # keep last mode in state for UI feedback
                                self.state.tool_center_demo_mode = current_mode
                            if mode_req is None or mode_req == "" or mode_req == current_mode:
                                should_start = False
                        if should_start and mode_req:
                            if self.tool_center_demo_thread and self.tool_center_demo_thread.is_alive():
                                self._stop_tool_center_demo()
                            self.tool_center_demo_stop.clear()
                            self.tool_center_demo_resume.set()
                            with self.state.lock:
                                self.state.tool_center_demo = True
                                self.tool_center_demo_mode = mode_req
                                self.state.tool_center_demo_mode = mode_req
                                self.state.tool_center_demo_waiting = False
                            t = threading.Thread(target=self._tool_center_demo_loop, daemon=True)
                            self.tool_center_demo_thread = t
                            t.start()
                        continue

                    elif typ == "j5_test_toggle":
                        start_req = bool(cmd.get("start", False))
                        print(f"[Tool] j5_test_toggle start={start_req}")
                        if start_req:
                            if self.robot is None:
                                print("⚠️ J5 test: robot not connected.")
                                with self.state.lock:
                                    self.state.j5_test_active = False
                                continue
                            if self.j5_test_thread and self.j5_test_thread.is_alive():
                                print("[Tool] J5 test already running.")
                                continue
                            self.j5_test_stop.clear()
                            with self.state.lock:
                                self.state.j5_test_active = True
                            t = threading.Thread(target=self._run_j5_test, daemon=True)
                            self.j5_test_thread = t
                            t.start()
                            print("[Tool] Tool center rotation test started.")
                        else:
                            self.j5_test_stop.set()
                            try:
                                if self.robot:
                                    self.robot.play_dict({"cmd": "halt"})
                            except Exception:
                                pass
                            with self.state.lock:
                                self.state.j5_test_active = False
                            print("[Tool] Tool center rotation test stopped.")
                        continue

                    elif typ == "tool_center_demo_stop":
                        self._stop_tool_center_demo()
                        with self.state.lock:
                            self.state.tool_center_demo = False
                            self.state.tool_center_demo_waiting = False
                        continue

                    elif typ == "tool_center_demo_resume":
                        # Allow UI to resume after a pause
                        self.tool_center_demo_resume.set()
                        with self.state.lock:
                            self.state.tool_center_demo_waiting = False
                        continue

                    elif typ == "set_alarm_pid":
                        try:
                            threshold = float(cmd.get("threshold"))
                            duration = float(cmd.get("duration"))
                        except Exception:
                            threshold = DEFAULT_PID_THRESHOLD_MAIN
                            duration = DEFAULT_PID_DURATION_MAIN
                        try:
                            self._apply_alarm_pid(threshold, duration, persist=True)
                        except Exception as e:
                            print(f"⚠️ set_alarm_pid failed: {e}")
                        continue

                    elif typ == "auto_tune_alarm_pid":
                        try:
                            threshold = float(cmd.get("threshold", self.state.settings.get("alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN)))
                            duration = float(cmd.get("duration", self.state.settings.get("alarm_duration", DEFAULT_PID_DURATION_MAIN)))
                        except Exception:
                            threshold = DEFAULT_PID_THRESHOLD_MAIN
                            duration = DEFAULT_PID_DURATION_MAIN
                        try:
                            self._auto_tune_alarm_pid(threshold, duration, persist=True)
                        except Exception as e:
                            print(f"⚠️ auto_tune_alarm_pid failed: {e}")
                        continue

                    elif typ == "disarm_alarm":
                        try:
                            self._clear_alarm_latch("manual disarm")
                            print("[Halt] Cleared latched controller alarm.")
                        except Exception as e:
                            print(f"⚠️ disarm_alarm failed: {e}")
                        continue

                    elif typ == "set_alarm_sensitivity":
                        try:
                            val = float(cmd.get("sens"))
                        except Exception:
                            val = DEFAULT_ALARM_SENSITIVITY
                        try:
                            self._apply_alarm_sensitivity(val, persist=True)
                        except Exception as e:
                            print(f"⚠️ set_alarm_sensitivity failed: {e}")
                        continue

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
                            print(f"[Routine] Final pose dwell set to {s_val} s (≤0 = none)")
                        except Exception as e:
                            print(f"⚠️ set_pose_wait failed: {e}")

                    elif typ == "save_reload_with_midway":
                        with self.state.lock:
                            approach_mm = float(self.state.approach_mm)
                        ok = self._save_current_pose_with_midway("Reload", approach_mm, demonstrate=bool(cmd.get("demo", False)))
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
                        ok = self._save_current_pose_with_midway(
                            name,
                            approach_mm,
                            demonstrate=bool(cmd.get("demo", False)),
                        )
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
                        ok = self._save_current_pose_with_midway(
                            name,
                            approach_mm,
                            demonstrate=bool(cmd.get("demo", False)),
                        )
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

            lx_raw, ly_raw = self._preprocess_left_stick(lx, ly)
            lx, ly = self._shape_left_stick(lx_raw, ly_raw)
            rx = _apply_deadzone(rx, self.orient_deadzone)
            ry = _apply_deadzone(ry, self.orient_deadzone)
            sx, sj5, sb, sc, sh = (5.0*sens, 5.0*sens, 0.5*sens, 0.5*sens, 5.0*sens)
            manual_enabled = (time.time() >= self.skip_manual_until) and (not waiting)
            if manual_enabled:
                left_stick_mode = self._resolve_left_stick_mode(lx_raw, ly_raw)
            else:
                self.left_stick_mode = None
                self.left_stick_lx_cmd = 0.0
                self.left_stick_ly_cmd = 0.0
                left_stick_mode = None
            live_motion_cmd_sent = False
            live_motion_requested = False
            orientation_requested = False

            if manual_enabled and left_stick_mode == "y":
                d = -ly * sx * loop_dt
                tz = self.R[:,2]
                dx = tz[0] * d
                dy = tz[1] * d
                dz = tz[2] * d
                if abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dz) > 1e-9:
                    self.x0 += dx
                    self.y0 += dy
                    self.z0 += dz
                    # Use absolute TCP targets for the main tool-axis live motion
                    # path, but avoid active pose pullback feedback here because
                    # it can introduce visible oscillation.
                    self.live_abs_pose_dirty = True
                    self.live_last_motion_mode = "tool_axis_translation"
                    live_motion_requested = True

            if manual_enabled and left_stick_mode == "x":
                delta = lx * sj5 * loop_dt
                if abs(delta) > 1e-9:
                    self.j5v += delta
                    self.R = self.R @ axis_angle_to_R(0, 0, delta)
                    self.R = orthonormalize_R(self.R)
                    self.live_j5_pending += delta
                    with self.state.lock:
                        self.state.j5 = self.j5v
                    self.live_last_motion_mode = "tool_roll"
                    live_motion_requested = True

            moved = False
            if manual_enabled:
                at_default = (self.current_named == "Default")
                rx_eff = -rx if at_default else rx
                ry_eff = ry
                if abs(rx_eff) > self.DZ:
                    self.R = axis_angle_to_R(0, 0, -rx_eff * sc * loop_dt) @ self.R
                    self.R = orthonormalize_R(self.R)
                    moved = True
                    orientation_requested = True
                if abs(ry_eff) > self.DZ:
                    self.R = self.R @ axis_angle_to_R(0, ry_eff * sb * loop_dt, 0)
                    self.R = orthonormalize_R(self.R)
                    moved = True
                    orientation_requested = True

            if moved:
                self.live_abs_pose_dirty = True
                self.live_last_motion_mode = "orientation"
                live_motion_requested = True

            if manual_enabled:
                tz = self.R[:,2]
                tx = self.R[:,0]
                ty = self.R[:,1]
                proj = np.array([tz[0], tz[1], 0.0])
                n = np.linalg.norm(proj)
                at_default = (self.current_named == "Default")

                if hy != 0:
                    step = float(hy) * sh * DPAD_Z_SIGN * loop_dt
                    if at_default:
                        if abs(step) > 1e-9:
                            self.z0 += step
                            self.live_rel_xyz_pending += np.array([0.0, 0.0, step], dtype=float)
                            self.live_last_motion_mode = "hat_translation"
                            live_motion_requested = True
                    else:
                        up_axis = tx if abs(tx[2]) >= abs(ty[2]) else ty
                        dx, dy, dz = up_axis[0]*step, up_axis[1]*step, up_axis[2]*step
                        if abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dz) > 1e-9:
                            self.x0 += dx
                            self.y0 += dy
                            self.z0 += dz
                            self.live_rel_xyz_pending += np.array([dx, dy, dz], dtype=float)
                            self.live_last_motion_mode = "hat_translation"
                            live_motion_requested = True

                hx_eff = -hx
                if hx_eff != 0:
                    if n > 1e-3:
                        perp = np.array([-proj[1], proj[0], 0.0]) / n * hx_eff
                    else:
                        perp = np.array([0.0, -hx_eff, 0.0])
                    dx, dy = perp[0] * sh * loop_dt, perp[1] * sh * loop_dt
                    if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                        self.x0 += dx
                        self.y0 += dy
                        self.live_rel_xyz_pending += np.array([dx, dy, 0.0], dtype=float)
                        self.live_last_motion_mode = "hat_translation"
                        live_motion_requested = True

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
                    self.R = orthonormalize_R(self.R)
                    self.live_j5_pending += delta
                    with self.state.lock:
                        self.state.j5 = self.j5v
                    self.live_last_motion_mode = "tool_roll"
                    live_motion_requested = True
            else:
                # Reset so next enable starts relative to current pose
                self.saw_prev_offset = 0.0
                self.saw_phase = 0.0
                self.saw_last_t = time.time()

            if live_motion_requested:
                try:
                    live_motion_cmd_sent = self._flush_live_motion(now)
                except Exception:
                    pass

            if manual_enabled:
                if live_motion_requested or live_motion_cmd_sent:
                    self.live_motion_active = True
                elif self.live_motion_active:
                    self._reset_live_motion_pending()
                    self._soft_stop_live_motion()
                    self.live_motion_active = False
                    self.last_pose_refresh = 0.0
            else:
                if self.live_motion_active:
                    self._reset_live_motion_pending()
                    self._soft_stop_live_motion()
                    self.last_pose_refresh = 0.0
                else:
                    self._reset_live_motion_pending()
                self.live_motion_active = False

            # Do not overwrite the live manual target with controller feedback
            # while the operator is actively holding input.
            if (
                not (live_motion_requested or self.live_motion_active)
                and time.time() - self.last_pose_refresh >= 0.2
            ):
                try:
                    self._refresh_from_robot()
                except Exception:
                    pass
                self.last_pose_refresh = time.time()

            # Periodic joint snapshot for logging
            if time.time() - joint_poll_time >= 0.05:
                joints = self._try_get_current_joints()
                with self.state.lock:
                    self.state.last_joints = joints
                now_joint_t = time.time()
                if joints is not None:
                    j4_now = float(joints.get("j4", 0.0))
                    if (
                        manual_enabled
                        and (live_motion_requested or self.live_motion_active)
                        and not orientation_requested
                        and self.last_j4_poll is not None
                        and self.last_j4_poll_t is not None
                    ):
                        dj4 = j4_now - self.last_j4_poll
                        if abs(dj4) >= self.j4_jump_guard_deg:
                            print(
                                f"[Guard] Unexpected j4 jump {dj4:+.3f} deg without wrist input; "
                                "halting live motion to avoid slip."
                            )
                            self._reset_live_motion_pending()
                            self._halt_live_motion()
                            self.live_motion_active = False
                            self.skip_manual_until = max(self.skip_manual_until, time.time() + 0.35)
                            self.last_pose_refresh = 0.0
                    self.last_j4_poll = j4_now
                    self.last_j4_poll_t = now_joint_t
                joint_poll_time = time.time()

        if not self.keep_motors_on_exit:
            try:
                self.robot.set_motor(0)
            except Exception:
                pass
        try:
            self.robot.close()
        except Exception:
            pass
        self._stop_tool_center_demo()

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

def detect_needle_tip(frame_bgr, entry_side="right", prefer="auto"):
    """Return normalized (x,y) of the bright needle tip or None.

    entry_side: "right"/"left"/"bottom"/"top" for the clipped side the body sits on.
    prefer: "left", "right", or "auto" (pick farthest extremum from center).
    """
    if frame_bgr is None:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold (Otsu) to pick the bright line against dark background
    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_val = max(20, min(240, int(otsu_val * 0.9)))
    _, mask = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY)
    # Ignore top/bottom margins so a tip exiting the frame doesn't trigger
    h, w = mask.shape[:2]
    margin = max(2, int(h * 0.05))
    mask[:margin, :] = 0
    mask[h - margin:, :] = 0

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts:
        return None
    # Largest contour, pick extremum based on entry_side/preference
    c = max(cnts, key=cv2.contourArea)
    pts = c.reshape(-1, 2)
    if pts.size == 0:
        return None
    x_left = pts[np.argmin(pts[:, 0])]
    x_right = pts[np.argmax(pts[:, 0])]
    y_top = pts[np.argmin(pts[:, 1])]
    y_bot = pts[np.argmax(pts[:, 1])]
    if entry_side == "right":
        tip = x_left  # body clipped on right, tip sweeps left
    elif entry_side == "left":
        tip = x_right  # body clipped on left, tip sweeps right
    elif entry_side == "bottom":
        tip = y_top   # body clipped on bottom, tip sweeps upward
    elif entry_side == "top":
        tip = y_bot   # body clipped on top, tip sweeps downward
    elif prefer == "left":
        tip = x_left
    elif prefer == "right":
        tip = x_right
    else:
        # pick point farthest from image borders to bias toward the true tip in-frame
        h, w = mask.shape[:2]
        d_border = [min(p[0], w - 1 - p[0], p[1], h - 1 - p[1]) for p in pts]
        tip = pts[int(np.argmax(d_border))]
    x, y = float(tip[0]), float(tip[1])
    h, w = mask.shape[:2]
    if w <= 0 or h <= 0:
        return None
    return (x / w, y / h)

def map_norm_to_rect(pt_norm, frame_shape, rect: pygame.Rect):
    """Map normalized (x,y) in frame space to screen coords inside rect using cover scaling."""
    if pt_norm is None or frame_shape is None:
        return None
    fh, fw = frame_shape[:2]
    if fw <= 0 or fh <= 0:
        return None
    scale = max(rect.w / float(fw), rect.h / float(fh))
    new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
    crop_x = max(0, (new_w - rect.w) // 2)
    crop_y = max(0, (new_h - rect.h) // 2)
    sx = rect.x + int(round(pt_norm[0] * fw * scale - crop_x))
    sy = rect.y + int(round(pt_norm[1] * fh * scale - crop_y))
    return (sx, sy)

def compute_trace_offsets(trace_list, frame):
    """Return (dx_px, dy_px) of avg tip vs center in pixel units."""
    if frame is None or not trace_list:
        return None
    fh, fw = frame.shape[:2]
    if fw <= 0 or fh <= 0:
        return None
    arr = np.array(trace_list, dtype=float)
    avg = np.mean(arr, axis=0)
    dx_px = (avg[0] - 0.5) * fw
    dy_px = (avg[1] - 0.5) * fh
    return dx_px, dy_px

def draw_persistent_lines(frame, lines_px, color=(0, 255, 255), thickness=1):
    """
    Draw cached lines extended to frame bounds so intersections remain visible.
    """
    if frame is None or not lines_px:
        return
    h, w = frame.shape[:2]
    bounds = (0, 0, w - 1, h - 1)
    for x1, y1, x2, y2 in lines_px:
        ok, p1, p2 = cv2.clipLine(bounds, (int(x1), int(y1)), (int(x2), int(y2)))
        if ok:
            cv2.line(frame, p1, p2, color, thickness)

def extend_line_to_frame(shape, line):
    """
    Extend the infinite line through (x1,y1)-(x2,y2) to the frame borders.
    Returns a pair of intersection points ((x1e,y1e),(x2e,y2e)) or None if no intersection.
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = map(float, line)
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None

    ts = []
    # Intersections with x = 0 and x = w-1
    if abs(dx) > 1e-9:
        t = (0 - x1) / dx
        y_int = y1 + t * dy
        if 0 <= y_int <= h - 1:
            ts.append((t, 0.0, y_int))
        t = ((w - 1) - x1) / dx
        y_int = y1 + t * dy
        if 0 <= y_int <= h - 1:
            ts.append((t, w - 1, y_int))
    # Intersections with y = 0 and y = h-1
    if abs(dy) > 1e-9:
        t = (0 - y1) / dy
        x_int = x1 + t * dx
        if 0 <= x_int <= w - 1:
            ts.append((t, x_int, 0.0))
        t = ((h - 1) - y1) / dy
        x_int = x1 + t * dx
        if 0 <= x_int <= w - 1:
            ts.append((t, x_int, h - 1))

    if len(ts) < 2:
        return None
    # Pick two farthest apart along the line parameter t
    ts_sorted = sorted(ts, key=lambda v: v[0])
    p_start = ts_sorted[0]
    p_end = ts_sorted[-1]
    return (p_start[1], p_start[2], p_end[1], p_end[2])

def intersections_from_lines(lines_px, shape):
    """
    Compute intersection points of all line pairs that fall within frame bounds.
    Returns list of (x,y) floats.
    """
    pts = []
    h, w = shape[:2]
    n = len(lines_px)
    for i in range(n):
        x1, y1, x2, y2 = lines_px[i]
        for j in range(i + 1, n):
            x3, y3, x4, y4 = lines_px[j]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            if 0 <= px <= w - 1 and 0 <= py <= h - 1:
                pts.append((px, py))
    return pts

def estimate_pivot_from_lines(lines_px):
    """
    Least-squares intersection of multiple lines (each (x1,y1,x2,y2)).
    Returns (x,y) in pixels or None if ill-conditioned.
    """
    if not lines_px:
        return None
    A_rows = []
    c_vals = []
    for x1, y1, x2, y2 in lines_px:
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-3:
            continue
        # Line normal (a,b) normalized; equation a*x + b*y + c = 0
        a = dy / length
        b = -dx / length
        c = -(a * x1 + b * y1)
        A_rows.append([a, b])
        c_vals.append(c)
    if not A_rows:
        return None
    A = np.array(A_rows, dtype=float)
    c_vec = np.array(c_vals, dtype=float)
    ATA = A.T @ A
    if np.linalg.cond(ATA) > 1e6:
        return None
    try:
        pivot = -np.linalg.inv(ATA) @ A.T @ c_vec
        return float(pivot[0]), float(pivot[1])
    except Exception:
        return None

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

    wrap_w = max(1, width_chars)
    visual_lines = []
    line_to_vis = []
    for i, ln in enumerate(lines):
        start_vis_idx = len(visual_lines)
        if ln == "":
            visual_lines.append((i, 0, ""))
        else:
            for start in range(0, len(ln), wrap_w):
                visual_lines.append((i, start, ln[start:start + wrap_w]))
        line_to_vis.append(start_vis_idx)

    caret_row = line_to_vis[line_idx] + (col_idx // wrap_w)
    caret_col_in_row = col_idx % wrap_w

    if caret_row < editor.scroll:
        editor.scroll = caret_row
    if caret_row >= editor.scroll + height_lines:
        editor.scroll = caret_row - height_lines + 1

    view_lines = visual_lines[editor.scroll: editor.scroll + height_lines]
    prev_clip = screen.get_clip()
    screen.set_clip(box)
    for i, (_, _, ln) in enumerate(view_lines):
        t = font.render(ln, True, (255,255,255))
        screen.blit(t, (box.x + 6, box.y + 4 + i*char_h))

    caret_y_in_view = caret_row - editor.scroll
    caret_x_px = box.x + 6 + caret_col_in_row * char_w
    caret_y_px = box.y + 4 + caret_y_in_view * char_h

    if 0 <= caret_y_in_view < height_lines:
        if (pygame.time.get_ticks() // 500) % 2 == 0:
            pygame.draw.rect(screen, (255,255,255),
                             (caret_x_px, caret_y_px, max(2, char_w//9), char_h), 0)
    screen.set_clip(prev_clip)

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

    panel_w = 600
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

    btn_w = 180
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
        tt_lines = wrap_lines(font, text, rect.w - 12)
        ly = rect.y + (rect.h - len(tt_lines) * font.get_height()) // 2
        for tline in tt_lines:
            surf = font.render(tline, True, (230,230,230))
            screen.blit(surf, (rect.x + (rect.w - surf.get_width())//2, ly))
            ly += font.get_height()

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
        max_w = box.w - 12
        if tt.get_width() > max_w:
            wrapped = wrap_lines(font, text or "", max_w)
            ly = box.y + 4
            for wln in wrapped:
                surf = font.render(wln, True, (255,255,255))
                screen.blit(surf, (box.x + 6, ly))
                ly += font.get_height()
        else:
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
    btn_w = 200
    btn_h = 36
    gap = 10
    n_btn = 3 if allow_advance else 2
    total_w = n_btn*btn_w + (n_btn-1)*gap
    bx = x + (panel_w - total_w)//2
    by = y + panel_h - btn_h - 18

    btn_confirm = pygame.Rect(bx, by, btn_w, btn_h)
    pygame.draw.rect(screen, (50,80,50), btn_confirm)
    pygame.draw.rect(screen, (0,255,127), btn_confirm, 2)
    lines_btn1 = wrap_lines(font, "Confirm & Begin Recording", btn_w - 12)
    ly = btn_confirm.y + (btn_h - len(lines_btn1) * font.get_height()) // 2
    for ln in lines_btn1:
        t1 = font.render(ln, True, (230,230,230))
        screen.blit(t1, (btn_confirm.x + (btn_w - t1.get_width())//2,
                         ly))
        ly += font.get_height()

    btn_confirm_adv = None
    if allow_advance:
        bx2 = bx + btn_w + gap
        btn_confirm_adv = pygame.Rect(bx2, by, btn_w, btn_h)
        pygame.draw.rect(screen, (50,60,80), btn_confirm_adv)
        pygame.draw.rect(screen, (0,255,200), btn_confirm_adv, 2)
        lines_btn2 = wrap_lines(font, "Confirm & Advance", btn_w - 12)
        ly2 = btn_confirm_adv.y + (btn_h - len(lines_btn2) * font.get_height()) // 2
        for ln in lines_btn2:
            t2 = font.render(ln, True, (230,230,230))
            screen.blit(t2, (btn_confirm_adv.x + (btn_w - t2.get_width())//2,
                             ly2))
            ly2 += font.get_height()
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
    hint = "A = Confirm & Begin Recording, B = Cancel"
    if allow_advance:
        hint += ", X = Confirm & Advance + Record"
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
    startup_settings = load_startup_settings()
    args = _resolve_startup_args(args, startup_settings)
    if args.launcher:
        launched = show_startup_launcher(args)
        if launched is None:
            return
        args = launched
    _persist_startup_args(startup_settings, args)

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
    rs_thread = RealSenseThread(width=args.rs_width, height=args.rs_height, fps=args.rs_fps)
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
        cands = discover_uvc_inventory(limit=6)
        uvc1_path = ""
        uvc2_path = ""
        if len(cands) >= 2:
            by_bus = {}
            for c in cands:
                by_bus.setdefault(c["busnum"] or c["path"], []).append(c)
            if len(by_bus) >= 2:
                buses = list(by_bus.keys())
                pickA = by_bus[buses[0]][0]
                pickB = by_bus[buses[1]][0]
            else:
                pickA, pickB = cands[0], cands[1]
            uvc1_path = pickA["path"]
            uvc2_path = pickB["path"]
            print("[UVC auto-pick]")
            print(f"  Cam1: {_format_uvc_inventory_label(pickA)}")
            print(f"  Cam2: {_format_uvc_inventory_label(pickB)}")
        else:
            uvc1_path = (cands[0]["path"] if len(cands) >= 1 else "")
            uvc2_path = (cands[1]["path"] if len(cands) >= 2 else "")
            if len(cands) == 1:
                print("[UVC auto-pick] Only one UVC camera group was detected.")
                print(f"  Cam1: {_format_uvc_inventory_label(cands[0])}")

    if uvc1_path and uvc2_path and os.path.realpath(uvc1_path) == os.path.realpath(uvc2_path):
        print("[UVC] Auto-pick selected the same device twice; using a single UVC camera.")
        uvc2_path = ""

    uvc_threads = []
    if uvc1_path:
        uvc1 = UvcThread(
            uvc1_path, args.uvc_width, args.uvc_height, args.uvc_fps, False,
            name="UVC#1", try_index1_fallback=args.uvc_try_index1
        )
        uvc1.start()
        uvc_threads.append(uvc1)
        print(f"[UVC#1] {_format_uvc_selected_path(uvc1_path)}")
    else:
        uvc1 = None
        print("[UVC#1] Not found (placeholder)")

    if uvc2_path:
        uvc2 = UvcThread(
            uvc2_path, args.uvc_width, args.uvc_height, args.uvc_fps, False,
            name="UVC#2", try_index1_fallback=args.uvc_try_index1
        )
        uvc2.start()
        uvc_threads.append(uvc2)
        print(f"[UVC#2] {_format_uvc_selected_path(uvc2_path)}")
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
                    uvc1_path, args.uvc_width, args.uvc_height, 15, False,
                    name="UVC#1", try_index1_fallback=args.uvc_try_index1
                )
                uvc1.start()
                uvc_threads.append(uvc1)
            if uvc2_path:
                uvc2 = UvcThread(
                    uvc2_path, args.uvc_width, args.uvc_height, 15, False,
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
        threshold, duration = _clamp_alarm_pid(
            getattr(args, "alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN),
            getattr(args, "alarm_duration", DEFAULT_PID_DURATION_MAIN),
        )
        state.settings["alarm_threshold"] = threshold
        state.settings["alarm_duration"] = duration
        state.alarm_threshold = threshold
        state.alarm_duration = duration
    save_poses(state.poses)
    save_settings(state.settings)

    rt = RobotThread(
        state,
        host=args.host,
        port=args.port,
        clear_alarm_on_launch=bool(getattr(args, "clear_alarm_startup", True)),
        apply_halt_settings_startup=bool(getattr(args, "apply_halt_settings_startup", True)),
        auto_tune_halt_startup=bool(getattr(args, "auto_tune_halt_startup", False)),
    )
    rt.start()

    # ─────────────────────────────────────────────────────────────
    # Routine editor / executor
    # ─────────────────────────────────────────────────────────────
    routines = load_routines()
    routine_names_sorted = sorted(routines.keys(), key=str.lower)
    current_routine_name = routine_names_sorted[0] if routine_names_sorted else "Default"
    initial_routine_text = routines.get(current_routine_name, ROUTINE_SAMPLE.strip())
    routine_editor = EditorState(text=initial_routine_text,
                                 caret=len(initial_routine_text))
    routine_edit_active = False
    routine_action_modal_active = False
    routine_action_target = ""
    routine_edit_modal_active = False
    routine_edit_modal_target = ""
    routine_rename_target = ""
    routine_name_mode = "save_as"
    routine_executor = None

    # ─────────────────────────────────────────────────────────────
    # UI modal flags / text fields
    # ─────────────────────────────────────────────────────────────
    toollen_input_active  = False
    approach_input_active = False
    toolcx_input_active   = False
    toolcy_input_active   = False
    yaw_input_active      = False
    yaw_text              = ""
    angle_input_active    = False
    angle_text            = ""
    pitch_input_active    = False
    pitch_text            = ""
    tool_center_radius_drag = False
    alarm_threshold_drag = False
    alarm_duration_drag = False
    settings_path_input_active = False
    settings_path_mode = "load"
    naming_input_active   = False
    rename_input_active   = False
    delete_confirm_active = False
    overwrite_confirm_active = False
    edit_menu_active      = False
    edit_menu_target      = ""
    save_mode_modal_active = False
    pending_save_pose_name = ""
    toolpreset_input_active = False
    toolpreset_name_text    = ""
    toolpreset_rename_input_active = False
    toolpreset_rename_text  = ""
    toolpreset_rename_old   = ""
    tool_preset_menu_active = False
    tool_preset_menu_target = ""
    edit_menu_target_reserved = False

    fwd_rate_input_active = False
    bwd_rate_input_active = False
    endstop_timeout_input_active = False
    pose_wait_input_active       = False
    retract_input_active         = False
    routine_name_input_active = False
    routine_name_text         = ""
    routine_action_modal_active = False
    routine_action_target = ""
    routine_edit_modal_active = False
    routine_edit_modal_target = ""
    routine_rename_target = ""
    routine_name_mode = "save_as"

    syringe_calib_active  = False
    syringe_field_active  = "vol"

    injection_info_active  = False
    injection_field_active = "study"
    post_step_prompt_active = False
    subject_prompt_active   = False

    # text fields
    toollen_text          = f"{state.tool_lz:.1f}"
    approach_text         = f"{state.approach_mm:.1f}"
    toolcx_text           = f"{getattr(state, 'tool_cx', DEFAULT_TOOL_CX):.3f}"
    toolcy_text           = f"{getattr(state, 'tool_cy', DEFAULT_TOOL_CY):.3f}"
    tool_center_radius    = float(state.settings.get("tool_center_demo_radius_mm", DEFAULT_TOOL_CENTER_RADIUS))
    alarm_threshold_val = float(state.settings.get("alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN))
    alarm_duration_val = float(state.settings.get("alarm_duration", DEFAULT_PID_DURATION_MAIN))
    settings_path_text    = SETTINGS_PATH
    tool_center_radius_rect = None
    alarm_threshold_rect = None
    alarm_duration_rect = None
    # Needle tip traces (normalized coords in respective UVC frames)
    needle_trace_h = []
    needle_trace_v = []
    # Needle line collections for pivot estimation during oscillation demos
    needle_lines_h = []  # list of (x1,y1,x2,y2) in px for horizontal demo (left cam)
    needle_lines_v = []  # list of (x1,y1,x2,y2) in px for vertical demo (right cam)
    needle_line_count_h = 0
    needle_line_count_v = 0
    # Accumulators for averaging batches of detected lines
    needle_line_acc_h = np.zeros(4, dtype=float)
    needle_line_acc_v = np.zeros(4, dtype=float)
    last_trace_mode_h = ""
    last_trace_mode_v = ""
    needle_trace_prev_h = []
    needle_trace_prev_v = []
    needle_trace_total_h = []
    needle_trace_total_v = []
    trace_snap_h_done = False
    trace_snap_v_done = False
    corr_hint_text = ""
    j5_test_active = False
    naming_text           = ""
    rename_text           = ""
    rename_old_name       = ""
    delete_target_name    = ""
    overwrite_target_name = ""

    fwd_rate_text         = str(state.settings.get("plunger_fwd_rate", MAX_RATE))
    bwd_rate_text         = str(state.settings.get("plunger_bwd_rate", MAX_RATE))
    endstop_timeout_text  = str(state.settings.get("endstop_timeout_ms", 0))
    pose_wait_text        = f"{state.settings.get('pose_final_wait_s', 0.0):.1f}"
    retract_text          = f"{state.settings.get('injection_retract_mm', 10.0):.1f}"

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

    # Track the upcoming step index once the subject is confirmed
    pending_step_index = None

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

    # Needle tracking during rotation test (UVC)
    needle_baseline = {"uvc1": None, "uvc2": None}  # (angle, line, len)
    needle_last = {"uvc1": None, "uvc2": None}
    prev_j5_test_active = False
    prev_j5_wait_flag = False
    needle_guidance_until = 0.0
    last_rec_msgs_by_cam = {"uvc1": [], "uvc2": []}
    last_rec_shifts = {"uvc1": 0.0, "uvc2": 0.0}
    motion_stats_by_cam = {
        "uvc1": {"sum": 0.0, "count": 0},
        "uvc2": {"sum": 0.0, "count": 0},
    }
    last_shift_by_cam = {"uvc1": 0.0, "uvc2": 0.0}  # latest per-frame shift
    shift_history = {"uvc1": [], "uvc2": []}  # keep recent shifts for trend
    needle_correction_note = ""
    needle_correction_until = 0.0
    rotation_measure = {"uvc1": None, "uvc2": None}  # store mid offsets at + rotation
    last_decision_msgs = []
    last_good_shifts = {"uvc1": 0.0, "uvc2": 0.0}
    last_nonzero_shifts = {"uvc1": 0.0, "uvc2": 0.0}
    motion_med_shifts = {"uvc1": 0.0, "uvc2": 0.0}
    last_shift_log = {"uvc1": 0.0, "uvc2": 0.0}  # time of last console log per cam
    motion_max_shifts = {"uvc1": 0.0, "uvc2": 0.0}

    def delete_pose_local(name: str):
        """Delete a pose (and its __midway) directly from shared state."""
        if not name:
            return False
        with state.lock:
            poses = state.poses
            reserved_list = list(state.settings.get("reserved_poses", []))
            if name in RESERVED_POSES or name in reserved_list:
                return False
            removed_any = False
            if name in poses:
                poses.pop(name, None)
                removed_any = True
            mid = midway_name(name)
            if mid in poses:
                poses.pop(mid, None)
                removed_any = True
            if removed_any:
                save_poses(poses)
                # Clear current selection if it was deleted
                if state.current_pose_name in (name, mid):
                    state.current_pose_name = ""
            return removed_any
    quit_confirm_active = False

    # Short-lived UI notices
    ui_notice_text = ""
    ui_notice_until = 0.0

    # ─────────────────────────────────────────────────────────────
    # Helper closures local to main()
    # ─────────────────────────────────────────────────────────────
    def normalize_reserved_list(raw):
        """Return a list of pose names considered reserved."""
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple, set)):
            return [str(x) for x in raw]
        try:
            return [str(x) for x in list(raw)]
        except Exception:
            return [str(raw)]

    def visible_pose_names(poses: dict):
        return [name for name in poses.keys() if not name.endswith(MIDWAY_SUFFIX)]

    def sorted_pose_names(poses: dict):
        names = visible_pose_names(poses)
        def key(n):
            if n == "Default": return (0, "")
            if n == "Reload":  return (1, "")
            return (2, n.lower())
        return sorted(names, key=key)

    def make_buttons(current_tool_lz, current_approach, poses, settings, routines, current_routine_name, tool_center_demo_on, tool_center_demo_mode, tool_center_radius, alarm_threshold, alarm_duration):
        syringe_volume_ul = settings.get("syringe_volume_ul", 10.0)
        syringe_step_ul = settings.get("syringe_step_ul", 1.0)
        syringe_remaining_ul = settings.get("syringe_remaining_ul", syringe_volume_ul)
        ammo = (
            max(0, int(syringe_remaining_ul // syringe_step_ul))
            if syringe_step_ul > 0
            else 0
        )
        reserved_names = set(normalize_reserved_list(settings.get("reserved_poses", [])))
        alarm_threshold = float(alarm_threshold)
        alarm_duration = float(alarm_duration)

        items = [
            {"label": "— Tool Head —", "kind": "header"},
            {"label": f"Tool length (mm): {current_tool_lz:.1f}  [Edit]", "kind": "edit_tool"},
            {"label": f"Tool center X (mm): {settings.get('tool_cx', DEFAULT_TOOL_CX):.3f}  [Edit]", "kind": "edit_toolcx"},
            {"label": f"Tool center Y (mm): {settings.get('tool_cy', DEFAULT_TOOL_CY):.3f}  [Edit]", "kind": "edit_toolcy"},
            {"label": f"Approach distance (mm): {current_approach:.1f}  [Edit]", "kind": "edit_approach"},
            {"label": f"Tool center demo (circle): {'ON' if (tool_center_demo_on and tool_center_demo_mode=='circle') else 'OFF'}", "kind": "tool_center_demo_circle"},
            {"label": f"Tool center demo (vertical osc): {'ON' if (tool_center_demo_on and tool_center_demo_mode in ('vertical','oscillate_vertical')) else 'OFF'}", "kind": "tool_center_demo_vert"},
            {"label": f"Tool center demo (horizontal osc): {'ON' if (tool_center_demo_on and tool_center_demo_mode in ('horizontal','oscillate_horizontal')) else 'OFF'}", "kind": "tool_center_demo_horz"},
            {"label": f"Tool center rotation test: {'ON' if getattr(state, 'j5_test_active', False) else 'OFF'}", "kind": "tool_center_j5_test"},
            {"label": f"Tool center demo deviation (deg): {tool_center_radius:.1f}", "kind": "tool_center_radius"},
            {"label": "Save current tool setup as preset", "kind": "tool_preset_save"},
        ]
        tool_presets = settings.get("tool_presets", []) or []
        if tool_presets:
            items.append({"label": "— Presets —", "kind": "header"})
            for p in tool_presets:
                nm = str(p.get("name", "")).strip()
                try:
                    lz_p = float(p.get("lz", current_tool_lz))
                    cx_p = float(p.get("cx", settings.get("tool_cx", DEFAULT_TOOL_CX)))
                    cy_p = float(p.get("cy", settings.get("tool_cy", DEFAULT_TOOL_CY)))
                except Exception:
                    continue
                if not nm:
                    continue
                items.append({
                    "label": f"{nm}  (LZ={lz_p:.2f}, CX={cx_p:.3f}, CY={cy_p:.3f})",
                    "kind": "tool_preset_row",
                    "preset": nm,
                })

        items += [
            {"label": "— Routine —", "kind": "header"},
            {"label": "Stop Routine (halt motion only)", "kind": "routine_stop"},
            {"label": f"Wash FWD speed (steps/s): {settings.get('plunger_fwd_rate', MAX_RATE)}  [Edit]", "kind": "edit_fwd_rate"},
            {"label": f"Wash BWD speed (steps/s): {settings.get('plunger_bwd_rate', MAX_RATE)}  [Edit]", "kind": "edit_bwd_rate"},
            {"label": f"Failsafe timeout (ms) [0 = infinite]: {settings.get('endstop_timeout_ms', 0)}  [Edit]", "kind": "edit_endstop_timeout"},
            {"label": f"Final pose dwell (s) [0 = none]: {settings.get('pose_final_wait_s', 0.0):.1f}  [Edit]", "kind": "edit_pose_wait"},
            {"label": "— Saved routines —", "kind": "header"},
            {"label": "Create NEW routine", "kind": "routine_new"},
            # NOTE: swap_endstops still supported in settings.json, but UI toggle removed
        ]

        for rname in sorted(routines.keys(), key=str.lower):
            items.append({
                "label": f"Use: {rname}",
                "kind": "routine_row",
                "routine": rname,
                "current": (rname == current_routine_name),
            })

        items += [
            {"label": "— Injection —", "kind": "header"},
            {"label": (
                f"Syringe calib: stroke={settings.get('syringe_full_travel_time_s', 0.0):.2f}s, "
                f"vol={syringe_volume_ul:.3f} µL, step={syringe_step_ul:.3f} µL"
            ), "kind": "syringe_calib"},
            {"label": f"Retract distance after step (mm): {settings.get('injection_retract_mm', 10.0):.1f}  [Edit]", "kind": "edit_retract"},
            {"label": f"Start Injection (remaining ≈ {ammo} steps)", "kind": "start_injection"},
            {"label": "Stop Injection (exit injection mode)", "kind": "stop_injection"},

            {"label": "— Saved poses —", "kind": "header"},
            {"label": "New saved pose from CURRENT robot position", "kind": "new_pose"},
        ]

        for name in sorted_pose_names(poses):
            items.append({
                "label": f"Go: {name}",
                "kind": "pose_row",
                "pose": name,
                "reserved": (
                    (name in RESERVED_POSES)
                    or (name in reserved_names)
                ),
            })

        items += [
            {"label": "— Settings —", "kind": "header"},
            {"label": "Load settings.json", "kind": "settings_load"},
            {"label": "Save settings.json", "kind": "settings_save"},
            {"label": "Halt settings apply automatically on startup.", "kind": "alarm_status"},
            {"label": "Apply Halt Settings Now", "kind": "alarm_arm_now"},
            {"label": "Auto-Tune Halt at Default", "kind": "alarm_auto_tune"},
            {"label": "Clear Latched Alarm", "kind": "alarm_disarm_now"},
            {"label": f"Halt threshold: {int(round(alarm_threshold))}", "kind": "alarm_threshold"},
            {"label": f"Halt duration: {int(round(alarm_duration))}", "kind": "alarm_duration"},
            {"label": "Quit", "kind": "quit_app"},
        ]

        return items

    def _norm_angle_deg(a):
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    def expected_needle_angle(cam_idx, rot90, flip_h, flip_v):
        """
        Both cameras use the same logic: baseline horizontal (0°) for right,
        vertical (90°) for left, then apply the display rotation/flip.
        """
        base = 90.0 if cam_idx == 1 else 0.0
        ang = base + 90.0 * (int(rot90) % 4)
        if flip_h:
            ang = 180.0 - ang
        if flip_v:
            ang = -ang
        return _norm_angle_deg(ang)

    def detect_needle_line(frame_bgr, expected_angle_deg, tol_deg=30.0, entry="any", touch="any"):
        """
        Detect dominant line near expected angle using bright-line bias (needle on dark bg).
        Returns (angle_deg, (x1,y1,x2,y2), length_px) or None.
        entry: hint for origin ("bottom"/"left"/"any"); touch: required edge contact ("bottom"/"right"/"any"/"none").
        """
        if frame_bgr is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Bright mask: emphasize bright needle against dark background
        g_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = max(10, int(np.percentile(g_blur, 97)))
        _, bright = cv2.threshold(g_blur, thr, 255, cv2.THRESH_BINARY)

        # Edges from bright mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel, iterations=1)
        edges = cv2.Canny(mask, 5, 40, apertureSize=3)

        min_len = max(8, int(min(h, w) * 0.15))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=8, minLineLength=min_len, maxLineGap=20)

        best = None
        best_score = 1e9
        if lines is not None:
            for l in lines:
                if isinstance(l, (list, tuple)) and len(l) == 1 and isinstance(l[0], (list, tuple)):
                    l = l[0]
                if not (isinstance(l, (list, tuple)) and len(l) >= 4):
                    continue
                x1, y1, x2, y2 = l[:4]
                ang = _norm_angle_deg(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                alt = ang + 180.0 if ang < 0 else ang - 180.0
                cand = ang if abs(_norm_angle_deg(ang - expected_angle_deg)) < abs(_norm_angle_deg(alt - expected_angle_deg)) else alt
                diff = abs(_norm_angle_deg(cand - expected_angle_deg))
                length = np.hypot(x2 - x1, y2 - y1)
                if diff > tol_deg:
                    continue
                mid_x = 0.5 * (x1 + x2)
                mid_y = 0.5 * (y1 + y2)
                # Entry gating
                if entry == "bottom" and mid_y < h * 0.25:
                    continue
                if entry == "left" and mid_x > w * 0.75:
                    continue
                # Touch requirement
                if touch == "bottom" and max(y1, y2) < h - 3:
                    continue
                if touch == "right" and max(x1, x2) < w - 3:
                    continue
                score = diff + (0.02 * (1.0 / max(1.0, length)))
                if score < best_score:
                    best_score = score
                    best = (cand, (int(x1), int(y1), int(x2), int(y2)), length)

        # Fallback: use contour min-area rect to estimate orientation if Hough failed
        if best is None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 50:
                    rect = cv2.minAreaRect(largest)
                    (cxr, cyr), (wr, hr), angle = rect
                    # minAreaRect angle is relative; convert to our convention
                    if wr < hr:
                        angle += 90
                    cand_ang = _norm_angle_deg(angle)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    # derive a line from the longest edge
                    p0 = box[0]; p1 = box[1]
                    p2 = box[2]; p3 = box[3]
                    len01 = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
                    len12 = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
                    if len01 >= len12:
                        x1, y1, x2, y2 = p0[0], p0[1], p1[0], p1[1]
                    else:
                        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
                    length = np.hypot(x2 - x1, y2 - y1)
                    best = (cand_ang, (int(x1), int(y1), int(x2), int(y2)), length)

        return best
    def _wrap_text_cv(msg, font, scale, thick, max_w):
        words = msg.split()
        lines = []
        cur = ""
        for w in words:
            test = w if not cur else f"{cur} {w}"
            (tw, _), _ = cv2.getTextSize(test, font, scale, thick)
            if tw <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def overlay_recommendations_on_frame(frame, messages):
        """Draw correction recommendations at the bottom of a UVC frame with wrapping."""
        if frame is None or not messages:
            return
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        max_w = int(w * 0.9)
        y = h - 8
        shown = 0
        for msg in reversed(messages[-3:]):  # show up to 3 messages
            for line in reversed(_wrap_text_cv(msg, font, scale, thick, max_w)):
                cv2.putText(frame, line, (10, y), font, scale, (255, 220, 120), thick, cv2.LINE_AA)
                y -= 16
            y -= 2
            shown += 1
            if shown >= 3:
                break


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

    def persist_current_routine():
        nonlocal routines, current_routine_name, routine_editor
        text = routine_editor.text
        routines[current_routine_name] = text
        save_routines(routines)
        # Legacy single-file save for backwards compatibility
        save_routine_text(text)

    def apply_settings_dict(new_s: dict):
        """Apply loaded settings into state and refresh key cached fields."""
        with state.lock:
            state.settings = new_s.copy()
            state.tool_lz = float(new_s.get("tool_lz", DEFAULT_TOOL_LZ))
            state.tool_cx = float(new_s.get("tool_cx", DEFAULT_TOOL_CX))
            state.tool_cy = float(new_s.get("tool_cy", DEFAULT_TOOL_CY))
            state.approach_mm = float(new_s.get("approach_mm", DEFAULT_APPROACH_MM))
            state.saw_amp_deg = float(new_s.get("saw_amp_deg", 3.0))
            state.saw_freq_hz = float(new_s.get("saw_freq_hz", 1.0))
            state.idx = min(state.idx, len(state.levels) - 1)

    def cycle_routine(delta=1):
        nonlocal routines, current_routine_name, routine_editor
        names = sorted(routines.keys(), key=str.lower)
        if not names:
            routines["Default"] = ROUTINE_SAMPLE.strip()
            names = ["Default"]
        if current_routine_name not in names:
            current_routine_name = names[0]
        idx = names.index(current_routine_name)
        idx = (idx + delta) % len(names)
        current_routine_name = names[idx]
        txt = routines.get(current_routine_name, ROUTINE_SAMPLE.strip())
        routine_editor = EditorState(text=txt, caret=len(txt))
        with state.lock:
            state.routine_status_text = f"Selected routine '{current_routine_name}'"

    def run_routine_by_name(name: str):
        nonlocal routine_executor, routine_editor, current_routine_name
        if not name:
            return
        persist_current_routine()
        txt = routines.get(name, ROUTINE_SAMPLE.strip())
        current_routine_name = name
        routine_editor = EditorState(text=txt, caret=len(txt))
        if routine_executor and routine_executor.is_alive():
            routine_executor.stop()
        with state.lock:
            kw = state.settings.get("endstop_keywords", ["ENDSTOP", "LIMIT", "LIM", "HIT", "END"])
            tms = state.settings.get("endstop_timeout_ms", 0)
            vf_rate = state.settings.get("plunger_fwd_rate", MAX_RATE)
            vb_rate = state.settings.get("plunger_bwd_rate", MAX_RATE)
            wait_s = state.settings.get("pose_final_wait_s", 0.0)
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
        with state.lock:
            state.routine_phase = "RUNNING"
            state.routine_status_text = f"Running routine '{name}'"

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

    def take_step_index():
        """Use pending index from subject confirm, or increment when missing."""
        nonlocal pending_step_index
        with state.lock:
            if pending_step_index is not None:
                state.current_injection_step = pending_step_index
                step_idx = pending_step_index
            else:
                state.current_injection_step += 1
                step_idx = state.current_injection_step
        pending_step_index = None
        return step_idx

    def start_step_recording(step_index: int):
        """Begin per-step recording as soon as the subject is confirmed."""
        nonlocal current_step_recorder
        with state.lock:
            rec_tel = state.record_telemetry
            rec_vid = state.record_video
            rec_ui  = state.record_ui
            s = state.settings
            inj_meta = {
                "study":   s.get("injection_study",   ""),
                "subject": s.get("injection_subject", ""),
                "material":s.get("injection_material",""),
                "date":    s.get("injection_date",    ""),
            }

        if not (rec_tel or rec_vid or rec_ui):
            return

        if current_step_recorder:
            try:
                current_step_recorder.stop()
            except Exception:
                pass
            current_step_recorder = None

        try:
            current_step_recorder = StepRecorder(
                DATA_ROOT_DIR,
                inj_meta,
                step_index,
                record_telemetry=rec_tel,
                record_video=rec_vid,
                record_ui=rec_ui,
            )
            print(f"[Recorder] Step {step_index} recording started after subject confirm.")
        except Exception as e:
            print(f"[Recorder] Failed to create recorder: {e}")
            current_step_recorder = None

    def start_injection_session():
        nonlocal injection_info_active, injection_step_in_progress
        nonlocal injection_step_done, post_step_prompt_active
        nonlocal injection_last_step_small
        nonlocal injection_record_telemetry, injection_record_video, injection_record_ui
        nonlocal subject_prompt_active, subject_field_text, injection_subject_text
        nonlocal pending_step_index, current_step_recorder

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
        pending_step_index = None
        if current_step_recorder:
            try:
                current_step_recorder.stop()
            except Exception:
                pass
            current_step_recorder = None
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
    quit_confirm_active = False
    loop_prev_time = time.time()

    prev_a = prev_b = prev_x = prev_y = False
    prev_lb = prev_rb = False
    last_sig_axis_time = 0.0
    prev_lt = 0.0
    prev_rt = 0.0

    click_pos = None

    while running:
        click_pos = None
        tool_center_radius_rect = None
        alarm_threshold_rect = None
        alarm_duration_rect = None
        tool_center_demo_mode = getattr(state, "tool_center_demo_mode", "circle")

        # ───────────── Event pump ─────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                rt.stop_event.set()
                rt.keep_motors_on_exit = True
                break

            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                flags ^= pygame.FULLSCREEN
                screen = pygame.display.set_mode((0, 0), flags)
                continue

            if ev.type == pygame.KEYDOWN and save_mode_modal_active:
                if ev.key == pygame.K_ESCAPE:
                    save_mode_modal_active = False
                    pending_save_pose_name = ""
                elif ev.key == pygame.K_RETURN:
                    if pending_save_pose_name:
                        rt.enqueue({"type": "save_named_with_midway", "name": pending_save_pose_name, "demo": False})
                    save_mode_modal_active = False
                    pending_save_pose_name = ""
                continue

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
                if tool_center_radius_rect and tool_center_radius_rect.collidepoint(ev.pos):
                    tool_center_radius_drag = True
                    mx = ev.pos[0]
                    rel = max(0.0, min(1.0, (mx - tool_center_radius_rect.x) / max(1, tool_center_radius_rect.w)))
                    tool_center_radius = rel * 50.0
                    with state.lock:
                        state.settings["tool_center_demo_radius_mm"] = tool_center_radius
                    save_settings(state.settings)
                    continue
                if alarm_threshold_rect and alarm_threshold_rect.collidepoint(ev.pos):
                    alarm_threshold_drag = True
                    mx = ev.pos[0]
                    rel = max(0.0, min(1.0, (mx - alarm_threshold_rect.x) / max(1, alarm_threshold_rect.w)))
                    alarm_threshold_val = DEFAULT_PID_THRESHOLD_MIN + rel * (DEFAULT_PID_THRESHOLD_MAX - DEFAULT_PID_THRESHOLD_MIN)
                    alarm_threshold_val, alarm_duration_val = _clamp_alarm_pid(alarm_threshold_val, alarm_duration_val)
                    with state.lock:
                        state.settings["alarm_threshold"] = alarm_threshold_val
                        state.settings["alarm_duration"] = alarm_duration_val
                    save_settings(state.settings)
                    rt.enqueue({"type": "set_alarm_pid", "threshold": alarm_threshold_val, "duration": alarm_duration_val})
                    continue
                if alarm_duration_rect and alarm_duration_rect.collidepoint(ev.pos):
                    alarm_duration_drag = True
                    mx = ev.pos[0]
                    rel = max(0.0, min(1.0, (mx - alarm_duration_rect.x) / max(1, alarm_duration_rect.w)))
                    alarm_duration_val = DEFAULT_PID_DURATION_MIN + rel * (DEFAULT_PID_DURATION_MAX - DEFAULT_PID_DURATION_MIN)
                    alarm_threshold_val, alarm_duration_val = _clamp_alarm_pid(alarm_threshold_val, alarm_duration_val)
                    with state.lock:
                        state.settings["alarm_threshold"] = alarm_threshold_val
                        state.settings["alarm_duration"] = alarm_duration_val
                    save_settings(state.settings)
                    rt.enqueue({"type": "set_alarm_pid", "threshold": alarm_threshold_val, "duration": alarm_duration_val})
                    continue
                click_pos = ev.pos

            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                tool_center_radius_drag = False
                alarm_threshold_drag = False
                alarm_duration_drag = False

            if ev.type == pygame.MOUSEMOTION:
                if tool_center_radius_drag and tool_center_radius_rect:
                    mx = ev.pos[0]
                    rel = max(0.0, min(1.0, (mx - tool_center_radius_rect.x) / max(1, tool_center_radius_rect.w)))
                    tool_center_radius = rel * 50.0
                    with state.lock:
                        state.settings["tool_center_demo_radius_mm"] = tool_center_radius
                    save_settings(state.settings)
                if alarm_threshold_drag and alarm_threshold_rect:
                    mx = ev.pos[0]
                    rel = max(0.0, min(1.0, (mx - alarm_threshold_rect.x) / max(1, alarm_threshold_rect.w)))
                    alarm_threshold_val = DEFAULT_PID_THRESHOLD_MIN + rel * (DEFAULT_PID_THRESHOLD_MAX - DEFAULT_PID_THRESHOLD_MIN)
                    alarm_threshold_val, alarm_duration_val = _clamp_alarm_pid(alarm_threshold_val, alarm_duration_val)
                    with state.lock:
                        state.settings["alarm_threshold"] = alarm_threshold_val
                        state.settings["alarm_duration"] = alarm_duration_val
                    save_settings(state.settings)
                    rt.enqueue({"type": "set_alarm_pid", "threshold": alarm_threshold_val, "duration": alarm_duration_val})
                if alarm_duration_drag and alarm_duration_rect:
                    mx = ev.pos[0]
                    rel = max(0.0, min(1.0, (mx - alarm_duration_rect.x) / max(1, alarm_duration_rect.w)))
                    alarm_duration_val = DEFAULT_PID_DURATION_MIN + rel * (DEFAULT_PID_DURATION_MAX - DEFAULT_PID_DURATION_MIN)
                    alarm_threshold_val, alarm_duration_val = _clamp_alarm_pid(alarm_threshold_val, alarm_duration_val)
                    with state.lock:
                        state.settings["alarm_threshold"] = alarm_threshold_val
                        state.settings["alarm_duration"] = alarm_duration_val
                    save_settings(state.settings)
                    rt.enqueue({"type": "set_alarm_pid", "threshold": alarm_threshold_val, "duration": alarm_duration_val})

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

                # Tool center X
                if toolcx_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            new_cx = float(toolcx_text.strip())
                            with state.lock:
                                state.tool_cx = new_cx
                                state.settings["tool_cx"] = new_cx
                            save_settings(state.settings)
                            rt.enqueue({"type": "set_tool_center", "cx": new_cx, "cy": getattr(state, "tool_cy", DEFAULT_TOOL_CY)})
                        except ValueError:
                            print("⚠️ Invalid tool center X.")
                        toolcx_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        toolcx_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        toolcx_text = toolcx_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            toolcx_text += ch
                    continue

                # Tool center Y
                if toolcy_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            new_cy = float(toolcy_text.strip())
                            with state.lock:
                                state.tool_cy = new_cy
                                state.settings["tool_cy"] = new_cy
                            save_settings(state.settings)
                            rt.enqueue({"type": "set_tool_center", "cx": getattr(state, "tool_cx", DEFAULT_TOOL_CX), "cy": new_cy})
                        except ValueError:
                            print("⚠️ Invalid tool center Y.")
                        toolcy_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        toolcy_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        toolcy_text = toolcy_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            toolcy_text += ch
                    continue

                # Angle input (J5)
                if angle_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            val = float(angle_text.strip())
                            with state.lock:
                                mode = getattr(state, "j5_angle_mode", "actual")
                                base_ref = getattr(state, "j5_base_ref", None)
                                cur_j5 = float(getattr(state, "j5", 0.0))
                            if mode == "base":
                                if base_ref is None:
                                    base_ref = cur_j5
                                    with state.lock:
                                        state.j5_base_ref = base_ref
                                target_j5 = base_ref + val
                            else:
                                target_j5 = val
                            rt.enqueue({"type": "set_j5_angle", "j5": target_j5})
                        except ValueError:
                            print("⚠️ Invalid angle value.")
                        angle_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        angle_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        angle_text = angle_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            angle_text += ch
                    continue

                # Pitch input (degrees)
                if pitch_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            target_p = float(pitch_text.strip())
                            rt.enqueue({"type": "set_pitch_deg", "pitch": target_p})
                        except ValueError:
                            print("⚠️ Invalid pitch value.")
                        pitch_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        pitch_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        pitch_text = pitch_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            pitch_text += ch
                    continue

                # Yaw input (degrees heading)
                if yaw_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            target_yaw = float(yaw_text.strip())
                            rt.enqueue({"type": "set_yaw_heading", "yaw": target_yaw})
                        except ValueError:
                            print("⚠️ Invalid yaw value.")
                        yaw_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        yaw_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        yaw_text = yaw_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            yaw_text += ch
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

                # Injection retract distance
                if retract_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            new_r = float(retract_text.strip())
                            with state.lock:
                                state.settings["injection_retract_mm"] = new_r
                            save_settings(state.settings)
                        except ValueError:
                            print("⚠️ Invalid retract distance.")
                        retract_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        retract_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        retract_text = retract_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            retract_text += ch
                    continue

                # Routine editor
                if routine_edit_active:
                    mods = pygame.key.get_mods()
                    if ev.key == pygame.K_RETURN and (mods & pygame.KMOD_CTRL):
                        persist_current_routine()
                        routine_edit_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        routine_edit_active = False
                    else:
                        editor_handle_key(routine_editor, ev)
                    continue

                # Settings path input
                if settings_path_input_active:
                    if ev.key == pygame.K_RETURN:
                        path = settings_path_text.strip() or SETTINGS_PATH
                        settings_path_input_active = False
                        if settings_path_mode == "load":
                            new_s = load_settings(path)
                            apply_settings_dict(new_s)
                            save_settings(state.settings)
                            with state.lock:
                                state.routine_status_text = f"Loaded settings from {path}"
                        else:
                            save_settings(state.settings, path=path)
                            with state.lock:
                                state.routine_status_text = f"Saved settings to {path}"
                    elif ev.key == pygame.K_ESCAPE:
                        settings_path_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        settings_path_text = settings_path_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and ch.isprintable() and ch not in "\r\n\t":
                            settings_path_text += ch
                    continue

                # Routine name input (save as)
                if routine_name_input_active:
                    if ev.key == pygame.K_RETURN:
                        name = routine_name_text.strip()
                        routine_name_input_active = False
                        if name:
                            exists = name in routines
                            if routine_name_mode == "rename":
                                if not routine_rename_target:
                                    pass
                                elif exists and name != routine_rename_target:
                                    with state.lock:
                                        state.routine_status_text = "Name already exists; rename cancelled."
                                else:
                                    txt = routines.pop(routine_rename_target, ROUTINE_SAMPLE.strip())
                                    routines[name] = txt
                                    current_routine_name = name
                                    routine_editor = EditorState(text=txt, caret=len(txt))
                                    save_routines(routines)
                                    save_routine_text(txt)
                                    with state.lock:
                                        state.routine_status_text = f"Renamed routine to '{name}'"
                            elif routine_name_mode == "create":
                                if exists:
                                    with state.lock:
                                        state.routine_status_text = "Name already exists; choose another."
                                else:
                                    txt = ROUTINE_SAMPLE.strip()
                                    routines[name] = txt
                                    current_routine_name = name
                                    routine_editor = EditorState(text=txt, caret=len(txt))
                                    persist_current_routine()
                                    with state.lock:
                                        state.routine_status_text = f"Created routine '{name}'"
                            else:  # save_as / general save
                                current_routine_name = name
                                persist_current_routine()
                                with state.lock:
                                    state.routine_status_text = f"Saved routine '{name}'"
                        else:
                            with state.lock:
                                state.routine_status_text = "Routine save cancelled (empty name)"
                    elif ev.key == pygame.K_ESCAPE:
                        routine_name_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        routine_name_text = routine_name_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and ch.isprintable() and ch not in "\r\n\t":
                            routine_name_text += ch
                    continue

                # Tool preset name input
                if toolpreset_input_active:
                    if ev.key == pygame.K_RETURN:
                        name = toolpreset_name_text.strip()
                        toolpreset_input_active = False
                        if name:
                            rt.enqueue({"type": "save_tool_preset", "name": name})
                            try:
                                joy.rumble(0.5, 0.5, 140)
                            except Exception:
                                pass
                    elif ev.key == pygame.K_ESCAPE:
                        toolpreset_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        toolpreset_name_text = toolpreset_name_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and ch.isprintable() and ch not in "\r\n\t":
                            toolpreset_name_text += ch
                    continue

                # Apply tool correction button via Enter when focused (optional future)

                # Tool preset rename input
                if toolpreset_rename_input_active:
                    if ev.key == pygame.K_RETURN:
                        new_name = toolpreset_rename_text.strip()
                        toolpreset_rename_input_active = False
                        if new_name:
                            rt.enqueue({"type": "rename_tool_preset", "old": toolpreset_rename_old, "new": new_name})
                            try:
                                joy.rumble(0.5, 0.5, 140)
                            except Exception:
                                pass
                    elif ev.key == pygame.K_ESCAPE:
                        toolpreset_rename_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        toolpreset_rename_text = toolpreset_rename_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and ch.isprintable() and ch not in "\r\n\t":
                            toolpreset_rename_text += ch
                    continue

                # Quit confirm (keyboard)
                if quit_confirm_active:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        quit_confirm_active = False
                    elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        running = False
                    continue

                # New pose name
                if naming_input_active:
                    if ev.key == pygame.K_RETURN:
                        name = naming_text.strip()
                        naming_input_active = False
                        if name:
                            pending_save_pose_name = name
                            save_mode_modal_active = True
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

        # Manual speed scale change via LB/RB on press
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
            endstop_timeout_input_active or pose_wait_input_active or
            retract_input_active or routine_name_input_active or pitch_input_active or angle_input_active or yaw_input_active or
            toolcx_input_active or toolcy_input_active or settings_path_input_active
        )
        any_modal_active = (
            syringe_calib_active or injection_info_active or subject_prompt_active or
            final_small_warning_active or post_step_prompt_active or
            routine_edit_active or routine_action_modal_active or routine_edit_modal_active or
            edit_menu_active or routine_name_input_active or
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
                # Stay in injection mode but pause actions; require user to recalibrate
                with state.lock:
                    state.calibration_status = (
                        "Injection paused: syringe stroke/volume missing. Recalibrate to continue."
                    )
                injection_step_in_progress = False
                injection_step_done = False
                post_step_prompt_active = False
                final_small_warning_active = False
                pending_step_index = None
                if current_step_recorder:
                    try:
                        current_step_recorder.stop()
                    except Exception:
                        pass
                    current_step_recorder = None

            elif syringe_remaining_ul <= 1e-9:
                with state.lock:
                    state.settings["injection_active"] = False
                save_settings(state.settings)
                # Keep post-step prompt visible even on the final step so user can retract
                if not post_step_prompt_active:
                    injection_step_in_progress = False
                    injection_step_done = False
                final_small_warning_active = False
                pending_step_index = None
                if current_step_recorder and not post_step_prompt_active:
                    try:
                        current_step_recorder.stop()
                    except Exception:
                        pass
                    current_step_recorder = None

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

                            step_index = take_step_index()
                            if current_step_recorder is None:
                                start_step_recording(step_index)

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
            j5_angle_mode = getattr(state, "j5_angle_mode", "actual")
            j5_base_ref = getattr(state, "j5_base_ref", None)
            yaw_now = getattr(state, "yaw", 0.0)
            yaw_base = getattr(state, "yaw_base", None)
            sens = state.levels[state.idx]
            tool_lz = state.tool_lz
            tool_cx = getattr(state, "tool_cx", DEFAULT_TOOL_CX)
            tool_cy = getattr(state, "tool_cy", DEFAULT_TOOL_CY)
            approach_mm = state.approach_mm
            poses_snapshot = state.poses.copy()
            settings_snapshot = state.settings.copy()
            routine_phase = state.routine_phase
            routine_status_text = state.routine_status_text
            last_plunger_note = state.last_plunger_note or "—"
            pose_final_wait_s = float(state.settings.get("pose_final_wait_s", 0.0) or 0.0)
            saw_amp_deg = float(state.settings.get("saw_amp_deg", 3.0) or 0.0)
            saw_freq_hz = float(state.settings.get("saw_freq_hz", 1.0) or 0.0)
            saw_enabled_flag = bool(getattr(state, "saw_enabled", False))
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
            j5_test_active_now     = bool(getattr(state, "j5_test_active", False))
            j5_waiting_flag        = bool(getattr(state, "j5_test_waiting", False))
            tool_center_demo_on    = bool(getattr(state, "tool_center_demo", False))
            tool_center_demo_mode  = getattr(state, "tool_center_demo_mode", "circle")
            tool_center_demo_wait  = bool(getattr(state, "tool_center_demo_waiting", False))
            tool_center_demo_mode  = getattr(state, "tool_center_demo_mode", "circle")

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

        rs_clickables = []

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
        screen.blit(
            vf_.render("click to set", True, (180, 220, 255)),
            (x1, y1 + panel_size + padding + 16 + padding + 14),
        )
        pitch_rect = pygame.Rect(
            x1, y1,
            panel_size,
            panel_size + padding + 16 + padding + 32,
        )
        rs_clickables.append((pitch_rect, {"kind": "pitch_input"}))

        # Yaw indicator (tool Z heading) above pitch, same size/style as pitch but rotated right
        yaw_size = panel_size
        yaw_x = tl_rect.x + m
        extra_gap = padding * 4 + 12  # add breathing room for labels/text
        yaw_y = max(tl_rect.y + m, y1 - yaw_size - extra_gap)
        yaw_y = min(yaw_y, tl_rect.y + tl_h - yaw_size - m)
        yaw_ref = yaw_base if yaw_base is not None else 0.0
        yaw_offset = yaw_now - yaw_ref
        # Draw same style as pitch dial but rotated 90° to the right
        yaw_pan = pygame.Surface((yaw_size, yaw_size), pygame.SRCALPHA)
        yaw_pan.fill((10, 10, 30, 180))
        pivot_yaw = (yaw_size // 2, yaw_size - 6)  # bottom center keeps round end at bottom
        r_len_yaw = yaw_size // 2 - 6
        yaw_ang = np.radians(-90 - yaw_offset)  # 0 offset -> tip at top
        exy = pivot_yaw[0] + r_len_yaw * np.cos(yaw_ang)
        eyy = pivot_yaw[1] + r_len_yaw * np.sin(yaw_ang)
        pygame.draw.circle(yaw_pan, (0, 255, 127), (yaw_size // 2, yaw_size // 2), yaw_size // 2 - 6, 2)
        pygame.draw.line(yaw_pan, (0, 255, 127), pivot_yaw, (exy, eyy), 2)
        pygame.draw.circle(yaw_pan, (0, 255, 127), (int(exy), int(eyy)), 4)
        yaw_pan_rot = pygame.transform.rotate(yaw_pan, -90)
        rot_rect = yaw_pan_rot.get_rect(center=(yaw_x + yaw_size // 2, yaw_y + yaw_size // 2))
        screen.blit(yaw_pan_rot, rot_rect.topleft)
        yaw_label = vf_.render("Yaw (base)", True, (200, 230, 255))
        yaw_val_txt = vf_.render(f"{yaw_offset:+.3f}°", True, (0, 255, 127))
        screen.blit(yaw_label, (yaw_x, yaw_y - yaw_label.get_height() - 2))
        screen.blit(yaw_val_txt, (yaw_x + yaw_size - yaw_val_txt.get_width(),
                                  yaw_y + yaw_size + 2))
        # Place “click to set yaw” beneath dial like pitch
        set_txt_yaw = vf_.render("click to set yaw", True, (180, 220, 255))
        set_yaw_y = yaw_y + yaw_size + padding + 14
        screen.blit(set_txt_yaw, (yaw_x, set_yaw_y))
        yaw_rect = pygame.Rect(yaw_x, yaw_y, yaw_size, yaw_size + set_txt_yaw.get_height() + padding + 16)
        rs_clickables.append((yaw_rect, {"kind": "yaw_set"}))

        # J5 dial
        x2 = x1 + panel_size + diag_gap
        y2 = y1
        pan2 = pygame.Surface((panel_size, panel_size), pygame.SRCALPHA)
        pan2.fill((10, 10, 30, 180))
        center = (panel_size // 2, panel_size // 2)
        radius = panel_size // 2 - 5
        pygame.draw.circle(pan2, (0, 255, 127), center, radius, 2)
        angle_draw = j5v if j5_angle_mode == "actual" else (j5v - (j5_base_ref if j5_base_ref is not None else j5v))
        ang2 = np.radians(angle_draw - 90)
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
        angle_label = "Angle (J5)" if j5_angle_mode == "actual" else "Angle (base)"
        base_ref = j5_base_ref if j5_base_ref is not None else j5v
        angle_val = -j5v if j5_angle_mode == "actual" else -(j5v - base_ref)
        screen.blit(lf.render(angle_label, True, (0, 255, 127)),
                    (x2, y2 + panel_size + padding))
        screen.blit(
            vf_.render(f"{angle_val:+.5f}°", True, (0, 255, 127)),
            (x2, y2 + panel_size + padding + 16 + padding),
        )
        screen.blit(
            vf_.render("click to toggle", True, (180, 220, 255)),
            (x2, y2 + panel_size + padding + 16 + padding + 14),
        )
        toggle_txt = vf_.render("click to toggle", True, (180, 220, 255))
        set_txt = vf_.render("click to set", True, (220, 255, 220))
        btn_w = max(panel_size // 2, set_txt.get_width() + 12)
        btn_h = set_txt.get_height() + 8
        # place above dial to avoid overlap
        set_rect = pygame.Rect(
            x2 + panel_size - btn_w,
            y2 - btn_h - 6,
            btn_w,
            btn_h,
        )
        pygame.draw.rect(screen, (30, 50, 50), set_rect)
        pygame.draw.rect(screen, (0, 255, 127), set_rect, 2)
        screen.blit(
            set_txt,
            (set_rect.x + (set_rect.w - set_txt.get_width()) // 2,
             set_rect.y + (set_rect.h - set_txt.get_height()) // 2),
        )
        # Extend toggle hit area to include dial + toggle label
        angle_rect_h = max(panel_size, (y2 + panel_size + padding + 16 + padding + 14) - y2)
        angle_rect = pygame.Rect(x2, y2, panel_size, angle_rect_h)
        # order: set click first so it wins hit-testing over toggle (no overlap with set_rect)
        rs_clickables.append((set_rect, {"kind": "angle_set"}))
        rs_clickables.append((angle_rect, {"kind": "angle_toggle"}))

        inj_label = (
            f"Injection: {'ACTIVE' if injection_active_flag else 'idle'}; "
            f"ammo={ammo_steps} steps (retract A={inj_retract_mm_val:.1f} mm)"
        )

        texts = [
            vf_.render(f"Manual speed scale ({sens*100:.0f}%)", True, (200, 230, 255)),
            vf_.render(f"Move speed: ±{(5.0 * sens):.3f} mm/s", True, (0, 255, 127)),
            vf_.render(f"J5 speed:   ±{(5.0 * sens):.3f} deg/s", True, (0, 255, 127)),
            vf_.render(f"Approach: {approach_mm:.2f} mm", True, (200, 255, 200)),
            vf_.render(f"Tool center: X={tool_cx:.3f} Y={tool_cy:.3f} LZ={tool_lz:.2f}", True, (220, 240, 255)),
            vf_.render(f"Saw: {'ON' if saw_enabled_flag else 'OFF'} amp={saw_amp_deg:.2f}° freq={saw_freq_hz:.2f} Hz", True, (255, 220, 200)),
            vf_.render(
                f"Tool center demo: "
                f"{'ON (' + getattr(state, 'tool_center_demo_mode', 'circle') + ')' if getattr(state, 'tool_center_demo', False) else 'OFF'}",
                True,
                (220, 230, 255)
            ),
            vf_.render(f"Final dwell: {pose_final_wait_s:.2f} s", True, (220, 220, 255)),
            vf_.render(f"Routine: {current_routine_name}", True, (220, 220, 255)),
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

        # Transient notice (e.g., failed unreserve)
        if ui_notice_text and time.time() < ui_notice_until:
            pad = 8
            notice_font = pygame.font.SysFont("Consolas", 16, bold=True)
            surf = notice_font.render(ui_notice_text, True, (255, 200, 120))
            bg = pygame.Surface((surf.get_width() + pad * 2, surf.get_height() + pad * 2), pygame.SRCALPHA)
            bg.fill((40, 10, 10, 200))
            nx = tl_rect.x + (tl_rect.w - bg.get_width()) // 2
            ny = tl_rect.y + 10
            screen.blit(bg, (nx, ny))
            screen.blit(surf, (nx + pad, ny + pad))
        elif ui_notice_text and time.time() >= ui_notice_until:
            ui_notice_text = ""
            ui_notice_until = 0.0

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
        if not alarm_threshold_drag:
            with state.lock:
                alarm_threshold_val = float(state.settings.get("alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN))
        if not alarm_duration_drag:
            with state.lock:
                alarm_duration_val = float(state.settings.get("alarm_duration", DEFAULT_PID_DURATION_MAIN))
        buttons = make_buttons(
            tool_lz,
            approach_mm,
            poses_snapshot,
            settings_snapshot,
            routines,
            current_routine_name,
            getattr(state, "tool_center_demo", False),
            getattr(state, "tool_center_demo_mode", "circle"),
            tool_center_radius,
            alarm_threshold_val,
            alarm_duration_val,
        )

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
            if kind == "alarm_status":
                surf_txt = placeholder_font.render(
                    b["label"],
                    True,
                    (255, 220, 120) if getattr(state, "alarm_armed", False) else (180, 220, 255),
                )
                screen.blit(surf_txt, (tr_rect.x + m_ui, y_off_btn))
                bh = surf_txt.get_height()
                total_h += bh + btn_gap
                y_off_btn += bh + btn_gap
                continue

            if kind == "tool_preset_row":
                nm = b.get("preset", "")
                info_lines = wrap_text(b.get("label", ""), placeholder_font, btn_w - 10 - 90)
                btn_w_edit = min(90, max(70, int(btn_w * 0.28)))
                main_w = btn_w - btn_w_edit - 6
                main_rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, main_w, 34)
                pygame.draw.rect(screen, (40, 40, 60), main_rect)
                pygame.draw.rect(screen, (0, 255, 127), main_rect, 2)
                ly2 = main_rect.y + 4
                for ln in info_lines:
                    txt = placeholder_font.render(ln, True, (220, 220, 230))
                    screen.blit(txt, (main_rect.x + 6, ly2))
                    ly2 += placeholder_font.get_height()
                edit_rect = pygame.Rect(main_rect.right + 6, y_off_btn, btn_w_edit, 34)
                pygame.draw.rect(screen, (55, 55, 85), edit_rect)
                pygame.draw.rect(screen, (0, 255, 127), edit_rect, 2)
                etxt = placeholder_font.render("Edit", True, (230, 230, 230))
                screen.blit(
                    etxt,
                    (edit_rect.x + (edit_rect.w - etxt.get_width()) // 2,
                     edit_rect.y + (edit_rect.h - etxt.get_height()) // 2),
                )
                ui_clickables.append((main_rect, {"kind": "tool_preset_apply", "preset": nm}))
                ui_clickables.append((edit_rect, {"kind": "tool_preset_menu", "preset": nm}))
                y_off_btn += 34 + btn_gap
                total_h += 34 + btn_gap
                continue
            if kind in ("tool_center_demo_circle", "tool_center_demo_vert", "tool_center_demo_horz"):
                label = b["label"]
                if kind == "tool_center_demo_circle":
                    demo_on = bool(getattr(state, "tool_center_demo", False) and tool_center_demo_mode == "circle")
                elif kind == "tool_center_demo_vert":
                    demo_on = bool(getattr(state, "tool_center_demo", False) and tool_center_demo_mode in ("vertical","oscillate_vertical"))
                else:
                    demo_on = bool(getattr(state, "tool_center_demo", False) and tool_center_demo_mode in ("horizontal","oscillate_horizontal"))
                btn_txt_lines = wrap_text(label, placeholder_font, btn_w - 10)
                bh = len(btn_txt_lines) * (placeholder_font.get_height() + 2) + 10
                rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, btn_w, bh)
                pygame.draw.rect(screen, (40, 40, 60), rect)
                pygame.draw.rect(screen, (0, 255, 127) if demo_on else (0, 200, 200), rect, 2)
                ly2 = y_off_btn + 5
                for ln in btn_txt_lines:
                    surf_txt = placeholder_font.render(ln, True, (220, 220, 230))
                    tx = tr_rect.x + m_ui + (btn_w - surf_txt.get_width()) // 2
                    screen.blit(surf_txt, (tx, ly2))
                    ly2 += placeholder_font.get_height() + 2
                ui_clickables.append((rect, {"kind": kind}))
                y_off_btn += bh + btn_gap
                total_h   += bh + btn_gap
                continue
            if kind == "tool_center_j5_test":
                label = b["label"]
                test_on = bool(getattr(state, "j5_test_active", False))
                btn_txt_lines = wrap_text(label, placeholder_font, btn_w - 10)
                bh = len(btn_txt_lines) * (placeholder_font.get_height() + 2) + 10
                rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, btn_w, bh)
                pygame.draw.rect(screen, (40, 40, 60), rect)
                pygame.draw.rect(screen, (0, 255, 127) if test_on else (0, 200, 200), rect, 2)
                ly2 = y_off_btn + 5
                for ln in btn_txt_lines:
                    surf_txt = placeholder_font.render(ln, True, (220, 220, 230))
                    tx = tr_rect.x + m_ui + (btn_w - surf_txt.get_width()) // 2
                    screen.blit(surf_txt, (tx, ly2))
                    ly2 += placeholder_font.get_height() + 2
                ui_clickables.append((rect, {"kind": "tool_center_j5_test"}))
                y_off_btn += bh + btn_gap
                total_h   += bh + btn_gap
                continue

            if kind == "tool_center_radius":
                track_w = btn_w
                track_h = 10
                track_rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, track_w, 40)
                pygame.draw.rect(screen, (35, 35, 50), track_rect)
                pygame.draw.rect(screen, (0, 255, 127), track_rect, 2)
                bar_rect = pygame.Rect(track_rect.x + 10, track_rect.centery - track_h//2, track_w - 20, track_h)
                pygame.draw.rect(screen, (70, 70, 90), bar_rect)
                max_radius = 50.0
                radius_val = max(0.0, min(max_radius, tool_center_radius))
                knob_x = bar_rect.x + int((radius_val / max_radius) * bar_rect.w)
                knob = pygame.Rect(knob_x - 6, bar_rect.centery - 8, 12, 16)
                pygame.draw.rect(screen, (0, 255, 200), knob)
                val_txt = placeholder_font.render(f"{radius_val:.1f} deg", True, (220, 230, 240))
                screen.blit(val_txt, (track_rect.x + (track_w - val_txt.get_width()) // 2, track_rect.y + track_rect.h - val_txt.get_height() - 4))
                ui_clickables.append((bar_rect, {"kind": "tool_center_radius", "rect": bar_rect}))
                tool_center_radius_rect = bar_rect
                y_off_btn += track_rect.h + btn_gap
                total_h += track_rect.h + btn_gap
                continue

            if kind in ("alarm_threshold", "alarm_duration"):
                track_w = btn_w
                track_h = 10
                track_rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, track_w, 40)
                pygame.draw.rect(screen, (35, 35, 50), track_rect)
                pygame.draw.rect(screen, (0, 200, 200), track_rect, 2)
                bar_rect = pygame.Rect(track_rect.x + 10, track_rect.centery - track_h//2, track_w - 20, track_h)
                pygame.draw.rect(screen, (70, 70, 90), bar_rect)
                if kind == "alarm_threshold":
                    min_val = DEFAULT_PID_THRESHOLD_MIN
                    max_val = DEFAULT_PID_THRESHOLD_MAX
                    cur_val = max(min_val, min(max_val, alarm_threshold_val))
                    alarm_label = f"{int(round(cur_val))}"
                else:
                    min_val = DEFAULT_PID_DURATION_MIN
                    max_val = DEFAULT_PID_DURATION_MAX
                    cur_val = max(min_val, min(max_val, alarm_duration_val))
                    alarm_label = f"{int(round(cur_val))}"
                rel_val = (cur_val - min_val) / max(1e-6, (max_val - min_val))
                knob_x = bar_rect.x + int(rel_val * bar_rect.w)
                knob = pygame.Rect(knob_x - 6, bar_rect.centery - 8, 12, 16)
                pygame.draw.rect(screen, (0, 255, 127), knob)
                val_txt = placeholder_font.render(alarm_label, True, (220, 230, 240))
                screen.blit(val_txt, (track_rect.x + (track_w - val_txt.get_width()) // 2, track_rect.y + track_rect.h - val_txt.get_height() - 4))
                ui_clickables.append((bar_rect, {"kind": kind, "rect": bar_rect}))
                if kind == "alarm_threshold":
                    alarm_threshold_rect = bar_rect
                else:
                    alarm_duration_rect = bar_rect
                y_off_btn += track_rect.h + btn_gap
                total_h += track_rect.h + btn_gap
                continue

            if kind == "routine_row":
                rname = b["routine"]
                is_current = bool(b.get("current"))
                edit_btn_w = min(90, max(70, int(btn_w * 0.28)))
                use_w = btn_w - edit_btn_w - 6
                use_rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, use_w, 34)
                pygame.draw.rect(screen, (40, 50, 60), use_rect)
                border_col = (0, 255, 180) if is_current else (0, 200, 200)
                pygame.draw.rect(screen, border_col, use_rect, 2)
                use_txt_lines = wrap_text(
                    f"{'Current' if is_current else 'Use'}: {rname}",
                    placeholder_font,
                    use_w - 10
                )
                ly2 = use_rect.y + 4
                for ln in use_txt_lines:
                    use_txt = placeholder_font.render(ln, True, (220, 240, 240))
                    screen.blit(use_txt, (use_rect.x + 8, ly2))
                    ly2 += placeholder_font.get_height()
                ui_clickables.append((use_rect, {"kind": "routine_select", "routine": rname}))

                edit_rect = pygame.Rect(use_rect.right + 6, y_off_btn, edit_btn_w, 34)
                pygame.draw.rect(screen, (55, 55, 85), edit_rect)
                pygame.draw.rect(screen, (0, 255, 127), edit_rect, 2)
                etxt = placeholder_font.render("Edit", True, (230, 230, 230))
                screen.blit(
                    etxt,
                    (edit_rect.x + (edit_rect.w - etxt.get_width()) // 2,
                     edit_rect.y + (edit_rect.h - etxt.get_height()) // 2),
                )
                ui_clickables.append((edit_rect, {"kind": "routine_edit_row", "routine": rname}))

                y_off_btn += 34 + btn_gap
                total_h += 34 + btn_gap
                continue

            if kind == "pose_row":
                name = b["pose"]
                reserved_pose = bool(b.get("reserved"))
                edit_btn_w = min(90, max(70, int(btn_w * 0.28)))
                go_w = btn_w - edit_btn_w - 6
                go_rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, go_w, 34)
                pygame.draw.rect(screen, (40, 40, 60), go_rect)
                pygame.draw.rect(screen, (0, 255, 127), go_rect, 2)
                txt_color = (255, 220, 120) if reserved_pose else (220, 220, 230)
                go_txt_lines = wrap_text(
                    f"Go: {name}", placeholder_font, go_w - 10
                )
                ly2 = go_rect.y + 4
                for ln in go_txt_lines:
                    go_txt = placeholder_font.render(ln, True, txt_color)
                    screen.blit(go_txt, (go_rect.x + 8, ly2))
                    ly2 += placeholder_font.get_height()
                ui_clickables.append((go_rect, {"kind": "goto", "pose": name}))

                edit_rect = pygame.Rect(go_rect.right + 6, y_off_btn, edit_btn_w, 34)
                pygame.draw.rect(screen, (55, 55, 85), edit_rect)
                pygame.draw.rect(screen, (0, 255, 127), edit_rect, 2)
                etxt = placeholder_font.render(
                    "Edit", True, (255, 220, 120) if reserved_pose else (230, 230, 230)
                )
                screen.blit(
                    etxt,
                    (edit_rect.x + (edit_rect.w - etxt.get_width()) // 2,
                     edit_rect.y + (edit_rect.h - etxt.get_height()) // 2),
                )
                ui_clickables.append(
                    (edit_rect, {"kind": "edit_menu", "pose": name, "reserved": reserved_pose})
                )

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
        uvc_clickables = list(rs_clickables)

        f2_raw = uvc2.latest() if uvc2 else None
        f2 = transform_uvc_frame(f2_raw, u2_rot, u2_hf, u2_vf) if f2_raw is not None else None
        # BL camera: needle enters from bottom, tip moves upward across frame
        tip_h_norm = detect_needle_tip(f2, entry_side="bottom") if f2 is not None else None

        f1_raw = uvc1.latest() if uvc1 else None
        f1 = transform_uvc_frame(f1_raw, u1_rot, u1_hf, u1_vf) if f1_raw is not None else None
        tip_v_norm = detect_needle_tip(f1, entry_side="auto") if f1 is not None else None

        pivot_h = pivot_v = None
        pivot_offset_h = pivot_offset_v = None

        # Manage traces for demo modes
        if not tool_center_demo_on:
            needle_trace_h.clear()
            needle_trace_v.clear()
            needle_lines_h.clear()
            needle_lines_v.clear()
            needle_line_count_h = 0
            needle_line_count_v = 0
            needle_line_acc_h[:] = 0.0
            needle_line_acc_v[:] = 0.0
            last_trace_mode_h = ""
            last_trace_mode_v = ""
        else:
            if tool_center_demo_mode != last_trace_mode_h and tool_center_demo_mode == "horizontal":
                needle_trace_h.clear()
                needle_lines_h.clear()
                needle_line_count_h = 0
                needle_line_acc_h[:] = 0.0
            if tool_center_demo_mode != last_trace_mode_v and tool_center_demo_mode == "vertical":
                needle_trace_v.clear()
                needle_lines_v.clear()
                needle_line_count_v = 0
                needle_line_acc_v[:] = 0.0
            if tool_center_demo_mode != "horizontal":
                last_trace_mode_h = ""
            if tool_center_demo_mode != "vertical":
                last_trace_mode_v = ""

        if tool_center_demo_on and tool_center_demo_mode == "horizontal":
            if tip_h_norm:
                needle_trace_h.append(tip_h_norm)
                last_trace_mode_h = "horizontal"
            exp_ang_h = expected_needle_angle(1, u2_rot, u2_hf, u2_vf)
            det_line_h = detect_needle_line(f2, exp_ang_h, tol_deg=45.0, entry="bottom", touch="any") if f2 is not None else None
            if det_line_h:
                ang_h, (x1, y1, x2, y2), ln_h = det_line_h
                ext_h = extend_line_to_frame(f2.shape, (x1, y1, x2, y2)) if f2 is not None else None
                if ext_h:
                    x1e, y1e, x2e, y2e = ext_h
                else:
                    x1e, y1e, x2e, y2e = x1, y1, x2, y2
                needle_line_count_h += 1
                # Accumulate and keep/draw only the average of each batch of 100 detections
                needle_line_acc_h += np.array([x1e, y1e, x2e, y2e], dtype=float)
                if needle_line_count_h % 100 == 0:
                    avg_line = (needle_line_acc_h / 100.0).tolist()
                    needle_line_acc_h[:] = 0.0
                    needle_lines_h.append(tuple(avg_line))
                    if len(needle_lines_h) > 500:
                        needle_lines_h = needle_lines_h[-500:]
                    cv2.line(f2, (int(avg_line[0]), int(avg_line[1])), (int(avg_line[2]), int(avg_line[3])), (0, 255, 0), 1)

        if tool_center_demo_on and tool_center_demo_mode == "vertical":
            if tip_v_norm:
                needle_trace_v.append(tip_v_norm)
                last_trace_mode_v = "vertical"
            exp_ang_v = expected_needle_angle(2, u1_rot, u1_hf, u1_vf)
            det_line_v = detect_needle_line(f1, exp_ang_v, tol_deg=45.0, entry="any", touch="any") if f1 is not None else None
            if det_line_v:
                ang_v, (x1, y1, x2, y2), ln_v = det_line_v
                ext_v = extend_line_to_frame(f1.shape, (x1, y1, x2, y2)) if f1 is not None else None
                if ext_v:
                    x1e, y1e, x2e, y2e = ext_v
                else:
                    x1e, y1e, x2e, y2e = x1, y1, x2, y2
                needle_line_count_v += 1
                # Accumulate and keep/draw only the average of each batch of 100 detections
                needle_line_acc_v += np.array([x1e, y1e, x2e, y2e], dtype=float)
                if needle_line_count_v % 100 == 0:
                    avg_line = (needle_line_acc_v / 100.0).tolist()
                    needle_line_acc_v[:] = 0.0
                    needle_lines_v.append(tuple(avg_line))
                    if len(needle_lines_v) > 500:
                        needle_lines_v = needle_lines_v[-500:]
                    cv2.line(f1, (int(avg_line[0]), int(avg_line[1])), (int(avg_line[2]), int(avg_line[3])), (0, 255, 0), 1)

        # Pivot estimation from accumulated lines (only relevant during demos)
        if tool_center_demo_on and tool_center_demo_mode == "horizontal" and needle_lines_h and f2 is not None:
            # Draw all accumulated lines across the full frame so intersections stay visible
            draw_persistent_lines(f2, needle_lines_h, color=(0, 200, 255), thickness=1)
            # Highlight intersections of all line pairs (green dots)
            pts_int = intersections_from_lines(needle_lines_h, f2.shape)
            for px, py in pts_int:
                cv2.circle(f2, (int(px), int(py)), 4, (0, 255, 0), -1)
            pivot_h = estimate_pivot_from_lines(needle_lines_h)
            if pivot_h:
                cv2.circle(f2, (int(pivot_h[0]), int(pivot_h[1])), 5, (0, 0, 255), -1)
                pivot_offset_h = (pivot_h[0] - f2.shape[1] / 2.0, pivot_h[1] - f2.shape[0] / 2.0)

        if tool_center_demo_on and tool_center_demo_mode == "vertical" and needle_lines_v and f1 is not None:
            # Draw all accumulated lines across the full frame so intersections stay visible
            draw_persistent_lines(f1, needle_lines_v, color=(0, 200, 255), thickness=1)
            # Highlight intersections of all line pairs (green dots)
            pts_int_v = intersections_from_lines(needle_lines_v, f1.shape)
            for px, py in pts_int_v:
                cv2.circle(f1, (int(px), int(py)), 4, (0, 255, 0), -1)
            pivot_v = estimate_pivot_from_lines(needle_lines_v)
            if pivot_v:
                cv2.circle(f1, (int(pivot_v[0]), int(pivot_v[1])), 5, (0, 0, 255), -1)
                pivot_offset_v = (pivot_v[0] - f1.shape[1] / 2.0, pivot_v[1] - f1.shape[0] / 2.0)

        # Snapshot traces when waiting so we can compare with next cycle
        if tool_center_demo_wait:
            if needle_trace_h and not trace_snap_h_done:
                needle_trace_prev_h = needle_trace_h.copy()
                trace_snap_h_done = True
            if needle_trace_v and not trace_snap_v_done:
                needle_trace_prev_v = needle_trace_v.copy()
                trace_snap_v_done = True
        else:
            trace_snap_h_done = False
            trace_snap_v_done = False

        # Needle tracking during tool rotation test (overlay on UVC frames)
        rec_msgs_by_cam = {"uvc1": [], "uvc2": []}
        motion_avg_by_cam = {"uvc1": 0.0, "uvc2": 0.0}
        detected_now = {"uvc1": False, "uvc2": False}
        # Reset histories on request from rotation thread
        with state.lock:
            reset_hist = bool(getattr(state, "j5_reset_history", False))
            if reset_hist:
                state.j5_reset_history = False
        if reset_hist:
            shift_history = {"uvc1": [], "uvc2": []}
            motion_stats_by_cam = {
                "uvc1": {"sum": 0.0, "count": 0},
                "uvc2": {"sum": 0.0, "count": 0},
            }
            motion_max_shifts = {"uvc1": 0.0, "uvc2": 0.0}
            with state.lock:
                state.j5_max_shift_left = 0.0
                state.j5_max_shift_right = 0.0
                state.j5_start_shift_left = None
                state.j5_start_shift_right = None
                state.j5_end_shift_left = None
                state.j5_end_shift_right = None
                state.j5_last_dev = None
                state.j5_start_shift_left = None
                state.j5_start_shift_right = None
        if j5_test_active_now:
            # Reset baseline on rising edge
            if not prev_j5_test_active:
                needle_baseline["uvc1"] = None
                needle_baseline["uvc2"] = None
                needle_guidance_until = 0.0
                motion_stats_by_cam = {
                    "uvc1": {"sum": 0.0, "count": 0},
                    "uvc2": {"sum": 0.0, "count": 0},
                }
                shift_history = {"uvc1": [], "uvc2": []}
                motion_med_shifts = {"uvc1": 0.0, "uvc2": 0.0}
                rotation_measure = {"uvc1": None, "uvc2": None}
            # Map by layout: bottom-left = UVC2 (left camera, vertical), bottom-right = UVC1 (right camera, horizontal)
            cam_params = [
                ("uvc2", f2, u2_rot, u2_hf, u2_vf, 1, "bottom", "bottom"),  # left cam: use displayed transforms
                ("uvc1", f1, u1_rot, u1_hf, u1_vf, 2, "left", "right"),     # right cam: enters from left
            ]
            for key, frame_ref, rot_val, hf_val, vf_val, idx, entry_side, touch_edge in cam_params:
                if frame_ref is None:
                    continue
                det = None
                # For left UVC, auto-flip horizontally for detection and force 0 rotation so vertical stays vertical
                if key == "uvc2":
                    frame_proc = cv2.flip(frame_ref, 1)
                    exp_ang = 90.0  # vertical
                    tol = 45.0
                    w_det = frame_proc.shape[1]
                    det = detect_needle_line(frame_proc, exp_ang, tol_deg=tol, entry=entry_side, touch=touch_edge)
                else:
                    frame_proc = frame_ref
                    exp_ang = expected_needle_angle(idx, rot_val, hf_val, vf_val)
                    tol = 30.0 if idx == 2 else 45.0
                    w_det = frame_proc.shape[1]
                    det = detect_needle_line(frame_proc, exp_ang, tol_deg=tol, entry=entry_side, touch=touch_edge)
                if det:
                    ang, (x1, y1, x2, y2), ln = det
                    if key == "uvc2":
                        # map back to unflipped coords for overlay
                        x1 = w_det - 1 - x1
                        x2 = w_det - 1 - x2
                        ang = -ang  # flip reverses sign
                    if touch_edge == "right":
                        x1, y1, x2, y2 = extend_line_to_edge(frame_ref.shape, (x1, y1, x2, y2), "right")
                    elif touch_edge == "bottom":
                        x1, y1, x2, y2 = extend_line_to_edge(frame_ref.shape, (x1, y1, x2, y2), "bottom")
                    det = (ang, (x1, y1, x2, y2), ln)
                    prev_line = needle_last.get(key)
                    needle_last[key] = det
                    if needle_baseline[key] is None:
                        needle_baseline[key] = det
                    detected_now[key] = True
                    # Track frame-to-frame translation perpendicular to previous line
                    shift_perp = 0.0
                    shift_thr = 0.05  # sensitive to tiny movements
                    if prev_line is not None:
                        px1, py1, px2, py2 = prev_line[1]
                        pcx = 0.5 * (px1 + px2)
                        pcy = 0.5 * (py1 + py2)
                        vx = px2 - px1
                        vy = py2 - py1
                        vlen = math.hypot(vx, vy)
                        if vlen > 1e-3:
                            nx = -vy / vlen
                            ny = vx / vlen
                            cx = 0.5 * (x1 + x2)
                            cy = 0.5 * (y1 + y2)
                            shift_perp = (cx - pcx) * nx + (cy - pcy) * ny
                    # Accumulate average direction for persistence
                    stats = motion_stats_by_cam[key]
                    stats["sum"] += shift_perp
                    stats["count"] += 1
                    avg_shift = stats["sum"] / max(1, stats["count"])
                    motion_avg_by_cam[key] = avg_shift
                    last_shift_by_cam[key] = shift_perp
                    if abs(shift_perp) >= shift_thr:
                        last_nonzero_shifts[key] = shift_perp
                    # Track max |shift| during motion (only when both cams detected)
                    if not j5_waiting_flag and detected_now["uvc1"] and detected_now["uvc2"]:
                        motion_max_shifts[key] = max(motion_max_shifts.get(key, 0.0), abs(shift_perp))
                        with state.lock:
                            if key == "uvc2":
                                state.j5_max_shift_left = motion_max_shifts[key]
                            else:
                                state.j5_max_shift_right = motion_max_shifts[key]
                    with state.lock:
                        # capture first and latest shifts per cam for net displacement
                        if key == "uvc2":
                            if state.j5_start_shift_left is None:
                                state.j5_start_shift_left = shift_perp
                            state.j5_end_shift_left = shift_perp
                        else:
                            if state.j5_start_shift_right is None:
                                state.j5_start_shift_right = shift_perp
                            state.j5_end_shift_right = shift_perp
                    # Track trend history (cap to recent N)
                    hist = shift_history[key]
                    hist.append(shift_perp)
                    if len(hist) > 120:
                        del hist[0:len(hist)-120]
                    # Console log needle line + shift occasionally for debugging
                    now_log = time.time()
                    if now_log - last_shift_log[key] > 0.2:
                        last_shift_log[key] = now_log
                        print(f"[Needle][{key}] line=({x1},{y1})-({x2},{y2}) shift_perp={shift_perp:+.3f}px")
                    # Map to rotation_measure for downstream correction
                if key == "uvc2":  # left feed: perpendicular corresponds to horizontal motion
                    rotation_measure[key] = (avg_shift, 0.0)
                else:             # right feed: perpendicular corresponds to vertical motion
                    rotation_measure[key] = (0.0, avg_shift)
                # Update motion medians during movement (not waiting) when both cams detect
                if not j5_waiting_flag and detected_now["uvc1"] and detected_now["uvc2"]:
                    med_left = median_or_default(shift_history["uvc2"], last_good_shifts["uvc2"])
                    med_right = median_or_default(shift_history["uvc1"], last_good_shifts["uvc1"])
                    motion_med_shifts = {"uvc2": med_left, "uvc1": med_right}
                    with state.lock:
                        state.j5_motion_med_left = med_left
                        state.j5_motion_med_right = med_right
                    cv2.line(frame_ref, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(frame_ref, (cx, cy), 4, (0, 0, 255), -1)
                    # Update motion medians from histories when available
                    if shift_history["uvc1"] and shift_history["uvc2"]:
                        med_left = median_or_default(shift_history["uvc2"], last_good_shifts["uvc2"])
                        med_right = median_or_default(shift_history["uvc1"], last_good_shifts["uvc1"])
                        motion_med_shifts = {"uvc2": med_left, "uvc1": med_right}
                        with state.lock:
                            state.j5_motion_med_left = med_left
                            state.j5_motion_med_right = med_right
                else:
                    # fallback to last detected line to keep overlay visible if it moves out partially
                    if needle_last.get(key):
                        ang, (x1, y1, x2, y2), ln = needle_last[key]
                        cv2.line(frame_ref, (x1, y1), (x2, y2), (0, 180, 180), 2)
                    else:
                        needle_last[key] = None
                        cv2.putText(frame_ref, "Needle line not detected", (10, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            prev_j5_test_active = True
            # Reset per-cycle auto-apply state on waiting edge
            if j5_waiting_flag and not prev_j5_wait_flag:
                j5_auto_apply_sent = False
            prev_j5_wait_flag = j5_waiting_flag

            # Auto-apply if toggle is on and we are paused at +45°
            # Persist guidance for a short window
            now_t = time.time()
            if j5_waiting_flag:
                # Use motion-era medians (captured during movement), not pause readings
                med_left = motion_med_shifts.get("uvc2", last_good_shifts["uvc2"])
                med_right = motion_med_shifts.get("uvc1", last_good_shifts["uvc1"])
                thr_dec = 0.02
                # If medians are too small, fall back to last non-zero signals
                if abs(med_left) < thr_dec and abs(last_nonzero_shifts["uvc2"]) >= thr_dec:
                    med_left = last_nonzero_shifts["uvc2"]
                if abs(med_right) < thr_dec and abs(last_nonzero_shifts["uvc1"]) >= thr_dec:
                    med_right = last_nonzero_shifts["uvc1"]
                dx_dec, dy_dec, reason_dec = compute_correction(
                    med_left, med_right, step_mm=0.5, thr=thr_dec
                )
                last_decision_msgs = [
                    f"Decision: med L={med_left:+.2f}px R={med_right:+.2f}px -> ΔX={dx_dec:+.2f}mm ΔY={dy_dec:+.2f}mm ({reason_dec})"
                ]
                # Record shifts for robot thread (use medians) and store as last good when we have a clear decision
                last_rec_shifts = {"uvc2": med_left, "uvc1": med_right}
                with state.lock:
                    state.j5_last_shift_x = float(med_left)
                    state.j5_last_shift_y = float(med_right)
                if reason_dec != "no clear shift":
                    last_good_shifts = {"uvc2": med_left, "uvc1": med_right}
                needle_guidance_until = max(needle_guidance_until, now_t + 3.0)
        if needle_correction_note and now_t < needle_correction_until:
            pass  # drawn after UVC blits
        else:
            prev_j5_test_active = False
            rotation_measure = {"uvc1": None, "uvc2": None}

            # Always overlay latest recommendations directly onto UVC feeds
            # Left feed (f2) shows left-cam guidance; right feed (f1) shows right-cam guidance
            overlay_recommendations_on_frame(f2, last_decision_msgs)
            overlay_recommendations_on_frame(f1, last_decision_msgs)

        # Persist recent guidance overlays even when no new detections
        now_for_overlay = time.time()
        if now_for_overlay < needle_guidance_until:
            overlay_recommendations_on_frame(f2, last_decision_msgs)
            overlay_recommendations_on_frame(f1, last_decision_msgs)

        # Plot combined decision history on both UVC frames
        with state.lock:
            dev_hist = list(getattr(state, "j5_dev_history", []))
        draw_dev_plot(f2, dev_hist)
        draw_dev_plot(f1, dev_hist)

        if not blit_frame_cover(screen, f2, bl_rect):
            draw_placeholder(
                screen, bl_rect,
                title=(uvc2.status() if uvc2 else "UVC#2 not connected"),
                value_font=value_font, big_font=big_font
            )
        for r, op in draw_uvc_controls(screen, bl_rect, u2_rot, u2_hf, u2_vf, small_font):
            uvc_clickables.append((r, {"cam": 2, "op": op}))

        if not blit_frame_cover(screen, f1, br_rect):
            draw_placeholder(
                screen, br_rect,
                title=(uvc1.status() if uvc1 else "UVC#1 not connected"),
                value_font=value_font, big_font=big_font
            )
        for r, op in draw_uvc_controls(screen, br_rect, u1_rot, u1_hf, u1_vf, small_font):
            uvc_clickables.append((r, {"cam": 1, "op": op}))

        # Display pivot estimates for oscillation demos (intersection of needle lines)
        def draw_pivot_label(rect, pivot, offset, label):
            if not rect:
                return
            lines = []
            if pivot and offset:
                dx, dy = offset
                lines.append(f"{label} pivot dx={dx:+.1f}px dy={dy:+.1f}px")
                lines.append("Intersection away from center = tool center offset")
            else:
                lines.append(f"{label} pivot: not detected")
            y = rect.y + 8
            for ln in lines:
                surf = small_font.render(ln, True, (0, 255, 127))
                screen.blit(surf, (rect.x + 10, y))
                y += surf.get_height() + 2

        if tool_center_demo_on and tool_center_demo_mode == "horizontal":
            draw_pivot_label(bl_rect, pivot_h, pivot_offset_h, "Horizontal demo")
        if tool_center_demo_on and tool_center_demo_mode == "vertical":
            draw_pivot_label(br_rect, pivot_v, pivot_offset_v, "Vertical demo")

        # Resume button when waiting between cycles
        if tool_center_demo_wait and tool_center_demo_on:
            btn_w = min(200, bl_rect.w - 20)
            btn_h = 28
            btn_rect = pygame.Rect(bl_rect.x + 10, bl_rect.y + 10, btn_w, btn_h)
            pygame.draw.rect(screen, (50, 60, 90), btn_rect)
            pygame.draw.rect(screen, (0, 255, 127), btn_rect, 2)
            txt = small_font.render("Resume demo", True, (230, 230, 230))
            screen.blit(txt, (btn_rect.x + (btn_w - txt.get_width()) // 2,
                              btn_rect.y + (btn_h - txt.get_height()) // 2))
            uvc_clickables.append((btn_rect, {"kind": "demo_resume"}))

        # Draw traces / arcs on top of UVC views
        def draw_trace(trace_list, prev_trace, frame, rect, label_prefix, axis_hint, mode_hint=None, cross_offsets=None):
            if frame is None or not trace_list:
                return
            pts_prev = [map_norm_to_rect(p, frame.shape, rect) for p in (prev_trace or [])]
            pts_prev = [p for p in pts_prev if p is not None]
            if len(pts_prev) >= 2:
                pygame.draw.lines(screen, (120, 120, 120), False, pts_prev, 1)
            for p in pts_prev[-6:]:
                pygame.draw.circle(screen, (150, 150, 150), p, 3, 1)

            pts_screen = [map_norm_to_rect(p, frame.shape, rect) for p in trace_list]
            pts_screen = [p for p in pts_screen if p is not None]
            # Color the current arc based on magnitude of drift
            arr = np.array(trace_list, dtype=float)
            avg = np.mean(arr, axis=0)
            dx_px = (avg[0] - 0.5) * frame.shape[1]
            dy_px = (avg[1] - 0.5) * frame.shape[0]
            mag = np.sqrt(dx_px * dx_px + dy_px * dy_px)
            if mag < 2:
                arc_color = (0, 200, 120)   # small drift
            elif mag < 6:
                arc_color = (255, 200, 80)  # moderate
            else:
                arc_color = (255, 100, 80)  # large

            if len(pts_screen) >= 2:
                pygame.draw.lines(screen, arc_color, False, pts_screen, 2)
            for p in pts_screen[-6:]:
                pygame.draw.circle(screen, arc_color, p, 3)
            # Target crosshair at center
            ctr = map_norm_to_rect((0.5, 0.5), frame.shape, rect)
            if ctr:
                pygame.draw.circle(screen, (255, 200, 120), ctr, 5, 1)
                pygame.draw.line(screen, (255, 200, 120), (ctr[0]-8, ctr[1]), (ctr[0]+8, ctr[1]), 1)
                pygame.draw.line(screen, (255, 200, 120), (ctr[0], ctr[1]-8), (ctr[0], ctr[1]+8), 1)

            info_lines = [
                f"{label_prefix} offset: dx={dx_px:+.1f}px dy={dy_px:+.1f}px",
                f"Suggest: {axis_hint}",
            ]
            # Mode-based nudges
            suggestions = []
            if mode_hint == "horizontal":
                if abs(dx_px) > 2:
                    suggestions.append(f"tool_cy: tip drifts {'right' if dx_px > 0 else 'left'} → shift TCP {'left' if dx_px > 0 else 'right'} (adjust tool_cy)")
                if abs(dy_px) > 2:
                    suggestions.append(f"tool_cx/LZ: tip {'down' if dy_px > 0 else 'up'} → adjust tool center X or tool length")
            elif mode_hint == "vertical":
                # Vertical osc logic: dy tracks tool length; dx tracks tool_cx
                if abs(dy_px) > 2:
                    suggestions.append(
                        f"tool length: tip {'drops' if dy_px > 0 else 'rises'} when pitching down → LZ too {'long' if dy_px > 0 else 'short'}"
                    )
                if abs(dx_px) > 2:
                    suggestions.append(
                        f"tool_cx: tip drifts {'right' if dx_px > 0 else 'left'} during pitch → move TCP {'left' if dx_px > 0 else 'right'} (adjust tool_cx)"
                    )
            if cross_offsets:
                cx_dx, cx_dy = cross_offsets
                info_lines.append(f"Cross-check: dx={cx_dx:+.1f}px dy={cx_dy:+.1f}px")
            info_lines += suggestions
            bx = rect.x + 8
            by = rect.y + 8
            for ln in info_lines:
                txt = small_font.render(ln, True, (255, 230, 200))
                screen.blit(txt, (bx, by))
                by += txt.get_height() + 2

        # Cross-check offsets (tip vs center) if available
        h_offsets = compute_trace_offsets(needle_trace_h, f2) if needle_trace_h and f2 is not None else None
        v_offsets = compute_trace_offsets(needle_trace_v, f1) if needle_trace_v and f1 is not None else None

        # Bottom-left (UVC2) used for horizontal displacement check
        draw_trace(
            needle_trace_h,
            needle_trace_prev_h,
            f2,
            bl_rect,
            label_prefix="Horizontal demo",
            axis_hint="tune tool center X; vertical drift → check tool length",
            mode_hint="horizontal",
            cross_offsets=v_offsets if v_offsets else None,
        )
        # Bottom-right (UVC1) used for vertical displacement check
        draw_trace(
            needle_trace_v,
            needle_trace_prev_v,
            f1,
            br_rect,
            label_prefix="Vertical demo",
            axis_hint="tune tool center Y; horizontal drift → check tool length",
            mode_hint="vertical",
            cross_offsets=h_offsets if h_offsets else None,
        )

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
            and pending_step_index is None
        ):
            current_step_recorder.stop()
            current_step_recorder = None

        # ───────────── Click handling (no modals) ─────────────
        modals_open = (
            edit_menu_active or delete_confirm_active or overwrite_confirm_active
            or toollen_input_active or approach_input_active
            or toolcx_input_active or toolcy_input_active or pitch_input_active or angle_input_active or yaw_input_active
            or naming_input_active or rename_input_active
            or toolpreset_input_active or toolpreset_rename_input_active
            or fwd_rate_input_active or bwd_rate_input_active
            or endstop_timeout_input_active or pose_wait_input_active
            or retract_input_active or routine_name_input_active or settings_path_input_active
            or routine_edit_active or routine_action_modal_active or routine_edit_modal_active
            or syringe_calib_active or quit_confirm_active
            or injection_info_active or subject_prompt_active
            or post_step_prompt_active or final_small_warning_active
            or tool_preset_menu_active
            or save_mode_modal_active
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
                        if act.get("kind") == "pitch_input":
                            with state.lock:
                                current_pitch = float(state.pitch)
                            pitch_input_active = True
                            pitch_text = f"{current_pitch:.3f}"
                            handled = True
                            break
                        if act.get("kind") == "angle_set":
                            with state.lock:
                                mode = getattr(state, "j5_angle_mode", "actual")
                                base_ref = getattr(state, "j5_base_ref", None)
                                cur_j5 = float(getattr(state, "j5", 0.0))
                            if mode == "base":
                                if base_ref is None:
                                    base_ref = cur_j5
                                    with state.lock:
                                        state.j5_base_ref = base_ref
                                angle_text = f"{(cur_j5 - base_ref):.3f}"
                            else:
                                angle_text = f"{cur_j5:.3f}"
                            angle_input_active = True
                            handled = True
                            break
                        if act.get("kind") == "yaw_set":
                            with state.lock:
                                yaw_text = f"{getattr(state, 'yaw', 0.0):.3f}"
                            yaw_input_active = True
                            handled = True
                            break
                        if act.get("kind") == "yaw_base_set":
                            with state.lock:
                                state.yaw_base = float(getattr(state, "yaw", 0.0))
                            handled = True
                            break
                        if act.get("kind") == "angle_toggle":
                            with state.lock:
                                mode = getattr(state, "j5_angle_mode", "actual")
                                if mode == "actual":
                                    state.j5_angle_mode = "base"
                                    state.j5_base_ref = float(getattr(state, "j5", 0.0))
                                else:
                                    state.j5_angle_mode = "actual"
                            handled = True
                            break
                        if act.get("kind") == "auto_correct":
                            mode = act.get("mode")
                            target = act.get("target")
                            delta = float(act.get("delta", 0.0) or 0.0)
                            # Pause/stop demo before applying corrections
                            if getattr(state, "tool_center_demo", False):
                                rt.enqueue({"type": "tool_center_demo_toggle", "mode": None})
                            with state.lock:
                                cur_cx = getattr(state, "tool_cx", DEFAULT_TOOL_CX)
                                cur_cy = getattr(state, "tool_cy", DEFAULT_TOOL_CY)
                                cur_lz = getattr(state, "tool_lz", DEFAULT_TOOL_LZ)
                            if target == "cx":
                                new_cx = cur_cx + delta
                                rt.enqueue({"type": "set_tool_center", "cx": new_cx, "cy": cur_cy})
                            elif target == "cy":
                                new_cy = cur_cy + delta
                                rt.enqueue({"type": "set_tool_center", "cx": cur_cx, "cy": new_cy})
                            elif target == "lz":
                                new_lz = cur_lz + delta
                                rt.enqueue({"type": "set_tool_lz", "lz": new_lz})
                            handled = True
                            break
                        if act.get("kind") == "demo_resume":
                            rt.enqueue({"type": "tool_center_demo_resume"})
                            # Clear current traces to start a fresh cycle
                            needle_trace_h.clear()
                            needle_trace_v.clear()
                            handled = True
                            break
                        if act.get("kind") == "tool_center_j5_test":
                            test_on = bool(getattr(state, "j5_test_active", False))
                            if test_on:
                                print("[UI] Stopping tool center rotation test.")
                                rt.enqueue({"type": "j5_test_toggle", "start": False})
                            else:
                                print("[UI] Starting tool center rotation test request.")
                            rt.enqueue({"type": "j5_test_toggle", "start": True})
                            handled = True
                            break
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
                    if act.get("kind") == "auto_correct":
                        mode = act.get("mode")
                        target = act.get("target")
                        delta = float(act.get("delta", 0.0) or 0.0)
                        with state.lock:
                            cur_cx = getattr(state, "tool_cx", DEFAULT_TOOL_CX)
                            cur_cy = getattr(state, "tool_cy", DEFAULT_TOOL_CY)
                            cur_lz = getattr(state, "tool_lz", DEFAULT_TOOL_LZ)
                        if target == "cx":
                            new_cx = cur_cx + delta
                            rt.enqueue({"type": "set_tool_center", "cx": new_cx, "cy": cur_cy})
                        elif target == "cy":
                            new_cy = cur_cy + delta
                            rt.enqueue({"type": "set_tool_center", "cx": cur_cx, "cy": new_cy})
                        elif target == "lz":
                            new_lz = cur_lz + delta
                            rt.enqueue({"type": "set_tool_lz", "lz": new_lz})
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
                                elif kind == "edit_toolcx":
                                    toolcx_input_active = True
                                    toolcx_text = f"{getattr(state, 'tool_cx', DEFAULT_TOOL_CX):.3f}"
                                elif kind == "edit_toolcy":
                                    toolcy_input_active = True
                                    toolcy_text = f"{getattr(state, 'tool_cy', DEFAULT_TOOL_CY):.3f}"
                                elif kind == "tool_center_demo_circle":
                                    mode_now = getattr(state, "tool_center_demo_mode", "circle")
                                    demo_on = bool(getattr(state, "tool_center_demo", False) and mode_now == "circle")
                                    if demo_on:
                                        rt.tool_center_demo_stop.set()
                                        rt.enqueue({"type": "tool_center_demo_stop"})
                                        with state.lock:
                                            state.tool_center_demo_waiting = False
                                            state.tool_center_demo = False
                                        needle_trace_h.clear()
                                        needle_trace_v.clear()
                                    else:
                                        rt.enqueue({"type": "tool_center_demo_toggle", "mode": "circle"})
                                    click_pos = None
                                elif kind == "tool_center_demo_vert":
                                    mode_now = getattr(state, "tool_center_demo_mode", "circle")
                                    demo_on = bool(getattr(state, "tool_center_demo", False) and mode_now in ("vertical","oscillate_vertical"))
                                    if demo_on:
                                        rt.tool_center_demo_stop.set()
                                        rt.enqueue({"type": "tool_center_demo_stop"})
                                        with state.lock:
                                            state.tool_center_demo_waiting = False
                                            state.tool_center_demo = False
                                            state.tool_center_demo_mode = mode_now
                                        needle_trace_h.clear()
                                        needle_trace_v.clear()
                                    else:
                                        rt.enqueue({"type": "tool_center_demo_toggle", "mode": "vertical"})
                                    click_pos = None
                                elif kind == "tool_center_demo_horz":
                                    mode_now = getattr(state, "tool_center_demo_mode", "circle")
                                    demo_on = bool(getattr(state, "tool_center_demo", False) and mode_now in ("horizontal","oscillate_horizontal"))
                                    if demo_on:
                                        rt.tool_center_demo_stop.set()
                                        rt.enqueue({"type": "tool_center_demo_stop"})
                                        with state.lock:
                                            state.tool_center_demo_waiting = False
                                            state.tool_center_demo = False
                                            state.tool_center_demo_mode = mode_now
                                        needle_trace_h.clear()
                                        needle_trace_v.clear()
                                    else:
                                        rt.enqueue({"type": "tool_center_demo_toggle", "mode": "horizontal"})
                                    click_pos = None
                                elif kind == "tool_center_j5_test":
                                    test_on = bool(getattr(state, "j5_test_active", False))
                                    if test_on:
                                        print("[UI] Stopping tool center rotation test.")
                                        rt.enqueue({"type": "j5_test_toggle", "start": False})
                                    else:
                                        print("[UI] Starting tool center rotation test request.")
                                        rt.enqueue({"type": "j5_test_toggle", "start": True})
                                    click_pos = None
                                elif kind == "tool_center_radius":
                                    bar = act.get("rect")
                                    if bar and bar.w > 0:
                                        rel = max(0.0, min(1.0, (mx - bar.x) / bar.w))
                                        new_radius = rel * 50.0
                                        tool_center_radius = new_radius
                                        with state.lock:
                                            state.settings["tool_center_demo_radius_mm"] = new_radius
                                        save_settings(state.settings)
                                elif kind in ("alarm_threshold", "alarm_duration"):
                                    bar = act.get("rect")
                                    if bar and bar.w > 0:
                                        rel = max(0.0, min(1.0, (mx - bar.x) / bar.w))
                                        if kind == "alarm_threshold":
                                            alarm_threshold_val = DEFAULT_PID_THRESHOLD_MIN + rel * (DEFAULT_PID_THRESHOLD_MAX - DEFAULT_PID_THRESHOLD_MIN)
                                        else:
                                            alarm_duration_val = DEFAULT_PID_DURATION_MIN + rel * (DEFAULT_PID_DURATION_MAX - DEFAULT_PID_DURATION_MIN)
                                        alarm_threshold_val, alarm_duration_val = _clamp_alarm_pid(
                                            alarm_threshold_val,
                                            alarm_duration_val,
                                        )
                                        with state.lock:
                                            state.settings["alarm_threshold"] = alarm_threshold_val
                                            state.settings["alarm_duration"] = alarm_duration_val
                                        save_settings(state.settings)
                                        rt.enqueue({"type": "set_alarm_pid", "threshold": alarm_threshold_val, "duration": alarm_duration_val})
                                elif kind == "edit_approach":
                                    approach_input_active = True
                                    approach_text = f"{state.approach_mm:.1f}"
                                elif kind == "tool_preset_save":
                                    toolpreset_input_active = True
                                    toolpreset_name_text = ""
                                    click_pos = None
                                elif kind == "tool_preset_apply":
                                    preset_name = str(act.get("preset", "")).strip()
                                    if preset_name:
                                        rt.enqueue({"type": "apply_tool_preset", "name": preset_name})
                                        try:
                                            joy.rumble(0.4, 0.4, 120)
                                        except Exception:
                                            pass
                                    click_pos = None
                                elif kind == "tool_preset_menu":
                                    preset_name = str(act.get("preset", "")).strip()
                                    if preset_name:
                                        tool_preset_menu_active = True
                                        tool_preset_menu_target = preset_name
                                    click_pos = None
                                elif kind == "tool_preset_delete":
                                    preset_name = str(act.get("preset", "")).strip()
                                    if preset_name:
                                        rt.enqueue({"type": "delete_tool_preset", "name": preset_name})
                                    click_pos = None
                                    click_pos = None
                                elif kind == "goto":
                                    rt.enqueue({"type": "goto", "pose_name": act["pose"]})
                                    try:
                                        joy.rumble(0.4, 0.4, 120)
                                    except Exception:
                                        pass
                                elif kind == "edit_menu":
                                    edit_menu_active = True
                                    edit_menu_target = act["pose"]
                                    edit_menu_target_reserved = bool(act.get("reserved", False))
                                    click_pos = None
                                elif kind == "routine_new":
                                    routine_name_mode = "create"
                                    routine_name_text = ""
                                    routine_rename_target = ""
                                    routine_name_input_active = True
                                    click_pos = None
                                elif kind == "routine_select":
                                    rname = act.get("routine") or ""
                                    if rname:
                                        routine_action_target = rname
                                        routine_action_modal_active = True
                                        click_pos = None
                                elif kind == "routine_edit_row":
                                    rname = act.get("routine") or ""
                                    if rname:
                                        routine_edit_modal_target = rname
                                        routine_edit_modal_active = True
                                        click_pos = None
                                elif kind == "quit_app":
                                    quit_confirm_active = True
                                    click_pos = None
                                elif kind == "settings_load":
                                    settings_path_mode = "load"
                                    settings_path_text = SETTINGS_PATH
                                    settings_path_input_active = True
                                    click_pos = None
                                elif kind == "settings_save":
                                    settings_path_mode = "save"
                                    settings_path_text = SETTINGS_PATH
                                    settings_path_input_active = True
                                    click_pos = None
                                elif kind == "alarm_arm_now":
                                    rt.enqueue({
                                        "type": "set_alarm_pid",
                                        "threshold": alarm_threshold_val,
                                        "duration": alarm_duration_val,
                                    })
                                    click_pos = None
                                elif kind == "alarm_auto_tune":
                                    rt.enqueue({
                                        "type": "auto_tune_alarm_pid",
                                        "threshold": alarm_threshold_val,
                                        "duration": alarm_duration_val,
                                    })
                                    click_pos = None
                                elif kind == "alarm_disarm_now":
                                    rt.enqueue({"type": "disarm_alarm"})
                                    click_pos = None
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
                                        f"{state.settings.get('pose_final_wait_s', 0.0):.1f}"
                                    )
                                elif kind == "new_pose":
                                    naming_input_active = True
                                    naming_text = ""
                                elif kind == "syringe_calib":
                                    syringe_calib_active = True
                                    syringe_field_active = "vol"
                                    click_pos = None
                                elif kind == "edit_retract":
                                    retract_input_active = True
                                    retract_text = f"{state.settings.get('injection_retract_mm', 10.0):.1f}"
                                elif kind == "start_injection":
                                    injection_info_active = True
                                    injection_field_active = "study"
                                    click_pos = None
                                elif kind == "stop_injection":
                                    with state.lock:
                                        state.settings["injection_active"] = False
                                        state.calibration_status = "Injection stopped by user."
                                        state.settings["syringe_remaining_ul"] = 0.0
                                        state.current_injection_step = 0
                                    save_settings(state.settings)
                                    injection_step_in_progress = False
                                    injection_step_done = False
                                    post_step_prompt_active = False
                                    final_small_warning_active = False
                                    if current_step_recorder:
                                        current_step_recorder.stop()
                                        current_step_recorder = None
                                    print("[Injection] Injection stopped via UI button.")
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
        if toolcx_input_active:
            draw_center_input_box(
                screen, "Enter tool center X (mm)", toolcx_text
            )
        if toolcy_input_active:
            draw_center_input_box(
                screen, "Enter tool center Y (mm)", toolcy_text
            )
        if yaw_input_active:
            draw_center_input_box(
                screen, "Set yaw (deg)", yaw_text
            )
        if angle_input_active:
            draw_center_input_box(
                screen, "Set J5 angle (deg)", angle_text
            )
        if pitch_input_active:
            draw_center_input_box(
                screen, "Set pitch (deg)", pitch_text
            )
        if naming_input_active:
            draw_center_input_box(
                screen, "Enter NEW pose name", naming_text or ""
            )
        if rename_input_active:
            draw_center_input_box(
                screen, f"Rename '{rename_old_name}' →", rename_text or ""
            )
        if toolpreset_input_active:
            draw_center_input_box(
                screen, "Save tool preset (name)", toolpreset_name_text or ""
            )
        if toolpreset_rename_input_active:
            draw_center_input_box(
                screen, f"Rename tool preset '{toolpreset_rename_old}' →", toolpreset_rename_text or ""
            )
        if tool_preset_menu_active:
            pass  # menu drawn separately
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
        if retract_input_active:
            draw_center_input_box(
                screen,
                "Retract distance after each step (mm)",
                retract_text,
            )
        if routine_name_input_active:
            draw_center_input_box(
                screen,
                ("Rename routine" if routine_name_mode == "rename" else
                 "Create new routine" if routine_name_mode == "create" else
                 "Save routine as…"),
                routine_name_text,
            )
        if settings_path_input_active:
            draw_center_input_box(
                screen,
                "Settings file path",
                settings_path_text,
            )
        if quit_confirm_active:
            pass  # handled in modal drawing above
        if endstop_timeout_input_active:
            draw_center_input_box(
                screen,
                "Failsafe timeout (ms) — 0 = infinite",
                endstop_timeout_text,
            )
        if pose_wait_input_active:
            draw_center_input_box(
                screen,
                "Final pose dwell (seconds) — 0 = none",
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
                    state.current_injection_step = step_idx
                save_settings(state.settings)

                pending_step_index = step_idx
                injection_step_in_progress = False
                injection_step_done = False
                injection_step_distance = 0.0
                injection_step_target = 0.0
                injection_step_volume_ul = 0.0
                start_step_recording(step_idx)

                if choice == "confirm_adv" and advance_mm > 0.0:
                    rt.enqueue({"type": "inject_advance", "dist": advance_mm})

                subject_prompt_active = False

            elif choice == "cancel":
                subject_prompt_active = False

        # Routine run confirm modal
        if routine_action_modal_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            pan_w, pan_h = 520, 200
            px = (screen_w - pan_w) // 2
            py = (screen_h - pan_h) // 2
            panel = pygame.Rect(px, py, pan_w, pan_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (0, 255, 127), panel, 2)

            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            small_modal = pygame.font.SysFont("Consolas", 15)
            title = font_modal.render(f"Run routine '{routine_action_target}'?", True, (230, 230, 230))
            screen.blit(title, (px + 14, py + 16))
            info = small_modal.render("This will start the selected routine immediately.", True, (200, 200, 200))
            screen.blit(info, (px + 14, py + 48))

            btn_w, btn_h, gap = 160, 36, 16
            bx = px + (pan_w - (btn_w * 2 + gap)) // 2
            by = py + pan_h - btn_h - 18
            run_rect = pygame.Rect(bx, by, btn_w, btn_h)
            cancel_rect = pygame.Rect(bx + btn_w + gap, by, btn_w, btn_h)
            for rect, txt, col in [
                (run_rect, "Run", (0, 200, 120)),
                (cancel_rect, "Cancel", (200, 80, 80)),
            ]:
                pygame.draw.rect(screen, (40, 50, 60), rect)
                pygame.draw.rect(screen, col, rect, 2)
                t = font_modal.render(txt, True, (230, 230, 230))
                screen.blit(t, (rect.x + (rect.w - t.get_width()) // 2,
                                rect.y + (rect.h - t.get_height()) // 2))

            if click_pos is not None:
                mx, my = click_pos
                if run_rect.collidepoint(mx, my):
                    run_routine_by_name(routine_action_target)
                    routine_action_modal_active = False
                elif cancel_rect.collidepoint(mx, my) or not panel.collidepoint(mx, my):
                    routine_action_modal_active = False
                click_pos = None

        # Routine edit options modal
        if routine_edit_modal_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            pan_w, pan_h = 520, 260
            px = (screen_w - pan_w) // 2
            py = (screen_h - pan_h) // 2
            panel = pygame.Rect(px, py, pan_w, pan_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (0, 255, 127), panel, 2)

            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            title = font_modal.render(f"Routine '{routine_edit_modal_target}'", True, (230, 230, 230))
            screen.blit(title, (px + 14, py + 16))

            opts = [
                ("Edit script", "edit"),
                ("Rename", "rename"),
                ("Delete", "delete"),
                ("Close", "close"),
            ]
            btn_w = pan_w - 40
            btn_h = 34
            gap = 10
            by = py + 56
            buttons_modal = {}
            for label, kind in opts:
                rect = pygame.Rect(px + 20, by, btn_w, btn_h)
                pygame.draw.rect(screen, (45, 45, 70), rect)
                pygame.draw.rect(screen, (0, 255, 127), rect, 2)
                t = font_modal.render(label, True, (230, 230, 230))
                screen.blit(t, (rect.x + (rect.w - t.get_width()) // 2,
                                rect.y + (rect.h - t.get_height()) // 2))
                buttons_modal[kind] = rect
                by += btn_h + gap

            if click_pos is not None:
                mx, my = click_pos
                if not panel.collidepoint(mx, my):
                    routine_edit_modal_active = False
                else:
                    if buttons_modal["edit"].collidepoint(mx, my):
                        txt = routines.get(routine_edit_modal_target, ROUTINE_SAMPLE.strip())
                        current_routine_name = routine_edit_modal_target
                        routine_editor = EditorState(text=txt, caret=len(txt))
                        routine_edit_active = True
                        routine_edit_modal_active = False
                    elif buttons_modal["rename"].collidepoint(mx, my):
                        routine_name_mode = "rename"
                        routine_rename_target = routine_edit_modal_target
                        routine_name_text = routine_edit_modal_target
                        routine_name_input_active = True
                        routine_edit_modal_active = False
                    elif buttons_modal["delete"].collidepoint(mx, my):
                        if len(routines) <= 1 or routine_edit_modal_target in ("Default",):
                            with state.lock:
                                state.routine_status_text = "Cannot delete this routine."
                        else:
                            routines.pop(routine_edit_modal_target, None)
                            save_routines(routines)
                            cycle_routine(+0)
                            with state.lock:
                                state.routine_status_text = "Routine deleted."
                        routine_edit_modal_active = False
                    elif buttons_modal["close"].collidepoint(mx, my):
                        routine_edit_modal_active = False
                click_pos = None

        # Quit confirmation modal
        if quit_confirm_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (0, 0))
            pan_w, pan_h = 480, 180
            px = (screen_w - pan_w) // 2
            py = (screen_h - pan_h) // 2
            panel = pygame.Rect(px, py, pan_w, pan_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (255, 80, 80), panel, 2)
            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            small_modal = pygame.font.SysFont("Consolas", 15)
            title = font_modal.render("Quit application?", True, (230, 230, 230))
            screen.blit(title, (px + 16, py + 16))
            info = small_modal.render("Unsaved settings are written immediately.", True, (210, 210, 210))
            screen.blit(info, (px + 16, py + 52))
            btn_w, btn_h, gap = 150, 36, 16
            bx = px + (pan_w - (btn_w * 2 + gap)) // 2
            by = py + pan_h - btn_h - 16
            quit_rect = pygame.Rect(bx, by, btn_w, btn_h)
            cancel_rect = pygame.Rect(bx + btn_w + gap, by, btn_w, btn_h)
            for rect, txt, col in [
                (quit_rect, "Quit", (255, 80, 80)),
                (cancel_rect, "Cancel", (0, 200, 120)),
            ]:
                pygame.draw.rect(screen, (45, 45, 70), rect)
                pygame.draw.rect(screen, col, rect, 2)
                t = font_modal.render(txt, True, (230, 230, 230))
                screen.blit(t, (rect.x + (rect.w - t.get_width()) // 2,
                                rect.y + (rect.h - t.get_height()) // 2))

            if click_pos is not None:
                mx, my = click_pos
                if quit_rect.collidepoint(mx, my):
                    running = False
                    rt.stop_event.set()
                    rt.keep_motors_on_exit = True
                elif cancel_rect.collidepoint(mx, my) or not panel.collidepoint(mx, my):
                    quit_confirm_active = False
                click_pos = None

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
                    step_index = take_step_index()
                    if current_step_recorder is None:
                        start_step_recording(step_index)

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
                pending_step_index = None
                injection_step_in_progress = False
                injection_step_done = False
                if current_step_recorder:
                    try:
                        current_step_recorder.stop()
                    except Exception:
                        pass
                    current_step_recorder = None
                print("[Injection] Final small step cancelled; remaining volume discarded.")
            elif y_edge:
                if current_step_recorder:
                    current_step_recorder.stop()
                    current_step_recorder = None
                print("[Injection] Recording stopped via Y during final-step warning.")

        # Post-step retract / skip overlay (A / B only)
        syringe_remaining_ul_overlay = 0.0
        with state.lock:
            syringe_remaining_ul_overlay = state.settings.get("syringe_remaining_ul", 0.0)
        if post_step_prompt_active and syringe_remaining_ul_overlay > 1e-9:
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
            reserved_list_snapshot = settings_snapshot.get("reserved_poses", [])
            reserved_list_snapshot = normalize_reserved_list(reserved_list_snapshot)
            built_in_reserved = (edit_menu_target in RESERVED_POSES)
            user_reserved = (
                edit_menu_target in reserved_list_snapshot
                or edit_menu_target_reserved
            )
            rects, panel_rect = draw_edit_menu(
                screen,
                edit_menu_target,
                disable_rename_delete=built_in_reserved or user_reserved,
                reserved=(built_in_reserved or user_reserved),
                can_toggle_reserved=(not built_in_reserved) or user_reserved,
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
                                click_pos = None
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
                                if edit_menu_target in RESERVED_POSES:
                                    print(f"⚠️ '{edit_menu_target}' is built-in and cannot be unreserved.")
                                    with state.lock:
                                        state.routine_status_text = (
                                            f"Cannot unreserve built-in pose '{edit_menu_target}'."
                                        )
                                    ui_notice_text = (
                                        f"Built-in pose '{edit_menu_target}' cannot be unreserved."
                                    )
                                    ui_notice_until = time.time() + 4.0
                                    edit_menu_active = False
                                else:
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
            try:
                if not delete_target_name:
                    delete_confirm_active = False
                else:
                    rects, panel = draw_confirm_delete(screen, delete_target_name)
                    if click_pos is not None:
                        mx, my = click_pos
                        if not panel.collidepoint(mx, my):
                            delete_confirm_active = False
                            delete_target_name = ""
                            edit_menu_target = ""
                        elif rects["yes"].collidepoint(click_pos):
                            # Close modal before deleting to avoid re-entry issues
                            delete_confirm_active = False
                            pose_to_delete = delete_target_name
                            delete_target_name = ""
                            edit_menu_target = ""
                            # Attempt immediate delete; fallback to robot thread
                            if not delete_pose_local(pose_to_delete):
                                rt.enqueue({"type": "delete_pose", "pose": pose_to_delete})
                        elif rects["no"].collidepoint(click_pos):
                            delete_confirm_active = False
                            delete_target_name = ""
                            edit_menu_target = ""
                        click_pos = None
            except Exception as e:
                print(f"[UI] delete confirm handling failed: {e}")
                delete_confirm_active = False
                delete_target_name = ""
                click_pos = None

        # Tool preset edit menu
        if tool_preset_menu_active and tool_preset_menu_target:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            pan_w, pan_h = 520, 240
            px = (screen_w - pan_w) // 2
            py = (screen_h - pan_h) // 2
            panel = pygame.Rect(px, py, pan_w, pan_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (0, 255, 127), panel, 2)

            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            title = font_modal.render(f"Tool preset '{tool_preset_menu_target}'", True, (230, 230, 230))
            screen.blit(title, (px + 14, py + 14))

            opts = [
                ("Apply preset", "apply"),
                ("Overwrite with current tool", "overwrite"),
                ("Rename", "rename"),
                ("Delete", "delete"),
                ("Close", "close"),
            ]
            btn_w = pan_w - 40
            btn_h = 34
            gap = 10
            by = py + 56
            buttons_modal = {}
            for label, kind in opts:
                rect = pygame.Rect(px + 20, by, btn_w, btn_h)
                pygame.draw.rect(screen, (45, 45, 70), rect)
                pygame.draw.rect(screen, (0, 255, 127), rect, 2)
                t = font_modal.render(label, True, (230, 230, 230))
                screen.blit(t, (rect.x + (rect.w - t.get_width()) // 2,
                                rect.y + (rect.h - t.get_height()) // 2))
                buttons_modal[kind] = rect
                by += btn_h + gap

            if click_pos is not None:
                mx, my = click_pos
                if not panel.collidepoint(mx, my):
                    tool_preset_menu_active = False
                else:
                    if buttons_modal["apply"].collidepoint(mx, my):
                        rt.enqueue({"type": "apply_tool_preset", "name": tool_preset_menu_target})
                        tool_preset_menu_active = False
                    elif buttons_modal["overwrite"].collidepoint(mx, my):
                        rt.enqueue({"type": "save_tool_preset", "name": tool_preset_menu_target})
                        tool_preset_menu_active = False
                    elif buttons_modal["rename"].collidepoint(mx, my):
                        toolpreset_rename_input_active = True
                        toolpreset_rename_old = tool_preset_menu_target
                        toolpreset_rename_text = tool_preset_menu_target
                        tool_preset_menu_active = False
                    elif buttons_modal["delete"].collidepoint(mx, my):
                        rt.enqueue({"type": "delete_tool_preset", "name": tool_preset_menu_target})
                        tool_preset_menu_active = False
                    elif buttons_modal["close"].collidepoint(mx, my):
                        tool_preset_menu_active = False
                click_pos = None

        # Save pose mode chooser (demo vs no-demo)
        if save_mode_modal_active and pending_save_pose_name:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            pan_w, pan_h = 620, 220
            px = (screen_w - pan_w) // 2
            py = (screen_h - pan_h) // 2
            panel = pygame.Rect(px, py, pan_w, pan_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (0, 255, 127), panel, 2)

            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            small_modal = pygame.font.SysFont("Consolas", 15)
            title = font_modal.render(f"Save pose '{pending_save_pose_name}'", True, (230, 230, 230))
            screen.blit(title, (px + 16, py + 14))
            msg = small_modal.render("Choose whether to demonstrate the midway retract/return during save.", True, (210, 210, 210))
            screen.blit(msg, (px + 16, py + 48))

            btn_h, gap, pad = 38, 12, 20
            avail_w = pan_w - pad * 2
            btn_w = max(170, int((avail_w - gap * 2) / 3))
            total_w = btn_w * 3 + gap * 2
            bx = px + (pan_w - total_w) // 2
            by = py + pan_h - btn_h - 20
            no_demo_rect = pygame.Rect(bx, by, btn_w, btn_h)
            demo_rect    = pygame.Rect(bx + btn_w + gap, by, btn_w, btn_h)
            cancel_rect  = pygame.Rect(bx + (btn_w + gap) * 2, by, btn_w, btn_h)
            for rect, txt, col in [
                (no_demo_rect, "Save (no demo)",   (0, 200, 120)),
                (demo_rect,    "Save (show retract)", (0, 180, 255)),
                (cancel_rect,  "Cancel",           (200, 80, 80)),
            ]:
                pygame.draw.rect(screen, (45, 45, 70), rect)
                pygame.draw.rect(screen, col, rect, 2)
                t = small_modal.render(txt, True, (230, 230, 230))
                screen.blit(t, (rect.x + (rect.w - t.get_width()) // 2,
                                rect.y + (rect.h - t.get_height()) // 2))

            # Clicking outside closes modal
            if click_pos is not None:
                mx, my = click_pos
                if no_demo_rect.collidepoint(mx, my):
                    rt.enqueue({"type": "save_named_with_midway", "name": pending_save_pose_name, "demo": False})
                    save_mode_modal_active = False
                    pending_save_pose_name = ""
                elif demo_rect.collidepoint(mx, my):
                    rt.enqueue({"type": "save_named_with_midway", "name": pending_save_pose_name, "demo": True})
                    save_mode_modal_active = False
                    pending_save_pose_name = ""
                elif cancel_rect.collidepoint(mx, my):
                    save_mode_modal_active = False
                    pending_save_pose_name = ""
                elif not panel.collidepoint(mx, my):
                    save_mode_modal_active = False
                    pending_save_pose_name = ""
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
