#!/usr/bin/env python3
import sys
import argparse
import threading
import time
import os
import json
import csv
import queue
import re, glob
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
import math
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import numpy as np
import pygame.surfarray
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2
# Quiet down OpenCV warnings from the V4L2 backend. This build exposes the
# global logging API directly on cv2 rather than cv2.utils.logging.
try:
    cv2.setLogLevel(0)
except Exception:
    try:
        cv2.setLogLevel(1)
    except Exception:
        pass
try:
    cv2.setNumThreads(1)
except Exception:
    pass

import pyrealsense2 as rs
import serial
from dorna2 import Dorna

# Where to place step recordings
DATA_ROOT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "RobotInjectionData")
POSTER_ASSET_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "poster_assets")
CHARACTERIZATION_ROOT_DIR = os.path.join(POSTER_ASSET_DIR, "characterization")
REPEATABILITY_ROOT_DIR = os.path.join(POSTER_ASSET_DIR, "repeatability")
RIGHT_STICK_DEMO_ROOT_DIR = os.path.join(POSTER_ASSET_DIR, "right_stick_demo")
JOG_BENCHMARK_ROOT_DIR = os.path.join(POSTER_ASSET_DIR, "jog_benchmark")
DEFAULT_DORNA_HOST = "10.42.0.11"
DEFAULT_DORNA_PORT = 443
DEFAULT_UVC_FPS = 15
DEFAULT_UVC_WIDTH = 640
DEFAULT_UVC_HEIGHT = 480
DEFAULT_RS_WIDTH = 640
DEFAULT_RS_HEIGHT = 480
DEFAULT_RS_FPS = 30
DEFAULT_UI_FPS = 30
DEFAULT_RECORDING_FPS = 30.0
THERMAL_WARN_TEMP_C = 95.0
THERMAL_CLEAR_TEMP_C = 90.0
THERMAL_POLL_INTERVAL_S = 3.0
THERMAL_CRITICAL_TEMP_C = 99.0
_V4L2_CTL = shutil.which("v4l2-ctl")
_V4L2_CAPS_CACHE = {}

def _profile_label(width: int, height: int, fps: int, note: str = "") -> str:
    label = f"{int(width)}x{int(height)} @ {int(fps)} fps"
    if note:
        label += f" {note}"
    return label

UVC_QUALITY_PRESETS = [
    (_profile_label(640, 480, 15, "(Default)"), (640, 480, 15)),
    (_profile_label(640, 480, 30), (640, 480, 30)),
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
    ("Default start (30 / 50)", (30, 50)),
    ("Stock (200 / 10000)", (200, 10000)),
    ("Mild (180 / 7000)", (180, 7000)),
    ("Moderate (160 / 5000)", (160, 5000)),
    ("Sensitive (140 / 3000)", (140, 3000)),
    ("Aggressive (120 / 2000)", (120, 2000)),
]

JOG_PROFILE_PRESETS = {
    "balanced": {
        "label": "Balanced (default)",
        "send_hz": 40.0,
        "tool_axis_hz": 80.0,
        "vel_filter_alpha": 0.45,
        "active_feedback_s": 0.20,
        "halt_accel": 12.0,
        "halt_accel_translation": 16.0,
    },
    "smooth": {
        "label": "Smooth (more filtering)",
        "send_hz": 35.0,
        "tool_axis_hz": 60.0,
        "vel_filter_alpha": 0.25,
        "active_feedback_s": 0.25,
        "halt_accel": 8.0,
        "halt_accel_translation": 12.0,
    },
    "fast": {
        "label": "Fast (less filtering)",
        "send_hz": 40.0,
        "tool_axis_hz": 50.0,
        "vel_filter_alpha": 0.75,
        "active_feedback_s": 0.75,
        "halt_accel": 24.0,
        "halt_accel_translation": 32.0,
    },
    "legacy": {
        "label": "Legacy timing",
        "send_hz": 40.0,
        "tool_axis_hz": 80.0,
        "vel_filter_alpha": 1.0,
        "active_feedback_s": 0.05,
        "halt_accel": 8.0,
        "halt_accel_translation": 12.0,
    },
}
DEFAULT_JOG_PROFILE = "fast"
INJECTION_TRIGGER_DEADZONE = 0.03
INJECTION_TRIGGER_START = 0.02
INJECTION_TRIGGER_EXPONENT = 1.6
STARTUP_CALIBRATION_TESTS = {
    "none": "Do not run calibration",
    "tool_center_circle": "Tool-center circular sweep",
    "tool_center_vertical": "Tool-center vertical oscillation",
    "tool_center_horizontal": "Tool-center horizontal oscillation",
    "j5_rotation": "Roll tool-center rotation test",
}
DEFAULT_STARTUP_CALIBRATION_TEST = "none"

def _normalize_jog_profile(name: str) -> str:
    key = str(name or "").strip().lower()
    return key if key in JOG_PROFILE_PRESETS else DEFAULT_JOG_PROFILE

def _jog_profile_label(name: str) -> str:
    key = _normalize_jog_profile(name)
    return JOG_PROFILE_PRESETS[key]["label"]

def _jog_profile_from_label(label: str) -> str:
    label = str(label or "").strip()
    for key, profile in JOG_PROFILE_PRESETS.items():
        if label == profile["label"]:
            return key
    return _normalize_jog_profile(label)

def _jog_profile_labels() -> list:
    return [profile["label"] for profile in JOG_PROFILE_PRESETS.values()]

def _normalize_startup_calibration_test(name: str) -> str:
    key = str(name or "").strip().lower()
    return key if key in STARTUP_CALIBRATION_TESTS else DEFAULT_STARTUP_CALIBRATION_TEST

def _startup_calibration_label(name: str) -> str:
    return STARTUP_CALIBRATION_TESTS[_normalize_startup_calibration_test(name)]

def _startup_calibration_from_label(label: str) -> str:
    label = str(label or "").strip()
    for key, text in STARTUP_CALIBRATION_TESTS.items():
        if label == text:
            return key
    return _normalize_startup_calibration_test(label)

def _startup_calibration_labels() -> list:
    return list(STARTUP_CALIBRATION_TESTS.values())

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

def _read_thermal_zones():
    zones = []
    for zone_dir in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        zone_type = _read_text(os.path.join(zone_dir, "type"))
        raw_temp = _read_text(os.path.join(zone_dir, "temp"))
        if not zone_type or not raw_temp:
            continue
        try:
            temp_c = float(raw_temp)
        except Exception:
            continue
        if temp_c > 1000.0:
            temp_c /= 1000.0
        if temp_c <= 0.0:
            continue
        zones.append({
            "type": zone_type,
            "temp_c": temp_c,
        })
    return zones

def _pick_thermal_hotspot():
    zones = _read_thermal_zones()
    if not zones:
        return None
    preferred_names = ("x86_pkg_temp", "acpitz", "cpu-thermal", "soc_thermal")
    preferred = [
        zone for zone in zones
        if zone.get("type", "").lower() in preferred_names
    ]
    sample_pool = preferred or zones
    hottest = max(sample_pool, key=lambda zone: float(zone.get("temp_c", 0.0)))
    return {
        "type": str(hottest.get("type", "")),
        "temp_c": float(hottest.get("temp_c", 0.0)),
    }

def _current_thread_id():
    try:
        return int(threading.get_native_id())
    except Exception:
        try:
            return int(os.getpid())
        except Exception:
            return 0

def _apply_linux_thread_priority(label: str, nice: int | None = None, rt_priority: int | None = None):
    """
    Best-effort Linux thread prioritization.

    Real-time priority and negative nice require CAP_SYS_NICE/root. Camera
    positive nice normally succeeds as a regular user, which still protects
    the robot/control threads when the CPU is thermally throttling.
    """
    if os.name != "posix":
        return
    tid = _current_thread_id()
    notes = []

    if rt_priority is not None and hasattr(os, "sched_setscheduler"):
        try:
            os.sched_setscheduler(tid, os.SCHED_FIFO, os.sched_param(int(rt_priority)))
            notes.append(f"SCHED_FIFO {int(rt_priority)}")
        except PermissionError:
            notes.append("RT denied")
        except Exception as e:
            notes.append(f"RT unavailable: {e}")

    if nice is not None and hasattr(os, "setpriority"):
        try:
            os.setpriority(os.PRIO_PROCESS, tid, int(nice))
            actual = os.getpriority(os.PRIO_PROCESS, tid)
            notes.append(f"nice {actual}")
        except PermissionError:
            notes.append(f"nice {int(nice)} denied")
        except Exception as e:
            notes.append(f"nice unavailable: {e}")

    if notes:
        print(f"[LinuxSched] {label} tid={tid}: {', '.join(notes)}")

def _read_cpu_governors():
    governors = []
    for path in sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")):
        gov = _read_text(path)
        if gov:
            governors.append(gov)
    return sorted(set(governors))

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
        self._thermal_throttle = threading.Event()

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
            # Return the latest immutable frame reference directly. Capture
            # threads replace self._frame with a new ndarray; they do not mutate
            # previously published frames in place. Avoiding a full copy here
            # materially reduces per-frame memory churn in the live UI.
            return self._frame

    def status(self):
        return self._status

    def stop(self):
        self._stop_event.set()

    def set_thermal_throttle(self, enabled: bool):
        if enabled:
            self._thermal_throttle.set()
        else:
            self._thermal_throttle.clear()

    def run(self):
        _apply_linux_thread_priority(self.name, nice=8)
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
                time.sleep(0.045 if self._thermal_throttle.is_set() else idle_sleep)

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
    VIEW_MODES = ("color", "fusion", "depth", "depth_crop", "pointcloud")

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
        self._desired_mode = "depth_crop"
        self.colorizer = None
        self.pointcloud = None
        self.align = None
        self._mode_lock = threading.Lock()
        self._thermal_throttle = threading.Event()

    def latest(self):
        with self._lock:
            return self._frame

    def status(self):
        return self._status

    def view_mode(self):
        with self._mode_lock:
            return self._desired_mode

    def set_view_mode(self, mode):
        mode = str(mode or "").strip().lower()
        if mode not in self.VIEW_MODES:
            return False
        with self._mode_lock:
            self._desired_mode = mode
        return True

    def cycle_view_mode(self):
        with self._mode_lock:
            idx = self.VIEW_MODES.index(self._desired_mode) if self._desired_mode in self.VIEW_MODES else 0
            self._desired_mode = self.VIEW_MODES[(idx + 1) % len(self.VIEW_MODES)]
            return self._desired_mode

    def stop(self):
        self._stop_event.set()

    def set_thermal_throttle(self, enabled: bool):
        if enabled:
            self._thermal_throttle.set()
        else:
            self._thermal_throttle.clear()

    def _start_with(self, kind):
        self.colorizer = None
        self.pointcloud = None
        self.align = None
        self.pipeline = rs.pipeline()
        config = rs.config()
        try:
            if kind == "color":
                config.enable_stream(rs.stream.color, self.req_w, self.req_h, rs.format.bgr8, self.req_fps)
            elif kind == "fusion":
                config.enable_stream(rs.stream.color, self.req_w, self.req_h, rs.format.bgr8, self.req_fps)
                config.enable_stream(rs.stream.depth, self.req_w, self.req_h, rs.format.z16, self.req_fps)
                self.colorizer = rs.colorizer()
                self.align = rs.align(rs.stream.color)
            elif kind == "infrared":
                config.enable_stream(rs.stream.infrared, self.req_w, self.req_h, rs.format.y8, self.req_fps)
            elif kind == "depth":
                config.enable_stream(rs.stream.depth, self.req_w, self.req_h, rs.format.z16, self.req_fps)
                self.colorizer = rs.colorizer()
            elif kind == "depth_crop":
                config.enable_stream(rs.stream.color, self.req_w, self.req_h, rs.format.bgr8, self.req_fps)
                config.enable_stream(rs.stream.depth, self.req_w, self.req_h, rs.format.z16, self.req_fps)
                self.colorizer = rs.colorizer()
                self.align = rs.align(rs.stream.color)
            elif kind == "pointcloud":
                config.enable_stream(rs.stream.depth, self.req_w, self.req_h, rs.format.z16, self.req_fps)
                self.pointcloud = rs.pointcloud()
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

    def _stop_pipeline(self):
        try:
            if self.pipeline:
                self.pipeline.stop()
        except Exception:
            pass
        self.pipeline = None
        self.mode = None
        self.align = None

    def _render_pointcloud(self, depth_frame):
        if depth_frame is None or self.pointcloud is None:
            return None
        try:
            points = self.pointcloud.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        except Exception:
            return None
        if verts.size == 0:
            return None

        h, w = self.req_h, self.req_w
        img = np.zeros((h, w, 3), dtype=np.uint8)
        stride = max(1, int(len(verts) / 30000))
        pts = verts[::stride]
        z = pts[:, 2]
        mask = np.isfinite(z) & (z > 0.05) & (z < 3.0)
        if not np.any(mask):
            return img
        pts = pts[mask]
        z = pts[:, 2]
        px = (w * 0.5 + (pts[:, 0] / z) * w * 0.7).astype(np.int32)
        py = (h * 0.5 - (pts[:, 1] / z) * h * 0.7).astype(np.int32)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if not np.any(inside):
            return img
        px, py, z = px[inside], py[inside], z[inside]
        zn = np.clip((z - 0.05) / 2.5, 0.0, 1.0)
        colors = cv2.applyColorMap((255 - (zn * 255).astype(np.uint8)).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        img[py, px] = colors
        return cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)

    def _center_zoom(self, img_bgr, zoom=2.4):
        if img_bgr is None:
            return None
        h, w = img_bgr.shape[:2]
        if h <= 4 or w <= 4:
            return img_bgr
        zoom = max(1.0, float(zoom))
        crop_w = max(4, int(w / zoom))
        crop_h = max(4, int(h / zoom))
        x0 = max(0, (w - crop_w) // 2)
        y0 = max(0, (h - crop_h) // 2)
        crop = img_bgr[y0:y0 + crop_h, x0:x0 + crop_w]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

    def run(self):
        _apply_linux_thread_priority("RealSense", nice=8)
        if not self._start_with(self.view_mode()):
            self.set_view_mode("color")
        if self.pipeline is None and not self._start_with("color"):
            if not self._start_with("infrared"):
                if not self._start_with("depth"):
                    self._status = "RealSense: no usable stream"
                    return

        last_frame_t = 0.0
        try:
            while not self._stop_event.is_set():
                desired = self.view_mode()
                if desired != self.mode:
                    self._stop_pipeline()
                    if not self._start_with(desired):
                        with self._lock:
                            self._frame = None
                        time.sleep(0.2)
                        continue

                if self._thermal_throttle.is_set():
                    now_m = time.monotonic()
                    if now_m - last_frame_t < 0.10:
                        time.sleep(0.01)
                        continue

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
                elif self.mode == "fusion":
                    try:
                        if self.align:
                            frames = self.align.process(frames)
                    except Exception:
                        pass
                    cf = frames.get_color_frame()
                    df = frames.get_depth_frame()
                    if cf and df and self.colorizer:
                        color = np.asanyarray(cf.get_data())
                        depth_rgb = np.asanyarray(self.colorizer.process(df).get_data())
                        depth_bgr = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
                        if depth_bgr.shape[:2] != color.shape[:2]:
                            depth_bgr = cv2.resize(depth_bgr, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
                        img_bgr = cv2.addWeighted(color, 0.68, depth_bgr, 0.32, 0.0)
                    elif cf:
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
                elif self.mode == "depth_crop":
                    try:
                        if self.align:
                            frames = self.align.process(frames)
                    except Exception:
                        pass
                    cf = frames.get_color_frame()
                    df = frames.get_depth_frame()
                    if cf:
                        img_bgr = np.asanyarray(cf.get_data()).copy()
                    if df and self.colorizer:
                        depth_rgb = np.asanyarray(self.colorizer.process(df).get_data())
                        depth_bgr = cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
                        if img_bgr is None:
                            img_bgr = np.zeros_like(depth_bgr)
                        if depth_bgr.shape[:2] != img_bgr.shape[:2]:
                            depth_bgr = cv2.resize(depth_bgr, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                        h, w = img_bgr.shape[:2]
                        inset_size = max(8, int(min(w, h) * 0.48))
                        fusion = cv2.addWeighted(img_bgr, 0.68, depth_bgr, 0.32, 0.0)
                        zoomed_fusion = self._center_zoom(fusion, zoom=2.4)
                        x0 = (w - inset_size) // 2
                        y0 = (h - inset_size) // 2
                        inset = cv2.resize(zoomed_fusion, (inset_size, inset_size), interpolation=cv2.INTER_NEAREST)
                        img_bgr[y0:y0 + inset_size, x0:x0 + inset_size] = inset
                elif self.mode == "pointcloud":
                    img_bgr = self._render_pointcloud(frames.get_depth_frame())

                if img_bgr is not None:
                    with self._lock:
                        self._frame = img_bgr
                    last_frame_t = time.monotonic()
                else:
                    time.sleep(0.005)

        finally:
            self._stop_pipeline()


def _safe_joy_rumble(joy, low=0.25, high=0.25, duration_ms=80):
    try:
        if hasattr(joy, "rumble"):
            joy.rumble(float(low), float(high), int(duration_ms))
    except Exception:
        pass

DEFAULT_MANUAL_SPEED_LEVELS = (
    0.01,  # 1%: finest practical manual jog
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,  # 100%: nominal surgical jog speed
    2.00,
    5.00,
    10.00, # 1000%: coarse positioning only
)
BASE_MANUAL_TRANSLATE_MM_S = 5.0
DEFAULT_LIVE_LINEAR_EPSILON_MM = 0.010
STABLE_LIVE_LINEAR_EPSILON_MM = DEFAULT_LIVE_LINEAR_EPSILON_MM
LIVE_LINEAR_EPSILON_PRESETS_MM = (0.010, 0.005, 0.002, 0.001)
MANUAL_SPEED_EPSILON_MULTIPLIERS = (1.0, 2.0, 5.0)

def _clamp_live_linear_epsilon_mm(value):
    try:
        value = float(value)
    except Exception:
        value = DEFAULT_LIVE_LINEAR_EPSILON_MM
    return max(0.001, min(0.050, value))

def _live_linear_epsilon_label(value):
    value = _clamp_live_linear_epsilon_mm(value)
    um = value * 1000.0
    if abs(um - round(um)) < 1e-6:
        um_text = str(int(round(um)))
    else:
        um_text = f"{um:.1f}".rstrip("0").rstrip(".")
    suffix = "default" if abs(value - DEFAULT_LIVE_LINEAR_EPSILON_MM) < 1e-9 else "experimental"
    return f"{um_text} um ({value:.3f} mm, {suffix})"

def _stable_live_linear_epsilon_mm(value):
    return max(STABLE_LIVE_LINEAR_EPSILON_MM, _clamp_live_linear_epsilon_mm(value))

def _manual_speed_levels_for_epsilon(epsilon_mm):
    epsilon_mm = _clamp_live_linear_epsilon_mm(epsilon_mm)
    threshold_levels = [
        (epsilon_mm * mult) / BASE_MANUAL_TRANSLATE_MM_S
        for mult in MANUAL_SPEED_EPSILON_MULTIPLIERS
    ]
    levels = list(DEFAULT_MANUAL_SPEED_LEVELS) + threshold_levels
    levels = sorted({round(max(0.0001, min(10.0, float(level))), 6) for level in levels})
    return levels

def _manual_speed_levels_for_audio():
    levels = set(DEFAULT_MANUAL_SPEED_LEVELS)
    for epsilon_mm in LIVE_LINEAR_EPSILON_PRESETS_MM:
        levels.update(_manual_speed_levels_for_epsilon(epsilon_mm))
    return sorted(levels)

def _nearest_speed_level_index(levels, target_level):
    if not levels:
        return 0
    try:
        target_level = float(target_level)
    except Exception:
        target_level = 1.0
    return min(range(len(levels)), key=lambda idx: abs(float(levels[idx]) - target_level))

def _apply_manual_speed_levels_for_epsilon_locked(state, epsilon_mm):
    try:
        current_level = float(state.levels[state.idx])
    except Exception:
        current_level = 1.0
    state.levels = _manual_speed_levels_for_epsilon(epsilon_mm)
    state.idx = _nearest_speed_level_index(state.levels, current_level)

def _speed_scale_percent_label(level):
    try:
        percent = float(level) * 100.0
    except Exception:
        percent = 100.0
    if percent >= 10.0:
        return f"{percent:.0f}%"
    if percent >= 1.0:
        return f"{percent:.1f}".rstrip("0").rstrip(".") + "%"
    return f"{percent:.2f}".rstrip("0").rstrip(".") + "%"

def _speed_scale_cue_name(level):
    try:
        key = f"{float(level):.6f}".rstrip("0").rstrip(".")
    except Exception:
        key = "1"
    key = key.replace(".", "p").replace("-", "m")
    return f"speed_scale_{key}"

def _speed_scale_phrase(level):
    try:
        speed_mm_s = BASE_MANUAL_TRANSLATE_MM_S * float(level)
    except Exception:
        speed_mm_s = BASE_MANUAL_TRANSLATE_MM_S
    if speed_mm_s >= 1.0:
        if abs(speed_mm_s - round(speed_mm_s)) < 1e-6:
            value = str(int(round(speed_mm_s)))
        else:
            value = f"{speed_mm_s:.1f}".rstrip("0").rstrip(".")
        unit = "millimeter" if value == "1" else "millimeters"
        return f"Speed {value} {unit}."
    speed_um_s = speed_mm_s * 1000.0
    if abs(speed_um_s - round(speed_um_s)) < 1e-6:
        value = str(int(round(speed_um_s)))
    else:
        value = f"{speed_um_s:.1f}".rstrip("0").rstrip(".")
    unit = "micron" if value == "1" else "microns"
    return f"Speed {value} {unit}."


class JoystickPollThread(threading.Thread):
    def __init__(self, joy, state, poll_hz=240.0):
        super().__init__(daemon=True, name="JoystickPoll")
        self.joy = joy
        self.state = state
        self.poll_hz = max(30.0, float(poll_hz))
        self._stop_event = threading.Event()
        self._prev_lb = False
        self._prev_rb = False

    def stop(self):
        self._stop_event.set()

    def run(self):
        _apply_linux_thread_priority("Joystick poll", nice=-5, rt_priority=10)
        period = 1.0 / self.poll_hz
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
                now = time.monotonic()
            next_t = now + period

            try:
                pygame.event.pump()
            except Exception:
                pass

            try:
                raw_hx, raw_hy = self.joy.get_hat(0)
            except Exception:
                raw_hx, raw_hy = 0, 0

            try:
                lb = bool(self.joy.get_button(4))
                rb = bool(self.joy.get_button(5))
            except Exception:
                lb, rb = False, False

            lx = _axis(self.joy, LS_X_AXIS, INVERT_LS_X)
            ly = _axis(self.joy, LS_Y_AXIS, INVERT_LS_Y)
            rx = _axis(self.joy, RS_X_AXIS, INVERT_RS_X)
            ry = _axis(self.joy, RS_Y_AXIS, INVERT_RS_Y)
            lt = _axis(self.joy, LEFT_AXIS)
            rt = _axis(self.joy, RIGHT_AXIS)

            speed_scale_changed = False
            speed_scale_level = None
            with self.state.lock:
                self.state.lx, self.state.ly = lx, ly
                self.state.rx, self.state.ry = rx, ry
                self.state.hx, self.state.hy = raw_hx, raw_hy
                self.state.lb, self.state.rb = lb, rb
                self.state.lt, self.state.rt = lt, rt

                lb_edge = lb and not self._prev_lb
                rb_edge = rb and not self._prev_rb
                prev_idx = self.state.idx
                if lb_edge:
                    self.state.idx = max(0, self.state.idx - 1)
                if rb_edge:
                    self.state.idx = min(len(self.state.levels) - 1, self.state.idx + 1)
                speed_scale_changed = self.state.idx != prev_idx
                if speed_scale_changed:
                    speed_scale_level = self.state.levels[self.state.idx]
                    if self.state.audio_cue is None:
                        self.state.audio_cue = _speed_scale_cue_name(speed_scale_level)

            self._prev_lb = lb
            self._prev_rb = rb
            if speed_scale_changed:
                _safe_joy_rumble(self.joy, 0.20, 0.35, 70)

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
MENU_BUTTON_CANDIDATES = (7, 6)  # Xbox/SDL commonly maps Menu/Start to 7; View/Back to 6.
UI_NAV_DEADZONE = 0.55
UI_NAV_REPEAT_S = 0.18

AXIS_GUARD_DELTA  = 0.20
AXIS_GUARD_WINDOW = 0.15

POSES_PATH     = "poses.json"
SETTINGS_PATH  = "settings.json"
STARTUP_SETTINGS_PATH = ".dorna_launcher.json"
MIDWAY_SUFFIX  = "__midway"

DEFAULT_TOOL_LZ     = 205.0
DEFAULT_APPROACH_MM = 25.0
RELOAD_SAFE_RETRACT_MM = 10.0
DEFAULT_TOOL_CX     = 0.0
DEFAULT_TOOL_CY     = 0.0
DEFAULT_TOOL_CENTER_RADIUS = 10.0
DEFAULT_TUNNEL_TCP_Y_MM = 0.0
DEFAULT_TUNNEL_RETRACT_MM = 10.0
DEFAULT_TUNNEL_RETURN_MM = 8.0
DEFAULT_TUNNEL_TCP_MODE = "injector"
DEFAULT_TTS_VOICE_TYPE = "default"
DEFAULT_TTS_RATE = -10
DEFAULT_AIRCRAFT_CALLOUTS_ENABLED = True
DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG = 30.0
MIN_HIGH_ANGLE_PITCH_WARNING_DEG = 0.0
MAX_HIGH_ANGLE_PITCH_WARNING_DEG = 90.0
TTS_VOICE_TYPE_CHOICES = [
    ("default", "System default"),
    ("female1", "Female 1"),
    ("female2", "Female 2"),
    ("female3", "Female 3"),
    ("male1", "Male 1"),
    ("male2", "Male 2"),
    ("male3", "Male 3"),
    ("child_female", "Child female"),
    ("child_male", "Child male"),
]
DEFAULT_ALARM_SENSITIVITY  = 1.0
DEFAULT_PID_THRESHOLD_MAIN = 200.0
DEFAULT_PID_DURATION_MAIN = 10000.0
DEFAULT_STARTUP_ALARM_THRESHOLD = 30.0
DEFAULT_STARTUP_ALARM_DURATION = 50.0
DEFAULT_PID_THRESHOLD_MIN = 1.0
DEFAULT_PID_THRESHOLD_MAX = 400.0
DEFAULT_PID_DURATION_MIN = 1.0
DEFAULT_PID_DURATION_MAX = 20000.0
HALT_TUNE_THRESHOLD_MIN = DEFAULT_PID_THRESHOLD_MIN
HALT_TUNE_DURATION_MIN = DEFAULT_PID_DURATION_MIN
HALT_TUNE_DURATION_MAX = 1000.0
HALT_TUNE_BASELINE_DURATION_MAX = 10000.0
HALT_TUNE_MOVE_MM = 50.0
HALT_TUNE_MOVE_VEL = 25.0  # 500% manual speed scale equivalent
HALT_TUNE_HOLD_S = 0.75
COLLISION_JOINT_AXES = ("j0", "j1", "j2", "j3", "j4", "j5")
COLLISION_TCP_AXES = ("x", "y", "z", "a", "b", "c")
COLLISION_PATH_STEP_DEG = 6.0

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

def _clamp_high_angle_pitch_warning_deg(value):
    try:
        value = float(value)
    except Exception:
        value = DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG
    return max(
        MIN_HIGH_ANGLE_PITCH_WARNING_DEG,
        min(MAX_HIGH_ANGLE_PITCH_WARNING_DEG, value),
    )

def _normalize_tts_voice_type(value):
    key = str(value or DEFAULT_TTS_VOICE_TYPE).strip().lower()
    valid = {k for k, _ in TTS_VOICE_TYPE_CHOICES}
    return key if key in valid else DEFAULT_TTS_VOICE_TYPE

def _tts_voice_label(value):
    key = _normalize_tts_voice_type(value)
    for cand, label in TTS_VOICE_TYPE_CHOICES:
        if cand == key:
            return label
    return "System default"

def _next_tts_voice_type(value):
    key = _normalize_tts_voice_type(value)
    keys = [k for k, _ in TTS_VOICE_TYPE_CHOICES]
    try:
        idx = keys.index(key)
    except ValueError:
        idx = 0
    return keys[(idx + 1) % len(keys)]

def _speech_backend_name():
    for candidate in ("spd-say", "espeak-ng", "espeak"):
        if shutil.which(candidate):
            return candidate
    return None

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
        tunnel_tcp_y_mm = float(
            data.get("tunnel_tcp_y_mm", data.get("tunnel_tcp_x_mm", DEFAULT_TUNNEL_TCP_Y_MM))
        )
        tunnel_retract_mm = max(
            0.0,
            float(data.get("tunnel_retract_mm", DEFAULT_TUNNEL_RETRACT_MM)),
        )
        tunnel_return_mm = max(
            0.0,
            float(data.get("tunnel_return_mm", DEFAULT_TUNNEL_RETURN_MM)),
        )
        if tunnel_return_mm > tunnel_retract_mm:
            tunnel_return_mm = tunnel_retract_mm
        tunnel_tcp_mode = str(data.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)).strip().lower()
        if tunnel_tcp_mode not in ("injector", "tunnel"):
            tunnel_tcp_mode = DEFAULT_TUNNEL_TCP_MODE
        tunnel_injector_cy_mm = float(
            data.get("tunnel_injector_cy_mm", data.get("tunnel_injector_cx_mm", tool_cy))
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

        starred_poses = data.get("starred_poses", ["Default", "Reload"])
        if not isinstance(starred_poses, list):
            starred_poses = ["Default", "Reload"]
        starred_poses = [str(n) for n in starred_poses]

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
        injection_use_tunnel_workflow = bool(data.get("injection_use_tunnel_workflow", False))
        injection_material_loaded = bool(data.get("injection_material_loaded", False))

        # NEW: distance settings for injection retract / advance
        injection_retract_mm = float(data.get("injection_retract_mm", 10.0))
        injection_advance_mm = float(data.get("injection_advance_mm", 0.0))
        if injection_advance_mm > injection_retract_mm:
            injection_advance_mm = injection_retract_mm

        tts_voice_type = _normalize_tts_voice_type(data.get("tts_voice_type", DEFAULT_TTS_VOICE_TYPE))
        try:
            tts_rate = int(data.get("tts_rate", DEFAULT_TTS_RATE))
        except Exception:
            tts_rate = DEFAULT_TTS_RATE
        tts_rate = max(-100, min(100, tts_rate))
        aircraft_callouts_enabled = bool(
            data.get(
                "aircraft_callouts_enabled",
                data.get("gpws_callouts_enabled", DEFAULT_AIRCRAFT_CALLOUTS_ENABLED),
            )
        )
        high_angle_pitch_warning_deg = _clamp_high_angle_pitch_warning_deg(
            data.get(
                "high_angle_pitch_warning_deg",
                data.get("high_angle_pitch_deg", DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG),
            )
        )
        live_linear_epsilon_mm = _clamp_live_linear_epsilon_mm(
            data.get("live_linear_epsilon_mm", DEFAULT_LIVE_LINEAR_EPSILON_MM)
        )

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

        return {
            "tool_lz": tool_lz,
            "tool_cx": tool_cx,
            "tool_cy": tool_cy,
            "tool_center_demo_radius_mm": tool_center_demo_radius_mm,
            "tunnel_tcp_y_mm": tunnel_tcp_y_mm,
            "tunnel_retract_mm": tunnel_retract_mm,
            "tunnel_return_mm": tunnel_return_mm,
            "tunnel_tcp_mode": tunnel_tcp_mode,
            "tunnel_injector_cy_mm": tunnel_injector_cy_mm,
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
            "starred_poses": starred_poses,
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
            "injection_use_tunnel_workflow": injection_use_tunnel_workflow,
            "injection_material_loaded": injection_material_loaded,
            "injection_retract_mm": injection_retract_mm,
            "injection_advance_mm": injection_advance_mm,
            "tts_voice_type": tts_voice_type,
            "tts_rate": tts_rate,
            "aircraft_callouts_enabled": aircraft_callouts_enabled,
            "high_angle_pitch_warning_deg": high_angle_pitch_warning_deg,
            "live_linear_epsilon_mm": live_linear_epsilon_mm,
            "tool_presets": norm_presets,
            "collision_zones": collision_zones,
        }
    except Exception as e:
        print(f"⚠️ Using default settings (could not load {path}: {e})")
        return {
            "tool_lz": DEFAULT_TOOL_LZ,
            "tool_cx": DEFAULT_TOOL_CX,
            "tool_cy": DEFAULT_TOOL_CY,
            "tool_center_demo_radius_mm": DEFAULT_TOOL_CENTER_RADIUS,
            "tunnel_tcp_y_mm": DEFAULT_TUNNEL_TCP_Y_MM,
            "tunnel_retract_mm": DEFAULT_TUNNEL_RETRACT_MM,
            "tunnel_return_mm": DEFAULT_TUNNEL_RETURN_MM,
            "tunnel_tcp_mode": DEFAULT_TUNNEL_TCP_MODE,
            "tunnel_injector_cy_mm": DEFAULT_TOOL_CY,
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
            "starred_poses": ["Default", "Reload"],
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
            "injection_use_tunnel_workflow": False,
            "injection_material_loaded": False,
            "injection_retract_mm": 10.0,
            "injection_advance_mm": 0.0,
            "tts_voice_type": DEFAULT_TTS_VOICE_TYPE,
            "tts_rate": DEFAULT_TTS_RATE,
            "aircraft_callouts_enabled": DEFAULT_AIRCRAFT_CALLOUTS_ENABLED,
            "high_angle_pitch_warning_deg": DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG,
            "live_linear_epsilon_mm": DEFAULT_LIVE_LINEAR_EPSILON_MM,
            "tool_presets": [],
            "collision_zones": [],
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
        "startup_fullscreen": True,
        "startup_clear_alarm": True,
        "startup_apply_halt_settings": True,
        "startup_auto_tune_halt": False,
        "startup_alarm_threshold": DEFAULT_STARTUP_ALARM_THRESHOLD,
        "startup_alarm_duration": DEFAULT_STARTUP_ALARM_DURATION,
        "startup_jog_profile": DEFAULT_JOG_PROFILE,
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

LAUNCHER_RESULT_KEYS = (
    "host",
    "port",
    "uvc1",
    "uvc2",
    "uvc_width",
    "uvc_height",
    "uvc_fps",
    "rs_width",
    "rs_height",
    "rs_fps",
    "uvc_try_index1",
    "fullscreen",
    "clear_alarm_startup",
    "apply_halt_settings_startup",
    "auto_tune_halt_startup",
    "alarm_threshold",
    "alarm_duration",
    "jog_profile",
    "startup_calibration_test",
    "tool_center_demo_radius",
    "launcher",
)

def _launcher_result_payload(args, ok: bool):
    payload = {"ok": bool(ok)}
    for key in LAUNCHER_RESULT_KEYS:
        payload[key] = getattr(args, key, None)
    return payload

def _apply_launcher_result(args, payload):
    for key in LAUNCHER_RESULT_KEYS:
        if key in payload:
            setattr(args, key, payload[key])
    return args

def _write_launcher_result_file(result_path, args, ok: bool):
    if not result_path:
        return
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(_launcher_result_payload(args, ok=ok), f, indent=2)

def run_startup_launcher_subprocess():
    fd, result_path = tempfile.mkstemp(prefix="dorna_launcher_", suffix=".json")
    os.close(fd)
    try:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            *sys.argv[1:],
            "--launcher-subprocess",
            "--launcher-result-path",
            result_path,
        ]
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            print(f"[Launcher] Launcher subprocess exited with code {completed.returncode}.")
            return None
        if not os.path.exists(result_path):
            print("[Launcher] Launcher subprocess did not produce a result file.")
            return None
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or not payload.get("ok"):
            return None
        return payload
    except Exception as e:
        print(f"[Launcher] Could not run launcher subprocess: {e}")
        return None
    finally:
        try:
            if os.path.exists(result_path):
                os.remove(result_path)
        except Exception:
            pass

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
        else settings.get("startup_alarm_threshold", DEFAULT_STARTUP_ALARM_THRESHOLD)
    )
    args.alarm_duration = float(
        args.alarm_duration
        if getattr(args, "alarm_duration", None) is not None
        else settings.get("startup_alarm_duration", DEFAULT_STARTUP_ALARM_DURATION)
    )
    args.jog_profile = _normalize_jog_profile(
        args.jog_profile
        if getattr(args, "jog_profile", None) is not None
        else settings.get("startup_jog_profile", DEFAULT_JOG_PROFILE)
    )
    args.startup_calibration_test = _normalize_startup_calibration_test(
        getattr(args, "startup_calibration_test", None)
    )
    if getattr(args, "tool_center_demo_radius", None) is None:
        try:
            app_settings = load_settings()
            args.tool_center_demo_radius = float(
                app_settings.get("tool_center_demo_radius_mm", DEFAULT_TOOL_CENTER_RADIUS)
            )
        except Exception:
            args.tool_center_demo_radius = DEFAULT_TOOL_CENTER_RADIUS
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
    settings["startup_jog_profile"] = _normalize_jog_profile(
        getattr(args, "jog_profile", DEFAULT_JOG_PROFILE)
    )
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
    vals = [int(HALT_TUNE_THRESHOLD_MIN)]
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
    vals = [int(HALT_TUNE_DURATION_MIN)]
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


def _alarm_tune_ladder(builder, max_value: int, start_value: float):
    start_value = int(round(start_value))
    values = list(builder(int(max_value)))
    values.append(start_value)
    values = sorted(set(int(v) for v in values if int(v) > 0 and int(v) <= int(max_value)))
    if start_value not in values:
        values.append(start_value)
        values = sorted(set(values))
    return values, int(start_value)


def _run_alarm_tune_backoff_search(
    start_threshold: float,
    start_duration: float,
    test_candidate,
    log_func,
    max_threshold: int,
    max_duration: int,
    baseline_max_duration: int = None,
):
    max_threshold = int(max_threshold)
    max_duration = int(max_duration)
    if baseline_max_duration is None:
        baseline_max_duration = max_duration
    baseline_max_duration = int(max(max_duration, baseline_max_duration))
    start_threshold, start_duration = _clamp_alarm_pid(start_threshold, start_duration)
    start_threshold = max(1, min(max_threshold, int(start_threshold)))
    start_duration = max(1, min(baseline_max_duration, int(start_duration)))

    threshold_values, start_threshold = _alarm_tune_ladder(
        _build_alarm_threshold_values,
        max_threshold,
        start_threshold,
    )
    duration_values, start_duration = _alarm_tune_ladder(
        _build_alarm_duration_values,
        baseline_max_duration,
        start_duration,
    )

    def log(msg: str):
        try:
            log_func(str(msg))
        except Exception:
            pass

    baseline_threshold = start_threshold
    baseline_duration = start_duration
    log(
        "[HaltTune] Sequential tune: validate the starting pair, "
        "reduce threshold until a halt is encountered, back off, then do the same for duration."
    )
    baseline_ok = bool(test_candidate(baseline_threshold, baseline_duration))
    if not baseline_ok:
        log(
            "[HaltTune] Starting pair was not stable; increasing threshold/duration until a stable baseline is found."
        )
        if baseline_max_duration > max_duration:
            log(
                f"[HaltTune] Widening baseline duration search up to {int(baseline_max_duration)} "
                "so movement testing can proceed from a stable pair."
            )
        found = False
        asc_thresholds = [v for v in threshold_values if v >= baseline_threshold]
        asc_durations = [v for v in duration_values if v >= baseline_duration]
        for threshold_candidate in asc_thresholds:
            if threshold_candidate == baseline_threshold:
                continue
            if test_candidate(threshold_candidate, baseline_duration):
                baseline_threshold = int(threshold_candidate)
                found = True
                break
        if not found:
            for duration_candidate in asc_durations:
                for threshold_candidate in asc_thresholds:
                    if duration_candidate == baseline_duration and threshold_candidate == baseline_threshold:
                        continue
                    if test_candidate(threshold_candidate, duration_candidate):
                        baseline_threshold = int(threshold_candidate)
                        baseline_duration = int(duration_candidate)
                        found = True
                        break
                if found:
                    break
        if not found:
            log("⚠️ [HaltTune] Could not find a stable baseline pair within the allowed range.")
            return None

    log(
        f"[HaltTune] Stable baseline found at threshold={int(baseline_threshold)}, duration={int(baseline_duration)}."
    )

    tuned_threshold = int(baseline_threshold)
    threshold_desc = [v for v in threshold_values if v < tuned_threshold]
    threshold_desc.reverse()
    if threshold_desc:
        log("[HaltTune] Reducing threshold until a halt is encountered, then backing off.")
    threshold_failed = False
    for threshold_candidate in threshold_desc:
        if test_candidate(threshold_candidate, baseline_duration):
            tuned_threshold = int(threshold_candidate)
            continue
        log(
            f"[HaltTune] Threshold reduction halted at threshold={int(threshold_candidate)}, "
            f"duration={int(baseline_duration)}. Backing off to threshold={int(tuned_threshold)}."
        )
        threshold_failed = True
        break
    if threshold_desc and not threshold_failed:
        log(f"[HaltTune] Threshold remained stable down to {int(tuned_threshold)}.")

    tuned_duration = int(baseline_duration)
    duration_desc = [v for v in duration_values if v < tuned_duration]
    duration_desc.reverse()
    if duration_desc:
        log("[HaltTune] Reducing duration until a halt is encountered, then backing off.")
    duration_failed = False
    for duration_candidate in duration_desc:
        if test_candidate(tuned_threshold, duration_candidate):
            tuned_duration = int(duration_candidate)
            continue
        log(
            f"[HaltTune] Duration reduction halted at threshold={int(tuned_threshold)}, "
            f"duration={int(duration_candidate)}. Backing off to duration={int(tuned_duration)}."
        )
        duration_failed = True
        break
    if duration_desc and not duration_failed:
        log(f"[HaltTune] Duration remained stable down to {int(tuned_duration)}.")

    return int(tuned_threshold), int(tuned_duration)


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
    reload_pose = poses.get("Reload", DEFAULT_POSES["Reload"]).copy()
    go = {"cmd": "jmove", "rel": 0, "vel": HALT_TUNE_MOVE_VEL}
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
    stable = bool(settled) and (not alarm_latch.is_latched())
    if not stable:
        _launcher_tune_log(progress_cb, "[HaltTune] Candidate failed immediately after arming at Default; skipping movement phase.")
    if stable:
        for pose_name, pose, label in (
            ("Reload", reload_pose, "Default -> Reload"),
            ("Default", default_pose, "Reload -> Default"),
        ):
            try:
                _launcher_tune_log(
                    progress_cb,
                    f"[HaltTune] Movement test: {label} pose transition.",
                )
                go = {"cmd": "jmove", "rel": 0, "vel": HALT_TUNE_MOVE_VEL}
                go.update(pose)
                robot.play_dict(go)
                settled = _launcher_wait_for_joint_settle(
                    robot,
                    progress_cb=progress_cb,
                    max_wait_s=8.0,
                    stable_for_s=0.35,
                    tol_deg=0.05,
                )
                stable = bool(settled) and _launcher_hold_without_alarm(
                    alarm_latch,
                    hold_s=min(hold_s, 0.35),
                    poll_s=0.05,
                )
                if not stable:
                    _launcher_tune_log(progress_cb, f"[HaltTune] Candidate failed after the {pose_name} movement phase.")
            except Exception as e:
                _launcher_tune_log(progress_cb, f"⚠️ [HaltTune] Movement test failed during {pose_name} move: {e}")
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
        try:
            _launcher_prepare_relaxed_alarm_pid(robot, alarm_latch)
            _launcher_clear_alarm_latch(
                robot,
                alarm_latch,
                "before returning to Default after failed test",
                progress_cb=progress_cb,
            )
            robot.set_motor(1)
            go = {"cmd": "jmove", "rel": 0, "vel": HALT_TUNE_MOVE_VEL}
            go.update(default_pose)
            _launcher_tune_log(progress_cb, "[HaltTune] Returning to Default after failed candidate test.")
            robot.play_dict(go)
            _launcher_wait_for_joint_settle(
                robot,
                progress_cb=progress_cb,
                max_wait_s=8.0,
                stable_for_s=0.35,
                tol_deg=0.05,
            )
        except Exception as e:
            _launcher_tune_log(
                progress_cb,
                f"⚠️ [HaltTune] Could not return to Default after failed candidate: {e}",
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
        "[HaltTune] Using sequential backoff tuning "
        f"(threshold {int(HALT_TUNE_THRESHOLD_MIN)}..{int(DEFAULT_PID_THRESHOLD_MAX)}, "
        f"duration {int(HALT_TUNE_DURATION_MIN)}..{int(HALT_TUNE_DURATION_MAX)}).",
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

        tuned_pair = _run_alarm_tune_backoff_search(
            requested_threshold,
            requested_duration,
            lambda threshold_candidate, duration_candidate: _launcher_test_alarm_pid_candidate(
                robot,
                alarm_latch,
                threshold_candidate,
                duration_candidate,
                progress_cb=progress_cb,
                hold_s=0.75,
            ),
            lambda msg: _launcher_tune_log(progress_cb, msg),
            int(DEFAULT_PID_THRESHOLD_MAX),
            int(HALT_TUNE_DURATION_MAX),
            baseline_max_duration=int(HALT_TUNE_BASELINE_DURATION_MAX),
        )

        if tuned_pair is None:
            _launcher_tune_log(
                progress_cb,
                "⚠️ [HaltTune] No stable halt settings were found at Default pose within the allowed range.",
            )
            return None
        tuned_threshold, tuned_duration = tuned_pair

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
            robot.close()
        except Exception:
            pass
        _launcher_tune_log(
            progress_cb,
            "[HaltTune] Launcher auto-tune worker disconnected from the robot without disabling motors.",
        )

def _launcher_tcp_from_joint_dict(robot, joints: dict):
    if not isinstance(joints, dict):
        return None
    try:
        joint_list = [float(joints[f"j{i}"]) for i in range(6)]
        T_flange = np.array(robot.kinematic.t_flange_r_world(joint=joint_list), dtype=float)
        T_tcp = T_flange @ robot.kinematic.T_tcp_r_flange
        xyzabc = robot.kinematic.mat_to_xyzabc(T_tcp)
        return [float(v) for v in list(xyzabc[:6])]
    except Exception:
        return None


def _launcher_tcp_matrix_from_joint_list(robot, joint_list):
    T_flange = np.array(robot.kinematic.t_flange_r_world(joint=joint_list), dtype=float)
    return T_flange @ robot.kinematic.T_tcp_r_flange


def _launcher_rotvec_from_R(R):
    R = np.array(R, dtype=float)
    tr = float(np.trace(R))
    theta = math.acos(max(min((tr - 1.0) / 2.0, 1.0), -1.0))
    if abs(theta) < 1e-9:
        return np.zeros(3, dtype=float)
    denom = 2.0 * math.sin(theta)
    axis = np.array([
        (R[2, 1] - R[1, 2]) / denom,
        (R[0, 2] - R[2, 0]) / denom,
        (R[1, 0] - R[0, 1]) / denom,
    ], dtype=float)
    return axis * theta


def _launcher_numerical_tcp_jacobian(robot, joint_list, delta_deg=0.05):
    """Central-difference geometric Jacobian for the configured TCP.

    The translational block is in mm/rad. The angular block is rad/rad and is
    estimated from the SO(3) logarithm of the finite rotation increment.
    """
    q0 = np.array([float(v) for v in joint_list[:6]], dtype=float)
    delta_rad = math.radians(float(delta_deg))
    if delta_rad <= 0:
        delta_rad = math.radians(0.05)
    J = np.zeros((6, 6), dtype=float)
    for idx in range(6):
        q_plus = q0.copy()
        q_minus = q0.copy()
        q_plus[idx] += float(delta_deg)
        q_minus[idx] -= float(delta_deg)
        T_plus = _launcher_tcp_matrix_from_joint_list(robot, q_plus.tolist())
        T_minus = _launcher_tcp_matrix_from_joint_list(robot, q_minus.tolist())
        p_plus = np.array(T_plus[:3, 3], dtype=float).reshape(3)
        p_minus = np.array(T_minus[:3, 3], dtype=float).reshape(3)
        J[:3, idx] = (p_plus - p_minus) / (2.0 * delta_rad)

        R_plus = np.array(T_plus[:3, :3], dtype=float)
        R_minus = np.array(T_minus[:3, :3], dtype=float)
        R_delta_world = R_plus @ R_minus.T
        J[3:, idx] = _launcher_rotvec_from_R(R_delta_world) / (2.0 * delta_rad)
    return J


def _launcher_jacobian_metrics(robot, joints: dict):
    metrics = {
        "sigma_min": "",
        "sigma_max": "",
        "condition": "",
        "manip_6d": "",
        "manip_trans": "",
    }
    if not isinstance(joints, dict):
        return metrics
    try:
        joint_list = [float(joints[f"j{i}"]) for i in range(6)]
        J = None
        try:
            J = _launcher_numerical_tcp_jacobian(robot, joint_list)
        except Exception:
            J = None
        if J is None or J.ndim != 2 or J.shape[0] < 6 or J.shape[1] < 6:
            J = np.array(robot.kinematic.jacobian(joint=joint_list), dtype=float)

        # Normalize the translational rows from mm/rad to m/rad before computing
        # the 6D metric so rotational and translational rows are not separated by
        # three orders of magnitude. Keep the raw translational metric separately.
        Jn = J[:6, :6].copy()
        Jn[:3, :] /= 1000.0
        sigma = np.linalg.svd(Jn, compute_uv=False)
        sigma_min = float(np.min(sigma))
        sigma_max = float(np.max(sigma))
        if sigma_min > 1e-12:
            metrics["condition"] = sigma_max / sigma_min
        else:
            metrics["condition"] = float("inf")
        det_6d = float(np.linalg.det(Jn @ Jn.T))
        metrics["manip_6d"] = math.sqrt(max(0.0, det_6d))
        metrics["sigma_min"] = sigma_min
        metrics["sigma_max"] = sigma_max

        Jv = J[:3, :6]
        det_trans = float(np.linalg.det(Jv @ Jv.T))
        metrics["manip_trans"] = math.sqrt(max(0.0, det_trans))
    except Exception:
        pass
    return metrics


def _launcher_characterization_row(
    robot,
    t0,
    segment,
    target,
    sample_idx,
    alarm_latch,
    prev_sample=None,
    target_pose=None,
):
    now = time.time()
    t_rel = now - t0
    joints = _launcher_try_get_current_joints(robot)
    if joints is None:
        return None, prev_sample
    tcp = _launcher_tcp_from_joint_dict(robot, joints) or [""] * 6
    metrics = _launcher_jacobian_metrics(robot, joints)
    q = [float(joints[f"j{i}"]) for i in range(6)]
    xyz = [float(v) for v in tcp[:3]] if all(v != "" for v in tcp[:3]) else None
    joint_target_error_norm, joint_target_error_max = _launcher_joint_error_metrics(joints, target_pose)

    q_speed_norm = ""
    tcp_speed_norm = ""
    if isinstance(prev_sample, dict):
        dt = max(1e-9, t_rel - float(prev_sample.get("t", t_rel)))
        prev_q = prev_sample.get("q")
        prev_xyz = prev_sample.get("xyz")
        if prev_q:
            q_speed_norm = math.sqrt(sum((q[i] - prev_q[i]) ** 2 for i in range(6))) / dt
        if xyz and prev_xyz:
            tcp_speed_norm = math.sqrt(sum((xyz[i] - prev_xyz[i]) ** 2 for i in range(3))) / dt

    row = {
        "t": t_rel,
        "segment": segment,
        "target": target,
        "sample": sample_idx,
        "alarm": int(alarm_latch.is_latched()) if alarm_latch else 0,
        "q_speed_norm_deg_s": q_speed_norm,
        "tcp_speed_norm_mm_s": tcp_speed_norm,
        "joint_target_error_norm_deg": joint_target_error_norm,
        "joint_target_error_max_deg": joint_target_error_max,
        **{f"j{i}": q[i] for i in range(6)},
        "x": tcp[0],
        "y": tcp[1],
        "z": tcp[2],
        "a": tcp[3],
        "b": tcp[4],
        "c": tcp[5],
        **metrics,
    }
    next_prev = {"t": t_rel, "q": q, "xyz": xyz}
    return row, next_prev


def _launcher_write_characterization_row(writer, row):
    formatted = {}
    for key, value in row.items():
        if isinstance(value, float):
            if math.isinf(value):
                formatted[key] = "inf"
            else:
                formatted[key] = f"{value:.9g}"
        else:
            formatted[key] = value
    writer.writerow(formatted)


def _launcher_record_characterization_segment(
    robot,
    writer,
    t0,
    segment,
    target,
    min_duration_s,
    max_duration_s,
    sample_hz,
    alarm_latch,
    prev_sample=None,
    progress_cb=None,
    target_pose=None,
):
    period = 1.0 / max(1.0, float(sample_hz))
    start = time.time()
    next_sample = start
    sample_idx = 0
    stable_since = None
    prev_q_for_settle = None
    last_log_second = -1
    max_delta_seen = 0.0

    while True:
        now = time.time()
        if now < next_sample:
            time.sleep(next_sample - now)
            now = time.time()
        next_sample = now + period

        row, prev_sample = _launcher_characterization_row(
            robot,
            t0,
            segment,
            target,
            sample_idx,
            alarm_latch,
            prev_sample=prev_sample,
            target_pose=target_pose,
        )
        sample_idx += 1
        if row is not None:
            _launcher_write_characterization_row(writer, row)
            cur_q = [float(row[f"j{i}"]) for i in range(6)]
            if prev_q_for_settle is not None:
                max_delta = max(abs(cur_q[i] - prev_q_for_settle[i]) for i in range(6))
                max_delta_seen = max(max_delta_seen, max_delta)
                if max_delta <= 0.04 and (time.time() - start) >= float(min_duration_s):
                    if stable_since is None:
                        stable_since = time.time()
                else:
                    stable_since = None
            prev_q_for_settle = cur_q

        elapsed = time.time() - start
        elapsed_second = int(elapsed)
        if elapsed_second != last_log_second and elapsed_second % 2 == 0:
            last_log_second = elapsed_second
            _launcher_tune_log(
                progress_cb,
                f"[Characterization] {segment}: t={elapsed:.1f}s samples={sample_idx}",
            )

        if stable_since is not None and (time.time() - stable_since) >= 0.35:
            break
        if elapsed >= float(max_duration_s):
            _launcher_tune_log(
                progress_cb,
                f"[Characterization] {segment}: max duration reached; continuing to next segment.",
            )
            break
        if alarm_latch and alarm_latch.is_latched():
            _launcher_tune_log(
                progress_cb,
                f"[Characterization] {segment}: alarm latched; stopping characterization.",
            )
            break

    return prev_sample, sample_idx, max_delta_seen


def _launcher_joint_delta_deg(a: dict, b: dict):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    vals = []
    for axis in COLLISION_JOINT_AXES:
        try:
            vals.append(abs(float(a[axis]) - float(b[axis])))
        except Exception:
            pass
    return max(vals) if vals else 0.0


def _launcher_joint_error_metrics(measured: dict, target: dict):
    diffs = []
    if not isinstance(measured, dict) or not isinstance(target, dict):
        return "", ""
    for axis in COLLISION_JOINT_AXES:
        try:
            diffs.append(float(measured[axis]) - float(target[axis]))
        except Exception:
            pass
    if not diffs:
        return "", ""
    return math.sqrt(sum(d * d for d in diffs)), max(abs(d) for d in diffs)


def _launcher_sync_tcp_settings(robot, settings: dict, progress_cb=None, context="launcher study"):
    try:
        cx = float(settings.get("tool_cx", DEFAULT_TOOL_CX))
        cy = float(settings.get("tool_cy", DEFAULT_TOOL_CY))
        lz = float(settings.get("tool_lz", DEFAULT_TOOL_LZ))
        robot.kinematic.set_tcp_xyzabc([cx, cy, lz, 0.0, 0.0, 0.0])
        robot.play_dict(cmd={"cmd": "tool", "id": 1, "lx": cx, "ly": cy, "lz": lz}, timeout=1)
        return True
    except Exception as e:
        _launcher_tune_log(progress_cb, f"⚠️ [{context}] Could not sync TCP settings: {e}")
        return False


def _launcher_repeatability_row(robot, t0, cycle, move_index, target_name, target_pose, settle_s, alarm_latch):
    joints = _launcher_try_get_current_joints(robot)
    if joints is None:
        return None
    tcp = _launcher_tcp_from_joint_dict(robot, joints) or [""] * 6
    joint_norm, joint_max = _launcher_joint_error_metrics(joints, target_pose)
    row = {
        "t": time.time() - t0,
        "cycle": int(cycle),
        "move_index": int(move_index),
        "target": target_name,
        "settle_s": settle_s,
        "alarm": int(alarm_latch.is_latched()) if alarm_latch else 0,
        "joint_error_norm_deg": joint_norm,
        "joint_error_max_deg": joint_max,
        **{f"j{i}": float(joints[f"j{i}"]) for i in range(6)},
        **{f"target_j{i}": float(target_pose[f"j{i}"]) for i in range(6)},
        "x": tcp[0],
        "y": tcp[1],
        "z": tcp[2],
        "a": tcp[3],
        "b": tcp[4],
        "c": tcp[5],
    }
    return row


def _launcher_orientation_demo_row(
    robot,
    t0,
    segment,
    command_label,
    command_rx,
    command_ry,
    sample_idx,
    alarm_latch,
    base_q,
    base_xyz,
    target_xyzabc,
    prev_sample=None,
):
    now = time.time()
    t_rel = now - t0
    joints = _launcher_try_get_current_joints(robot)
    if joints is None:
        return None, prev_sample
    tcp = _launcher_tcp_from_joint_dict(robot, joints) or [""] * 6
    q = [float(joints[f"j{i}"]) for i in range(6)]
    xyz = [float(v) for v in tcp[:3]] if all(v != "" for v in tcp[:3]) else None
    q_delta = [q[i] - float(base_q[i]) for i in range(6)]
    joint_delta_norm = math.sqrt(sum(v * v for v in q_delta))
    joint_delta_max = max(abs(v) for v in q_delta)
    tcp_drift_norm = ""
    drift_xyz = ["", "", ""]
    if xyz and base_xyz:
        drift_xyz = [xyz[i] - float(base_xyz[i]) for i in range(3)]
        tcp_drift_norm = math.sqrt(sum(v * v for v in drift_xyz))

    q_speed_norm = ""
    tcp_speed_norm = ""
    if isinstance(prev_sample, dict):
        dt = max(1e-9, t_rel - float(prev_sample.get("t", t_rel)))
        prev_q = prev_sample.get("q")
        prev_xyz = prev_sample.get("xyz")
        if prev_q:
            q_speed_norm = math.sqrt(sum((q[i] - prev_q[i]) ** 2 for i in range(6))) / dt
        if xyz and prev_xyz:
            tcp_speed_norm = math.sqrt(sum((xyz[i] - prev_xyz[i]) ** 2 for i in range(3))) / dt

    target_xyzabc = list(target_xyzabc or [""] * 6)
    row = {
        "t": t_rel,
        "segment": segment,
        "command": command_label,
        "right_stick_x": float(command_rx),
        "right_stick_y": float(command_ry),
        "sample": sample_idx,
        "alarm": int(alarm_latch.is_latched()) if alarm_latch else 0,
        **{f"j{i}": q[i] for i in range(6)},
        "x": tcp[0],
        "y": tcp[1],
        "z": tcp[2],
        "a": tcp[3],
        "b": tcp[4],
        "c": tcp[5],
        "target_x": target_xyzabc[0],
        "target_y": target_xyzabc[1],
        "target_z": target_xyzabc[2],
        "target_a": target_xyzabc[3],
        "target_b": target_xyzabc[4],
        "target_c": target_xyzabc[5],
        "joint_delta_norm_deg": joint_delta_norm,
        "joint_delta_max_deg": joint_delta_max,
        "tcp_drift_norm_mm": tcp_drift_norm,
        "tcp_drift_x_mm": drift_xyz[0],
        "tcp_drift_y_mm": drift_xyz[1],
        "tcp_drift_z_mm": drift_xyz[2],
        "q_speed_norm_deg_s": q_speed_norm,
        "tcp_speed_norm_mm_s": tcp_speed_norm,
    }
    next_prev = {"t": t_rel, "q": q, "xyz": xyz}
    return row, next_prev


def _launcher_record_orientation_demo_segment(
    robot,
    writer,
    t0,
    segment,
    command_label,
    command_rx,
    command_ry,
    target_xyzabc,
    base_q,
    base_xyz,
    min_duration_s,
    max_duration_s,
    sample_hz,
    alarm_latch,
    prev_sample=None,
    progress_cb=None,
):
    period = 1.0 / max(1.0, float(sample_hz))
    start = time.time()
    next_sample = start
    sample_idx = 0
    stable_since = None
    prev_q_for_settle = None
    rows_written = 0
    max_joint_delta = 0.0
    max_tcp_drift = 0.0

    while True:
        now = time.time()
        if now < next_sample:
            time.sleep(next_sample - now)
            now = time.time()
        next_sample = now + period

        row, prev_sample = _launcher_orientation_demo_row(
            robot,
            t0,
            segment,
            command_label,
            command_rx,
            command_ry,
            sample_idx,
            alarm_latch,
            base_q,
            base_xyz,
            target_xyzabc,
            prev_sample=prev_sample,
        )
        sample_idx += 1
        if row is not None:
            _launcher_write_characterization_row(writer, row)
            rows_written += 1
            cur_q = [float(row[f"j{i}"]) for i in range(6)]
            max_joint_delta = max(max_joint_delta, float(row.get("joint_delta_norm_deg") or 0.0))
            try:
                max_tcp_drift = max(max_tcp_drift, float(row.get("tcp_drift_norm_mm") or 0.0))
            except Exception:
                pass
            if prev_q_for_settle is not None:
                max_step = max(abs(cur_q[i] - prev_q_for_settle[i]) for i in range(6))
                if max_step <= 0.035 and (time.time() - start) >= float(min_duration_s):
                    if stable_since is None:
                        stable_since = time.time()
                else:
                    stable_since = None
            prev_q_for_settle = cur_q

        elapsed = time.time() - start
        if stable_since is not None and (time.time() - stable_since) >= 0.25:
            break
        if elapsed >= float(max_duration_s):
            _launcher_tune_log(
                progress_cb,
                f"[RightStickDemo] {segment}: max duration reached; continuing.",
            )
            break
        if alarm_latch and alarm_latch.is_latched():
            _launcher_tune_log(
                progress_cb,
                f"[RightStickDemo] {segment}: alarm latched; stopping demo.",
            )
            break

    return prev_sample, rows_written, max_joint_delta, max_tcp_drift


def run_launcher_right_stick_tcp_demo(
    host: str,
    port: int,
    threshold: float,
    duration: float,
    progress_cb=None,
    sample_hz: float = 40.0,
    move_vel: float = 8.0,
    tilt_deg: float = None,
):
    """Emulate right-stick orientation commands while holding TCP xyz fixed."""
    poses = load_poses()
    settings = load_settings()
    default_pose = poses.get("Default", DEFAULT_POSES["Default"]).copy()
    if not default_pose:
        raise RuntimeError("Default pose is required for the right-stick TCP demo.")

    if tilt_deg is None:
        try:
            tilt_deg = float(settings.get("tool_center_demo_radius_mm", DEFAULT_TOOL_CENTER_RADIUS))
        except Exception:
            tilt_deg = DEFAULT_TOOL_CENTER_RADIUS
    tilt_deg = max(1.0, min(12.0, abs(float(tilt_deg))))

    os.makedirs(RIGHT_STICK_DEMO_ROOT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RIGHT_STICK_DEMO_ROOT_DIR, ts)
    os.makedirs(out_dir, exist_ok=False)
    csv_path = os.path.join(out_dir, "right_stick_tcp_demo.csv")
    meta_path = os.path.join(out_dir, "metadata.json")

    fieldnames = [
        "t", "segment", "command", "right_stick_x", "right_stick_y", "sample", "alarm",
        "j0", "j1", "j2", "j3", "j4", "j5",
        "x", "y", "z", "a", "b", "c",
        "target_x", "target_y", "target_z", "target_a", "target_b", "target_c",
        "joint_delta_norm_deg", "joint_delta_max_deg",
        "tcp_drift_norm_mm", "tcp_drift_x_mm", "tcp_drift_y_mm", "tcp_drift_z_mm",
        "q_speed_norm_deg_s", "tcp_speed_norm_mm_s",
    ]

    def rot_y(deg):
        return axis_angle_to_R(0.0, float(deg), 0.0)

    def rot_z(deg):
        return axis_angle_to_R(0.0, 0.0, float(deg))

    def fixed_tcp_target(base_xyz, target_R):
        a, b, c = R_to_axis_angle(orthonormalize_R(target_R))
        return [float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2]), float(a), float(b), float(c)]

    alarm_latch = _AlarmEventLatch()
    robot = Dorna(model="dorna_ta")
    connected = False
    rows_written = 0
    segment_summaries = []
    try:
        _launcher_tune_log(progress_cb, "[RightStickDemo] Connecting to robot for fixed-TCP right-stick demo.")
        for attempt in range(5):
            if robot.connect(host=host, port=port):
                connected = True
                break
            _launcher_tune_log(
                progress_cb,
                f"⚠️ [RightStickDemo] Connection attempt {attempt + 1}/5 failed. Retrying...",
            )
            time.sleep(2)
        if not connected:
            raise RuntimeError(f"All connection attempts to {host}:{port} failed.")

        try:
            robot.add_event(alarm_latch.on_robot_event)
        except Exception as e:
            _launcher_tune_log(progress_cb, f"⚠️ [RightStickDemo] Could not add alarm hook: {e}")

        _launcher_sync_tcp_settings(robot, settings, progress_cb=progress_cb, context="RightStickDemo")
        _launcher_clear_alarm_latch(robot, alarm_latch, "before right-stick demo", progress_cb=progress_cb)
        _launcher_apply_alarm_pid(robot, alarm_latch, threshold, duration, progress_cb=progress_cb)
        robot.set_motor(1)

        go_default = {"cmd": "jmove", "rel": 0, "vel": min(float(move_vel), 12.0)}
        go_default.update(default_pose)
        _launcher_tune_log(progress_cb, "[RightStickDemo] Moving to Default before fixed-TCP orientation pulses.")
        robot.play_dict(go_default)
        if not _launcher_wait_for_joint_settle(
            robot,
            progress_cb=progress_cb,
            max_wait_s=6.0,
            stable_for_s=0.75,
            tol_deg=0.05,
        ):
            raise RuntimeError("Robot did not settle at Default before right-stick demo.")

        base_joints = _launcher_try_get_current_joints(robot)
        base_tcp = _launcher_tcp_from_joint_dict(robot, base_joints)
        if base_joints is None or base_tcp is None:
            raise RuntimeError("Could not read base joints/TCP pose for right-stick demo.")
        base_q = [float(base_joints[f"j{i}"]) for i in range(6)]
        base_xyz = [float(base_tcp[i]) for i in range(3)]
        base_R = axis_angle_to_R(float(base_tcp[3]), float(base_tcp[4]), float(base_tcp[5]))
        center_target = fixed_tcp_target(base_xyz, base_R)

        pulses = [
            ("right_stick_up", "up", 0.0, 1.0, base_R @ rot_y(+tilt_deg)),
            ("return_center_after_up", "center", 0.0, 0.0, base_R),
            ("right_stick_down", "down", 0.0, -1.0, base_R @ rot_y(-tilt_deg)),
            ("return_center_after_down", "center", 0.0, 0.0, base_R),
            ("right_stick_left", "left", -1.0, 0.0, rot_z(-tilt_deg) @ base_R),
            ("return_center_after_left", "center", 0.0, 0.0, base_R),
            ("right_stick_right", "right", 1.0, 0.0, rot_z(+tilt_deg) @ base_R),
            ("return_center_after_right", "center", 0.0, 0.0, base_R),
        ]

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            t0 = time.time()
            prev_sample = None
            prev_sample, count, q_max, tcp_max = _launcher_record_orientation_demo_segment(
                robot,
                writer,
                t0,
                "baseline_fixed_tcp",
                "center",
                0.0,
                0.0,
                center_target,
                base_q,
                base_xyz,
                0.65,
                1.0,
                sample_hz,
                alarm_latch,
                prev_sample=prev_sample,
                progress_cb=progress_cb,
            )
            rows_written += count
            segment_summaries.append({
                "segment": "baseline_fixed_tcp",
                "command": "center",
                "rows": count,
                "max_joint_delta_norm_deg": q_max,
                "max_tcp_drift_norm_mm": tcp_max,
            })

            for segment, command, rx, ry, target_R in pulses:
                if alarm_latch.is_latched():
                    break
                target_xyzabc = fixed_tcp_target(base_xyz, target_R)
                cmd = {
                    "cmd": "lmove",
                    "rel": 0,
                    "x": target_xyzabc[0],
                    "y": target_xyzabc[1],
                    "z": target_xyzabc[2],
                    "a": target_xyzabc[3],
                    "b": target_xyzabc[4],
                    "c": target_xyzabc[5],
                    "vel": float(move_vel),
                }
                _launcher_tune_log(
                    progress_cb,
                    f"[RightStickDemo] Commanding {command} pulse at fixed TCP xyz; orientation amplitude={tilt_deg:.1f} deg.",
                )
                robot.play_dict(cmd, timeout=0)
                prev_sample, count, q_max, tcp_max = _launcher_record_orientation_demo_segment(
                    robot,
                    writer,
                    t0,
                    segment,
                    command,
                    rx,
                    ry,
                    target_xyzabc,
                    base_q,
                    base_xyz,
                    0.45,
                    max(1.8, tilt_deg / max(1.0, float(move_vel)) + 1.2),
                    sample_hz,
                    alarm_latch,
                    prev_sample=prev_sample,
                    progress_cb=progress_cb,
                )
                rows_written += count
                segment_summaries.append({
                    "segment": segment,
                    "command": command,
                    "rows": count,
                    "max_joint_delta_norm_deg": q_max,
                    "max_tcp_drift_norm_mm": tcp_max,
                })

        metadata = {
            "created": ts,
            "csv": csv_path,
            "host": host,
            "port": port,
            "sample_hz": float(sample_hz),
            "move_vel": float(move_vel),
            "tilt_deg": float(tilt_deg),
            "alarm_threshold": int(_clamp_alarm_pid(threshold, duration)[0]),
            "alarm_duration": int(_clamp_alarm_pid(threshold, duration)[1]),
            "rows": rows_written,
            "base_joint_deg": base_q,
            "base_tcp_xyzabc": [float(v) for v in base_tcp[:6]],
            "segments": segment_summaries,
            "settings_subset": {
                "tool_lz": settings.get("tool_lz", DEFAULT_TOOL_LZ),
                "tool_cx": settings.get("tool_cx", DEFAULT_TOOL_CX),
                "tool_cy": settings.get("tool_cy", DEFAULT_TOOL_CY),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        _emit_launcher_tune_progress(progress_cb, "artifact", path=csv_path)
        _launcher_tune_log(progress_cb, f"[RightStickDemo] Wrote {rows_written} samples to {csv_path}")
        _launcher_tune_log(progress_cb, "[RightStickDemo] Run `python3 generate_poster_assets.py` to update the fixed-TCP figure.")
        return {"csv": csv_path, "metadata": meta_path, "rows": rows_written, "dir": out_dir}
    finally:
        try:
            if connected:
                robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        _launcher_tune_log(progress_cb, "[RightStickDemo] Right-stick demo worker disconnected from the robot.")


def _launcher_pick_fixed_tcp_ik_solution(robot, target_tcp_mat, seed_joint_list, max_pos_err_mm=0.05, max_rot_err_deg=0.5):
    try:
        target_xyzabc = robot.kinematic.mat_to_xyzabc(target_tcp_mat)
        sols = robot.kinematic.inv(target_xyzabc, joint_current=seed_joint_list, all_sol=True)
    except Exception:
        return None, (float("inf"), float("inf"), float("inf"))
    if sols is None:
        return None, (float("inf"), float("inf"), float("inf"))
    try:
        if isinstance(sols, np.ndarray) and sols.ndim == 1:
            sols_iter = [sols.tolist()]
        else:
            sols_iter = list(sols)
    except Exception:
        sols_iter = [sols]

    seed = np.asarray(seed_joint_list[:6], dtype=float)
    target_pos = np.asarray(target_tcp_mat[:3, 3], dtype=float).reshape(3)
    target_R = np.asarray(target_tcp_mat[:3, :3], dtype=float)
    best = None
    best_cost = float("inf")
    best_metrics = (float("inf"), float("inf"), float("inf"))

    def rot_err_deg(target, actual):
        try:
            val = (float(np.trace(np.asarray(target, dtype=float) @ np.asarray(actual, dtype=float).T)) - 1.0) / 2.0
            return float(math.degrees(math.acos(max(-1.0, min(1.0, val)))))
        except Exception:
            return float("inf")

    for sol in sols_iter:
        try:
            cand = np.asarray(list(sol)[:6], dtype=float)
        except Exception:
            continue
        if cand.size < 6 or not np.all(np.isfinite(cand[:6])):
            continue
        try:
            T_flange = np.array(robot.kinematic.t_flange_r_world(joint=cand[:6].tolist()), dtype=float)
            T_cand = T_flange @ np.array(robot.kinematic.T_tcp_r_flange, dtype=float)
        except Exception:
            continue
        pos_err = float(np.linalg.norm(np.asarray(T_cand[:3, 3], dtype=float).reshape(3) - target_pos))
        rot_err = rot_err_deg(target_R, T_cand[:3, :3])
        joint_dist = float(np.linalg.norm(cand[:6] - seed))
        cost = pos_err * 100.0 + rot_err * 2.0 + joint_dist * 0.1
        if cost < best_cost:
            best = cand[:6].copy()
            best_cost = cost
            best_metrics = (pos_err, rot_err, joint_dist)

    if best is None:
        return None, best_metrics
    if best_metrics[0] > float(max_pos_err_mm) or best_metrics[1] > float(max_rot_err_deg):
        return None, best_metrics
    return best.tolist(), best_metrics


def run_launcher_fixed_tip_cone_demo(
    host: str,
    port: int,
    threshold: float,
    duration: float,
    progress_cb=None,
    diameter_mm: float = 50.0,
    rotations: int = 5,
    move_vel: float = 8.0,
):
    poses = load_poses()
    settings = load_settings()
    default_pose = poses.get("Default", DEFAULT_POSES["Default"]).copy()
    if not default_pose:
        raise RuntimeError("Default pose is required for the fixed-tip cone demo.")

    alarm_latch = _AlarmEventLatch()
    robot = Dorna(model="dorna_ta")
    connected = False

    def rot_z_rad(rad):
        c = math.cos(rad)
        s = math.sin(rad)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    def rot_y_deg(deg):
        rad = math.radians(deg)
        c = math.cos(rad)
        s = math.sin(rad)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)

    def rot_err_deg(target, actual):
        try:
            val = (float(np.trace(np.asarray(target, dtype=float) @ np.asarray(actual, dtype=float).T)) - 1.0) / 2.0
            return float(math.degrees(math.acos(max(-1.0, min(1.0, val)))))
        except Exception:
            return 0.0

    try:
        _launcher_tune_log(progress_cb, "[ConeDemo] Connecting to robot for fixed-tip cone rotation demo.")
        for attempt in range(5):
            if robot.connect(host=host, port=port):
                connected = True
                break
            _launcher_tune_log(progress_cb, f"⚠️ [ConeDemo] Connection attempt {attempt + 1}/5 failed. Retrying...")
            time.sleep(2)
        if not connected:
            raise RuntimeError(f"All connection attempts to {host}:{port} failed.")

        try:
            robot.add_event(alarm_latch.on_robot_event)
        except Exception as e:
            _launcher_tune_log(progress_cb, f"⚠️ [ConeDemo] Could not add alarm hook: {e}")

        _launcher_sync_tcp_settings(robot, settings, progress_cb=progress_cb, context="ConeDemo")
        _launcher_clear_alarm_latch(robot, alarm_latch, "before cone demo", progress_cb=progress_cb)
        _launcher_apply_alarm_pid(robot, alarm_latch, threshold, duration, progress_cb=progress_cb)
        robot.set_motor(1)

        go_default = {"cmd": "jmove", "rel": 0, "vel": min(float(move_vel), 12.0)}
        go_default.update(default_pose)
        _launcher_tune_log(progress_cb, "[ConeDemo] Moving to Default before fixed-tip cone rotation.")
        robot.play_dict(go_default)
        if not _launcher_wait_for_joint_settle(
            robot,
            progress_cb=progress_cb,
            max_wait_s=6.0,
            stable_for_s=0.75,
            tol_deg=0.05,
        ):
            raise RuntimeError("Robot did not settle at Default before cone demo.")

        base_joints = _launcher_try_get_current_joints(robot)
        if base_joints is None:
            raise RuntimeError("Could not read base joints for cone demo.")
        base_joint = [float(base_joints[f"j{i}"]) for i in range(6)]
        base_flange_mat = np.array(robot.kinematic.t_flange_r_world(joint=base_joint), dtype=float)
        active_tcp_r_flange = np.array(robot.kinematic.T_tcp_r_flange, dtype=float)

        tool_lz = float(settings.get("tool_lz", DEFAULT_TOOL_LZ))
        tool_cx = float(settings.get("tool_cx", DEFAULT_TOOL_CX))
        tool_cy = float(settings.get("tool_cy", DEFAULT_TOOL_CY))
        demo_lz = float(tool_lz)
        demo_source = "active"
        if abs(demo_lz) < 80.0:
            preset_lz_values = []
            for preset in settings.get("tool_presets", []) or []:
                try:
                    preset_lz = float(preset.get("lz", 0.0) or 0.0)
                except Exception:
                    continue
                if abs(preset_lz) >= 80.0:
                    preset_lz_values.append(preset_lz)
            if preset_lz_values:
                demo_lz = max(preset_lz_values, key=lambda value: abs(value))
                demo_source = "saved preset"
            else:
                demo_lz = DEFAULT_TOOL_LZ if demo_lz >= 0.0 else -DEFAULT_TOOL_LZ
                demo_source = "default"

        demo_tip_r_flange = np.eye(4, dtype=float)
        demo_tip_r_flange[0, 3] = tool_cx
        demo_tip_r_flange[1, 3] = tool_cy
        demo_tip_r_flange[2, 3] = demo_lz
        inv_demo_tip_r_flange = np.linalg.inv(demo_tip_r_flange)
        base_tip_mat = base_flange_mat @ demo_tip_r_flange
        base_tip_pos = np.asarray(base_tip_mat[:3, 3], dtype=float).reshape(3)
        base_tip_R = orthonormalize_R(np.asarray(base_tip_mat[:3, :3], dtype=float))

        lever_mm = max(1.0, float(np.linalg.norm([tool_cx, tool_cy, demo_lz])))
        radius_mm = max(1.0, float(diameter_mm) * 0.5)
        raw_tilt_deg = math.degrees(math.asin(max(-0.999, min(0.999, radius_mm / lever_mm))))
        tilt_deg = max(0.5, min(20.0, raw_tilt_deg))
        actual_diameter_mm = 2.0 * lever_mm * math.sin(math.radians(tilt_deg))
        if demo_source != "active":
            _launcher_tune_log(
                progress_cb,
                f"[ConeDemo] Using {demo_source} tool length {demo_lz:.1f} mm "
                f"(active tool_lz={tool_lz:.1f} mm) for visible physical-tip precession.",
            )
        _launcher_tune_log(
            progress_cb,
            f"[ConeDemo] Planning {int(rotations)} rotations, diameter~{actual_diameter_mm:.1f} mm, tilt={tilt_deg:.2f} deg.",
        )

        def target_R_for(theta_rad, scale=1.0):
            tilt_R = rot_z_rad(theta_rad) @ rot_y_deg(tilt_deg * float(scale)) @ rot_z_rad(-theta_rad)
            return orthonormalize_R(base_tip_R @ tilt_R)

        ramp_steps = 12
        circle_steps = 72
        cone_rotations = max(1, int(rotations or 1))
        target_Rs = []
        for idx in range(1, ramp_steps + 1):
            target_Rs.append(target_R_for(0.0, idx / ramp_steps))
        for cycle in range(cone_rotations):
            for idx in range(1, circle_steps + 1):
                target_Rs.append(target_R_for(2.0 * math.pi * (cycle + idx / circle_steps), 1.0))
        for idx in range(ramp_steps - 1, -1, -1):
            target_Rs.append(target_R_for(0.0, idx / ramp_steps))

        seed = list(base_joint)
        cmd_list = []
        path_deg = 0.0
        prev_R = base_tip_R
        for R_tgt in target_Rs:
            target_tip_mat = np.eye(4, dtype=float)
            target_tip_mat[:3, :3] = R_tgt
            target_tip_mat[:3, 3] = base_tip_pos
            target_flange_mat = target_tip_mat @ inv_demo_tip_r_flange
            target_tcp_mat = target_flange_mat @ active_tcp_r_flange
            sol, metrics = _launcher_pick_fixed_tcp_ik_solution(robot, target_tcp_mat, seed)
            if sol is None:
                pos_err, rot_err, _ = metrics
                raise RuntimeError(f"Cone demo IK failed: pos_err={pos_err * 1000.0:.1f} um rot_err={rot_err:.3f} deg")
            cand_flange = np.array(robot.kinematic.t_flange_r_world(joint=sol), dtype=float)
            cand_tip = cand_flange @ demo_tip_r_flange
            demo_tip_err = float(np.linalg.norm(np.asarray(cand_tip[:3, 3], dtype=float).reshape(3) - base_tip_pos))
            if demo_tip_err > 0.10:
                raise RuntimeError(f"Cone demo physical-tip pivot error {demo_tip_err * 1000.0:.1f} um")
            cmd = {
                "cmd": "jmove",
                "rel": 0,
                "vel": min(max(3.0, float(move_vel)), 8.0),
                "acc": 80.0,
                "jerk": 350.0,
                "cont": 1,
                "corner": 100,
            }
            cmd.update({f"j{i}": float(sol[i]) for i in range(6)})
            cmd_list.append(cmd)
            path_deg += rot_err_deg(R_tgt, prev_R)
            prev_R = R_tgt
            seed = list(sol)

        if not cmd_list:
            raise RuntimeError("Cone demo produced no commands.")
        cmd_list[-1]["cont"] = 0
        cmd_list[-1]["corner"] = 0
        estimated_s = max(2.0, path_deg / max(1e-3, min(max(3.0, float(move_vel)), 8.0)) + 1.0)
        _launcher_tune_log(progress_cb, f"[ConeDemo] Sending {len(cmd_list)} blended joint targets.")
        robot.play_list(cmd_list, timeout=0)

        end_t = time.time() + estimated_s
        while time.time() < end_t:
            if alarm_latch.is_latched():
                raise RuntimeError("Cone demo stopped because the controller alarm latched.")
            time.sleep(0.05)

        end_joints = _launcher_try_get_current_joints(robot)
        drift_um = None
        if end_joints:
            end_list = [float(end_joints[f"j{i}"]) for i in range(6)]
            end_flange = np.array(robot.kinematic.t_flange_r_world(joint=end_list), dtype=float)
            end_tip = end_flange @ demo_tip_r_flange
            drift_um = float(np.linalg.norm(np.asarray(end_tip[:3, 3], dtype=float).reshape(3) - base_tip_pos) * 1000.0)
        if drift_um is None:
            _launcher_tune_log(progress_cb, "[ConeDemo] Fixed-tip cone demo complete.")
        else:
            _launcher_tune_log(progress_cb, f"[ConeDemo] Fixed-tip cone demo complete. physical-tip drift={drift_um:.1f} um.")
        return {
            "rows": len(cmd_list),
            "diameter_mm": actual_diameter_mm,
            "rotations": cone_rotations,
            "tilt_deg": tilt_deg,
            "drift_um": drift_um,
        }
    finally:
        try:
            if connected:
                robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        _launcher_tune_log(progress_cb, "[ConeDemo] Cone demo worker disconnected from the robot.")


def run_launcher_repeatability_characterization(
    host: str,
    port: int,
    threshold: float,
    duration: float,
    progress_cb=None,
    cycles: int = 5,
    move_vel: float = 20.0,
):
    """Measure settled endpoint repeatability for repeated Default/Reload returns."""
    poses = load_poses()
    settings = load_settings()
    default_pose = poses.get("Default", DEFAULT_POSES["Default"]).copy()
    reload_pose = poses.get("Reload", DEFAULT_POSES["Reload"]).copy()
    if not default_pose or not reload_pose:
        raise RuntimeError("Default and Reload poses are required for endpoint repeatability.")

    os.makedirs(REPEATABILITY_ROOT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPEATABILITY_ROOT_DIR, ts)
    os.makedirs(out_dir, exist_ok=False)
    csv_path = os.path.join(out_dir, "pose_repeatability.csv")
    meta_path = os.path.join(out_dir, "metadata.json")
    fieldnames = [
        "t", "cycle", "move_index", "target", "settle_s", "alarm",
        "joint_error_norm_deg", "joint_error_max_deg",
        "j0", "j1", "j2", "j3", "j4", "j5",
        "target_j0", "target_j1", "target_j2", "target_j3", "target_j4", "target_j5",
        "x", "y", "z", "a", "b", "c",
    ]

    alarm_latch = _AlarmEventLatch()
    robot = Dorna(model="dorna_ta")
    connected = False
    rows_written = 0
    try:
        _launcher_tune_log(progress_cb, "[Repeatability] Connecting to robot for endpoint repeatability study.")
        for attempt in range(5):
            if robot.connect(host=host, port=port):
                connected = True
                break
            _launcher_tune_log(
                progress_cb,
                f"⚠️ [Repeatability] Connection attempt {attempt + 1}/5 failed. Retrying...",
            )
            time.sleep(2)
        if not connected:
            raise RuntimeError(f"All connection attempts to {host}:{port} failed.")

        try:
            robot.add_event(alarm_latch.on_robot_event)
        except Exception as e:
            _launcher_tune_log(progress_cb, f"⚠️ [Repeatability] Could not add alarm hook: {e}")

        _launcher_sync_tcp_settings(robot, settings, progress_cb=progress_cb, context="Repeatability")
        _launcher_clear_alarm_latch(robot, alarm_latch, "before repeatability", progress_cb=progress_cb)
        _launcher_apply_alarm_pid(robot, alarm_latch, threshold, duration, progress_cb=progress_cb)
        robot.set_motor(1)

        go_default = {"cmd": "jmove", "rel": 0, "vel": min(float(move_vel), 20.0)}
        go_default.update(default_pose)
        _launcher_tune_log(progress_cb, "[Repeatability] Moving to Default before repeated endpoint measurements.")
        robot.play_dict(go_default)
        if not _launcher_wait_for_joint_settle(
            robot,
            progress_cb=progress_cb,
            max_wait_s=6.0,
            stable_for_s=0.75,
            tol_deg=0.05,
        ):
            raise RuntimeError("Robot did not settle at Default before repeatability study.")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            t0 = time.time()
            move_index = 0
            sequence = [("Reload", reload_pose), ("Default", default_pose)]
            for cycle in range(1, max(1, int(cycles)) + 1):
                for target_name, target_pose in sequence:
                    if alarm_latch.is_latched():
                        break
                    move_index += 1
                    go = {"cmd": "jmove", "rel": 0, "vel": float(move_vel)}
                    go.update(target_pose)
                    _launcher_tune_log(
                        progress_cb,
                        f"[Repeatability] Cycle {cycle}/{max(1, int(cycles))}: moving to {target_name}.",
                    )
                    start = time.time()
                    robot.play_dict(go)
                    settled = _launcher_wait_for_joint_settle(
                        robot,
                        progress_cb=progress_cb,
                        max_wait_s=8.0,
                        stable_for_s=0.5,
                        tol_deg=0.04,
                    )
                    settle_s = time.time() - start
                    if not settled:
                        _launcher_tune_log(progress_cb, f"⚠️ [Repeatability] {target_name} did not settle cleanly.")
                    row = _launcher_repeatability_row(
                        robot,
                        t0,
                        cycle,
                        move_index,
                        target_name,
                        target_pose,
                        settle_s,
                        alarm_latch,
                    )
                    if row is not None:
                        _launcher_write_characterization_row(writer, row)
                        rows_written += 1
                if alarm_latch.is_latched():
                    break

        metadata = {
            "created": ts,
            "csv": csv_path,
            "host": host,
            "port": port,
            "cycles": max(1, int(cycles)),
            "move_vel": float(move_vel),
            "alarm_threshold": int(_clamp_alarm_pid(threshold, duration)[0]),
            "alarm_duration": int(_clamp_alarm_pid(threshold, duration)[1]),
            "rows": rows_written,
            "poses": {
                "Default": default_pose,
                "Reload": reload_pose,
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        _emit_launcher_tune_progress(progress_cb, "artifact", path=csv_path)
        _launcher_tune_log(progress_cb, f"[Repeatability] Wrote {rows_written} settled endpoints to {csv_path}")
        _launcher_tune_log(progress_cb, "[Repeatability] Run `python3 generate_poster_assets.py` to update repeatability figures.")
        return {"csv": csv_path, "metadata": meta_path, "rows": rows_written, "dir": out_dir}
    finally:
        try:
            if connected:
                robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        _launcher_tune_log(progress_cb, "[Repeatability] Repeatability worker disconnected from the robot.")


def run_launcher_motion_characterization(
    host: str,
    port: int,
    threshold: float,
    duration: float,
    progress_cb=None,
    cycles: int = 1,
    sample_hz: float = 40.0,
    move_vel: float = 20.0,
):
    """Collect a publication-style motion log from a short Default/Reload sweep.

    The launcher calls this only when explicitly requested. It records measured
    joint feedback, forward-kinematic TCP pose, velocity norms, Jacobian
    singular values, manipulability, and alarm state to CSV.
    """
    poses = load_poses()
    settings = load_settings()
    default_pose = poses.get("Default", DEFAULT_POSES["Default"]).copy()
    reload_pose = poses.get("Reload", DEFAULT_POSES["Reload"]).copy()
    if not default_pose or not reload_pose:
        raise RuntimeError("Default and Reload poses are required for motion characterization.")

    os.makedirs(CHARACTERIZATION_ROOT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(CHARACTERIZATION_ROOT_DIR, ts)
    os.makedirs(out_dir, exist_ok=False)
    csv_path = os.path.join(out_dir, "motion_characterization.csv")
    meta_path = os.path.join(out_dir, "metadata.json")

    fieldnames = [
        "t", "segment", "target", "sample", "alarm",
        "j0", "j1", "j2", "j3", "j4", "j5",
        "x", "y", "z", "a", "b", "c",
        "q_speed_norm_deg_s", "tcp_speed_norm_mm_s",
        "joint_target_error_norm_deg", "joint_target_error_max_deg",
        "sigma_min", "sigma_max", "condition", "manip_6d", "manip_trans",
    ]

    alarm_latch = _AlarmEventLatch()
    robot = Dorna(model="dorna_ta")
    connected = False
    rows_written = 0
    try:
        _launcher_tune_log(progress_cb, "[Characterization] Connecting to robot for motion logging.")
        for attempt in range(5):
            if robot.connect(host=host, port=port):
                connected = True
                break
            _launcher_tune_log(
                progress_cb,
                f"⚠️ [Characterization] Connection attempt {attempt + 1}/5 failed. Retrying...",
            )
            time.sleep(2)
        if not connected:
            raise RuntimeError(f"All connection attempts to {host}:{port} failed.")

        try:
            robot.add_event(alarm_latch.on_robot_event)
        except Exception as e:
            _launcher_tune_log(progress_cb, f"⚠️ [Characterization] Could not add alarm hook: {e}")

        try:
            cx = float(settings.get("tool_cx", DEFAULT_TOOL_CX))
            cy = float(settings.get("tool_cy", DEFAULT_TOOL_CY))
            lz = float(settings.get("tool_lz", DEFAULT_TOOL_LZ))
            robot.kinematic.set_tcp_xyzabc([cx, cy, lz, 0.0, 0.0, 0.0])
            robot.play_dict(cmd={"cmd": "tool", "id": 1, "lx": cx, "ly": cy, "lz": lz}, timeout=1)
        except Exception as e:
            _launcher_tune_log(progress_cb, f"⚠️ [Characterization] Could not sync TCP settings: {e}")

        _launcher_clear_alarm_latch(robot, alarm_latch, "before characterization", progress_cb=progress_cb)
        _launcher_apply_alarm_pid(robot, alarm_latch, threshold, duration, progress_cb=progress_cb)
        robot.set_motor(1)

        go_default = {"cmd": "jmove", "rel": 0, "vel": min(float(move_vel), 20.0)}
        go_default.update(default_pose)
        _launcher_tune_log(progress_cb, "[Characterization] Moving to Default before logging.")
        robot.play_dict(go_default)
        if not _launcher_wait_for_joint_settle(
            robot,
            progress_cb=progress_cb,
            max_wait_s=6.0,
            stable_for_s=0.75,
            tol_deg=0.05,
        ):
            raise RuntimeError("Robot did not settle at Default before characterization.")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            t0 = time.time()
            prev_sample = None
            prev_sample, count, _ = _launcher_record_characterization_segment(
                robot,
                writer,
                t0,
                "baseline_default",
                "Default",
                0.75,
                1.2,
                sample_hz,
                alarm_latch,
                prev_sample=prev_sample,
                progress_cb=progress_cb,
                target_pose=default_pose,
            )
            rows_written += count

            current_pose = default_pose
            sequence = []
            for cycle in range(max(1, int(cycles))):
                sequence.append((f"cycle{cycle + 1}_default_to_reload", "Reload", reload_pose))
                sequence.append((f"cycle{cycle + 1}_reload_to_default", "Default", default_pose))

            for segment, target_name, target_pose in sequence:
                if alarm_latch.is_latched():
                    break
                delta = _launcher_joint_delta_deg(current_pose, target_pose)
                max_duration = max(3.0, delta / max(1.0, float(move_vel)) + 6.0)
                min_duration = min(max_duration * 0.35, 2.0)
                go = {"cmd": "jmove", "rel": 0, "vel": float(move_vel)}
                go.update(target_pose)
                _launcher_tune_log(
                    progress_cb,
                    f"[Characterization] Commanding {segment} at vel={float(move_vel):.1f}; expected max joint delta={delta:.1f} deg.",
                )
                robot.play_dict(go, timeout=0)
                prev_sample, count, max_delta_seen = _launcher_record_characterization_segment(
                    robot,
                    writer,
                    t0,
                    segment,
                    target_name,
                    min_duration,
                    max_duration,
                    sample_hz,
                    alarm_latch,
                    prev_sample=prev_sample,
                    progress_cb=progress_cb,
                    target_pose=target_pose,
                )
                rows_written += count
                _launcher_tune_log(
                    progress_cb,
                    f"[Characterization] {segment} complete; max inter-sample joint delta={max_delta_seen:.3f} deg.",
                )
                current_pose = target_pose

        metadata = {
            "created": ts,
            "csv": csv_path,
            "host": host,
            "port": port,
            "cycles": max(1, int(cycles)),
            "sample_hz": float(sample_hz),
            "move_vel": float(move_vel),
            "alarm_threshold": int(_clamp_alarm_pid(threshold, duration)[0]),
            "alarm_duration": int(_clamp_alarm_pid(threshold, duration)[1]),
            "rows": rows_written,
            "poses": {
                "Default": default_pose,
                "Reload": reload_pose,
            },
            "settings_subset": {
                "tool_lz": settings.get("tool_lz", DEFAULT_TOOL_LZ),
                "tool_cx": settings.get("tool_cx", DEFAULT_TOOL_CX),
                "tool_cy": settings.get("tool_cy", DEFAULT_TOOL_CY),
                "approach_mm": settings.get("approach_mm", DEFAULT_APPROACH_MM),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        _emit_launcher_tune_progress(progress_cb, "artifact", path=csv_path)
        _launcher_tune_log(progress_cb, f"[Characterization] Wrote {rows_written} samples to {csv_path}")
        _launcher_tune_log(progress_cb, "[Characterization] Run `python3 generate_poster_assets.py` to update the measured plots.")
        return {"csv": csv_path, "metadata": meta_path, "rows": rows_written, "dir": out_dir}
    finally:
        try:
            if connected:
                robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        _launcher_tune_log(progress_cb, "[Characterization] Characterization worker disconnected from the robot.")


def run_launcher_jog_profile_benchmark(
    host: str,
    port: int,
    threshold: float,
    duration: float,
    progress_cb=None,
    profiles=None,
    stroke_mm: float = 8.0,
    stroke_s: float = 1.2,
    cycles: int = 2,
    sample_hz: float = 40.0,
):
    """Benchmark live-style jog streaming profiles with small tool-axis strokes."""
    poses = load_poses()
    settings = load_settings()
    default_pose = poses.get("Default", DEFAULT_POSES["Default"]).copy()
    if not default_pose:
        raise RuntimeError("Default pose is required for jog profile benchmarking.")

    profiles = [_normalize_jog_profile(p) for p in (profiles or JOG_PROFILE_PRESETS.keys())]
    profiles = list(dict.fromkeys(profiles))
    os.makedirs(JOG_BENCHMARK_ROOT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(JOG_BENCHMARK_ROOT_DIR, ts)
    os.makedirs(out_dir, exist_ok=False)
    csv_path = os.path.join(out_dir, "jog_profile_benchmark.csv")
    meta_path = os.path.join(out_dir, "metadata.json")

    fieldnames = [
        "t", "profile", "cycle", "segment", "cmd_index", "send_dt_s",
        "target_send_hz", "cmd_vel", "cmd_dx", "cmd_dy", "cmd_dz",
        "sample_kind", "alarm",
        "j0", "j1", "j2", "j3", "j4", "j5",
        "x", "y", "z", "a", "b", "c",
        "axis_projection_mm", "off_axis_error_mm", "distance_from_base_mm",
    ]

    def smooth_velocity(prev, raw_vel, alpha):
        raw_vel = max(0.2, float(raw_vel))
        if prev is None:
            return raw_vel
        return max(0.2, prev + (raw_vel - prev) * alpha)

    def sample_row(robot, t0, profile, cycle, segment, cmd_index, send_dt, target_hz,
                   cmd_vel, cmd_vec, sample_kind, alarm_latch, base_xyz, axis):
        joints = _launcher_try_get_current_joints(robot)
        tcp = _launcher_tcp_from_joint_dict(robot, joints) if joints else None
        if tcp:
            xyz = np.array(tcp[:3], dtype=float)
            delta = xyz - base_xyz
            axis_projection = float(np.dot(delta, axis))
            off_axis = float(np.linalg.norm(delta - axis * axis_projection))
            dist = float(np.linalg.norm(delta))
        else:
            tcp = [""] * 6
            axis_projection = off_axis = dist = ""
        return {
            "t": time.time() - t0,
            "profile": profile,
            "cycle": cycle,
            "segment": segment,
            "cmd_index": cmd_index,
            "send_dt_s": send_dt,
            "target_send_hz": target_hz,
            "cmd_vel": cmd_vel,
            "cmd_dx": cmd_vec[0],
            "cmd_dy": cmd_vec[1],
            "cmd_dz": cmd_vec[2],
            "sample_kind": sample_kind,
            "alarm": int(alarm_latch.is_latched()) if alarm_latch else 0,
            **({f"j{i}": float(joints[f"j{i}"]) for i in range(6)} if joints else {f"j{i}": "" for i in range(6)}),
            "x": tcp[0],
            "y": tcp[1],
            "z": tcp[2],
            "a": tcp[3],
            "b": tcp[4],
            "c": tcp[5],
            "axis_projection_mm": axis_projection,
            "off_axis_error_mm": off_axis,
            "distance_from_base_mm": dist,
        }

    alarm_latch = _AlarmEventLatch()
    robot = Dorna(model="dorna_ta")
    connected = False
    rows_written = 0
    send_stats = {profile: [] for profile in profiles}
    try:
        _launcher_tune_log(progress_cb, "[JogBenchmark] Connecting to robot for autonomous jog profile benchmark.")
        for attempt in range(5):
            if robot.connect(host=host, port=port):
                connected = True
                break
            _launcher_tune_log(progress_cb, f"⚠️ [JogBenchmark] Connection attempt {attempt + 1}/5 failed. Retrying...")
            time.sleep(2)
        if not connected:
            raise RuntimeError(f"All connection attempts to {host}:{port} failed.")

        try:
            robot.add_event(alarm_latch.on_robot_event)
        except Exception as e:
            _launcher_tune_log(progress_cb, f"⚠️ [JogBenchmark] Could not add alarm hook: {e}")

        _launcher_sync_tcp_settings(robot, settings, progress_cb=progress_cb, context="JogBenchmark")
        _launcher_clear_alarm_latch(robot, alarm_latch, "before jog benchmark", progress_cb=progress_cb)
        _launcher_apply_alarm_pid(robot, alarm_latch, threshold, duration, progress_cb=progress_cb)
        robot.set_motor(1)

        t0 = time.time()
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for profile in profiles:
                if alarm_latch.is_latched():
                    break
                pconf = JOG_PROFILE_PRESETS[profile]
                send_hz = float(pconf["tool_axis_hz"])
                send_interval = 1.0 / max(1.0, send_hz)
                alpha = float(pconf["vel_filter_alpha"])
                steps_per_segment = max(4, int(round(float(stroke_s) / send_interval)))
                step_mm = float(stroke_mm) / steps_per_segment
                raw_vel = abs(step_mm) / send_interval

                go_default = {"cmd": "jmove", "rel": 0, "vel": 12.0}
                go_default.update(default_pose)
                _launcher_tune_log(
                    progress_cb,
                    f"[JogBenchmark] Profile {profile}: moving to Default, then streaming +/-{float(stroke_mm):.1f} mm tool-axis strokes at {send_hz:.0f} Hz.",
                )
                robot.play_dict(go_default)
                if not _launcher_wait_for_joint_settle(robot, progress_cb=progress_cb, max_wait_s=6.0, stable_for_s=0.5, tol_deg=0.05):
                    raise RuntimeError(f"Robot did not settle at Default before profile {profile}.")

                base_pose = robot.get_all_pose()[:6]
                bx, by, bz, ba, bb, bc = [float(v) for v in base_pose]
                base_xyz = np.array([bx, by, bz], dtype=float)
                axis = np.array(axis_angle_to_R(ba, bb, bc)[:, 2], dtype=float)
                axis = axis / max(1e-9, float(np.linalg.norm(axis)))

                smoothed_vel = None
                last_send = None
                next_sample = time.time()
                sample_interval = 1.0 / max(1.0, float(sample_hz))
                cmd_index = 0

                for cycle in range(1, max(1, int(cycles)) + 1):
                    for segment, sign in (("extend", 1.0), ("retract", -1.0)):
                        for _step_idx in range(steps_per_segment):
                            if alarm_latch.is_latched():
                                break
                            target_t = time.time() + send_interval
                            d = sign * step_mm
                            cmd_vec = axis * d
                            smoothed_vel = smooth_velocity(smoothed_vel, raw_vel, alpha)
                            now = time.time()
                            send_dt = "" if last_send is None else now - last_send
                            if send_dt != "":
                                send_stats[profile].append(float(send_dt))
                            robot.play_dict({
                                "cmd": "lmove",
                                "rel": 1,
                                "x": float(cmd_vec[0]),
                                "y": float(cmd_vec[1]),
                                "z": float(cmd_vec[2]),
                                "vel": float(smoothed_vel),
                                "cont": 1,
                                "queue": 0,
                            }, timeout=0)
                            last_send = now
                            cmd_index += 1

                            if now >= next_sample:
                                row = sample_row(
                                    robot, t0, profile, cycle, segment, cmd_index,
                                    send_dt, send_hz, smoothed_vel, cmd_vec,
                                    "feedback", alarm_latch, base_xyz, axis,
                                )
                                _launcher_write_characterization_row(writer, row)
                                rows_written += 1
                                next_sample = now + sample_interval

                            sleep_s = target_t - time.time()
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                        if alarm_latch.is_latched():
                            break
                    if alarm_latch.is_latched():
                        break

                robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
                time.sleep(0.25)
                robot.play_dict({
                    "cmd": "lmove",
                    "rel": 0,
                    "x": bx,
                    "y": by,
                    "z": bz,
                    "a": ba,
                    "b": bb,
                    "c": bc,
                    "vel": 10.0,
                })
                _launcher_wait_for_joint_settle(robot, progress_cb=progress_cb, max_wait_s=4.0, stable_for_s=0.25, tol_deg=0.06)

        summary = {}
        for profile, dts in send_stats.items():
            if dts:
                arr = np.array(dts, dtype=float)
                summary[profile] = {
                    "send_dt_mean_s": float(np.mean(arr)),
                    "send_dt_std_s": float(np.std(arr)),
                    "send_dt_max_s": float(np.max(arr)),
                    "effective_send_hz": float(1.0 / max(1e-9, np.mean(arr))),
                    "samples": int(len(arr)),
                }
            else:
                summary[profile] = {
                    "send_dt_mean_s": None,
                    "send_dt_std_s": None,
                    "send_dt_max_s": None,
                    "effective_send_hz": None,
                    "samples": 0,
                }

        metadata = {
            "created": ts,
            "csv": csv_path,
            "host": host,
            "port": port,
            "profiles": profiles,
            "stroke_mm": float(stroke_mm),
            "stroke_s": float(stroke_s),
            "cycles": max(1, int(cycles)),
            "sample_hz": float(sample_hz),
            "rows": rows_written,
            "summary": summary,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        _emit_launcher_tune_progress(progress_cb, "artifact", path=csv_path)
        _launcher_tune_log(progress_cb, f"[JogBenchmark] Wrote {rows_written} samples to {csv_path}")
        for profile in profiles:
            s = summary.get(profile, {})
            hz = s.get("effective_send_hz")
            if hz:
                _launcher_tune_log(
                    progress_cb,
                    f"[JogBenchmark] {profile}: effective={hz:.1f} Hz, send jitter std={(s.get('send_dt_std_s') or 0.0) * 1000.0:.2f} ms, max gap={(s.get('send_dt_max_s') or 0.0) * 1000.0:.2f} ms.",
                )
        return {"csv": csv_path, "metadata": meta_path, "rows": rows_written, "dir": out_dir, "summary": summary}
    finally:
        try:
            if connected:
                robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        _launcher_tune_log(progress_cb, "[JogBenchmark] Jog benchmark worker disconnected from the robot.")


def show_startup_launcher(args, result_path=None):
    if not os.environ.get("DISPLAY") and sys.platform not in ("win32", "darwin"):
        print("[Launcher] DISPLAY is not set; starting without the launcher UI.")
        _write_launcher_result_file(result_path, args, ok=True)
        return args

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as e:
        print(f"[Launcher] Could not start Tk launcher: {e}")
        _write_launcher_result_file(result_path, args, ok=True)
        return args

    result = {"ok": False}
    root = tk.Tk()
    root.title("Dorna Joy Control Launcher")
    root.configure(bg="#0f1117")
    root.resizable(True, True)
    root.minsize(820, 640)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    try:
        root.option_add("*Font", ("DejaVu Sans", 10))
    except Exception:
        pass
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    bg = "#0f1117"
    panel = "#171b22"
    panel_2 = "#202632"
    field = "#0b0d12"
    text = "#eef2f7"
    muted = "#a8b3c2"
    accent = "#19d18f"
    danger = "#ff6b6b"
    style.configure(".", background=bg, foreground=text, fieldbackground=field, font=("DejaVu Sans", 10))
    style.configure("Launcher.TFrame", background=bg)
    style.configure("Panel.TFrame", background=panel)
    style.configure("TLabel", background=bg, foreground=text)
    style.configure("Muted.TLabel", background=bg, foreground=muted)
    style.configure("Header.TLabel", background=bg, foreground=text, font=("DejaVu Sans", 16, "bold"))
    style.configure("Section.TLabelframe", background=panel, foreground=text, bordercolor="#2d3542")
    style.configure("Section.TLabelframe.Label", background=panel, foreground=text, font=("DejaVu Sans", 10, "bold"))
    style.configure("TCheckbutton", background=bg, foreground=text)
    style.map("TCheckbutton", background=[("active", bg)], foreground=[("active", text)])
    style.configure("TButton", background=panel_2, foreground=text, padding=(10, 6), bordercolor="#344050")
    style.map("TButton", background=[("active", "#2a3342"), ("disabled", "#151922")], foreground=[("disabled", "#657080")])
    style.configure("Accent.TButton", background="#126b4b", foreground=text, font=("DejaVu Sans", 10, "bold"))
    style.map("Accent.TButton", background=[("active", "#16865e"), ("disabled", "#153529")])
    style.configure("TEntry", fieldbackground=field, foreground=text, insertcolor=text, bordercolor="#344050")
    style.configure("TCombobox", fieldbackground=field, background=panel_2, foreground=text, arrowcolor=text, bordercolor="#344050")
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", field), ("disabled", "#151922")],
        foreground=[("readonly", text), ("disabled", "#657080")],
        background=[("readonly", panel_2)],
    )
    style.configure("Horizontal.TScale", background=bg, troughcolor="#2a3342")
    style.configure("TNotebook", background=bg, borderwidth=0)
    style.configure("TNotebook.Tab", background="#151922", foreground=muted, padding=(14, 7))
    style.map("TNotebook.Tab", background=[("selected", panel_2), ("active", "#1c2230")], foreground=[("selected", text), ("active", text)])
    style.configure("TSeparator", background="#2d3542")
    poll_after_id = {"id": None}
    close_requested = {"value": False}

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
    jog_profile_var = tk.StringVar(
        value=_jog_profile_label(getattr(args, "jog_profile", DEFAULT_JOG_PROFILE))
    )
    startup_calibration_var = tk.StringVar(
        value=_startup_calibration_label(
            getattr(args, "startup_calibration_test", DEFAULT_STARTUP_CALIBRATION_TEST)
        )
    )
    tool_center_radius_var = tk.DoubleVar(
        value=float(getattr(args, "tool_center_demo_radius", DEFAULT_TOOL_CENTER_RADIUS))
    )
    launcher_var = tk.BooleanVar(value=bool(args.launcher))
    status_var = tk.StringVar(value="Detecting cameras...")
    alarm_status_var = tk.StringVar(value="")
    tune_status_var = tk.StringVar(value="Launcher auto-tune is idle.")
    option_paths = [""]
    tune_queue = queue.Queue()
    tune_state = {"running": False}
    launcher_vars = [
        host_var,
        port_var,
        uvc1_var,
        uvc2_var,
        uvc_quality_var,
        rs_quality_var,
        try_index1_var,
        fullscreen_var,
        clear_alarm_var,
        apply_halt_settings_var,
        auto_tune_halt_var,
        alarm_threshold_var,
        alarm_duration_var,
        halt_preset_var,
        jog_profile_var,
        startup_calibration_var,
        tool_center_radius_var,
        launcher_var,
        status_var,
        alarm_status_var,
        tune_status_var,
    ]

    frame = ttk.Frame(root, padding=16, style="Launcher.TFrame")
    frame.grid(row=0, column=0, sticky="nsew")
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)

    ttk.Label(frame, text="Dorna Joy Control", style="Header.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(
        frame,
        text="Startup configuration for robot connection, imaging, halt behavior, and optional studies.",
        style="Muted.TLabel",
        wraplength=760,
    ).grid(row=0, column=0, sticky="e", padx=(260, 0))

    notebook = ttk.Notebook(frame)
    notebook.grid(row=1, column=0, sticky="nsew", pady=(14, 12))
    setup_tab = ttk.Frame(notebook, padding=14, style="Launcher.TFrame")
    advanced_tab = ttk.Frame(notebook, padding=14, style="Launcher.TFrame")
    diagnostics_tab = ttk.Frame(notebook, padding=14, style="Launcher.TFrame")
    notebook.add(setup_tab, text="Setup")
    notebook.add(advanced_tab, text="Advanced")
    notebook.add(diagnostics_tab, text="Diagnostics")
    for tab in (setup_tab, advanced_tab, diagnostics_tab):
        tab.columnconfigure(0, weight=1)

    connection_frame = ttk.LabelFrame(setup_tab, text="Robot Connection", padding=12, style="Section.TLabelframe")
    connection_frame.grid(row=0, column=0, sticky="we")
    connection_frame.columnconfigure(1, weight=1)
    ttk.Label(connection_frame, text="Robot host").grid(row=0, column=0, sticky="w")
    host_entry = ttk.Entry(connection_frame, textvariable=host_var, width=28)
    host_entry.grid(row=0, column=1, sticky="we", padx=(10, 20))
    ttk.Label(connection_frame, text="Port").grid(row=0, column=2, sticky="w")
    port_entry = ttk.Entry(connection_frame, textvariable=port_var, width=8)
    port_entry.grid(row=0, column=3, sticky="w", padx=(10, 0))

    video_frame = ttk.LabelFrame(setup_tab, text="Imaging", padding=12, style="Section.TLabelframe")
    video_frame.grid(row=1, column=0, sticky="we", pady=(12, 0))
    video_frame.columnconfigure(1, weight=1)
    ttk.Label(video_frame, text="UVC #1").grid(row=0, column=0, sticky="w")
    uvc1_combo = ttk.Combobox(video_frame, textvariable=uvc1_var, width=76)
    uvc1_combo.grid(row=0, column=1, columnspan=3, sticky="we", padx=(10, 0))
    ttk.Label(video_frame, text="UVC #2").grid(row=1, column=0, sticky="w", pady=(8, 0))
    uvc2_combo = ttk.Combobox(video_frame, textvariable=uvc2_var, width=76)
    uvc2_combo.grid(row=1, column=1, columnspan=3, sticky="we", padx=(10, 0), pady=(8, 0))
    ttk.Label(
        video_frame,
        text="Leave a UVC path blank to auto-detect. Use Diagnostics to inspect the Linux camera inventory.",
        style="Muted.TLabel",
        wraplength=720,
    ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))
    ttk.Label(video_frame, text="UVC quality").grid(row=3, column=0, sticky="w", pady=(12, 0))
    uvc_quality_combo = ttk.Combobox(
        video_frame,
        textvariable=uvc_quality_var,
        values=_profile_choice_values(UVC_QUALITY_PRESETS, uvc_quality_var.get()),
        width=30,
        state="readonly",
    )
    uvc_quality_combo.grid(row=3, column=1, sticky="w", padx=(10, 20), pady=(12, 0))
    ttk.Label(video_frame, text="RealSense quality").grid(row=3, column=2, sticky="w", pady=(12, 0))
    rs_quality_combo = ttk.Combobox(
        video_frame,
        textvariable=rs_quality_var,
        values=_profile_choice_values(RS_QUALITY_PRESETS, rs_quality_var.get()),
        width=30,
        state="readonly",
    )
    rs_quality_combo.grid(row=3, column=3, sticky="w", padx=(10, 0), pady=(12, 0))
    ttk.Checkbutton(video_frame, text="Try sibling video-index1 if index0 has no frames", variable=try_index1_var).grid(
        row=4, column=0, columnspan=4, sticky="w", pady=(10, 0)
    )

    control_frame = ttk.LabelFrame(setup_tab, text="Control And Safety", padding=12, style="Section.TLabelframe")
    control_frame.grid(row=2, column=0, sticky="we", pady=(12, 0))
    control_frame.columnconfigure(1, weight=1)
    ttk.Checkbutton(control_frame, text="Clear latched alarms on startup", variable=clear_alarm_var).grid(
        row=0, column=0, columnspan=4, sticky="w"
    )
    ttk.Checkbutton(control_frame, text="Apply halt settings on startup", variable=apply_halt_settings_var).grid(
        row=1, column=0, columnspan=4, sticky="w", pady=(6, 0)
    )
    ttk.Label(control_frame, text="Halt threshold").grid(row=2, column=0, sticky="w", pady=(12, 0))
    threshold_scale = ttk.Scale(
        control_frame,
        from_=DEFAULT_PID_THRESHOLD_MIN,
        to=DEFAULT_PID_THRESHOLD_MAX,
        variable=alarm_threshold_var,
    )
    threshold_scale.grid(row=2, column=1, columnspan=3, sticky="we", padx=(10, 0), pady=(12, 0))
    ttk.Label(control_frame, text="Halt duration").grid(row=3, column=0, sticky="w", pady=(6, 0))
    duration_scale = ttk.Scale(
        control_frame,
        from_=DEFAULT_PID_DURATION_MIN,
        to=DEFAULT_PID_DURATION_MAX,
        variable=alarm_duration_var,
    )
    duration_scale.grid(row=3, column=1, columnspan=3, sticky="we", padx=(10, 0), pady=(6, 0))
    ttk.Label(control_frame, text="Halt preset").grid(row=4, column=0, sticky="w", pady=(6, 0))
    halt_preset_combo = ttk.Combobox(
        control_frame,
        textvariable=halt_preset_var,
        values=_halt_preset_choice_values(halt_preset_var.get()),
        width=30,
        state="readonly",
    )
    halt_preset_combo.grid(row=4, column=1, columnspan=3, sticky="w", padx=(10, 0), pady=(6, 0))
    ttk.Label(control_frame, textvariable=alarm_status_var, style="Muted.TLabel").grid(
        row=5, column=1, columnspan=3, sticky="w", padx=(10, 0), pady=(2, 0)
    )
    ttk.Label(control_frame, text="Jog profile").grid(row=6, column=0, sticky="w", pady=(12, 0))
    jog_profile_combo = ttk.Combobox(
        control_frame,
        textvariable=jog_profile_var,
        values=_jog_profile_labels(),
        width=30,
        state="readonly",
    )
    jog_profile_combo.grid(row=6, column=1, columnspan=3, sticky="w", padx=(10, 0), pady=(12, 0))
    ttk.Checkbutton(control_frame, text="Start fullscreen", variable=fullscreen_var).grid(
        row=7, column=0, columnspan=4, sticky="w", pady=(10, 0)
    )
    ttk.Checkbutton(control_frame, text="Show this launcher on startup", variable=launcher_var).grid(
        row=8, column=0, columnspan=4, sticky="w", pady=(6, 0)
    )

    startup_cal_frame = ttk.LabelFrame(advanced_tab, text="Startup Calibration", padding=12, style="Section.TLabelframe")
    startup_cal_frame.grid(row=0, column=0, sticky="we")
    startup_cal_frame.columnconfigure(1, weight=1)
    ttk.Label(startup_cal_frame, text="Calibration to run").grid(row=0, column=0, sticky="w")
    startup_calibration_combo = ttk.Combobox(
        startup_cal_frame,
        textvariable=startup_calibration_var,
        values=_startup_calibration_labels(),
        width=38,
        state="readonly",
    )
    startup_calibration_combo.grid(row=0, column=1, sticky="w", padx=(10, 0))
    start_calibration_button = ttk.Button(
        startup_cal_frame,
        text="Start Control With This Calibration",
        command=lambda: start_with_startup_calibration(),
    )
    start_calibration_button.grid(row=0, column=2, sticky="e", padx=(10, 0))
    ttk.Label(startup_cal_frame, text="Calibration deviation").grid(row=1, column=0, sticky="w", pady=(10, 0))
    tool_center_radius_scale = ttk.Scale(
        startup_cal_frame,
        from_=0.0,
        to=50.0,
        variable=tool_center_radius_var,
    )
    tool_center_radius_scale.grid(row=1, column=1, sticky="we", padx=(10, 0), pady=(10, 0))
    tool_center_radius_label = ttk.Label(startup_cal_frame, text="")
    tool_center_radius_label.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(10, 0))

    advanced_actions = ttk.LabelFrame(advanced_tab, text="Demos, Studies, And Auto-Tune", padding=12, style="Section.TLabelframe")
    advanced_actions.grid(row=1, column=0, sticky="we", pady=(12, 0))
    for col in range(3):
        advanced_actions.columnconfigure(col, weight=1)
    auto_tune_check = ttk.Checkbutton(
        advanced_actions,
        text="Tune halt sensitivity before Start Control",
        variable=auto_tune_halt_var,
    )
    auto_tune_check.grid(row=0, column=0, columnspan=3, sticky="w")
    characterize_button = ttk.Button(advanced_actions, text="Record Default-Reload Motion", command=lambda: run_characterization_now())
    characterize_button.grid(row=1, column=0, sticky="we", padx=(0, 8), pady=(12, 0))
    repeatability_button = ttk.Button(advanced_actions, text="Measure Repeatability", command=lambda: run_repeatability_now())
    repeatability_button.grid(row=1, column=1, sticky="we", padx=(0, 8), pady=(12, 0))
    right_stick_button = ttk.Button(advanced_actions, text="Record Right-Stick Fixed-TCP Pulses", command=lambda: run_right_stick_demo_now())
    right_stick_button.grid(row=1, column=2, sticky="we", pady=(12, 0))
    jog_benchmark_button = ttk.Button(advanced_actions, text="Benchmark Jog Profiles", command=lambda: run_jog_benchmark_now())
    jog_benchmark_button.grid(row=2, column=0, sticky="we", padx=(0, 8), pady=(8, 0))
    fixed_tip_cone_button = ttk.Button(advanced_actions, text="Run Fixed-Tip Cone Rotation", command=lambda: run_fixed_tip_cone_demo_now())
    fixed_tip_cone_button.grid(row=2, column=1, sticky="we", padx=(0, 8), pady=(8, 0))
    autotune_button = ttk.Button(advanced_actions, text="Tune Halt Sensitivity Now", command=lambda: run_auto_tune_now())
    autotune_button.grid(row=2, column=2, sticky="we", pady=(8, 0))

    tune_frame = ttk.LabelFrame(advanced_tab, text="Advanced Progress", padding=12, style="Section.TLabelframe")
    tune_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
    advanced_tab.rowconfigure(2, weight=1)
    tune_frame.columnconfigure(0, weight=1)
    tune_frame.rowconfigure(0, weight=1)
    tune_text = tk.Text(
        tune_frame,
        width=92,
        height=10,
        wrap="word",
        bg=field,
        fg=text,
        insertbackground=text,
        relief="flat",
        borderwidth=8,
        font=("DejaVu Sans Mono", 9),
    )
    tune_text.grid(row=0, column=0, sticky="nsew")
    tune_text.configure(state="disabled")
    ttk.Label(tune_frame, textvariable=tune_status_var, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))

    diagnostics_tab.rowconfigure(1, weight=1)
    ttk.Label(
        diagnostics_tab,
        text="Detected UVC inventory",
        font=("DejaVu Sans", 11, "bold"),
    ).grid(row=0, column=0, sticky="w")
    inventory_text = tk.Text(
        diagnostics_tab,
        width=92,
        height=18,
        wrap="word",
        bg=field,
        fg=text,
        insertbackground=text,
        relief="flat",
        borderwidth=8,
        font=("DejaVu Sans Mono", 9),
    )
    inventory_text.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
    inventory_text.configure(state="disabled")
    ttk.Label(diagnostics_tab, textvariable=status_var, foreground=danger).grid(
        row=2, column=0, sticky="w"
    )

    button_bar = ttk.Frame(frame, style="Launcher.TFrame")
    button_bar.grid(row=2, column=0, sticky="e")
    refresh_button = ttk.Button(button_bar, text="Refresh Cameras", command=lambda: refresh_inventory())
    refresh_button.grid(row=0, column=0, padx=(0, 8))
    cancel_button = ttk.Button(button_bar, text="Cancel", command=lambda: cancel())
    cancel_button.grid(row=0, column=1, padx=(0, 8))
    start_button = ttk.Button(button_bar, text="Start Control", command=lambda: start(), style="Accent.TButton")
    start_button.grid(row=0, column=2)

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

    def refresh_calibration_radius(*_args):
        try:
            tool_center_radius_label.configure(text=f"{float(tool_center_radius_var.get()):.1f} deg")
        except Exception:
            tool_center_radius_label.configure(text="")

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
            (jog_profile_combo, "readonly"),
            (startup_calibration_combo, "readonly"),
            (tool_center_radius_scale, "normal"),
            (start_calibration_button, "normal"),
            (refresh_button, "normal"),
            (cancel_button, "normal"),
            (characterize_button, "normal"),
            (repeatability_button, "normal"),
            (right_stick_button, "normal"),
            (jog_benchmark_button, "normal"),
            (fixed_tip_cone_button, "normal"),
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
        startup_calibration_test = _startup_calibration_from_label(startup_calibration_var.get())
        try:
            tool_center_demo_radius = max(0.0, min(50.0, float(tool_center_radius_var.get())))
        except Exception:
            messagebox.showerror("Invalid calibration deviation", "Calibration deviation must be a number.")
            return None
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
            "jog_profile": _jog_profile_from_label(jog_profile_var.get()),
            "startup_calibration_test": startup_calibration_test,
            "tool_center_demo_radius": tool_center_demo_radius,
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
        args.jog_profile = config["jog_profile"]
        args.startup_calibration_test = config["startup_calibration_test"]
        args.tool_center_demo_radius = config["tool_center_demo_radius"]
        args.launcher = config["launcher"]
        result["ok"] = True
        close_requested["value"] = True
        try:
            if poll_after_id["id"] is not None:
                root.after_cancel(poll_after_id["id"])
        except Exception:
            pass
        poll_after_id["id"] = None
        try:
            root.quit()
        except Exception:
            pass

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
        close_requested["value"] = True
        try:
            if poll_after_id["id"] is not None:
                root.after_cancel(poll_after_id["id"])
        except Exception:
            pass
        poll_after_id["id"] = None
        try:
            root.quit()
        except Exception:
            pass

    def start():
        config = collect_launch_settings()
        if config is None:
            return
        if config["auto_tune_halt_startup"]:
            begin_auto_tune(start_after=True)
            return
        finalize_start(config)

    def start_with_startup_calibration():
        calibration_key = _startup_calibration_from_label(startup_calibration_var.get())
        if calibration_key == "none":
            messagebox.showerror(
                "No calibration selected",
                "Choose a startup calibration routine first, then use this button.",
            )
            return
        start()

    def run_auto_tune_now():
        begin_auto_tune(start_after=False)

    def run_characterization_now():
        if tune_state["running"]:
            return
        config = collect_launch_settings()
        if config is None:
            return
        append_tune_log(
            "[Characterization] Starting measured motion characterization. "
            "The robot will move Default -> Reload -> Default and log joint/TCP/Jacobian data.",
            reset=True,
        )
        tune_status_var.set("Connecting to robot and starting motion characterization...")
        set_launcher_busy(True)

        def progress_cb(kind, **payload):
            tune_queue.put({"kind": kind, **payload})

        def worker():
            try:
                artifact = run_launcher_motion_characterization(
                    config["host"],
                    config["port"],
                    config["alarm_threshold"],
                    config["alarm_duration"],
                    progress_cb=progress_cb,
                    cycles=1,
                    sample_hz=40.0,
                    move_vel=20.0,
                )
                tune_queue.put({
                    "kind": "characterization_done",
                    "ok": True,
                    "artifact": artifact,
                })
            except Exception as e:
                tune_queue.put({
                    "kind": "characterization_done",
                    "ok": False,
                    "error": str(e),
                })

        threading.Thread(target=worker, daemon=True).start()

    def run_repeatability_now():
        if tune_state["running"]:
            return
        config = collect_launch_settings()
        if config is None:
            return
        append_tune_log(
            "[Repeatability] Starting endpoint repeatability study. "
            "The robot will repeat Default <-> Reload and log settled TCP endpoints.",
            reset=True,
        )
        tune_status_var.set("Connecting to robot and starting endpoint repeatability study...")
        set_launcher_busy(True)

        def progress_cb(kind, **payload):
            tune_queue.put({"kind": kind, **payload})

        def worker():
            try:
                artifact = run_launcher_repeatability_characterization(
                    config["host"],
                    config["port"],
                    config["alarm_threshold"],
                    config["alarm_duration"],
                    progress_cb=progress_cb,
                    cycles=5,
                    move_vel=20.0,
                )
                tune_queue.put({
                    "kind": "repeatability_done",
                    "ok": True,
                    "artifact": artifact,
                })
            except Exception as e:
                tune_queue.put({
                    "kind": "repeatability_done",
                    "ok": False,
                    "error": str(e),
                })

        threading.Thread(target=worker, daemon=True).start()

    def run_right_stick_demo_now():
        if tune_state["running"]:
            return
        config = collect_launch_settings()
        if config is None:
            return
        append_tune_log(
            "[RightStickDemo] Starting fixed-TCP right-stick orientation demo. "
            "The robot will command up/down/left/right orientation pulses while holding TCP xyz fixed.",
            reset=True,
        )
        tune_status_var.set("Connecting to robot and starting fixed-TCP right-stick demo...")
        set_launcher_busy(True)

        def progress_cb(kind, **payload):
            tune_queue.put({"kind": kind, **payload})

        def worker():
            try:
                artifact = run_launcher_right_stick_tcp_demo(
                    config["host"],
                    config["port"],
                    config["alarm_threshold"],
                    config["alarm_duration"],
                    progress_cb=progress_cb,
                    sample_hz=40.0,
                    move_vel=8.0,
                )
                tune_queue.put({
                    "kind": "right_stick_demo_done",
                    "ok": True,
                    "artifact": artifact,
                })
            except Exception as e:
                tune_queue.put({
                    "kind": "right_stick_demo_done",
                    "ok": False,
                    "error": str(e),
                })

        threading.Thread(target=worker, daemon=True).start()

    def run_fixed_tip_cone_demo_now():
        if tune_state["running"]:
            return
        config = collect_launch_settings()
        if config is None:
            return
        append_tune_log(
            "[ConeDemo] Starting physical-tip cone rotation demo. "
            "The robot will move to Default, hold the tool tip fixed, and rotate the head through five cone rotations.",
            reset=True,
        )
        tune_status_var.set("Connecting to robot and starting fixed-tip cone rotation demo...")
        set_launcher_busy(True)

        def progress_cb(kind, **payload):
            tune_queue.put({"kind": kind, **payload})

        def worker():
            try:
                summary = run_launcher_fixed_tip_cone_demo(
                    config["host"],
                    config["port"],
                    config["alarm_threshold"],
                    config["alarm_duration"],
                    progress_cb=progress_cb,
                    diameter_mm=50.0,
                    rotations=5,
                    move_vel=8.0,
                )
                tune_queue.put({
                    "kind": "fixed_tip_cone_done",
                    "ok": True,
                    "summary": summary,
                })
            except Exception as e:
                tune_queue.put({
                    "kind": "fixed_tip_cone_done",
                    "ok": False,
                    "error": str(e),
                })

        threading.Thread(target=worker, daemon=True).start()

    def run_jog_benchmark_now():
        if tune_state["running"]:
            return
        config = collect_launch_settings()
        if config is None:
            return
        append_tune_log(
            "[JogBenchmark] Starting autonomous jog profile benchmark. "
            "The robot will move to Default and stream small +/- tool-axis jog strokes for each profile.",
            reset=True,
        )
        tune_status_var.set("Connecting to robot and starting jog profile benchmark...")
        set_launcher_busy(True)

        def progress_cb(kind, **payload):
            tune_queue.put({"kind": kind, **payload})

        def worker():
            try:
                artifact = run_launcher_jog_profile_benchmark(
                    config["host"],
                    config["port"],
                    config["alarm_threshold"],
                    config["alarm_duration"],
                    progress_cb=progress_cb,
                )
                tune_queue.put({
                    "kind": "jog_benchmark_done",
                    "ok": True,
                    "artifact": artifact,
                })
            except Exception as e:
                tune_queue.put({
                    "kind": "jog_benchmark_done",
                    "ok": False,
                    "error": str(e),
                })

        threading.Thread(target=worker, daemon=True).start()

    def poll_tune_queue():
        try:
            while True:
                item = tune_queue.get_nowait()
                kind = item.get("kind")
                if kind == "log":
                    message = str(item.get("message", "")).rstrip()
                    append_tune_log(message)
                    tune_status_var.set(message or "Launcher worker is running...")
                elif kind == "artifact":
                    path = str(item.get("path", "")).strip()
                    if path:
                        append_tune_log(f"[Launcher] Characterization artifact: {path}")
                        tune_status_var.set(f"Characterization artifact written: {path}")
                elif kind == "values":
                    alarm_threshold_var.set(float(item.get("threshold", alarm_threshold_var.get())))
                    alarm_duration_var.set(float(item.get("duration", alarm_duration_var.get())))
                    refresh_alarm_status()
                elif kind == "characterization_done":
                    set_launcher_busy(False)
                    if item.get("ok"):
                        artifact = item.get("artifact") or {}
                        csv_path = artifact.get("csv", "")
                        rows = artifact.get("rows", 0)
                        tune_status_var.set(f"Motion characterization complete ({rows} samples).")
                        append_tune_log(
                            f"[Launcher] Motion characterization complete: {csv_path}"
                        )
                        append_tune_log("[Launcher] Regenerate poster figures with: python3 generate_poster_assets.py")
                    else:
                        err = str(item.get("error") or "Motion characterization failed.")
                        tune_status_var.set(err)
                        append_tune_log(f"⚠️ {err}")
                elif kind == "repeatability_done":
                    set_launcher_busy(False)
                    if item.get("ok"):
                        artifact = item.get("artifact") or {}
                        csv_path = artifact.get("csv", "")
                        rows = artifact.get("rows", 0)
                        tune_status_var.set(f"Endpoint repeatability complete ({rows} endpoints).")
                        append_tune_log(
                            f"[Launcher] Endpoint repeatability complete: {csv_path}"
                        )
                        append_tune_log("[Launcher] Regenerate poster figures with: python3 generate_poster_assets.py")
                    else:
                        err = str(item.get("error") or "Endpoint repeatability failed.")
                        tune_status_var.set(err)
                        append_tune_log(f"⚠️ {err}")
                elif kind == "right_stick_demo_done":
                    set_launcher_busy(False)
                    if item.get("ok"):
                        artifact = item.get("artifact") or {}
                        csv_path = artifact.get("csv", "")
                        rows = artifact.get("rows", 0)
                        tune_status_var.set(f"Fixed-TCP right-stick demo complete ({rows} samples).")
                        append_tune_log(
                            f"[Launcher] Fixed-TCP right-stick demo complete: {csv_path}"
                        )
                        append_tune_log("[Launcher] Regenerate poster figures with: python3 generate_poster_assets.py")
                    else:
                        err = str(item.get("error") or "Fixed-TCP right-stick demo failed.")
                        tune_status_var.set(err)
                        append_tune_log(f"⚠️ {err}")
                elif kind == "fixed_tip_cone_done":
                    set_launcher_busy(False)
                    if item.get("ok"):
                        summary = item.get("summary") or {}
                        rotations_done = summary.get("rotations", 5)
                        diameter_done = summary.get("diameter_mm", 50.0)
                        drift_um = summary.get("drift_um")
                        if drift_um is None:
                            tune_status_var.set(f"Fixed-tip cone rotation complete ({rotations_done} rotations).")
                        else:
                            tune_status_var.set(
                                f"Fixed-tip cone rotation complete ({rotations_done} rotations, drift {float(drift_um):.1f} um)."
                            )
                        append_tune_log(
                            f"[Launcher] Fixed-tip cone rotation complete: "
                            f"diameter~{float(diameter_done):.1f} mm, rotations={rotations_done}."
                        )
                    else:
                        err = str(item.get("error") or "Fixed-tip cone rotation failed.")
                        tune_status_var.set(err)
                        append_tune_log(f"⚠️ {err}")
                elif kind == "jog_benchmark_done":
                    set_launcher_busy(False)
                    if item.get("ok"):
                        artifact = item.get("artifact") or {}
                        csv_path = artifact.get("csv", "")
                        rows = artifact.get("rows", 0)
                        summary = artifact.get("summary") or {}
                        tune_status_var.set(f"Jog benchmark complete ({rows} samples).")
                        append_tune_log(f"[Launcher] Jog benchmark complete: {csv_path}")
                        for profile, stats in summary.items():
                            hz = stats.get("effective_send_hz")
                            std_s = stats.get("send_dt_std_s")
                            max_s = stats.get("send_dt_max_s")
                            if hz:
                                append_tune_log(
                                    f"[Launcher] {profile}: effective={hz:.1f} Hz, "
                                    f"jitter std={(std_s or 0.0) * 1000.0:.2f} ms, "
                                    f"max gap={(max_s or 0.0) * 1000.0:.2f} ms"
                                )
                    else:
                        err = str(item.get("error") or "Jog benchmark failed.")
                        tune_status_var.set(err)
                        append_tune_log(f"⚠️ {err}")
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
            if not close_requested["value"]:
                poll_after_id["id"] = root.after(100, poll_tune_queue)
        except Exception:
            pass

    def cleanup_launcher():
        close_requested["value"] = True
        try:
            if poll_after_id["id"] is not None:
                root.after_cancel(poll_after_id["id"])
        except Exception:
            pass
        poll_after_id["id"] = None

        # Unset Tk variables on the main thread so their destructors do not try
        # to talk to Tcl later from an arbitrary worker-thread shutdown path.
        for var in launcher_vars:
            try:
                tk_app = getattr(var, "_tk", None)
                name = getattr(var, "_name", None)
                if tk_app is not None and name and tk_app.getboolean(tk_app.call("info", "exists", name)):
                    tk_app.globalunsetvar(name)
            except Exception:
                pass
            try:
                tk_app = getattr(var, "_tk", None)
                tcl_cmds = getattr(var, "_tclCommands", None) or []
                for cmd_name in tcl_cmds:
                    try:
                        tk_app.deletecommand(cmd_name)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                var._tclCommands = None
            except Exception:
                pass
            try:
                var._tk = None
            except Exception:
                pass

        try:
            root.update_idletasks()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass

    refresh_inventory()
    refresh_alarm_status()
    threshold_scale.configure(command=lambda _v: refresh_alarm_status())
    duration_scale.configure(command=lambda _v: refresh_alarm_status())
    tool_center_radius_scale.configure(command=lambda _v: refresh_calibration_radius())
    halt_preset_combo.bind("<<ComboboxSelected>>", apply_halt_preset)
    refresh_calibration_radius()
    poll_after_id["id"] = root.after(100, poll_tune_queue)
    host_entry.focus_set()
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    cleanup_launcher()

    launched = args if result["ok"] else None
    if result_path:
        try:
            _write_launcher_result_file(
                result_path,
                launched or args,
                ok=bool(result["ok"] and launched is not None),
            )
        except Exception as e:
            print(f"[Launcher] Could not write launcher result file: {e}")
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

def injection_trigger_curve(trigger_norm):
    trigger = max(0.0, min(1.0, float(trigger_norm or 0.0)))
    if trigger <= INJECTION_TRIGGER_DEADZONE:
        return 0.0
    span = max(1e-6, 1.0 - INJECTION_TRIGGER_DEADZONE)
    scaled = (trigger - INJECTION_TRIGGER_DEADZONE) / span
    return max(0.0, min(1.0, scaled ** INJECTION_TRIGGER_EXPONENT))

def wrap_text(text: str, font: pygame.font.Font, max_width: int):
    max_width = max(20, int(max_width))
    words = text.split(' ')
    lines, cur = [], ""
    for w in words:
        if font.size(w)[0] > max_width:
            chunks = []
            chunk = ""
            for ch in w:
                test_chunk = chunk + ch
                if chunk and font.size(test_chunk)[0] > max_width:
                    chunks.append(chunk)
                    chunk = ch
                else:
                    chunk = test_chunk
            if chunk:
                chunks.append(chunk)
        else:
            chunks = [w]
        for piece in chunks:
            test = (cur + " " + piece).strip()
            if font.size(test)[0] <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = piece
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
#   MOVE_TOOL_Z <mm> # Signed tool-axis move; + advances, - retracts
#   WASH [N]       # N defaults to 1
POSE Reload
ADVANCE_WAIT
WASH
"""

ROUTINE_BLANK_TEMPLATE = """\
# New routine
# Add steps from the visual Routine Builder, or edit this script directly.
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
            m = re.match(r'^WASH(?:\s+(\d+))?$', line, re.I)
            if m:
                steps.append(("wash", int(m.group(1) or 1)))
                continue
            m = re.match(r'^PLUNGER\s+(FWD|BWD)$', line, re.I)
            if m:
                steps.append(("plunger_end", m.group(1).upper()))
                continue
            m = re.match(r'^(MOVE_TOOL_Z|TOOL_Z|MOVE_Z)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+))$', line, re.I)
            if m:
                steps.append(("tool_z", float(m.group(2))))
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

            elif typ == "tool_z":
                dist = float(arg)
                if abs(dist) <= 1e-6:
                    self._set_status_line("MOVE_TOOL_Z 0 ignored")
                    continue
                self._wait_until_final_pose(last_pose)
                direction = "advance" if dist > 0 else "retract"
                self._set_status_line(f"MOVE_TOOL_Z {dist:.3f} mm ({direction})")
                cmd_type = "inject_advance" if dist > 0 else "inject_retract"
                self.rt.enqueue({"type": cmd_type, "dist": abs(dist)})
                try:
                    vt = float(getattr(self.rt, "VT", 10.0) or 10.0)
                except Exception:
                    vt = 10.0
                wait_s = abs(dist) / max(1e-6, vt) + 0.25
                t0 = time.time()
                while not self.stop_event.is_set() and (time.time() - t0) < wait_s:
                    time.sleep(0.02)

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
        success = False
        try:
            with self.state.lock:
                self.state.syringe_calibration_running = True
                self.state.calibration_status = "Calibrating syringe…"
                self.state.audio_cue = "syringe_calibration_start"
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
                self.state.audio_cue = "syringe_calibration_complete"

            save_settings(self.state.settings)
            success = True
            print(
                "[SyringeCalib] stroke="
                f"{full_time:.3f}s, rot_total={rotations_total:.1f}"
            )

        finally:
            with self.state.lock:
                if self.state.syringe_calibration_running and not success:
                    self.state.audio_cue = "syringe_calibration_failed"
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
        self._writer_queue = None
        self._writer_stop = threading.Event()
        self._writer_thread = None
        self._dropped_frames = 0

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
                "distance,step_target,step_fraction,step_volume,"
                "delivered_volume,trigger_attenuation,injection_rate,"
                "tcp_mode,tunnel_tcp_y_mm,tool_cx,tool_cy,tool_lz\n"
            )

        # Frames index + video writers (for RS / UVC / UI)
        if self.record_video or self.record_ui:
            path = os.path.join(self.step_dir, "frames.csv")
            self.frames_log_file = open(path, "w", encoding="utf-8")
            self.frames_log_file.write("t,epoch,camera,frame_index\n")
            self.video_writers = {"rs": None, "uvc1": None, "uvc2": None, "ui": None}
            self.frame_counts = {"rs": 0, "uvc1": 0, "uvc2": 0, "ui": 0}
            self._writer_queue = queue.Queue(maxsize=180)
            self._writer_stop.clear()
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name=f"StepRecorder-{self.step_index:02d}",
                daemon=True,
            )
            self._writer_thread.start()
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
            f"{extra.get('step_target',0.0):.6f}",
            f"{extra.get('step_fraction',0.0):.6f}",
            f"{extra.get('step_volume',0.0):.6f}",
            f"{extra.get('delivered_volume',0.0):.6f}",
            f"{extra.get('trigger_attenuation',0.0):.6f}",
            f"{extra.get('injection_rate',0.0):.6f}",
            str(extra.get("tcp_mode", "")),
            f"{extra.get('tunnel_tcp_y_mm',0.0):.6f}",
            f"{extra.get('tool_cx',0.0):.6f}",
            f"{extra.get('tool_cy',0.0):.6f}",
            f"{extra.get('tool_lz',0.0):.6f}",
        ]
        self.telemetry_file.write(",".join(row) + "\n")

    def mark_outcome(self, outcome, delivered_volume=0.0, target_volume=0.0,
                     distance=0.0, target_distance=0.0, note="",
                     subject="", subject_total_delivered=0.0,
                     subject_injection_count=0, subject_injection_volumes=None):
        if not self.step_dir:
            return
        payload = {
            "outcome": str(outcome or ""),
            "delivered_volume_ul": float(delivered_volume or 0.0),
            "target_volume_ul": float(target_volume or 0.0),
            "distance": float(distance or 0.0),
            "target_distance": float(target_distance or 0.0),
            "subject": str(subject or ""),
            "subject_total_delivered_ul": float(subject_total_delivered or 0.0),
            "subject_injection_count": int(subject_injection_count or 0),
            "subject_injection_volumes_ul": list(subject_injection_volumes or []),
            "note": str(note or ""),
            "timestamp": time.time(),
        }
        try:
            with open(os.path.join(self.step_dir, "outcome.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[Recorder] Failed to write step outcome: {e}")

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
        self.video_writers[cam_key] = cv2.VideoWriter(filename, fourcc, DEFAULT_RECORDING_FPS, (w, h))

    def _enqueue_frame(self, t_rel, cam_key, frame):
        if self._writer_queue is None:
            return
        item = (float(t_rel or 0.0), str(cam_key), frame)
        try:
            self._writer_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._writer_queue.get_nowait()
            self._dropped_frames += 1
        except queue.Empty:
            pass
        try:
            self._writer_queue.put_nowait(item)
        except queue.Full:
            self._dropped_frames += 1

    def _writer_loop(self):
        _apply_linux_thread_priority("Step recorder", nice=8)
        while not self._writer_stop.is_set() or (
            self._writer_queue is not None and not self._writer_queue.empty()
        ):
            try:
                t_rel, cam_key, frame = self._writer_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write_frame_sync(t_rel, cam_key, frame)
            except Exception as e:
                print(f"[Recorder] Failed to write {cam_key} frame: {e}")
            finally:
                try:
                    self._writer_queue.task_done()
                except Exception:
                    pass

    def _write_frame_sync(self, t_rel, cam_key, frame):
        if frame is None:
            return
        if cam_key not in self.video_writers:
            return
        frame_to_write = self._frame_with_timestamp(frame, t_rel, cam_key, copy_frame=False)
        self._ensure_writer(cam_key, frame_to_write)
        writer = self.video_writers.get(cam_key)
        if writer is None:
            return
        writer.write(frame_to_write)
        idx = self.frame_counts[cam_key]
        self.frame_counts[cam_key] = idx + 1
        if self.frames_log_file:
            self.frames_log_file.write(f"{t_rel:.6f},{self.t0 + float(t_rel or 0.0):.6f},{cam_key},{idx}\n")

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
        self._enqueue_frame(t_rel, cam_key, frame.copy())

    def _frame_with_timestamp(self, frame, t_rel, cam_key, copy_frame=True):
        """Return a BGR frame with absolute and relative time burned in."""
        if frame is None:
            return
        if len(frame.shape) == 2:
            out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            out = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif copy_frame:
            out = frame.copy()
        else:
            out = frame
        epoch = self.t0 + float(t_rel or 0.0)
        whole = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))
        ms = int(max(0.0, min(0.999, epoch - math.floor(epoch))) * 1000.0)
        label = f"{whole}.{ms:03d}  t={float(t_rel or 0.0):.3f}s  step={self.step_index:02d}  {cam_key}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.75, out.shape[1] / 1280.0 * 0.58))
        thickness = 1
        (tw, th), base = cv2.getTextSize(label, font, scale, thickness)
        x, y = 10, 10 + th
        cv2.rectangle(out, (6, 6), (min(out.shape[1] - 1, x + tw + 8), y + base + 6), (0, 0, 0), -1)
        cv2.putText(out, label, (x, y), font, scale, (235, 255, 235), thickness, cv2.LINE_AA)
        return out

    def write_ui_frame(self, t_rel, surface):
        """Capture full UI surface and write as 'ui.mp4'."""
        if not self.active or not self.record_ui or surface is None:
            return
        w, h = surface.get_size()
        try:
            raw = pygame.image.tostring(surface, "RGB")
            frame_rgb = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            arr = pygame.surfarray.array3d(surface)  # (w,h,3)
            frame_rgb = np.transpose(arr, (1, 0, 2))  # (h,w,3)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._enqueue_frame(t_rel, "ui", frame_bgr)

    def stop(self):
        if not self.active:
            return
        self.active = False
        self._writer_stop.set()
        if self._writer_thread is not None:
            try:
                self._writer_thread.join(timeout=5.0)
            except Exception:
                pass
            if self._writer_thread.is_alive():
                print("[Recorder] WARNING: video writer still draining; closing remaining frames.")
        if self._dropped_frames:
            print(f"[Recorder] Dropped {self._dropped_frames} video frame(s) because encoding fell behind.")
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
    p.add_argument(
        "--jog-profile",
        choices=tuple(JOG_PROFILE_PRESETS.keys()),
        default=None,
        help="Live joystick timing profile",
    )
    p.add_argument(
        "--startup-calibration-test",
        choices=tuple(STARTUP_CALIBRATION_TESTS.keys()),
        default=None,
        help="Start a calibration test after startup settles",
    )
    p.add_argument(
        "--tool-center-demo-radius",
        type=float,
        default=None,
        help="Tool-center demo deviation in degrees",
    )

    p.add_argument("--fullscreen", dest="fullscreen", action="store_true", help="Start in fullscreen (toggle with F11)")
    p.add_argument("--windowed", dest="fullscreen", action="store_false", help="Force windowed startup")
    p.add_argument("--launcher", dest="launcher", action="store_true", help="Show the startup launcher window")
    p.add_argument("--no-launcher", dest="launcher", action="store_false", help="Skip the startup launcher window")
    p.set_defaults(fullscreen=None, launcher=None)
    p.add_argument("--launcher-subprocess", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--launcher-result-path", default=None, help=argparse.SUPPRESS)
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
        self.lt = self.rt = 0.0
        self.ui_nav_active = False

        # Tool attitude
        self.pitch = 0.0
        self.j5    = 0.0

        # Poses + settings
        self.poses = load_poses()
        for k, v in DEFAULT_POSES.items():
            self.poses.setdefault(k, v.copy())
        self.settings = load_settings()

        # Manual speed scale levels (LB/RB to change). The lowest precision
        # speeds follow the live jog target threshold so sub-1% jogs remain
        # available when the operator selects 10/5/2/1 um command targets.
        self.levels = _manual_speed_levels_for_epsilon(
            self.settings.get("live_linear_epsilon_mm", DEFAULT_LIVE_LINEAR_EPSILON_MM)
        )
        self.idx = _nearest_speed_level_index(self.levels, 1.0)  # 100%

        self.tool_lz     = float(self.settings.get("tool_lz", DEFAULT_TOOL_LZ))
        self.tool_cx     = float(self.settings.get("tool_cx", DEFAULT_TOOL_CX))
        self.tool_cy     = float(self.settings.get("tool_cy", DEFAULT_TOOL_CY))
        self.tool_center_demo_radius_mm = float(
            self.settings.get("tool_center_demo_radius_mm", DEFAULT_TOOL_CENTER_RADIUS)
        )
        self.alarm_threshold = float(self.settings.get("alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN))
        self.alarm_duration = float(self.settings.get("alarm_duration", DEFAULT_PID_DURATION_MAIN))
        self.alarm_armed = False
        self.alarm_latched = False
        self.last_alarm_message = ""
        self.tool_center_demo_mode = "circle"
        self.approach_mm = float(self.settings.get("approach_mm", DEFAULT_APPROACH_MM))
        self.tunnel_tcp_y_mm = float(self.settings.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM))
        self.tunnel_retract_mm = float(self.settings.get("tunnel_retract_mm", DEFAULT_TUNNEL_RETRACT_MM))
        self.tunnel_return_mm = float(self.settings.get("tunnel_return_mm", DEFAULT_TUNNEL_RETURN_MM))
        self.tunnel_tcp_mode = str(self.settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE))
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
        self.injection_reload_retracted_pose = None

        # Syringe calibration status
        self.calibration_status  = "Not calibrated"
        self.syringe_calibration_running = False
        self.audio_cue = None

        # Last robot joints snapshot (for logging)
        self.last_joints = None

        # Injection logging toggles
        self.record_telemetry = False
        self.record_video     = False
        self.record_ui        = False  # NEW: record full UI overlay
        self.current_injection_step = 0
        self.injection_step_in_progress = False

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
        jog_profile: str = DEFAULT_JOG_PROFILE,
        startup_calibration_test: str = DEFAULT_STARTUP_CALIBRATION_TEST,
    ):
        super().__init__(daemon=True, name="RobotControl")
        self.state      = state
        self.host       = host
        self.port       = port
        self.clear_alarm_on_launch = bool(clear_alarm_on_launch)
        self.apply_halt_settings_startup = bool(apply_halt_settings_startup)
        self.auto_tune_halt_startup = bool(auto_tune_halt_startup)
        self.jog_profile = _normalize_jog_profile(jog_profile)
        self.startup_calibration_test = _normalize_startup_calibration_test(startup_calibration_test)
        self.stop_event = threading.Event()
        self.cmd_q      = queue.Queue()
        self.robot      = None
        self.keep_motors_on_exit = True
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
                latched = True
            elif alarm_val in (0, 0.0, False):
                self.alarm_latched = False
                self.last_alarm_msg = dict(msg)
                latched = False
            else:
                return
        with self.state.lock:
            self.state.alarm_latched = latched
            self.state.last_alarm_message = json.dumps(msg, sort_keys=True)

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
        self.BSB, self.BSC  = 0.5, 1.0
        self.BSH            = 5.0
        self.VT, self.VR    = 10.0, 5.0
        self.VR_POSE        = self.VR * 2.0
        self.DZ             = 0.1
        self.left_stick_deadzone = 0.14
        self.left_stick_filter_alpha = 0.50
        self.left_stick_cmd_threshold = 0.01
        self.left_stick_release_snap = 0.03
        self.left_stick_axis_margin = 0.08
        self.left_stick_reversal_threshold = 0.20
        self.left_stick_lx_cmd = 0.0
        self.left_stick_ly_cmd = 0.0
        self.left_stick_mode = None
        self.left_stick_y_sign = 0
        self.left_stick_x_sign = 0
        self.right_stick_filter_alpha = 0.30
        self.right_stick_rx_cmd = 0.0
        self.right_stick_ry_cmd = 0.0

        self.live_motion_active = False
        self.live_lmove_dirty = False
        self.live_abs_pose_dirty = False
        self.live_rel_xyz_pending = np.zeros(3, dtype=float)
        self.live_rel_abc_pending = np.zeros(3, dtype=float)
        self.live_abs_xyz_velocity = np.zeros(3, dtype=float)
        self.live_abs_abc_velocity = np.zeros(3, dtype=float)
        self.live_j5_pending = 0.0
        profile = JOG_PROFILE_PRESETS[_normalize_jog_profile(self.jog_profile)]
        self.live_send_interval = 1.0 / max(1.0, float(profile["send_hz"]))
        self.live_send_interval_tool_axis = 1.0 / max(1.0, float(profile["tool_axis_hz"]))
        self.live_send_interval_orientation = 1.0 / 50.0
        self.live_tool_axis_lookahead_s = 0.12
        self.live_tool_axis_lookahead_max_mm = 1.2
        self.live_orientation_lookahead_s = 0.22
        self.live_orientation_lookahead_max_deg = 1.00
        self.live_orientation_max_step_s = 0.12
        self.live_orientation_velocity_gain = 1.15
        self.live_orientation_jacobian_length_mm = max(80.0, abs(float(lz)) if "lz" in locals() else 80.0)
        self.live_orientation_jacobian_damping = 1e-3
        self.live_orientation_max_tip_error_mm = 0.25
        self.live_orientation_max_rot_error_deg = 5.0
        self.live_orientation_joint_state = None
        self.live_orientation_R_state = None
        self.live_orientation_last_warn_t = 0.0
        self.live_next_send_t = 0.0
        self.live_last_send_t = 0.0
        self.live_last_abs_pose = None
        self.live_smoothed_vel = None
        self.live_vel_filter_alpha = float(profile["vel_filter_alpha"])
        self.live_feedback_poll_idle_s = 0.05
        self.live_feedback_poll_active_s = float(profile["active_feedback_s"])
        print(
            "[Jog] Profile "
            f"{self.jog_profile}: send={float(profile['send_hz']):.0f}Hz, "
            f"tool-axis={float(profile['tool_axis_hz']):.0f}Hz, "
            f"vel-filter={self.live_vel_filter_alpha:.2f}, "
            f"active-feedback={self.live_feedback_poll_active_s:.2f}s, "
            f"halt-accel={float(profile.get('halt_accel', 8.0)):.0f}/"
            f"{float(profile.get('halt_accel_translation', 12.0)):.0f}"
        )
        self.live_linear_epsilon = _stable_live_linear_epsilon_mm(
            self.state.settings.get("live_linear_epsilon_mm", DEFAULT_LIVE_LINEAR_EPSILON_MM)
        )
        self.live_angular_epsilon = 0.002
        self.live_j5_epsilon = 0.01
        self.live_halt_accel = float(profile.get("halt_accel", 8.0))
        self.live_halt_accel_translation = float(profile.get("halt_accel_translation", 12.0))
        self.live_last_motion_mode = None
        self.last_collision_zone = ""
        self.last_collision_t = 0.0
        self.orient_deadzone = 0.18
        self.last_j4_poll = None
        self.last_j4_poll_t = None

    def _play_live(self, cmd: dict):
        """
        Dispatch a live/manual motion command without waiting for completion.
        Blocking on every incremental joystick move adds visible control lag.
        """
        cmd = dict(cmd)
        if cmd.get("cmd") in ("lmove", "jmove"):
            cmd.setdefault("queue", 0)
        self.robot.play_dict(cmd, timeout=0)
        return True

    def _halt_live_motion(self):
        accel = self.live_halt_accel
        if self.live_last_motion_mode == "tool_axis_translation":
            accel = max(accel, self.live_halt_accel_translation)
        try:
            alarm_threshold = float(
                self.state.settings.get(
                    "alarm_threshold",
                    getattr(self.state, "alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN),
                )
            )
            if alarm_threshold > 0:
                # Keep release braking crisp, but avoid commanding a decel above
                # the active alarm threshold. That can trip the controller during
                # normal stick release, especially on long tool-axis jogs.
                cap_scale = 0.45 if self.live_last_motion_mode == "tool_axis_translation" else 0.65
                accel = min(float(accel), max(4.0, alarm_threshold * cap_scale))
        except Exception:
            pass
        try:
            self.robot.play_dict({"cmd": "halt", "accel": accel, "queue": 0}, timeout=0)
            return
        except Exception:
            pass
        try:
            self.robot.halt(accel=accel, timeout=0)
            return
        except Exception:
            pass
        try:
            self.robot.play_dict({"cmd": "halt", "accel": accel}, timeout=0)
        except Exception:
            pass

    def _smooth_live_velocity(self, raw_vel: float):
        try:
            raw_vel = float(raw_vel)
        except Exception:
            raw_vel = 0.2
        raw_vel = max(0.2, raw_vel)
        if self.live_smoothed_vel is None:
            self.live_smoothed_vel = raw_vel
            return raw_vel
        alpha = max(0.05, min(1.0, float(self.live_vel_filter_alpha)))
        self.live_smoothed_vel += (raw_vel - self.live_smoothed_vel) * alpha
        return max(0.2, self.live_smoothed_vel)

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
            self.state.alarm_latched = False
            self.state.last_alarm_message = ""
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
        with self.state.lock:
            self.state.alarm_latched = False
            self.state.last_alarm_message = ""
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
                self.state.alarm_latched = False
                self.state.last_alarm_message = ""
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
                reload_pose = self.state.poses.get("Reload", {}).copy()
            if not default_pose:
                default_pose = DEFAULT_POSES["Default"].copy()
            if not reload_pose:
                reload_pose = DEFAULT_POSES["Reload"].copy()
            if default_pose:
                print("[HaltTune] Returning to Default pose before candidate test.")
                if not self._queue_jmove_to_pose(default_pose, vel=HALT_TUNE_MOVE_VEL):
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
        stable = bool(settled) and (not self._is_alarm_latched())
        if not stable:
            print("[HaltTune] Candidate failed immediately after arming at Default; skipping movement phase.")
        if stable:
            for pose_name, pose, label in (
                ("Reload", reload_pose, "Default -> Reload"),
                ("Default", default_pose, "Reload -> Default"),
            ):
                try:
                    print(f"[HaltTune] Movement test: {label} pose transition.")
                    if not self._queue_jmove_to_pose(pose, vel=HALT_TUNE_MOVE_VEL):
                        stable = False
                        break
                    self._set_current_named(pose_name)
                    settled = self._wait_for_joint_settle(
                        max_wait_s=8.0,
                        stable_for_s=0.35,
                        tol_deg=0.05,
                    )
                    stable = bool(settled) and self._hold_without_alarm(
                        hold_s=min(hold_s, 0.35),
                        poll_s=0.05,
                    )
                    if not stable:
                        print(f"[HaltTune] Candidate failed after the {pose_name} movement phase.")
                except Exception as e:
                    print(f"⚠️ [HaltTune] Movement test failed during {pose_name} move: {e}")
                    stable = False
                if not stable:
                    break
        if not stable:
            with self.alarm_state_lock:
                alarm_msg = dict(self.last_alarm_msg) if isinstance(self.last_alarm_msg, dict) else None
            if alarm_msg:
                print(f"[HaltTune] Alarm during test: {alarm_msg}")
            self._clear_alarm_latch(f"after testing {int(threshold)}/{int(duration)}")
            try:
                self._prepare_relaxed_alarm_pid()
                self._clear_alarm_latch("before returning to Default after failed test")
                self.robot.set_motor(1)
                print("[HaltTune] Returning to Default after failed candidate test.")
                if self._queue_jmove_to_pose(default_pose, vel=HALT_TUNE_MOVE_VEL):
                    self._set_current_named("Default")
                    self._wait_for_joint_settle(
                        max_wait_s=8.0,
                        stable_for_s=0.35,
                        tol_deg=0.05,
                    )
            except Exception as e:
                print(f"⚠️ [HaltTune] Could not return to Default after failed candidate: {e}")
        return stable

    def _auto_tune_alarm_pid(self, threshold: float, duration: float, persist: bool = True):
        requested_threshold, requested_duration = _clamp_alarm_pid(threshold, duration)
        print(
            "[HaltTune] Auto-tuning at Default pose. "
            f"Requested starting point threshold={int(requested_threshold)}, duration={int(requested_duration)}."
        )
        print(
            "[HaltTune] Using sequential backoff tuning "
            f"(threshold {int(HALT_TUNE_THRESHOLD_MIN)}..{int(DEFAULT_PID_THRESHOLD_MAX)}, "
            f"duration {int(HALT_TUNE_DURATION_MIN)}..{int(HALT_TUNE_DURATION_MAX)})."
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

        tuned_pair = _run_alarm_tune_backoff_search(
            requested_threshold,
            requested_duration,
            lambda threshold_candidate, duration_candidate: self._test_alarm_pid_candidate(
                threshold_candidate,
                duration_candidate,
                hold_s=0.75,
            ),
            lambda msg: print(msg),
            int(DEFAULT_PID_THRESHOLD_MAX),
            int(HALT_TUNE_DURATION_MAX),
            baseline_max_duration=int(HALT_TUNE_BASELINE_DURATION_MAX),
        )

        if tuned_pair is None:
            print("⚠️ [HaltTune] No stable halt settings were found at Default pose within the allowed range.")
            return None
        tuned_threshold, tuned_duration = tuned_pair

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

    def _tcp_transform_from_joint_list(self, joint_list):
        try:
            T_flange = np.array(self.robot.kinematic.t_flange_r_world(joint=list(joint_list)[:6]), dtype=float)
            return T_flange @ np.array(self.robot.kinematic.T_tcp_r_flange, dtype=float)
        except Exception:
            return None

    def _tcp_jacobian_for_joint_list(self, joint_list):
        """
        Geometric Jacobian at the configured TCP, not at the flange.

        Dorna's built-in Jacobian is flange-centered. For right-stick pitch/yaw
        we need the TCP linear rows to include the tool length so the solved
        joint velocity explicitly commands zero tool-tip velocity.
        """
        try:
            kin = self.robot.kinematic
            n_dof = int(getattr(kin, "n_dof", 6))
            joint_vals = [float(v) for v in list(joint_list)[:max(6, n_dof)]]
            theta = [math.radians(v) for v in joint_vals]
            if len(theta) < n_dof:
                return None

            T = np.array(getattr(kin, "T_rail_r_world", np.eye(4)), dtype=float).copy()
            if n_dof == 5 and len(theta) > 5:
                rail = getattr(kin, "rail_vec_r_base", [0.0, 0.0, 0.0])
                rail_vec = np.array(
                    [[float(rail[0]) * theta[5]], [float(rail[1]) * theta[5]], [float(rail[2]) * theta[5]], [0.0]],
                    dtype=float,
                )
                rail_d = T @ rail_vec
                T[0, 3] += float(rail_d[0, 0])
                T[1, 3] += float(rail_d[1, 0])
                T[2, 3] += float(rail_d[2, 0])

            T = T @ np.array(kin.T(0, 0.0), dtype=float)
            z_axes = [np.array(T[:3, 2], dtype=float).reshape(3)]
            origins = [np.array(T[:3, 3], dtype=float).reshape(3)]

            for idx in range(1, n_dof + 1):
                T = T @ np.array(kin.T(idx, theta[idx - 1]), dtype=float)
                z_axes.append(np.array(T[:3, 2], dtype=float).reshape(3))
                origins.append(np.array(T[:3, 3], dtype=float).reshape(3))

            T_tcp = T @ np.array(kin.T_tcp_r_flange, dtype=float)
            tcp_origin = np.array(T_tcp[:3, 3], dtype=float).reshape(3)
            J = np.zeros((6, n_dof), dtype=float)
            for idx in range(n_dof):
                axis = z_axes[idx]
                origin = origins[idx]
                J[:3, idx] = np.cross(axis, tcp_origin - origin)
                J[3:, idx] = axis
            return J, T_tcp
        except Exception:
            return None

    def _solve_fixed_tcp_orientation_qdot(self, joint_list, angular_velocity_world_rad_s):
        jac = self._tcp_jacobian_for_joint_list(joint_list)
        if jac is None:
            return None
        J, _ = jac
        try:
            w = np.asarray(angular_velocity_world_rad_s, dtype=float).reshape(3)
        except Exception:
            return None
        if not np.all(np.isfinite(w)):
            return None

        length_mm = max(30.0, float(self.live_orientation_jacobian_length_mm))
        A = np.array(J, dtype=float)
        A[3:, :] *= length_mm
        b = np.zeros(6, dtype=float)
        b[3:] = w * length_mm

        damping = max(1e-6, float(self.live_orientation_jacobian_damping))
        lhs = A.T @ A + (damping * damping) * np.eye(A.shape[1], dtype=float)
        rhs = A.T @ b
        try:
            qdot = np.linalg.solve(lhs, rhs)
        except Exception:
            try:
                qdot = np.linalg.lstsq(A, b, rcond=1e-4)[0]
            except Exception:
                return None
        if not np.all(np.isfinite(qdot)):
            return None
        return qdot

    def _rotation_error_deg(self, target_R, actual_R):
        try:
            err_R = np.asarray(target_R, dtype=float) @ np.asarray(actual_R, dtype=float).T
            val = (float(np.trace(err_R)) - 1.0) / 2.0
            return float(math.degrees(math.acos(max(-1.0, min(1.0, val)))))
        except Exception:
            return float("inf")

    def _orientation_advanced_matrix(self, base_R, dt_s: float):
        dt_s = max(0.0, float(dt_s))
        pitch_rate = float(self.live_abs_abc_velocity[1])
        yaw_rate = float(self.live_abs_abc_velocity[2])
        R_cmd = np.array(base_R, dtype=float)
        if abs(yaw_rate) > 1e-9:
            R_cmd = axis_angle_to_R(0.0, 0.0, yaw_rate * dt_s) @ R_cmd
        if abs(pitch_rate) > 1e-9:
            R_cmd = R_cmd @ axis_angle_to_R(0.0, pitch_rate * dt_s, 0.0)
        return orthonormalize_R(R_cmd)

    def _play_live_orientation_joint_target(self, pose_xyz, segment_dt: float):
        pitch_rate = float(self.live_abs_abc_velocity[1])
        yaw_rate = float(self.live_abs_abc_velocity[2])
        if max(abs(pitch_rate), abs(yaw_rate)) <= 1e-9:
            return False

        q_state = self.live_orientation_joint_state
        R_state = self.live_orientation_R_state
        if q_state is None:
            q_state = self._joint_dict_to_list(self._try_get_current_joints())
        if q_state is None:
            return False

        q_state = np.asarray(q_state, dtype=float)
        if R_state is None:
            T_state = self._tcp_transform_from_joint_list(q_state.tolist())
            if T_state is not None:
                R_state = orthonormalize_R(np.asarray(T_state[:3, :3], dtype=float))
            else:
                R_state = np.array(self.R, dtype=float)
        else:
            R_state = np.array(R_state, dtype=float)

        angular_velocity = np.zeros(3, dtype=float)
        if abs(yaw_rate) > 1e-9:
            angular_velocity += np.array([0.0, 0.0, math.radians(yaw_rate)], dtype=float)
        if abs(pitch_rate) > 1e-9:
            angular_velocity += np.asarray(R_state[:, 1], dtype=float).reshape(3) * math.radians(pitch_rate)

        qdot_rad = self._solve_fixed_tcp_orientation_qdot(q_state.tolist(), angular_velocity)
        if qdot_rad is None:
            return False
        qdot_deg = np.degrees(np.asarray(qdot_rad, dtype=float))
        if qdot_deg.size <= 0 or not np.all(np.isfinite(qdot_deg)):
            return False

        n = min(qdot_deg.size, q_state.size)
        advance_s = min(max(float(segment_dt), 0.0), float(self.live_orientation_max_step_s))
        q_next = q_state.copy()
        q_next[:n] += qdot_deg[:n] * advance_s
        R_next = self._orientation_advanced_matrix(R_state, advance_s)

        max_joint_rate = float(np.max(np.abs(qdot_deg[:n]))) if n > 0 else 0.0
        if max_joint_rate <= 1e-9:
            return False
        max_orientation_rate = max(abs(pitch_rate), abs(yaw_rate))
        lead_s = float(self.live_orientation_lookahead_s)
        if max_orientation_rate > 1e-9:
            lead_s = min(lead_s, float(self.live_orientation_lookahead_max_deg) / max_orientation_rate)
        lead_s = max(0.0, lead_s)

        q_cmd = q_next.copy()
        q_cmd[:n] += qdot_deg[:n] * lead_s
        R_cmd = self._orientation_advanced_matrix(R_next, lead_s)

        T_cmd = self._tcp_transform_from_joint_list(q_cmd.tolist())
        if T_cmd is not None:
            target_pos = np.asarray(pose_xyz, dtype=float).reshape(3)
            cmd_pos = np.asarray(T_cmd[:3, 3], dtype=float).reshape(3)
            tip_err = float(np.linalg.norm(cmd_pos - target_pos))
            rot_err = self._rotation_error_deg(R_cmd, T_cmd[:3, :3])
            if tip_err > self.live_orientation_max_tip_error_mm or rot_err > self.live_orientation_max_rot_error_deg:
                now = time.time()
                if now - self.live_orientation_last_warn_t > 0.5:
                    print(
                        "[Jog] Fixed-TCP orientation fallback: "
                        f"tip_err={tip_err * 1000.0:.1f}um rot_err={rot_err:.3f}deg"
                    )
                    self.live_orientation_last_warn_t = now
                self.live_orientation_joint_state = None
                self.live_orientation_R_state = None
                return False

        joint_target = {f"j{i}": float(q_cmd[i]) for i in range(min(6, q_cmd.size))}
        if not self._guard_live_joint_target(joint_target, "live fixed-TCP orientation"):
            self.live_orientation_joint_state = None
            self.live_orientation_R_state = None
            return False

        joint_delta = q_cmd - q_state
        if not np.all(np.isfinite(joint_delta)):
            self.live_orientation_joint_state = None
            self.live_orientation_R_state = None
            return False
        joint_delta_norm = float(np.max(np.abs(joint_delta[:min(6, joint_delta.size)])))
        if joint_delta_norm <= 1e-6:
            return False

        cmd_vel = max(
            0.2,
            min(
                float(self.VR_POSE),
                max_joint_rate * float(self.live_orientation_velocity_gain) + 0.05,
            ),
        )
        # Keep this as relative joint deltas. On 2026-05-20 this fixed-TCP
        # right-stick path was validated on the robot as smooth and responsive;
        # absolute joint targets reintroduced stepwise motion during testing.
        cmd = {"cmd": "jmove", "rel": 1, "vel": cmd_vel, "cont": 1, "corner": 100}
        cmd.update({f"j{i}": float(joint_delta[i]) for i in range(min(6, joint_delta.size))})
        sent = bool(self._play_live(cmd))
        if sent:
            self.live_orientation_joint_state = q_next.tolist()
            self.live_orientation_R_state = R_next
        return sent

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

    def _find_live_tcp_collision_zone(self, tcp_pose):
        """Fast TCP-only zone check for high-rate joystick motion."""
        if tcp_pose is None:
            return None
        try:
            tcp_values = {axis: float(tcp_pose[i]) for i, axis in enumerate(COLLISION_TCP_AXES)}
        except Exception:
            return None

        with self.state.lock:
            zones = list(self.state.settings.get("collision_zones", []))
        for zone in zones:
            if not isinstance(zone, dict) or not zone.get("enabled", True):
                continue
            tcp_ranges = zone.get("tcp_ranges", {})
            if tcp_ranges and self._dict_in_ranges(tcp_values, tcp_ranges):
                return str(zone.get("name") or "unnamed")
        return None

    def _find_live_joint_collision_zone(self, joints: dict):
        """Fast joint-only zone check for high-rate joystick motion."""
        if not isinstance(joints, dict):
            return None
        with self.state.lock:
            zones = list(self.state.settings.get("collision_zones", []))
        for zone in zones:
            if not isinstance(zone, dict) or not zone.get("enabled", True):
                continue
            joint_ranges = zone.get("joint_ranges", {})
            if joint_ranges and self._dict_in_ranges(joints, joint_ranges):
                return str(zone.get("name") or "unnamed")
        return None

    def _find_collision_zone_along_joint_path(self, joint_target: dict):
        target = self._joint_dict_to_list(joint_target)
        if target is None:
            return None
        current = self._joint_dict_to_list(self._try_get_current_joints())
        if current is None:
            return self._find_collision_zone(joints=joint_target)

        samples = self._joint_path_samples(
            current,
            target,
            COLLISION_PATH_STEP_DEG,
        )
        for sample in samples:
            sample_joints = {f"j{i}": float(sample[i]) for i in range(6)}
            hit = self._find_collision_zone(joints=sample_joints)
            if hit:
                return hit
        return None

    def _find_collision_zone_along_tcp_path(self, tcp_pose):
        if tcp_pose is None:
            return None
        try:
            target = np.array([float(v) for v in list(tcp_pose)[:6]], dtype=float)
        except Exception:
            return None
        if target.shape[0] < 6:
            return None

        current_pose = None
        try:
            current_pose = np.array([float(v) for v in self._get_tcp_pose_from_robot()[:6]], dtype=float)
        except Exception:
            current_pose = None
        if current_pose is None or current_pose.shape[0] < 6:
            return self._find_collision_zone(tcp_pose=target.tolist())

        linear_delta = float(np.linalg.norm(target[:3] - current_pose[:3]))
        angular_delta = float(np.max(np.abs(target[3:] - current_pose[3:])))
        steps = max(1, int(np.ceil(max(linear_delta / COLLISION_PATH_STEP_DEG, angular_delta / 5.0))))
        steps = min(steps, 30)
        for idx in range(1, steps + 1):
            alpha = idx / steps
            sample = current_pose + (target - current_pose) * alpha
            hit = self._find_collision_zone(tcp_pose=sample.tolist())
            if hit:
                return hit
        return None

    def _guard_tcp_target(self, tcp_pose, context: str, sweep: bool = False):
        if tcp_pose is None:
            return True

        zone_hit = (
            self._find_collision_zone_along_tcp_path(tcp_pose)
            if sweep else self._find_collision_zone(tcp_pose=tcp_pose)
        )
        if zone_hit:
            self._collision_print(zone_hit, context)
            return False

        return True

    def _guard_joint_target(self, joint_target: dict, context: str, sweep: bool = False):
        if not isinstance(joint_target, dict):
            return True

        zone_hit = (
            self._find_collision_zone_along_joint_path(joint_target)
            if sweep else self._find_collision_zone(joints=joint_target)
        )
        if zone_hit:
            self._collision_print(zone_hit, context)
            return False

        return True

    def _guard_live_tcp_target(self, tcp_pose, context: str):
        zone_hit = self._find_live_tcp_collision_zone(tcp_pose)
        if zone_hit:
            self._collision_print(zone_hit, context)
            return False
        return True

    def _guard_live_joint_target(self, joint_target: dict, context: str):
        zone_hit = self._find_live_joint_collision_zone(joint_target)
        if zone_hit:
            self._collision_print(zone_hit, context)
            return False
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

    def _stick_sign(self, value: float):
        if value > self.left_stick_reversal_threshold:
            return 1
        if value < -self.left_stick_reversal_threshold:
            return -1
        return 0

    def _handle_live_direction_sign(self, axis: str, sign: int):
        if axis == "y":
            prev = self.left_stick_y_sign
            if sign:
                if prev and sign != prev and self.live_motion_active:
                    self._reset_live_motion_pending()
                    self._halt_live_motion()
                    self.live_motion_active = False
                    self.last_pose_refresh = 0.0
                self.left_stick_y_sign = sign
            return

        if axis == "x":
            prev = self.left_stick_x_sign
            if sign:
                if prev and sign != prev and self.live_motion_active:
                    self._reset_live_motion_pending()
                    self._halt_live_motion()
                    self.live_motion_active = False
                    self.last_pose_refresh = 0.0
                self.left_stick_x_sign = sign

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
        self.live_orientation_joint_state = None
        self.live_orientation_R_state = None

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
        self.live_abs_xyz_velocity.fill(0.0)
        self.live_abs_abc_velocity.fill(0.0)
        self.live_j5_pending = 0.0
        self.live_next_send_t = 0.0
        self.live_last_send_t = 0.0
        self.live_last_abs_pose = None
        self.live_smoothed_vel = None
        self.live_orientation_joint_state = None
        self.live_orientation_R_state = None

    def _clear_left_stick_jog_state(self):
        self.left_stick_lx_cmd = 0.0
        self.left_stick_ly_cmd = 0.0
        self.left_stick_mode = None
        self.left_stick_y_sign = 0
        self.left_stick_x_sign = 0
        self.live_abs_xyz_velocity.fill(0.0)
        self.live_abs_abc_velocity.fill(0.0)

    def _shape_right_stick(self, rx: float, ry: float):
        if rx == 0.0:
            self.right_stick_rx_cmd = 0.0
        else:
            self.right_stick_rx_cmd += (
                rx - self.right_stick_rx_cmd
            ) * self.right_stick_filter_alpha

        if ry == 0.0:
            self.right_stick_ry_cmd = 0.0
        else:
            self.right_stick_ry_cmd += (
                ry - self.right_stick_ry_cmd
            ) * self.right_stick_filter_alpha

        return self.right_stick_rx_cmd, self.right_stick_ry_cmd

    def _flush_live_motion(self, now_t: float):
        if self.live_last_motion_mode == "tool_axis_translation":
            send_interval = self.live_send_interval_tool_axis
        elif self.live_last_motion_mode == "orientation":
            send_interval = self.live_send_interval_orientation
        else:
            send_interval = self.live_send_interval
        if self.live_last_motion_mode != "orientation":
            self.live_orientation_joint_state = None
            self.live_orientation_R_state = None
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
            if j_target is not None and not self._guard_live_joint_target(j_target, "live roll motion"):
                self.live_j5_pending = 0.0
                self._reset_live_motion_pending()
                self._soft_stop_live_motion()
                return True
            self.live_j5_pending = 0.0
            sent = bool(self._play_live({"cmd":"jmove","rel":1,"j5":delta,"vel":self.VR,"cont":1}))

        if self.live_abs_pose_dirty:
            a1, b1, c1 = R_to_axis_angle(self.R)
            pose_xyz = np.array([self.x0, self.y0, self.z0], dtype=float)
            if self.live_last_motion_mode == "tool_axis_translation":
                speed = float(np.linalg.norm(self.live_abs_xyz_velocity))
                if speed > 1e-9:
                    lead_s = min(
                        self.live_tool_axis_lookahead_s,
                        self.live_tool_axis_lookahead_max_mm / speed,
                    )
                    pose_xyz = pose_xyz + self.live_abs_xyz_velocity * max(0.0, lead_s)
            pose_now = (
                float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2]),
                float(a1), float(b1), float(c1),
            )
            if not self._guard_live_tcp_target(pose_now, "live TCP motion"):
                self._reset_live_motion_pending()
                self._soft_stop_live_motion()
                return True
            if self.live_last_abs_pose is None:
                linear_delta = float(np.linalg.norm(self.live_rel_xyz_pending))
                angular_delta = float(np.linalg.norm(self.live_rel_abc_pending))
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
                if self.live_last_abs_pose is None:
                    if self.live_last_motion_mode == "tool_axis_translation":
                        raw_vel = max(0.2, float(np.linalg.norm(self.live_abs_xyz_velocity)))
                    elif self.live_last_motion_mode == "orientation":
                        raw_vel = max(0.2, float(np.linalg.norm(self.live_abs_abc_velocity)))
                    else:
                        raw_vel = self.VR
                else:
                    raw_vel = max(0.2, linear_delta / segment_dt, angular_delta / segment_dt)
                cmd_vel = self._smooth_live_velocity(raw_vel)
                self.live_rel_xyz_pending.fill(0.0)
                self.live_rel_abc_pending.fill(0.0)
                cmd_sent = False
                if self.live_last_motion_mode == "orientation":
                    cmd_sent = self._play_live_orientation_joint_target(pose_xyz, segment_dt)
                if not cmd_sent:
                    cmd = {
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
                    }
                    cmd_sent = bool(self._play_live(cmd))
                if cmd_sent:
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
            if not self._guard_live_tcp_target(pose_now, "live TCP motion"):
                self._reset_live_motion_pending()
                self._soft_stop_live_motion()
                return True
            dx, dy, dz = [float(v) for v in self.live_rel_xyz_pending]
            da, db, dc = [float(v) for v in self.live_rel_abc_pending]
            self.live_rel_xyz_pending.fill(0.0)
            self.live_rel_abc_pending.fill(0.0)
            linear_speed = rel_xyz_norm / segment_dt if rel_xyz_norm > 1e-9 else 0.0
            angular_speed = rel_abc_norm / segment_dt if rel_abc_norm > 1e-9 else 0.0
            raw_vel = max(0.2, linear_speed, angular_speed)
            cmd_vel = self._smooth_live_velocity(raw_vel)
            sent = bool(self._play_live({
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
            }))
            if sent:
                tz = self.R[:,2]
                pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
                yaw_deg = np.degrees(np.arctan2(tz[1], tz[0]))
                with self.state.lock:
                    self.state.pitch = pitch
                    self.state.yaw = yaw_deg

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

    def _queue_jmove_to_pose(self, pose: dict, vel: float = None):
        target_joints = {axis: pose[axis] for axis in COLLISION_JOINT_AXES if axis in pose}
        merged_target = self._merge_joint_target(target_joints)
        if merged_target is not None and not self._guard_joint_target(merged_target, "joint move", sweep=True):
            return False
        if vel is None:
            vel = self.VR_POSE
        go = {"cmd":"jmove","rel":0,"vel":float(vel)}
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

    def _pick_fixed_tcp_ik_solution(self, target_tcp_mat, seed_joint_list, max_pos_err_mm=0.05, max_rot_err_deg=0.5):
        try:
            target_xyzabc = self.robot.kinematic.mat_to_xyzabc(target_tcp_mat)
            sols = self.robot.kinematic.inv(target_xyzabc, joint_current=seed_joint_list, all_sol=True)
        except Exception:
            return None, (float("inf"), float("inf"), float("inf"))
        if sols is None:
            return None, (float("inf"), float("inf"), float("inf"))
        try:
            if isinstance(sols, np.ndarray) and sols.ndim == 1:
                sols_iter = [sols.tolist()]
            else:
                sols_iter = list(sols)
        except Exception:
            sols_iter = [sols]

        seed = np.asarray(seed_joint_list[:6], dtype=float)
        target_pos = np.asarray(target_tcp_mat[:3, 3], dtype=float).reshape(3)
        target_R = np.asarray(target_tcp_mat[:3, :3], dtype=float)
        best = None
        best_cost = float("inf")
        best_metrics = (float("inf"), float("inf"), float("inf"))
        for sol in sols_iter:
            try:
                cand = np.asarray(list(sol)[:6], dtype=float)
            except Exception:
                continue
            if cand.size < 6 or not np.all(np.isfinite(cand[:6])):
                continue
            T_cand = self._tcp_transform_from_joint_list(cand[:6].tolist())
            if T_cand is None:
                continue
            pos_err = float(np.linalg.norm(np.asarray(T_cand[:3, 3], dtype=float).reshape(3) - target_pos))
            rot_err = self._rotation_error_deg(target_R, T_cand[:3, :3])
            joint_dist = float(np.linalg.norm(cand[:6] - seed))
            cost = pos_err * 100.0 + rot_err * 2.0 + joint_dist * 0.1
            if cost < best_cost:
                best = cand[:6].copy()
                best_cost = cost
                best_metrics = (pos_err, rot_err, joint_dist)

        if best is None:
            return None, best_metrics
        if best_metrics[0] > float(max_pos_err_mm) or best_metrics[1] > float(max_rot_err_deg):
            return None, best_metrics
        return best.tolist(), best_metrics

    def _fixed_tip_cone_demo_loop(self, diameter_mm=50.0, rotations=5):
        """
        One-shot robotics demo: keep the physical tool tip fixed while the
        flange/head orbits around it. This is the usual TCP stability cone
        demo; the 50 mm diameter is measured at the tool head/shaft, not as
        alternating independent pitch/yaw strokes.
        """
        self.tool_center_demo_resume.set()
        with self.state.lock:
            self.state.tool_center_demo = True
            self.state.tool_center_demo_mode = "cone"
            self.state.tool_center_demo_waiting = False

        stopped = False
        try:
            joints = self._try_get_current_joints()
            if not joints:
                print("⚠️ Fixed-tip cone demo: no joint feedback available.")
                return
            base_joint = [float(joints.get(f"j{i}", 0.0) or 0.0) for i in range(6)]
            try:
                base_flange_mat = np.array(self.robot.kinematic.t_flange_r_world(joint=base_joint), dtype=float)
                active_tcp_r_flange = np.array(self.robot.kinematic.T_tcp_r_flange, dtype=float)
            except Exception:
                print("⚠️ Fixed-tip cone demo: cannot compute current flange/TCP transform.")
                return

            with self.state.lock:
                tool_lz = float(self.state.settings.get("tool_lz", DEFAULT_TOOL_LZ))
                tool_cx = float(self.state.settings.get("tool_cx", DEFAULT_TOOL_CX))
                tool_cy = float(self.state.settings.get("tool_cy", DEFAULT_TOOL_CY))
                tool_presets = list(self.state.settings.get("tool_presets", []) or [])
            demo_lz = float(tool_lz)
            demo_source = "active"
            if abs(demo_lz) < 80.0:
                preset_lz_values = []
                for preset in tool_presets:
                    try:
                        preset_lz = float(preset.get("lz", 0.0) or 0.0)
                    except Exception:
                        continue
                    if abs(preset_lz) >= 80.0:
                        preset_lz_values.append(preset_lz)
                if preset_lz_values:
                    demo_lz = max(preset_lz_values, key=lambda value: abs(value))
                    demo_source = "saved preset"
                else:
                    demo_lz = DEFAULT_TOOL_LZ if demo_lz >= 0.0 else -DEFAULT_TOOL_LZ
                    demo_source = "default"

            demo_tip_r_flange = np.eye(4, dtype=float)
            demo_tip_r_flange[0, 3] = tool_cx
            demo_tip_r_flange[1, 3] = tool_cy
            demo_tip_r_flange[2, 3] = demo_lz
            inv_demo_tip_r_flange = np.linalg.inv(demo_tip_r_flange)
            base_tip_mat = base_flange_mat @ demo_tip_r_flange
            base_tip_pos = np.asarray(base_tip_mat[:3, 3], dtype=float).reshape(3)
            base_tip_R = orthonormalize_R(np.asarray(base_tip_mat[:3, :3], dtype=float))

            lever_mm = max(1.0, float(np.linalg.norm([tool_cx, tool_cy, demo_lz])))
            radius_mm = max(1.0, float(diameter_mm) * 0.5)
            raw_tilt_deg = math.degrees(math.asin(max(-0.999, min(0.999, radius_mm / lever_mm))))
            tilt_deg = max(0.5, min(20.0, raw_tilt_deg))
            actual_diameter_mm = 2.0 * lever_mm * math.sin(math.radians(tilt_deg))
            if demo_source != "active":
                print(
                    "[Tool] Fixed-tip cone demo using "
                    f"{demo_source} tool length {demo_lz:.1f}mm "
                    f"(active tool_lz={tool_lz:.1f}mm) for visible physical-tip precession."
                )
            if abs(tilt_deg - raw_tilt_deg) > 1e-6:
                print(
                    "[Tool] Fixed-tip cone demo tilt clamped: "
                    f"lever={lever_mm:.1f}mm requested_diameter={diameter_mm:.1f}mm "
                    f"actual_diameter~{actual_diameter_mm:.1f}mm"
                )

            def rot_z_rad(rad):
                c = math.cos(rad)
                s = math.sin(rad)
                return np.array(
                    [
                        [c, -s, 0.0],
                        [s,  c, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                )

            def rot_y_deg(deg):
                rad = math.radians(deg)
                c = math.cos(rad)
                s = math.sin(rad)
                return np.array(
                    [
                        [ c, 0.0, s],
                        [0.0, 1.0, 0.0],
                        [-s, 0.0, c],
                    ],
                    dtype=float,
                )

            def target_R_for(theta_rad, scale=1.0):
                # Classic TCP stability demo: keep the tip fixed while the tool
                # axis precesses around its starting direction. A point one
                # tool-length back from the tip traces the requested circle.
                tilt_R = (
                    rot_z_rad(theta_rad)
                    @ rot_y_deg(tilt_deg * float(scale))
                    @ rot_z_rad(-theta_rad)
                )
                return orthonormalize_R(base_tip_R @ tilt_R)

            ramp_steps = 12
            circle_steps = 72
            cone_rotations = max(1, int(rotations or 1))
            target_Rs = []
            for idx in range(1, ramp_steps + 1):
                target_Rs.append(target_R_for(0.0, idx / ramp_steps))
            for cycle in range(cone_rotations):
                for idx in range(1, circle_steps + 1):
                    target_Rs.append(target_R_for(2.0 * math.pi * (cycle + idx / circle_steps), 1.0))
            for idx in range(ramp_steps - 1, -1, -1):
                target_Rs.append(target_R_for(0.0, idx / ramp_steps))

            seed = list(base_joint)
            cmd_list = []
            path_deg = 0.0
            prev_R = base_tip_R
            for R_tgt in target_Rs:
                if self.tool_center_demo_stop.is_set() or self.stop_event.is_set():
                    stopped = True
                    break
                target_tip_mat = np.eye(4, dtype=float)
                target_tip_mat[:3, :3] = R_tgt
                target_tip_mat[0, 3] = base_tip_pos[0]
                target_tip_mat[1, 3] = base_tip_pos[1]
                target_tip_mat[2, 3] = base_tip_pos[2]
                target_flange_mat = target_tip_mat @ inv_demo_tip_r_flange
                target_tcp_mat = target_flange_mat @ active_tcp_r_flange
                sol, metrics = self._pick_fixed_tcp_ik_solution(target_tcp_mat, seed)
                if sol is None:
                    pos_err, rot_err, _ = metrics
                    print(
                        "[Tool] Fixed-tip cone demo aborted: IK could not hold TCP "
                        f"(pos_err={pos_err * 1000.0:.1f}um rot_err={rot_err:.3f}deg)."
                    )
                    return
                joint_target = {f"j{i}": float(sol[i]) for i in range(6)}
                if not self._guard_live_joint_target(joint_target, "fixed-tip cone demo"):
                    print("[Tool] Fixed-tip cone demo aborted by joint guard.")
                    return
                try:
                    cand_flange = np.array(self.robot.kinematic.t_flange_r_world(joint=sol), dtype=float)
                    cand_tip = cand_flange @ demo_tip_r_flange
                    demo_tip_err = float(np.linalg.norm(np.asarray(cand_tip[:3, 3], dtype=float).reshape(3) - base_tip_pos))
                    if demo_tip_err > 0.10:
                        print(
                            "[Tool] Fixed-tip cone demo aborted: physical-tip pivot error "
                            f"{demo_tip_err * 1000.0:.1f}um."
                        )
                        return
                except Exception:
                    pass
                cmd = {
                    "cmd": "jmove",
                    "rel": 0,
                    "vel": min(max(3.0, float(self.VR_POSE) * 0.8), 8.0),
                    "acc": 80.0,
                    "jerk": 350.0,
                    "cont": 1,
                    "corner": 100,
                }
                cmd.update(joint_target)
                cmd_list.append(cmd)
                path_deg += self._rotation_error_deg(R_tgt, prev_R)
                prev_R = R_tgt
                seed = list(sol)

            if stopped or not cmd_list:
                return
            cmd_list[-1]["cont"] = 0
            cmd_list[-1]["corner"] = 0

            demo_vel = float(cmd_list[0]["vel"])
            estimated_s = max(2.0, path_deg / max(1e-3, demo_vel) + 1.0)
            print(
                "[Tool] Fixed-tip cone demo started: "
                f"diameter~{actual_diameter_mm:.1f}mm tilt={tilt_deg:.2f}deg "
                f"rotations={cone_rotations} points={len(cmd_list)} physical-tip pivot."
            )
            self._mark_motion_for(estimated_s)
            try:
                self.robot.play_list(cmd_list, timeout=0)
            except Exception:
                for cmd in cmd_list:
                    if self.tool_center_demo_stop.is_set() or self.stop_event.is_set():
                        stopped = True
                        break
                    self.robot.play_dict(cmd, timeout=0)

            end_t = time.time() + estimated_s
            while time.time() < end_t:
                if self.tool_center_demo_stop.is_set() or self.stop_event.is_set():
                    stopped = True
                    try:
                        self.robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
                    except Exception:
                        pass
                    break
                time.sleep(0.05)

            if not stopped:
                try:
                    end_joints = self._try_get_current_joints()
                    if end_joints:
                        end_list = [float(end_joints.get(f"j{i}", 0.0) or 0.0) for i in range(6)]
                        end_flange = np.array(self.robot.kinematic.t_flange_r_world(joint=end_list), dtype=float)
                        end_tip = end_flange @ demo_tip_r_flange
                        drift = float(np.linalg.norm(np.asarray(end_tip[:3, 3], dtype=float).reshape(3) - base_tip_pos))
                        print(f"[Tool] Fixed-tip cone demo complete. physical-tip drift={drift * 1000.0:.1f}um.")
                except Exception:
                    print("[Tool] Fixed-tip cone demo complete.")
            else:
                print("[Tool] Fixed-tip cone demo stopped.")
        except Exception as e:
            print(f"⚠️ Fixed-tip cone demo failed: {e}")
        finally:
            self.tool_center_demo_stop.clear()
            with self.state.lock:
                self.state.tool_center_demo = False
                self.state.tool_center_demo_waiting = False
                self.state.tool_center_demo_mode = "cone"

    def _stop_tool_center_demo(self):
        self.tool_center_demo_stop.set()
        self.tool_center_demo_resume.set()
        try:
            if self.robot:
                self.robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
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
                    self.robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
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

    def _apply_tool_geometry_no_recenter(self, cx: float, cy: float, lz: float, persist: bool = True):
        """Apply controller TCP geometry without preserving the old TCP pose."""
        try:
            cx = float(cx)
            cy = float(cy)
            lz = float(lz)
            try:
                if hasattr(self.robot, "set_tool"):
                    self.robot.set_tool(lx=cx, ly=cy, lz=lz)
                else:
                    self.robot.play_dict(cmd={"cmd": "tool", "id": 1, "lx": cx, "ly": cy, "lz": lz})
            except Exception:
                self.robot.play_dict(cmd={"cmd": "tool", "id": 1, "lx": cx, "ly": cy, "lz": lz})
            with self.state.lock:
                self.state.tool_lz = lz
                self.state.tool_cx = cx
                self.state.tool_cy = cy
                self.state.settings["tool_lz"] = lz
                self.state.settings["tool_cx"] = cx
                self.state.settings["tool_cy"] = cy
            self._update_tcp_from_settings()
            if persist:
                save_settings(self.state.settings)
            return True
        except Exception as e:
            print(f"⚠️ Could not apply tool geometry: {e}")
            return False

    def _set_tunnel_status(self, text: str):
        with self.state.lock:
            self.state.routine_status_text = text

    def _safe_tunnel_tcp_shift(self, target_mode: str):
        """
        Switch between injector and side-mounted tunnel-needle TCPs.

        This deliberately performs the potentially lateral TCP compensation only
        after retracting from tissue: retract 10 mm by default, change TCP,
        recenter the new TCP at the retracted point, then return 8 mm so the
        final 2 mm remains under direct manual guidance.
        """
        target_mode = str(target_mode or "").strip().lower()
        if target_mode not in ("injector", "tunnel"):
            print(f"⚠️ Unknown tunnel TCP target: {target_mode!r}")
            return False

        with self.state.lock:
            settings = self.state.settings
            current_mode = str(settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)).strip().lower()
            if current_mode not in ("injector", "tunnel"):
                current_mode = DEFAULT_TUNNEL_TCP_MODE
            current_cx = float(getattr(self.state, "tool_cx", settings.get("tool_cx", DEFAULT_TOOL_CX)))
            current_cy = float(getattr(self.state, "tool_cy", settings.get("tool_cy", DEFAULT_TOOL_CY)))
            current_lz = float(getattr(self.state, "tool_lz", settings.get("tool_lz", DEFAULT_TOOL_LZ)))
            offset_y = float(settings.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM))
            retract_mm = max(0.0, float(settings.get("tunnel_retract_mm", DEFAULT_TUNNEL_RETRACT_MM)))
            return_mm = max(0.0, float(settings.get("tunnel_return_mm", DEFAULT_TUNNEL_RETURN_MM)))
            stored_injector_cy = float(settings.get("tunnel_injector_cy_mm", current_cy - offset_y))

        if abs(offset_y) < 0.001:
            msg = "Tunnel TCP shift refused: set the side needle Y offset first."
            print(f"⚠️ {msg}")
            self._set_tunnel_status(msg)
            return False
        if retract_mm < 1.0:
            msg = "Tunnel TCP shift refused: safety retract must be at least 1 mm."
            print(f"⚠️ {msg}")
            self._set_tunnel_status(msg)
            return False
        if return_mm > retract_mm:
            return_mm = retract_mm

        if current_mode == target_mode:
            msg = f"Tunnel TCP already in {target_mode} mode."
            print(f"[Tunnel] {msg}")
            self._set_tunnel_status(msg)
            return True

        if current_mode == "injector":
            injector_cy = current_cy
        else:
            injector_cy = stored_injector_cy

        if target_mode == "tunnel":
            target_cy = injector_cy + offset_y
            label = "tunnel needle"
        else:
            target_cy = injector_cy
            label = "injector"

        tool_cy_delta = target_cy - current_cy
        clearance_mm = max(0.0, retract_mm - return_mm)
        start_msg = (
            f"Switching TCP to {label}: retract {retract_mm:.1f} mm, "
            f"change tool-center Y by {tool_cy_delta:+.3f} mm, return {return_mm:.1f} mm "
            f"({clearance_mm:.1f} mm manual clearance)."
        )
        print(f"[Tunnel] {start_msg}")
        self._set_tunnel_status(start_msg)

        with self.state.lock:
            self.state.await_confirm = False
            self.state.await_target = None
            self.state.confirm_msg = ""

        if not self._tool_move_along_tz(-retract_mm, cont=0):
            msg = "Tunnel TCP shift stopped: safety retract failed."
            print(f"⚠️ {msg}")
            self._set_tunnel_status(msg)
            return False
        time.sleep(0.05)

        retracted_pose = self._capture_tcp_pose()
        if not self._apply_tool_geometry_no_recenter(current_cx, target_cy, current_lz, persist=False):
            msg = "Tunnel TCP shift stopped: controller TCP update failed."
            print(f"⚠️ {msg}")
            self._set_tunnel_status(msg)
            return False

        with self.state.lock:
            self.state.settings["tunnel_tcp_mode"] = target_mode
            self.state.settings["tunnel_injector_cy_mm"] = injector_cy
            self.state.settings["tunnel_tcp_y_mm"] = offset_y
            self.state.settings["tunnel_retract_mm"] = retract_mm
            self.state.settings["tunnel_return_mm"] = return_mm
            self.state.tunnel_tcp_mode = target_mode
            self.state.tunnel_tcp_y_mm = offset_y
            self.state.tunnel_retract_mm = retract_mm
            self.state.tunnel_return_mm = return_mm
        save_settings(self.state.settings)

        if not self._move_tcp_to_pose(retracted_pose):
            msg = "Tunnel TCP shift stopped: re-centering the new TCP failed."
            print(f"⚠️ {msg}")
            self._set_tunnel_status(msg)
            return False
        time.sleep(0.05)

        if return_mm > 0.0 and not self._tool_move_along_tz(+return_mm, cont=0):
            msg = "Tunnel TCP shift stopped: guided return failed."
            print(f"⚠️ {msg}")
            self._set_tunnel_status(msg)
            return False

        self._set_current_named(None)
        done_msg = (
            f"TCP switched to {label}; stopped {clearance_mm:.1f} mm short for manual guidance."
        )
        print(f"[Tunnel] {done_msg}")
        self._set_tunnel_status(done_msg)
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
        _apply_linux_thread_priority("Robot control", nice=-10, rt_priority=20)
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

        startup_cal = self.startup_calibration_test
        if startup_cal == "tool_center_circle":
            self.cmd_q.put({"type": "tool_center_demo_toggle", "mode": "circle"})
            print("[Startup] Queued tool center circle demo from launcher selection.")
        elif startup_cal == "tool_center_vertical":
            self.cmd_q.put({"type": "tool_center_demo_toggle", "mode": "vertical"})
            print("[Startup] Queued tool center vertical oscillation demo from launcher selection.")
        elif startup_cal == "tool_center_horizontal":
            self.cmd_q.put({"type": "tool_center_demo_toggle", "mode": "horizontal"})
            print("[Startup] Queued tool center horizontal oscillation demo from launcher selection.")
        elif startup_cal == "j5_rotation":
            self.cmd_q.put({"type": "j5_test_toggle", "start": True})
            print("[Startup] Queued tool center rotation test from launcher selection.")

        control_hz = 240.0
        next_t = time.time()
        joint_poll_time = next_t
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
                            self.robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
                            print("[Robot] Motion halted.")
                        except Exception as e:
                            print(f"⚠️ halt failed: {e}")

                    elif typ == "kill_motors":
                        try:
                            self.robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
                            print("[Robot] Kill-motors request ignored; motion halted only.")
                        except Exception as e:
                            print(f"⚠️ halt failed: {e}")

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

                    elif typ == "safe_tunnel_tcp_shift":
                        self._safe_tunnel_tcp_shift(cmd.get("target_mode", "tunnel"))
                        continue

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
                            # Pitch is derived from the tool-Z vector. Positive
                            # local-Y rotation raises the displayed pitch, which
                            # matches the right-stick pitch jog convention.
                            self.R = self.R @ axis_angle_to_R(0, delta, 0)
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
                            if merged_target is not None and not self._guard_joint_target(merged_target, "absolute roll move", sweep=True):
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

                    elif typ == "fixed_tip_cone_demo":
                        if self.robot is None:
                            print("⚠️ Fixed-tip cone demo: robot not connected.")
                            continue
                        if self.j5_test_thread and self.j5_test_thread.is_alive():
                            print("[Tool] Stop the tool rotation test before starting the cone demo.")
                            continue
                        if self.tool_center_demo_thread and self.tool_center_demo_thread.is_alive():
                            self._stop_tool_center_demo()
                        self.tool_center_demo_stop.clear()
                        self.tool_center_demo_resume.set()
                        self.tool_center_demo_mode = "cone"
                        with self.state.lock:
                            self.state.tool_center_demo = True
                            self.state.tool_center_demo_mode = "cone"
                            self.state.tool_center_demo_waiting = False
                        diameter = float(cmd.get("diameter_mm", 50.0) or 50.0)
                        rotations = int(cmd.get("rotations", 5) or 5)
                        t = threading.Thread(
                            target=self._fixed_tip_cone_demo_loop,
                            kwargs={"diameter_mm": diameter, "rotations": rotations},
                            daemon=True,
                        )
                        self.tool_center_demo_thread = t
                        t.start()
                        continue

                    elif typ == "j5_test_toggle":
                        start_req = bool(cmd.get("start", False))
                        print(f"[Tool] j5_test_toggle start={start_req}")
                        if start_req:
                            if self.robot is None:
                                print("⚠️ Roll test: robot not connected.")
                                with self.state.lock:
                                    self.state.j5_test_active = False
                                continue
                            if self.j5_test_thread and self.j5_test_thread.is_alive():
                                print("[Tool] Roll test already running.")
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
                                    self.robot.play_dict({"cmd": "halt", "queue": 0}, timeout=0)
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

                    elif typ == "set_live_linear_epsilon":
                        value = _clamp_live_linear_epsilon_mm(
                            cmd.get("value", DEFAULT_LIVE_LINEAR_EPSILON_MM)
                        )
                        stable_value = _stable_live_linear_epsilon_mm(value)
                        self.live_linear_epsilon = stable_value
                        with self.state.lock:
                            self.state.settings["live_linear_epsilon_mm"] = value
                            _apply_manual_speed_levels_for_epsilon_locked(self.state, value)
                            self.state.calibration_status = (
                                f"Fine speed reference set to {_live_linear_epsilon_label(value)}; "
                                f"live command floor held at {STABLE_LIVE_LINEAR_EPSILON_MM * 1000.0:.0f} um."
                            )
                        save_settings(self.state.settings)
                        print(
                            "[Jog] Fine speed reference set to "
                            f"{value:.3f} mm ({value * 1000.0:.0f} um); "
                            "stable live command epsilon remains "
                            f"{stable_value:.3f} mm ({stable_value * 1000.0:.0f} um)."
                        )
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
                            if bool(cmd.get("star", True)):
                                with self.state.lock:
                                    starred = list(self.state.settings.get("starred_poses", []))
                                    if name not in starred:
                                        starred.append(name)
                                    self.state.settings["starred_poses"] = starred
                                save_settings(self.state.settings)

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
                            starred = list(self.state.settings.get("starred_poses", []))
                            if old in starred:
                                starred = [new if x == old else x for x in starred]
                                self.state.settings["starred_poses"] = starred
                                save_settings(self.state.settings)
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
                            if removed_any:
                                starred = list(self.state.settings.get("starred_poses", []))
                                if name in starred:
                                    starred = [x for x in starred if x != name]
                                    self.state.settings["starred_poses"] = starred
                                    save_settings(self.state.settings)
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

                    elif typ == "injection_reload_capture_retracted":
                        pose = self._get_tcp_pose_from_robot()
                        with self.state.lock:
                            self.state.injection_reload_retracted_pose = pose
                        print("[Robot] Captured current pose as injection reload retracted pose.")

                    elif typ == "injection_reload_retract_capture":
                        try:
                            dist = float(cmd.get("dist", RELOAD_SAFE_RETRACT_MM) or RELOAD_SAFE_RETRACT_MM)
                        except Exception:
                            dist = RELOAD_SAFE_RETRACT_MM
                        if dist <= 0:
                            print("⚠️ injection_reload_retract_capture: non-positive distance; ignoring.")
                            continue
                        try:
                            if self._tool_move_along_tz(-dist, cont=0):
                                pose = self._capture_tcp_pose()
                                with self.state.lock:
                                    self.state.injection_reload_retracted_pose = pose
                                print(f"[Robot] Reload safety retract {dist:.1f} mm and captured retracted pose.")
                        except Exception as e:
                            print(f"⚠️ reload safety retract failed: {e}")

                    elif typ == "injection_reload_return_retracted":
                        with self.state.lock:
                            pose = self.state.injection_reload_retracted_pose
                        if not pose:
                            print("⚠️ Reload return skipped: no retracted pose captured.")
                            continue
                        try:
                            if self._move_tcp_to_pose(pose):
                                print("[Robot] Returned to captured reload retracted pose.")
                        except Exception as e:
                            print(f"⚠️ Reload return to retracted pose failed: {e}")

            except queue.Empty:
                pass

            with self.state.lock:
                lx, ly = self.state.lx, self.state.ly
                rx, ry = self.state.rx, self.state.ry
                hx, hy = self.state.hx, self.state.hy
                sens   = self.state.levels[self.state.idx]
                waiting = self.state.await_confirm
                alarm_active = bool(self.state.alarm_latched)
                ui_nav_active = bool(getattr(self.state, "ui_nav_active", False))

            lx_raw, ly_raw = self._preprocess_left_stick(lx, ly)
            lx, ly = self._shape_left_stick(lx_raw, ly_raw)
            rx = _apply_deadzone(rx, self.orient_deadzone)
            ry = _apply_deadzone(ry, self.orient_deadzone)
            sx, sj5, sb, sc, sh = (
                self.BSX * sens,
                self.BSJ5 * sens,
                self.BSB * sens,
                self.BSC * sens,
                self.BSH * sens,
            )
            manual_enabled = (
                time.time() >= self.skip_manual_until
                and (not waiting)
                and (not alarm_active)
                and (not ui_nav_active)
            )
            if manual_enabled:
                left_stick_mode = self._resolve_left_stick_mode(lx_raw, ly_raw)
                rx, ry = self._shape_right_stick(rx, ry)
            else:
                self._clear_left_stick_jog_state()
                self.right_stick_rx_cmd = 0.0
                self.right_stick_ry_cmd = 0.0
                left_stick_mode = None
            live_motion_cmd_sent = False
            live_motion_requested = False

            if manual_enabled and left_stick_mode == "y":
                self.left_stick_x_sign = 0
                self._handle_live_direction_sign("y", self._stick_sign(ly))
                tool_speed = -ly * sx
                d = tool_speed * loop_dt
                tz = self.R[:,2]
                self.live_abs_xyz_velocity = np.array(
                    [tz[0] * tool_speed, tz[1] * tool_speed, tz[2] * tool_speed],
                    dtype=float,
                )
                self.live_abs_abc_velocity.fill(0.0)
                dx = tz[0] * d
                dy = tz[1] * d
                dz = tz[2] * d
                if abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dz) > 1e-9:
                    self.x0 += dx
                    self.y0 += dy
                    self.z0 += dz
                    # Tool-axis jog uses latest absolute TCP target rather than
                    # accumulating small relative packets behind the stick.
                    self.live_abs_pose_dirty = True
                    self.live_last_motion_mode = "tool_axis_translation"
                    live_motion_requested = True

            if manual_enabled and left_stick_mode == "x":
                self.left_stick_y_sign = 0
                self._handle_live_direction_sign("x", self._stick_sign(lx))
                self.live_abs_xyz_velocity.fill(0.0)
                self.live_abs_abc_velocity.fill(0.0)
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
                yaw_rate = 0.0
                pitch_rate = 0.0
                pitch_active = abs(ry_eff) > self.DZ
                if abs(rx_eff) > self.DZ:
                    yaw_rate = -rx_eff * sc
                    self.R = axis_angle_to_R(0, 0, yaw_rate * loop_dt) @ self.R
                    self.R = orthonormalize_R(self.R)
                    moved = True
                if pitch_active:
                    pitch_rate = ry_eff * sb
                    pitch_delta = pitch_rate * loop_dt
                    self.R = self.R @ axis_angle_to_R(0, pitch_delta, 0)
                    self.R = orthonormalize_R(self.R)
                    moved = True
                self.live_abs_abc_velocity[:] = (0.0, pitch_rate, yaw_rate)

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

            # Robot feedback reads share the same connection/thread as live jog
            # commands. Poll only while idle; otherwise the operator feels the
            # blocking read as a periodic pause in the joystick stream.
            if live_motion_requested or self.live_motion_active:
                joint_poll_time = time.time()
            elif time.time() - joint_poll_time >= self.live_feedback_poll_idle_s:
                joints = self._try_get_current_joints()
                with self.state.lock:
                    self.state.last_joints = joints
                now_joint_t = time.time()
                if joints is not None:
                    j4_now = float(joints.get("j4", 0.0))
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

def _valid_display_mode(mode):
    try:
        w, h = int(mode[0]), int(mode[1])
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (w, h)

def _xrandr_primary_mode():
    if not shutil.which("xrandr"):
        return None
    try:
        cp = subprocess.run(
            ["xrandr", "--current"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    preferred = []
    fallback = []
    for line in cp.stdout.splitlines():
        if " connected" not in line:
            continue
        m = re.search(r"\b(\d+)x(\d+)\+\d+\+\d+", line)
        if not m:
            continue
        mode = (int(m.group(1)), int(m.group(2)))
        if " primary " in f" {line} ":
            preferred.append(mode)
        else:
            fallback.append(mode)
    modes = preferred or fallback
    if not modes:
        return None
    return max(modes, key=lambda item: item[0] * item[1])

def _desktop_display_mode():
    candidates = []
    try:
        for mode in pygame.display.get_desktop_sizes():
            valid = _valid_display_mode(mode)
            if valid:
                candidates.append(valid)
    except Exception:
        pass
    try:
        modes = pygame.display.list_modes()
        if modes and modes != -1:
            for mode in modes:
                valid = _valid_display_mode(mode)
                if valid:
                    candidates.append(valid)
    except Exception:
        pass
    try:
        info = pygame.display.Info()
        valid = _valid_display_mode((info.current_w, info.current_h))
        if valid:
            candidates.append(valid)
    except Exception:
        pass
    valid = _xrandr_primary_mode()
    if valid:
        candidates.append(valid)
    if not candidates:
        return (1280, 960)
    return max(candidates, key=lambda item: item[0] * item[1])

def create_display_surface(fullscreen: bool):
    if fullscreen:
        flags = pygame.NOFRAME | pygame.DOUBLEBUF
        try:
            mode = _desktop_display_mode()
            screen = pygame.display.set_mode(mode, flags)
            print(f"[Display] Borderless fullscreen {mode[0]}x{mode[1]}")
            return screen, flags, True
        except Exception as e:
            print(f"⚠️ Borderless fullscreen failed, falling back to windowed mode: {e}")
            screen = pygame.display.set_mode((1280, 960), pygame.RESIZABLE)
            return screen, pygame.RESIZABLE, False

    flags = pygame.RESIZABLE
    screen = pygame.display.set_mode((1280, 960), flags)
    print("[Display] Windowed 1280x960")
    return screen, flags, False

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
        surf = pygame.transform.scale(surf, (new_w, new_h))
    screen.blit(surf, (x + (w - new_w)//2, y + (h - new_h)//2))
    return True

class FrameBlitCache:
    def __init__(self):
        self._entries = {}

    def get_cover(self, key, frame_bgr, rect):
        x, y, w, h = rect
        fh, fw = frame_bgr.shape[:2]
        entry = self._entries.get(key)
        if (
            entry
            and entry["frame"] is frame_bgr
            and entry["src_size"] == (fw, fh)
            and entry["dest_size"] == (w, h)
        ):
            return entry["surface"], entry["area"]

        scale = max(w / float(fw), h / float(fh))
        new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
        if (new_w, new_h) != (fw, fh):
            surface = pygame.transform.scale(surface, (new_w, new_h))
        crop_x = max(0, (new_w - w) // 2)
        crop_y = max(0, (new_h - h) // 2)
        area = pygame.Rect(crop_x, crop_y, w, h)
        self._entries[key] = {
            "frame": frame_bgr,
            "src_size": (fw, fh),
            "dest_size": (w, h),
            "surface": surface,
            "area": area,
        }
        return surface, area

    def clear(self):
        self._entries.clear()

def blit_frame_cover(screen, frame_bgr, rect, cache=None, cache_key=None):
    x, y, w, h = rect
    pygame.draw.rect(screen, (25,25,25), rect)
    if frame_bgr is None:
        return False
    fh, fw = frame_bgr.shape[:2]
    if fh <= 0 or fw <= 0:
        return False
    if cache is not None and cache_key is not None:
        base, area = cache.get_cover(cache_key, frame_bgr, rect)
        screen.blit(base, (x, y), area)
        return True
    scale = max(w / float(fw), h / float(fh))
    new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
    if (new_w, new_h) != (fw, fh):
        base = pygame.transform.scale(base, (new_w, new_h))
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

def _build_tone_pcm(frequencies, duration_s=0.18, volume=0.72, sample_rate=44100):
    parts = []
    for freq in frequencies:
        count = max(1, int(sample_rate * float(duration_s)))
        t = np.linspace(0.0, float(duration_s), count, endpoint=False)
        wave = np.sin(2.0 * np.pi * float(freq) * t)
        ramp_n = max(1, min(80, count // 8))
        ramp = np.linspace(0.0, 1.0, ramp_n)
        if ramp_n * 2 < count:
            wave[:ramp_n] *= ramp
            wave[-ramp_n:] *= ramp[::-1]
        parts.append(wave)
    if not parts:
        return None
    arr = np.concatenate(parts)
    return np.asarray(arr * float(volume) * 32767.0, dtype=np.int16)

def _build_tone_sound(frequencies, duration_s=0.18, volume=0.72, sample_rate=44100):
    if not pygame.mixer.get_init():
        return None
    pcm = _build_tone_pcm(frequencies, duration_s=duration_s, volume=volume, sample_rate=sample_rate)
    if pcm is None:
        return None
    try:
        return pygame.sndarray.make_sound(pcm)
    except Exception:
        return None

def _write_tone_wav(name, frequencies, duration_s=0.18, volume=0.72, sample_rate=44100):
    pcm = _build_tone_pcm(frequencies, duration_s=duration_s, volume=volume, sample_rate=sample_rate)
    if pcm is None:
        return None
    path = os.path.join(tempfile.gettempdir(), f"dorna_{name}.wav")
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return path
    except Exception:
        return None

def init_audio_cues():
    specs = {
        "syringe_calibration_start": {
            "freqs": [660, 880],
            "duration": 0.16,
            "phrase": "Syringe calibration started.",
        },
        "syringe_calibration_complete": {
            "freqs": [660, 880, 1175],
            "duration": 0.18,
            "phrase": "Syringe calibration complete.",
        },
        "syringe_calibration_failed": {
            "freqs": [440, 330],
            "duration": 0.22,
            "phrase": "Syringe calibration failed.",
        },
        "injection_start": {
            "freqs": [587, 784, 988],
            "duration": 0.16,
            "phrase": "Injection mode started.",
        },
        "injection_stop": {
            "freqs": [523, 392],
            "duration": 0.22,
            "phrase": "Injection mode stopped.",
        },
        "tunnel_complete": {
            "freqs": [784, 988],
            "duration": 0.16,
            "phrase": "Tunnel complete. Preparing injector.",
        },
        "tunnel_mode_start": {
            "freqs": [587, 784, 587],
            "duration": 0.18,
            "phrase": "Tunnel mode started. Tunnel needle first.",
        },
        "gpws_controller_alarm": {
            "freqs": [220, 220, 330],
            "duration": 0.30,
            "phrase": "Robot alarm. Motion halted. Press X to reset when safe.",
        },
        "gpws_thermal_warning": {
            "freqs": [440, 440],
            "duration": 0.24,
            "phrase": "Thermal warning. Reduce load.",
        },
        "gpws_thermal_critical": {
            "freqs": [330, 247, 196],
            "duration": 0.34,
            "phrase": "Thermal critical. Control priority active.",
        },
        "gpws_record_outcome": {
            "freqs": [784, 587],
            "duration": 0.18,
            "phrase": "Record outcome.",
        },
        "gpws_retract_before_workflow": {
            "freqs": [587, 392],
            "duration": 0.22,
            "phrase": "Retract clear before workflow.",
        },
        "gpws_tunnel_tcp": {
            "freqs": [784, 784, 988],
            "duration": 0.16,
            "phrase": "Tunnel needle active.",
        },
        "gpws_injector_tcp": {
            "freqs": [988, 784],
            "duration": 0.16,
            "phrase": "Injector TCP active.",
        },
        "gpws_midway_hold": {
            "freqs": [660],
            "duration": 0.16,
            "phrase": "Midway hold. Advance or free control.",
        },
        "gpws_final_dose": {
            "freqs": [660, 494, 660],
            "duration": 0.20,
            "phrase": "Final dose. Confirm small injection.",
        },
        "gpws_injection_armed": {
            "freqs": [587, 784],
            "duration": 0.16,
            "phrase": "Injection armed.",
        },
        "gpws_high_angle": {
            "freqs": [392, 392, 392],
            "duration": 0.28,
            "phrase": "High angle. High angle.",
        },
        "gpws_test": {
            "freqs": [440, 660, 440],
            "duration": 0.22,
            "phrase": "Safety callout test. Maintain clearance.",
        },
        "tts_test": {
            "freqs": [660, 990],
            "duration": 0.14,
            "phrase": "Voice cue test.",
        },
    }
    for level in _manual_speed_levels_for_audio():
        specs[_speed_scale_cue_name(level)] = {
            "freqs": [494, 660],
            "duration": 0.12,
            "phrase": _speed_scale_phrase(level),
        }
    speech_cmd = _speech_backend_name()

    cues = {}
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    except Exception as e:
        print(f"⚠️ Tone cues disabled: {e}")
    for name, spec in specs.items():
        freqs = spec["freqs"]
        dur = spec["duration"]
        cues[name] = {
            "sound": _build_tone_sound(freqs, duration_s=dur),
            "wav": _write_tone_wav(name, freqs, duration_s=dur),
            "phrase": spec["phrase"],
            "speech_cmd": speech_cmd,
        }
    return {
        name: cue
        for name, cue in cues.items()
        if cue.get("speech_cmd") or cue.get("sound") is not None or cue.get("wav")
    }

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

def draw_alarm_popup(screen, alarm_message: str):
    sw, sh = screen.get_size()
    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 205))
    screen.blit(overlay, (0, 0))

    font_title = pygame.font.SysFont("Consolas", 24, bold=True)
    font = pygame.font.SysFont("Consolas", 17)
    small = pygame.font.SysFont("Consolas", 14)
    panel_w = min(760, sw - 48)
    panel_h = 260
    px = (sw - panel_w) // 2
    py = (sh - panel_h) // 2
    panel = pygame.Rect(px, py, panel_w, panel_h)
    pygame.draw.rect(screen, (55, 22, 22), panel)
    pygame.draw.rect(screen, (255, 80, 80), panel, 3)

    title = font_title.render("ROBOT ALARM LATCHED", True, (255, 230, 220))
    screen.blit(title, (px + (panel_w - title.get_width()) // 2, py + 20))

    lines = [
        "Motion is blocked until the alarm is cleared.",
        "Inspect the robot and workspace before resetting.",
        "Press X to clear/reset the controller alarm.",
    ]
    y = py + 72
    for line in lines:
        surf = font.render(line, True, (255, 240, 230))
        screen.blit(surf, (px + 28, y))
        y += font.get_height() + 8

    if alarm_message:
        msg_lines = wrap_text(f"Controller message: {alarm_message}", small, panel_w - 56)
        y += 4
        for line in msg_lines[:3]:
            surf = small.render(line, True, (245, 210, 190))
            screen.blit(surf, (px + 28, y))
            y += small.get_height() + 4

    hint = small.render("Reset clears the alarm only; it does not power off motors.", True, (255, 220, 160))
    screen.blit(hint, (px + 28, panel.bottom - 34))
    return panel

def draw_thermal_warning_popup(screen, message: str):
    if not message:
        return None
    sw, sh = screen.get_size()
    pad = 18
    notice_font = pygame.font.SysFont("Consolas", 20, bold=True)
    small_notice_font = pygame.font.SysFont("Consolas", 14)
    text_surf = notice_font.render(message, True, (255, 210, 170))
    btn_pad_x = 10
    btn_pad_y = 6
    dismiss_surf = small_notice_font.render("B  Dismiss", True, (255, 240, 220))
    btn_w = dismiss_surf.get_width() + btn_pad_x * 2
    btn_h = dismiss_surf.get_height() + btn_pad_y * 2
    bg_w = min(sw - 80, max(520, text_surf.get_width() + pad * 2))
    bg_h = text_surf.get_height() + btn_h + pad * 3
    bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
    bg.fill((70, 20, 20, 235))
    pygame.draw.rect(bg, (255, 170, 120), bg.get_rect(), 2)
    bg.blit(text_surf, ((bg_w - text_surf.get_width()) // 2, pad))
    btn_x = (bg_w - btn_w) // 2
    btn_y = bg_h - pad - btn_h
    btn_rect_local = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
    pygame.draw.rect(bg, (110, 35, 35), btn_rect_local)
    pygame.draw.rect(bg, (255, 200, 160), btn_rect_local, 2)
    bg.blit(
        dismiss_surf,
        (
            btn_x + (btn_w - dismiss_surf.get_width()) // 2,
            btn_y + (btn_h - dismiss_surf.get_height()) // 2,
        ),
    )
    nx = (sw - bg_w) // 2
    ny = (sh - bg_h) // 2
    screen.blit(bg, (nx, ny))
    return pygame.Rect(
        nx + btn_rect_local.x,
        ny + btn_rect_local.y,
        btn_rect_local.w,
        btn_rect_local.h,
    )

def draw_warning_strip(screen, warning_items, muted_keys, anchor_x=8, anchor_y=None):
    if not warning_items:
        return []
    font = pygame.font.SysFont("Consolas", 13, bold=True)
    small = pygame.font.SysFont("Consolas", 10, bold=True)
    _, sh = screen.get_size()
    blink_on = (pygame.time.get_ticks() // 360) % 2 == 0
    x = int(anchor_x)
    y = int(anchor_y) if anchor_y is not None else max(86, sh // 2 - min(len(warning_items), 7) * 24)
    clickables = []
    for item in warning_items[:7]:
        key = str(item.get("key", ""))
        label = str(item.get("label", "WARN"))[:8]
        color = tuple(item.get("color", (255, 200, 80)))
        muted = key in muted_keys
        rect = pygame.Rect(x, y, 96, 42)
        alpha = 235 if blink_on else 145
        bg = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        bg.fill((18, 12, 8, alpha))
        screen.blit(bg, rect.topleft)
        edge_col = color if blink_on else tuple(max(45, int(c * 0.55)) for c in color)
        pygame.draw.rect(screen, edge_col, rect, 2)
        tri_x = rect.x + 12
        tri_y = rect.y + 7
        tri = [(tri_x + 9, tri_y), (tri_x, tri_y + 18), (tri_x + 18, tri_y + 18)]
        pygame.draw.polygon(screen, edge_col, tri)
        pygame.draw.polygon(screen, (15, 15, 15), tri, 1)
        excl = small.render("!", True, (20, 20, 20))
        screen.blit(excl, (tri_x + 6, tri_y + 5))
        label_surf = font.render(label, True, edge_col)
        screen.blit(label_surf, (rect.x + 34, rect.y + 6))
        action_txt = str(item.get("action", "") or "").strip()
        mute_txt = "MUTED" if muted else (action_txt[:8] if action_txt else "AUDIO")
        mute_col = (210, 210, 210) if muted else (220, 255, 220)
        mute_surf = small.render(mute_txt, True, mute_col)
        screen.blit(mute_surf, (rect.x + 34, rect.y + 24))
        clickables.append((rect, item))
        y += rect.h + 6
    return clickables

def draw_controller_action_bar(screen, panel_rect, actions, position="bottom"):
    actions = [
        (str(btn or "").strip(), str(label or "").strip(), tuple(color))
        for btn, label, color in actions
        if str(btn or "").strip() and str(label or "").strip()
    ]
    if not actions:
        return None
    font_key = pygame.font.SysFont("Consolas", 14, bold=True)
    font_label = pygame.font.SysFont("Consolas", 12)
    max_actions = min(4, len(actions))
    actions = actions[:max_actions]

    margin = 8
    gap = 6
    strip_h = 42
    strip_w = min(panel_rect.w - margin * 2, max(300, min(520, 132 * max_actions)))
    if position == "top":
        strip = pygame.Rect(
            panel_rect.x + margin,
            panel_rect.y + margin,
            panel_rect.w - margin * 2,
            strip_h,
        )
    elif position == "top_right":
        strip = pygame.Rect(
            panel_rect.right - strip_w - margin,
            panel_rect.y + margin,
            strip_w,
            strip_h,
        )
    else:
        strip = pygame.Rect(
            panel_rect.x + margin,
            panel_rect.bottom - strip_h - margin,
            panel_rect.w - margin * 2,
            strip_h,
        )
    bg = pygame.Surface((strip.w, strip.h), pygame.SRCALPHA)
    bg.fill((10, 14, 18, 226))
    screen.blit(bg, strip.topleft)
    pygame.draw.rect(screen, (65, 95, 105), strip, 1)

    slot_w = max(1, (strip.w - gap * (max_actions - 1) - 12) // max_actions)
    x = strip.x + 6
    for btn, label, color in actions:
        slot = pygame.Rect(x, strip.y + 7, slot_w, strip.h - 14)
        key_rect = pygame.Rect(slot.x, slot.y + 2, 26, slot.h - 4)
        pygame.draw.rect(screen, (28, 34, 42), key_rect)
        pygame.draw.rect(screen, color, key_rect, 2)
        key_surf = font_key.render(btn[:1].upper(), True, (245, 245, 245))
        screen.blit(
            key_surf,
            (
                key_rect.x + (key_rect.w - key_surf.get_width()) // 2,
                key_rect.y + (key_rect.h - key_surf.get_height()) // 2,
            ),
        )

        label_lines = wrap_lines(font_label, label, max(20, slot.w - key_rect.w - 8))
        ly = slot.y + max(0, (slot.h - min(2, len(label_lines)) * font_label.get_height()) // 2)
        for ln in label_lines[:2]:
            surf = font_label.render(ln, True, (222, 230, 232))
            screen.blit(surf, (key_rect.right + 6, ly))
            ly += font_label.get_height()
        x += slot_w + gap
    return strip

def draw_ui_nav_focus(screen, nav_items, focus_index, show_hint=True):
    if not nav_items:
        return
    try:
        item = nav_items[int(focus_index) % len(nav_items)]
    except Exception:
        return
    rect = item.get("rect")
    if rect is None:
        return
    rect = pygame.Rect(rect)
    focus_rect = rect.inflate(8, 8)
    blink = (pygame.time.get_ticks() // 280) % 2 == 0
    col = (200, 95, 255) if blink else (140, 60, 220)
    pygame.draw.rect(screen, col, focus_rect, 3)
    pygame.draw.rect(screen, (20, 20, 20), focus_rect, 1)
    if not show_hint:
        return

    font = pygame.font.SysFont("Consolas", 13, bold=True)
    small = pygame.font.SysFont("Consolas", 11)
    label = str(item.get("label", "UI element") or "UI element")
    hint = "MENU UI MODE  RS move  A select  Menu exit"
    lines = [label, hint]
    max_w = 320
    wrapped = []
    for idx, line in enumerate(lines):
        wrapped.extend(wrap_lines(font if idx == 0 else small, line, max_w - 18)[:2])
    h = 10 + font.get_height() + small.get_height() + 4
    w = min(max_w, max(210, max((font.render(line, True, (0, 0, 0)).get_width() for line in wrapped), default=180) + 18))
    x = min(max(8, focus_rect.x), screen.get_width() - w - 8)
    y = focus_rect.bottom + 8
    if y + h > screen.get_height() - 8:
        y = focus_rect.y - h - 8
    if y < 8:
        y = 8
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((8, 12, 16, 230))
    screen.blit(panel, (x, y))
    pygame.draw.rect(screen, col, pygame.Rect(x, y, w, h), 1)
    title_surf = font.render(label[:64], True, (235, 205, 255))
    screen.blit(title_surf, (x + 9, y + 5))
    hint_surf = small.render(hint, True, (225, 230, 230))
    screen.blit(hint_surf, (x + 9, y + 5 + font.get_height() + 2))

def make_nav_item(rect, label, group="ui", kind=""):
    if rect is None:
        return None
    rect = pygame.Rect(rect)
    if rect.w <= 2 or rect.h <= 2:
        return None
    return {
        "rect": rect,
        "label": str(label or "UI element"),
        "group": str(group or "ui"),
        "kind": str(kind or ""),
    }

def add_nav_item(items, rect, label, group="ui", screen_rect=None, kind=""):
    item = make_nav_item(rect, label, group, kind=kind)
    if item is None:
        return
    if screen_rect is not None and not item["rect"].colliderect(screen_rect):
        return
    items.append(item)

def sort_nav_items(items):
    def key(item):
        rect = item.get("rect") or pygame.Rect(0, 0, 0, 0)
        group = item.get("group", "ui")
        group_rank = {"warning": 0, "modal": 1, "recording": 2, "camera": 3, "panel": 4}.get(group, 5)
        return (group_rank, rect.y, rect.x)
    return sorted(items, key=key)

def preferred_nav_focus_index(items):
    if not items:
        return 0
    preferred_kinds = (
        "start_injection",
        "stop_injection",
        "injection_record_outcome",
        "injection_tunnel_complete",
        "injection_reload",
        "injection_new_subject",
        "injection_tunnel_prepare",
        "injection_tunnel_injector",
    )
    for kind in preferred_kinds:
        for idx, item in enumerate(items):
            if item.get("group") == "panel" and item.get("kind") == kind:
                return idx
    for preferred_group in ("panel", "modal", "recording", "warning", "camera"):
        for idx, item in enumerate(items):
            if item.get("group") == preferred_group:
                return idx
    return 0

def estimate_right_panel_scroll_for_kinds(
    buttons,
    target_kinds,
    tr_rect,
    m_ui,
    btn_w,
    btn_gap,
    label_font,
    placeholder_font,
):
    """Return a scroll offset that brings a preferred main-menu action into view."""
    if not buttons or not target_kinds:
        return None

    def item_height(item):
        kind = str(item.get("kind", "") or "")
        if kind == "top_actions":
            return 38 + btn_gap + 2
        if kind == "header":
            return label_font.get_height() + 10 + btn_gap
        if kind == "alarm_status":
            return placeholder_font.get_height() + btn_gap
        if kind in ("tool_preset_row", "routine_row", "pose_row"):
            return 34 + btn_gap
        if kind in ("alarm_threshold", "alarm_duration"):
            return 40 + btn_gap
        lines = wrap_text(str(item.get("label", "") or ""), placeholder_font, btn_w - 10)
        bh = len(lines) * (placeholder_font.get_height() + 2) + 10
        return bh + btn_gap

    heights = [item_height(item) for item in buttons]
    total_h = sum(heights) + m_ui
    max_scroll = max(0, total_h - tr_rect.h + 12)
    if max_scroll <= 0:
        return 0

    for target_kind in target_kinds:
        offset = 0
        for item, height in zip(buttons, heights):
            if str(item.get("kind", "") or "") == target_kind:
                desired_y = max(48, min(110, tr_rect.h // 5))
                return max(0, min(max_scroll, int(offset - desired_y)))
            offset += height
    return None

def nav_label_from_action(action):
    if not isinstance(action, dict):
        return "UI element"
    kind = str(action.get("kind", "") or "")
    if action.get("label"):
        return str(action.get("label"))
    if kind == "goto":
        return f"Go to pose: {action.get('pose', '')}"
    if kind == "routine_select":
        return f"Run routine: {action.get('routine', '')}"
    if kind == "routine_edit_row":
        return f"Edit routine: {action.get('routine', '')}"
    if kind == "tool_preset_apply":
        return f"Apply tool preset: {action.get('preset', '')}"
    if kind == "tool_preset_menu":
        return f"Edit tool preset: {action.get('preset', '')}"
    return {
        "rs_hud_toggle": "Toggle status overlay",
        "rs_mode_cycle": "Cycle RealSense mode",
        "pitch_input": "Set pitch",
        "angle_set": "Set roll",
        "angle_toggle": "Toggle roll absolute/relative",
        "yaw_set": "Set yaw",
        "yaw_base_set": "Zero yaw reference",
        "settings_modal": "Open settings",
        "quit_app": "Quit application",
        "routine_stop": "Stop routine",
        "start_injection": "Start injection",
        "stop_injection": "Exit injection mode",
        "injection_record_outcome": "Record injection outcome",
        "injection_reload": "Reload/refill syringe",
        "injection_tunnel_prepare": "Prepare tunnel needle",
        "injection_tunnel_injector": "Prepare injector",
        "injection_tunnel_complete": "Tunnel complete -> injector",
        "injection_new_subject": "Next animal/sample",
        "settings_poses": "Open pose settings",
        "settings_routines": "Open routine settings",
        "demo_resume": "Resume demo",
    }.get(kind, kind.replace("_", " ").strip().title() or "UI element")

def joy_button_pressed(joy, button_index):
    try:
        if hasattr(joy, "get_numbuttons") and int(button_index) >= int(joy.get_numbuttons()):
            return False
        return bool(joy.get_button(int(button_index)))
    except Exception:
        return False

def joy_any_button_pressed(joy, button_indices):
    return any(joy_button_pressed(joy, idx) for idx in button_indices)

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

def draw_routine_editor(screen, editor: EditorState, title="Routine Script",
                        height_lines=12, width_chars=56):
    sw, sh = screen.get_size()
    font = pygame.font.SysFont("Consolas", 16)
    small = pygame.font.SysFont("Consolas", 13)
    char_w, char_h = font.size("M")[0], font.get_height()

    panel_w = int(width_chars * char_w + 40)
    panel_h = int(height_lines * char_h + 136)
    x = (sw - panel_w)//2
    y = (sh - panel_h)//2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    btn_h = 28
    close_rect = pygame.Rect(panel.right - 74, y + 12, 58, btn_h)
    apply_rect = pygame.Rect(close_rect.x - 68, y + 12, 60, btn_h)
    builder_rect = pygame.Rect(apply_rect.x - 104, y + 12, 96, btn_h)

    title_surf = font.render(title, True, (230,230,230))
    screen.blit(title_surf, (x + 12, y + 12))
    subtitle = small.render("Ctrl+Enter=Apply   Ctrl+B=GUI Builder   Esc=Close", True, (190, 205, 205))
    screen.blit(subtitle, (x + 12, y + 36))

    draw_modal_button(screen, builder_rect, "GUI Builder", small, border=(0, 220, 200), fill=(42, 52, 66))
    draw_modal_button(screen, apply_rect, "Apply", small, border=(0, 255, 127), fill=(45, 65, 52))
    draw_modal_button(screen, close_rect, "Close", small, border=(255, 120, 120), fill=(64, 42, 48))

    box = pygame.Rect(x + 12, y + 62, panel_w - 24, height_lines*char_h + 8)
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

    clickables = [
        (builder_rect, {"kind": "routine_script_builder"}),
        (apply_rect, {"kind": "routine_script_apply"}),
        (close_rect, {"kind": "routine_script_close"}),
    ]
    return panel, box, (char_w, char_h), clickables

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
    title_font = pygame.font.SysFont("Consolas", 18, bold=True)
    font = pygame.font.SysFont("Consolas", 15)
    small = pygame.font.SysFont("Consolas", 13)

    panel_w = min(720, max(620, sw - 80))
    panel_h = 340
    x = (sw - panel_w)//2
    y = (sh - panel_h)//2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    title = title_font.render("Syringe Calibration", True, (230,230,230))
    screen.blit(title, (x + 16, y + 14))

    label_color = (210,210,210)
    box_color   = (50,50,80)
    active_col  = (0,255,127)
    idle_col    = (0,150,90)

    field_rects = {}
    spinner_rects = {}

    def draw_numeric_row(row_y, key, label, text):
        label_surf = font.render(label, True, label_color)
        screen.blit(label_surf, (x + 18, row_y + 6))
        spinner_w = 32
        gap = 6
        box_x = x + 300
        box_w = panel_w - (box_x - x) - 18 - (spinner_w + gap) * 2
        box = pygame.Rect(box_x, row_y, box_w, 34)
        minus = pygame.Rect(box.right + gap, row_y, spinner_w, 34)
        plus = pygame.Rect(minus.right + gap, row_y, spinner_w, 34)
        pygame.draw.rect(screen, box_color, box)
        pygame.draw.rect(screen, active_col if active_field == key else idle_col, box, 2)
        prev_clip = screen.get_clip()
        screen.set_clip(box.inflate(-10, -4))
        txt_surf = font.render(text or "", True, (255, 255, 255))
        screen.blit(txt_surf, (box.x + 8, box.y + (box.h - txt_surf.get_height()) // 2))
        screen.set_clip(prev_clip)
        for rect, symbol in ((minus, "-"), (plus, "+")):
            pygame.draw.rect(screen, (45, 55, 75), rect)
            pygame.draw.rect(screen, active_col, rect, 1)
            sym = font.render(symbol, True, (235, 245, 245))
            screen.blit(sym, (rect.x + (rect.w - sym.get_width()) // 2,
                              rect.y + (rect.h - sym.get_height()) // 2))
        field_rects[key] = box
        spinner_rects[f"{key}_minus"] = minus
        spinner_rects[f"{key}_plus"] = plus

    y1 = y + 58
    draw_numeric_row(y1, "vol", "Total syringe volume (uL)", vol_text)
    draw_numeric_row(y1 + 48, "step", "Step size (uL / injection)", step_text)

    status_rect = pygame.Rect(x + 16, y + 158, panel_w - 32, 86)
    pygame.draw.rect(screen, (24, 24, 32), status_rect)
    pygame.draw.rect(screen, (80, 105, 105), status_rect, 1)
    status_title = small.render("Status", True, (180, 220, 210))
    screen.blit(status_title, (status_rect.x + 8, status_rect.y + 6))
    if status_text:
        lines = wrap_lines(small, str(status_text), status_rect.w - 16)
        sy = status_rect.y + 24
        max_lines = max(1, (status_rect.bottom - sy - 6) // (small.get_height() + 2))
        for ln in lines[-max_lines:]:
            stxt = small.render(ln, True, (220, 220, 220))
            screen.blit(stxt, (status_rect.x + 8, sy))
            sy += stxt.get_height() + 2

    btn_w = min(200, (panel_w - 32 - 20) // 3)
    btn_h = 38
    spacing = 10
    total_btn_w = btn_w * 3 + spacing * 2
    bx = x + (panel_w - total_btn_w) // 2
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

    buttons = {
        "calibrate": btn_cal,
        "apply": btn_apply,
        "cancel": btn_cancel,
    }
    buttons.update(spinner_rects)
    return field_rects, buttons, panel

def draw_modal_button(screen, rect, label, font, border=(0, 255, 127), fill=(50, 50, 80)):
    pygame.draw.rect(screen, fill, rect)
    pygame.draw.rect(screen, border, rect, 2)
    lines = wrap_lines(font, label, rect.w - 12)
    line_h = font.get_height()
    ly = rect.y + max(2, (rect.h - len(lines) * line_h) // 2)
    for line in lines:
        surf = font.render(line, True, (230, 230, 230))
        screen.blit(surf, (rect.x + (rect.w - surf.get_width()) // 2, ly))
        ly += line_h

def draw_settings_modal(screen, active_tab, items, alarm_threshold, alarm_duration, scroll=0):
    sw, sh = screen.get_size()
    title_font = pygame.font.SysFont("Consolas", 18, bold=True)
    font = pygame.font.SysFont("Consolas", 14)
    small = pygame.font.SysFont("Consolas", 12)

    panel_w = min(900, max(660, sw - 90))
    panel_h = min(620, max(500, sh - 70))
    x = (sw - panel_w) // 2
    y = (sh - panel_h) // 2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 185))
    screen.blit(overlay, (0, 0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (28, 30, 38), panel)
    pygame.draw.rect(screen, (0, 220, 200), panel, 2)

    title = title_font.render("Settings", True, (235, 245, 245))
    screen.blit(title, (x + 18, y + 14))

    close_rect = pygame.Rect(panel.right - 94, y + 12, 76, 28)
    draw_modal_button(
        screen,
        close_rect,
        "Close",
        small,
        border=(255, 120, 120),
        fill=(64, 42, 48),
    )

    tabs = [
        ("tool", "Tool / TCP"),
        ("poses", "Poses"),
        ("injection", "Injection"),
        ("routine", "Routine"),
        ("safety", "Safety"),
        ("audio", "Audio"),
        ("files", "Files"),
    ]
    tab_rects = {}
    tab_x = x + 18
    tab_y = y + 54
    tab_h = 34
    tab_gap = 8
    for key, label in tabs:
        tw = max(86, small.size(label)[0] + 18)
        rect = pygame.Rect(tab_x, tab_y, tw, tab_h)
        is_active = key == active_tab
        pygame.draw.rect(screen, (38, 58, 64) if is_active else (42, 44, 54), rect)
        pygame.draw.rect(screen, (0, 255, 200) if is_active else (90, 110, 120), rect, 2 if is_active else 1)
        txt = small.render(label, True, (235, 245, 245) if is_active else (195, 205, 205))
        screen.blit(txt, (rect.x + (rect.w - txt.get_width()) // 2, rect.y + (rect.h - txt.get_height()) // 2))
        tab_rects[key] = rect
        tab_x += tw + tab_gap

    content = pygame.Rect(x + 18, tab_y + tab_h + 14, panel_w - 36, panel_h - 124)
    pygame.draw.rect(screen, (22, 24, 30), content)
    pygame.draw.rect(screen, (70, 90, 95), content, 1)

    clickables = []
    prev_clip = screen.get_clip()
    screen.set_clip(content.inflate(-2, -2))
    row_start = content.y + 12 - int(scroll)
    row_y = row_start
    row_gap = 8
    for item in items:
        kind = item.get("kind", "")
        label = str(item.get("label", ""))
        if kind == "header":
            surf = font.render(label, True, (170, 220, 210))
            if row_y + surf.get_height() >= content.y and row_y <= content.bottom:
                screen.blit(surf, (content.x + 12, row_y))
            row_y += surf.get_height() + row_gap
            continue
        if kind == "tool_preset_manage_row":
            nm = str(item.get("preset", "")).strip()
            rect = pygame.Rect(content.x + 12, row_y, content.w - 24, 42)
            visible = rect.bottom >= content.y and rect.y <= content.bottom
            edit_w = 76
            main_rect = pygame.Rect(rect.x, rect.y, rect.w - edit_w - 8, rect.h)
            edit_rect = pygame.Rect(main_rect.right + 8, rect.y, edit_w, rect.h)
            if visible:
                pygame.draw.rect(screen, (42, 44, 58), main_rect)
                pygame.draw.rect(screen, (0, 220, 200), main_rect, 1)
                lines = wrap_lines(font, label, main_rect.w - 18)
                ly = main_rect.y + 6
                for line in lines[:2]:
                    surf = font.render(line, True, (225, 228, 230))
                    screen.blit(surf, (main_rect.x + 10, ly))
                    ly += font.get_height() + 2
                draw_modal_button(
                    screen,
                    edit_rect,
                    "Edit",
                    small,
                    border=(0, 255, 127),
                    fill=(52, 55, 76),
                )
                clickables.append((main_rect, {"kind": "tool_preset_apply", "preset": nm}))
                clickables.append((edit_rect, {"kind": "tool_preset_menu", "preset": nm}))
            row_y += rect.h + row_gap
            continue
        if kind == "pose_manage_row":
            name = str(item.get("pose", "")).strip()
            reserved = bool(item.get("reserved", False))
            starred = bool(item.get("starred", False))
            rect = pygame.Rect(content.x + 12, row_y, content.w - 24, 44)
            visible = rect.bottom >= content.y and rect.y <= content.bottom
            star_w = 84
            edit_w = 66
            go_w = 58
            gap = 7
            label_w = rect.w - star_w - edit_w - go_w - gap * 3
            label_rect = pygame.Rect(rect.x, rect.y, max(120, label_w), rect.h)
            go_rect = pygame.Rect(label_rect.right + gap, rect.y, go_w, rect.h)
            edit_rect = pygame.Rect(go_rect.right + gap, rect.y, edit_w, rect.h)
            star_rect = pygame.Rect(edit_rect.right + gap, rect.y, star_w, rect.h)
            if visible:
                pygame.draw.rect(screen, (36, 38, 50), rect)
                pygame.draw.rect(screen, (0, 220, 200) if starred else (80, 110, 115), rect, 1)
                name_col = (255, 220, 120) if reserved else (225, 228, 230)
                tag = "STARRED" if starred else "library"
                lines = wrap_lines(font, f"{name}  [{tag}]", label_rect.w - 14)
                ly = label_rect.y + 7
                for line in lines[:2]:
                    surf = font.render(line, True, name_col)
                    screen.blit(surf, (label_rect.x + 10, ly))
                    ly += font.get_height() + 2
                draw_modal_button(screen, go_rect, "Go", small, border=(0, 220, 200), fill=(38, 54, 64))
                draw_modal_button(screen, edit_rect, "Edit", small, border=(0, 255, 127), fill=(52, 55, 76))
                draw_modal_button(
                    screen,
                    star_rect,
                    "Unstar" if starred else "Star",
                    small,
                    border=(255, 220, 90) if starred else (150, 150, 150),
                    fill=(66, 58, 36) if starred else (48, 48, 58),
                )
                clickables.append((go_rect, {"kind": "goto", "pose": name}))
                clickables.append((edit_rect, {"kind": "edit_menu", "pose": name, "reserved": reserved}))
                clickables.append((star_rect, {"kind": "pose_star_toggle", "pose": name}))
            row_y += rect.h + row_gap
            continue
        if kind == "routine_manage_row":
            rname = str(item.get("routine", "")).strip()
            is_current = bool(item.get("current", False))
            rect = pygame.Rect(content.x + 12, row_y, content.w - 24, 44)
            visible = rect.bottom >= content.y and rect.y <= content.bottom
            use_w = 92
            run_w = 62
            edit_w = 66
            gap = 7
            label_w = rect.w - use_w - run_w - edit_w - gap * 3
            label_rect = pygame.Rect(rect.x, rect.y, max(120, label_w), rect.h)
            use_rect = pygame.Rect(label_rect.right + gap, rect.y, use_w, rect.h)
            run_rect = pygame.Rect(use_rect.right + gap, rect.y, run_w, rect.h)
            edit_rect = pygame.Rect(run_rect.right + gap, rect.y, edit_w, rect.h)
            if visible:
                pygame.draw.rect(screen, (36, 38, 50), rect)
                pygame.draw.rect(screen, (0, 255, 180) if is_current else (80, 110, 115), rect, 1)
                tag = "current" if is_current else "saved"
                lines = wrap_lines(font, f"{rname}  [{tag}]", label_rect.w - 14)
                ly = label_rect.y + 7
                for line in lines[:2]:
                    surf = font.render(line, True, (225, 240, 240))
                    screen.blit(surf, (label_rect.x + 10, ly))
                    ly += font.get_height() + 2
                draw_modal_button(screen, use_rect, "Current", small, border=(0, 220, 200), fill=(38, 54, 64))
                draw_modal_button(screen, run_rect, "Run", small, border=(0, 255, 127), fill=(45, 65, 52))
                draw_modal_button(screen, edit_rect, "Edit", small, border=(0, 255, 127), fill=(52, 55, 76))
                clickables.append((use_rect, {"kind": "routine_make_current", "routine": rname}))
                clickables.append((run_rect, {"kind": "routine_run", "routine": rname}))
                clickables.append((edit_rect, {"kind": "routine_edit_row", "routine": rname}))
            row_y += rect.h + row_gap
            continue
        if kind in ("alarm_threshold", "alarm_duration"):
            rect = pygame.Rect(content.x + 12, row_y, content.w - 24, 48)
            visible = rect.bottom >= content.y and rect.y <= content.bottom
            if kind == "alarm_threshold":
                min_val, max_val = DEFAULT_PID_THRESHOLD_MIN, DEFAULT_PID_THRESHOLD_MAX
                cur_val = max(min_val, min(max_val, float(alarm_threshold)))
                value_text = f"{int(round(cur_val))}"
            else:
                min_val, max_val = DEFAULT_PID_DURATION_MIN, DEFAULT_PID_DURATION_MAX
                cur_val = max(min_val, min(max_val, float(alarm_duration)))
                value_text = f"{int(round(cur_val))}"
            bar_rect = pygame.Rect(rect.x + 14, rect.bottom - 18, rect.w - 90, 8)
            if visible:
                pygame.draw.rect(screen, (36, 38, 50), rect)
                pygame.draw.rect(screen, (0, 180, 200), rect, 1)
                label_surf = small.render(label, True, (225, 230, 230))
                screen.blit(label_surf, (rect.x + 10, rect.y + 6))
                pygame.draw.rect(screen, (70, 70, 90), bar_rect)
                rel = (cur_val - min_val) / max(1e-6, (max_val - min_val))
                knob_x = bar_rect.x + int(rel * bar_rect.w)
                pygame.draw.rect(screen, (0, 255, 127), (knob_x - 6, bar_rect.y - 5, 12, 18))
                value_surf = small.render(value_text, True, (230, 240, 240))
                screen.blit(value_surf, (rect.right - value_surf.get_width() - 14, bar_rect.y - 5))
                clickables.append((bar_rect, {"kind": kind, "rect": bar_rect}))
            row_y += rect.h + row_gap
            continue
        if kind == "jog_epsilon_preset":
            current = bool(item.get("current", False))
            rect = pygame.Rect(content.x + 12, row_y, content.w - 24, 40)
            visible = rect.bottom >= content.y and rect.y <= content.bottom
            if visible:
                fill = (34, 58, 48) if current else (42, 44, 58)
                edge = (0, 255, 127) if current else (0, 220, 200)
                pygame.draw.rect(screen, fill, rect)
                pygame.draw.rect(screen, edge, rect, 2 if current else 1)
                tag = "ACTIVE" if current else "SET"
                tag_surf = small.render(tag, True, (10, 18, 16) if current else (220, 245, 245))
                tag_rect = pygame.Rect(rect.x + 10, rect.y + 8, 62, 24)
                pygame.draw.rect(screen, (0, 255, 127) if current else (38, 54, 64), tag_rect)
                pygame.draw.rect(screen, edge, tag_rect, 1)
                screen.blit(
                    tag_surf,
                    (
                        tag_rect.x + (tag_rect.w - tag_surf.get_width()) // 2,
                        tag_rect.y + (tag_rect.h - tag_surf.get_height()) // 2,
                    ),
                )
                text = (
                    label
                    .replace("Current jog threshold: ", "")
                    .replace("Set jog threshold: ", "")
                    .replace("Current fine speed reference: ", "")
                    .replace("Set fine speed reference: ", "")
                )
                txt = font.render(text, True, (235, 245, 245) if current else (225, 228, 230))
                screen.blit(txt, (tag_rect.right + 12, rect.y + (rect.h - txt.get_height()) // 2))
                clickables.append((rect, item))
            row_y += rect.h + row_gap
            continue

        lines = wrap_lines(font, label, content.w - 44)
        row_h = max(36, len(lines) * (font.get_height() + 2) + 14)
        rect = pygame.Rect(content.x + 12, row_y, content.w - 24, row_h)
        if rect.bottom >= content.y and rect.y <= content.bottom:
            pygame.draw.rect(screen, (42, 44, 58), rect)
            pygame.draw.rect(screen, (0, 220, 200), rect, 1)
            ly = rect.y + 7
            for line in lines:
                surf = font.render(line, True, (225, 228, 230))
                screen.blit(surf, (rect.x + 12, ly))
                ly += font.get_height() + 2
            clickables.append((rect, item))
        row_y += row_h + row_gap

    screen.set_clip(prev_clip)
    content_h = row_y - row_start + 12
    max_scroll = max(0, content_h - max(1, content.h - 10))
    if max_scroll > 0:
        track = pygame.Rect(content.right - 10, content.y + 6, 4, content.h - 12)
        pygame.draw.rect(screen, (55, 65, 70), track)
        knob_h = max(24, int(track.h * (content.h / max(content.h, content_h))))
        knob_y = track.y + int((track.h - knob_h) * (max(0, min(scroll, max_scroll)) / max_scroll))
        pygame.draw.rect(screen, (0, 220, 200), pygame.Rect(track.x, knob_y, track.w, knob_h))

    return tab_rects, clickables, close_rect, panel, max_scroll

def draw_checkbox_row(screen, x, y, width, label, checked, font, color=(0, 255, 127)):
    box_size = 18
    box = pygame.Rect(x, y, box_size, box_size)
    pygame.draw.rect(screen, (50, 50, 80), box)
    pygame.draw.rect(screen, color, box, 2)
    if checked:
        pygame.draw.line(screen, color, (box.x + 3, box.y + box_size // 2),
                         (box.x + box_size // 2, box.y + box_size - 3), 2)
        pygame.draw.line(screen, color, (box.x + box_size // 2, box.y + box_size - 3),
                         (box.x + box_size - 3, box.y + 3), 2)
    text_x = box.right + 8
    lines = wrap_lines(font, label, width - (text_x - x))
    ly = y + (box_size - font.get_height()) // 2
    for line in lines:
        surf = font.render(line, True, (210, 210, 210))
        screen.blit(surf, (text_x, ly))
        ly += font.get_height() + 1
    row_h = max(box_size, len(lines) * (font.get_height() + 1))
    return pygame.Rect(x, y, width, row_h), row_h

def draw_clipped_text(screen, rect, text, font, color=(255, 255, 255), pad=8):
    prev_clip = screen.get_clip()
    inner = rect.inflate(-pad * 2, -4)
    screen.set_clip(inner)
    surf = font.render(text or "", True, color)
    screen.blit(surf, (rect.x + pad, rect.y + (rect.h - surf.get_height()) // 2))
    screen.set_clip(prev_clip)

def draw_injection_info_modal(
    screen,
    study_text,
    material_text,
    date_text,
    active_field,
    record_telemetry,
    record_video,
    record_ui,
    use_tunnel_workflow,
    material_loaded,
    syringe_calibrated,
):
    """
    Start Injection popup: Study, Injection Material, Date
    plus readiness, workflow, and recording choices.
    """
    sw, sh = screen.get_size()
    title_font = pygame.font.SysFont("Consolas", 18, bold=True)
    font = pygame.font.SysFont("Consolas", 15)
    small = pygame.font.SysFont("Consolas", 13)

    panel_w = min(760, max(640, sw - 80))
    panel_h = min(560, max(544, sh - 40))
    x = (sw - panel_w)//2
    y = (sh - panel_h)//2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    title = title_font.render("Start Injection - Metadata & Logging", True, (230,230,230))
    screen.blit(title, (x + 16, y + 14))
    subtitle = small.render("Configure the session before enabling injection mode.", True, (200, 210, 210))
    screen.blit(subtitle, (x + 16, y + 40))

    label_color = (210,210,210)
    box_color   = (50,50,80)
    active_col  = (0,255,127)
    idle_col    = (0,150,90)

    def field(ypos, text, label, key):
        lbl = font.render(label, True, label_color)
        screen.blit(lbl, (x + 18, ypos + 7))
        box = pygame.Rect(x + 190, ypos, panel_w - 208, 34)
        pygame.draw.rect(screen, box_color, box)
        pygame.draw.rect(screen, active_col if active_field == key else idle_col, box, 2)
        draw_clipped_text(screen, box, text or "", font)
        return box

    y1 = y + 76
    box_study   = field(y1, study_text,        "Study:",              "study")
    y2 = y1 + 46
    box_mat     = field(y2, material_text,     "Injection Material:", "material")
    y3 = y2 + 46
    box_date    = field(y3, date_text,         "Date:",               "date")

    prep_panel = pygame.Rect(x + 16, y3 + 52, panel_w - 32, 126)
    pygame.draw.rect(screen, (24, 24, 32), prep_panel)
    pygame.draw.rect(screen, (80, 105, 105), prep_panel, 1)
    prep_title = small.render("Step 0 / Workflow", True, (180, 220, 210))
    screen.blit(prep_title, (prep_panel.x + 8, prep_panel.y + 6))
    calib_col = (0, 255, 127) if syringe_calibrated else (255, 170, 80)
    calib_text = "YES" if syringe_calibrated else "NO - calibrate syringe before injecting"
    calib_surf = small.render(f"Step 0 syringe calibrated: {calib_text}", True, calib_col)
    screen.blit(calib_surf, (prep_panel.x + 10, prep_panel.y + 28))
    prep_row_y = prep_panel.y + 52
    cb_loaded, h_loaded = draw_checkbox_row(
        screen,
        prep_panel.x + 10,
        prep_row_y,
        prep_panel.w - 20,
        "X  Syringe is loaded with injection material",
        material_loaded,
        small,
        color=(0, 220, 200),
    )
    prep_row_y += h_loaded + 7
    cb_tunnel, _ = draw_checkbox_row(
        screen,
        prep_panel.x + 10,
        prep_row_y,
        prep_panel.w - 20,
        "Y  Use optional tunnel needle workflow in this injection session",
        use_tunnel_workflow,
        small,
        color=(255, 210, 90),
    )

    log_panel = pygame.Rect(x + 16, prep_panel.bottom + 10, panel_w - 32, 116)
    pygame.draw.rect(screen, (24, 24, 32), log_panel)
    pygame.draw.rect(screen, (80, 105, 105), log_panel, 1)
    log_title = small.render("Recording", True, (180, 220, 210))
    screen.blit(log_title, (log_panel.x + 8, log_panel.y + 6))
    row_y = log_panel.y + 28
    cb1, h1 = draw_checkbox_row(screen, log_panel.x + 10, row_y, log_panel.w - 20,
                                "Record robot telemetry and controller inputs", record_telemetry, small)
    row_y += h1 + 7
    cb2, h2 = draw_checkbox_row(screen, log_panel.x + 10, row_y, log_panel.w - 20,
                                "Record camera video: RealSense and both UVC feeds", record_video, small)
    row_y += h2 + 7
    cb3, h3 = draw_checkbox_row(screen, log_panel.x + 10, row_y, log_panel.w - 20,
                                "Record full UI overlay per step", record_ui, small)

    # Buttons
    btn_w = min(220, (panel_w - 48) // 2)
    btn_h = 40
    bx = x + (panel_w - (btn_w * 2 + 12)) // 2
    by = y + panel_h - btn_h - 16
    btn_start  = pygame.Rect(bx,               by, btn_w, btn_h)
    btn_cancel = pygame.Rect(bx + btn_w + 12,  by, btn_w, btn_h)

    draw_modal_button(screen, btn_start, "A  Start Injection Mode", font, border=(0, 255, 127), fill=(45, 70, 55))
    draw_modal_button(screen, btn_cancel, "B  Cancel", font, border=(255, 100, 100), fill=(70, 45, 50))

    return (
        {"study": box_study, "material": box_mat, "date": box_date},
        {"start": btn_start, "cancel": btn_cancel},
        {
            "telemetry": cb1,
            "video": cb2,
            "ui": cb3,
            "material_loaded": cb_loaded,
            "tunnel": cb_tunnel,
        },
        panel,
    )

def current_injection_timestamp_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def split_subject_eye_text(value):
    raw = str(value or "").strip()
    if not raw:
        return "", "OD"
    for eye in ("OD", "OS"):
        for suffix in (f" [{eye}]", f" {eye}", f"_{eye}", f"-{eye}"):
            if raw.upper().endswith(suffix.upper()):
                return raw[:-len(suffix)].strip(" _-/"), eye
    return raw, "OD"

def compose_subject_eye_text(subject_text, eye_choice):
    subject = str(subject_text or "").strip()
    eye = str(eye_choice or "OD").strip().upper()
    if eye not in ("OD", "OS"):
        eye = "OD"
    return f"{subject} {eye}".strip() if subject else eye

def draw_subject_modal(screen, subject_text, eye_choice, use_tunnel_workflow, step_index, allow_advance):
    """
    Popup to capture animal/eye/sample for this injection step.

    step_index: 1-based index of the upcoming step.
    allow_advance: if True, show "Confirm & Advance" button.
    """
    sw, sh = screen.get_size()
    title_font = pygame.font.SysFont("Consolas", 18, bold=True)
    font = pygame.font.SysFont("Consolas", 15)
    small = pygame.font.SysFont("Consolas", 13)

    panel_w = min(760, max(640, sw - 80))
    panel_h = 384
    x = (sw - panel_w)//2
    y = (sh - panel_h)//2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0,0,0,190))
    screen.blit(overlay, (0,0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (30,30,40), panel)
    pygame.draw.rect(screen, (0,255,127), panel, 2)

    title = title_font.render(f"Animal / Eye / Sample - Step {step_index}", True, (230,230,230))
    screen.blit(title, (x + 16, y + 14))

    info_lines = wrap_lines(
        small,
        "Recording starts after confirmation and stays active through tunnel prep, injection, and outcome entry.",
        panel_w - 32,
    )
    ty = y + 48
    for ln in info_lines:
        t = small.render(ln, True, (210,210,210))
        screen.blit(t, (x + 16, ty))
        ty += t.get_height() + 2

    label = font.render("Animal / Sample ID:", True, (210,210,210))
    screen.blit(label, (x + 16, ty + 14))

    box = pygame.Rect(x + 16, ty + 40, panel_w - 32, 38)
    pygame.draw.rect(screen, (50,50,80), box)
    pygame.draw.rect(screen, (0,255,127), box, 2)
    draw_clipped_text(screen, box, subject_text or "", font)

    controls_y = box.bottom + 18
    eye_label = font.render("Eye:", True, (210, 210, 210))
    screen.blit(eye_label, (x + 16, controls_y + 8))
    eye_x = x + 86
    eye_w = 82
    eye_h = 34
    eye_choice = str(eye_choice or "OD").upper()
    if eye_choice not in ("OD", "OS"):
        eye_choice = "OD"
    od_rect = pygame.Rect(eye_x, controls_y, eye_w, eye_h)
    os_rect = pygame.Rect(od_rect.right + 8, controls_y, eye_w, eye_h)
    for rect, label_eye in ((od_rect, "OD"), (os_rect, "OS")):
        selected = eye_choice == label_eye
        pygame.draw.rect(screen, (45, 70, 55) if selected else (42, 44, 54), rect)
        pygame.draw.rect(screen, (0, 255, 127) if selected else (100, 120, 120), rect, 2 if selected else 1)
        t = font.render(label_eye, True, (235, 245, 235))
        screen.blit(t, (rect.x + (rect.w - t.get_width()) // 2, rect.y + (rect.h - t.get_height()) // 2))

    tunnel_rect, tunnel_h = draw_checkbox_row(
        screen,
        x + 16,
        controls_y + eye_h + 16,
        panel_w - 32,
        "Prepare tunnel needle workflow for this animal/eye after confirmation",
        bool(use_tunnel_workflow),
        small,
        color=(255, 210, 90),
    )

    # Buttons
    btn_w = min(200, (panel_w - 52) // (3 if allow_advance else 2))
    btn_h = 42
    gap = 12
    n_btn = 3 if allow_advance else 2
    total_w = n_btn*btn_w + (n_btn-1)*gap
    bx = x + (panel_w - total_w)//2
    by = y + panel_h - btn_h - 42

    btn_confirm = pygame.Rect(bx, by, btn_w, btn_h)
    draw_modal_button(screen, btn_confirm, "Confirm & Record", font, border=(0, 255, 127), fill=(45, 70, 55))

    btn_confirm_adv = None
    if allow_advance:
        bx2 = bx + btn_w + gap
        btn_confirm_adv = pygame.Rect(bx2, by, btn_w, btn_h)
        draw_modal_button(screen, btn_confirm_adv, "Confirm, Advance & Record", font, border=(0, 255, 200), fill=(45, 55, 75))
        bx_cancel = bx2 + btn_w + gap
    else:
        bx_cancel = bx + btn_w + gap

    btn_cancel = pygame.Rect(bx_cancel, by, btn_w, btn_h)
    draw_modal_button(screen, btn_cancel, "Cancel", font, border=(255, 80, 80), fill=(75, 45, 50))

    # Controller hint
    hint = "A = Confirm & Begin Recording, B = Cancel, X = OD/OS, Y = tunnel option"
    if allow_advance:
        hint += "; use the on-screen button for confirm + advance"
    hint_lines = wrap_lines(small, hint, panel_w - 32)
    if hint_lines:
        hh = small.render(hint_lines[-1], True, (200,200,200))
        screen.blit(hh, (x + 16, y + panel_h - hh.get_height() - 6))

    return box, {
        "confirm": btn_confirm,
        "confirm_adv": btn_confirm_adv,
        "cancel": btn_cancel,
    }, {
        "od": od_rect,
        "os": os_rect,
        "tunnel": tunnel_rect,
    }, panel

def draw_material_loaded_warning_modal(screen):
    sw, sh = screen.get_size()
    title_font = pygame.font.SysFont("Consolas", 18, bold=True)
    font = pygame.font.SysFont("Consolas", 14)
    panel_w = min(640, max(520, sw - 80))
    panel_h = 250
    x = (sw - panel_w) // 2
    y = (sh - panel_h) // 2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 195))
    screen.blit(overlay, (0, 0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (34, 30, 28), panel)
    pygame.draw.rect(screen, (255, 190, 80), panel, 2)
    title = title_font.render("Confirm Loaded Material", True, (255, 230, 180))
    screen.blit(title, (x + 16, y + 16))
    lines = [
        "The syringe material-loaded checkbox is not checked.",
        "Proceed only if injection material is actually loaded in the syringe.",
        "Continuing will mark the syringe as loaded and start injection mode.",
    ]
    dy = y + 58
    for msg in lines:
        for ln in wrap_lines(font, msg, panel_w - 32):
            surf = font.render(ln, True, (235, 230, 220))
            screen.blit(surf, (x + 16, dy))
            dy += surf.get_height() + 4
        dy += 2

    btn_w = min(250, (panel_w - 48) // 2)
    btn_h = 44
    gap = 14
    bx = x + (panel_w - (btn_w * 2 + gap)) // 2
    by = y + panel_h - btn_h - 18
    proceed_rect = pygame.Rect(bx, by, btn_w, btn_h)
    cancel_rect = pygame.Rect(proceed_rect.right + gap, by, btn_w, btn_h)
    draw_modal_button(screen, proceed_rect, "A  Proceed, Mark Loaded", title_font, border=(255, 210, 90), fill=(75, 58, 34))
    draw_modal_button(screen, cancel_rect, "B  Go Back", title_font, border=(255, 100, 100), fill=(70, 45, 50))
    return {"proceed": proceed_rect, "cancel": cancel_rect}, panel

def routine_node_kind(line):
    text = str(line or "").strip()
    upper = text.upper()
    if upper.startswith(("POSE ", "GO ")):
        return "pose"
    if upper.startswith("MOVE_TOOL_Z"):
        return "move"
    if upper.startswith("ADVANCE"):
        return "advance"
    if upper.startswith("WASH"):
        return "wash"
    if upper.startswith("PLUNGER"):
        return "plunger"
    if not text or text.startswith("#"):
        return "comment"
    return "custom"

def routine_command_lines(text):
    return [entry["line"] for entry in routine_command_entries(text)]

def routine_command_entries(text):
    entries = []
    for line_index, raw in enumerate(str(text or "").splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append({
            "line_index": line_index,
            "line": line,
            "kind": routine_node_kind(line),
        })
    return entries

def routine_node_display(line):
    text = str(line or "").strip()
    upper = text.upper()
    if upper.startswith("POSE "):
        return text, "saved robot pose"
    if upper.startswith("GO "):
        return text, "direct saved-pose move"
    if upper == "ADVANCE_WAIT":
        return "Wait for user advance", "pause at midway until A is pressed"
    if upper == "ADVANCE_AUTO":
        return "Auto-advance", "continue from midway to final pose"
    if upper == "ADVANCE_CANCEL":
        return "Skip next midway", "bypass the next midway pose"
    if upper.startswith("MOVE_TOOL_Z"):
        parts = text.split()
        dist = parts[1] if len(parts) > 1 else "0"
        return "Tool-axis move", f"{dist} mm (+ advance, - retract)"
    if upper.startswith("WASH"):
        parts = text.split()
        if len(parts) > 1:
            return "WASH", f"{parts[1]} cycle(s)"
        return "WASH", "default 1 cycle"
    if upper.startswith("PLUNGER"):
        return text, "plunger endstop move"
    return text, "custom command"

def draw_routine_builder_modal(screen, routine_name, routine_text, pose_names, scroll=0):
    sw, sh = screen.get_size()
    title_font = pygame.font.SysFont("Consolas", 18, bold=True)
    font = pygame.font.SysFont("Consolas", 14)
    small = pygame.font.SysFont("Consolas", 12)

    panel_w = min(980, max(760, sw - 70))
    panel_h = min(660, max(540, sh - 60))
    x = (sw - panel_w) // 2
    y = (sh - panel_h) // 2

    overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 190))
    screen.blit(overlay, (0, 0))

    panel = pygame.Rect(x, y, panel_w, panel_h)
    pygame.draw.rect(screen, (28, 30, 38), panel)
    pygame.draw.rect(screen, (0, 220, 200), panel, 2)

    title = title_font.render(f"Routine Builder - {routine_name}", True, (235, 245, 245))
    screen.blit(title, (x + 18, y + 14))
    subtitle = small.render("Click a step to edit it. Use Up/Down/Delete per step; Script is for precise manual edits.", True, (190, 210, 210))
    screen.blit(subtitle, (x + 18, y + 38))

    close_rect = pygame.Rect(panel.right - 88, y + 12, 70, 28)
    script_rect = pygame.Rect(close_rect.x - 86, y + 12, 76, 28)
    run_rect = pygame.Rect(script_rect.x - 76, y + 12, 66, 28)
    draw_modal_button(screen, run_rect, "Run", small, border=(0, 255, 127), fill=(45, 65, 52))
    draw_modal_button(screen, script_rect, "Script", small, border=(0, 220, 200), fill=(42, 52, 66))
    draw_modal_button(screen, close_rect, "Close", small, border=(255, 120, 120), fill=(64, 42, 48))

    content = pygame.Rect(x + 18, y + 70, panel_w - 36, panel_h - 92)
    chain_w = int(content.w * 0.62)
    chain_rect = pygame.Rect(content.x, content.y, chain_w, content.h)
    palette_rect = pygame.Rect(chain_rect.right + 14, content.y, content.w - chain_w - 14, content.h)
    pygame.draw.rect(screen, (22, 24, 30), chain_rect)
    pygame.draw.rect(screen, (70, 90, 95), chain_rect, 1)
    pygame.draw.rect(screen, (22, 24, 30), palette_rect)
    pygame.draw.rect(screen, (70, 90, 95), palette_rect, 1)

    clickables = [
        (close_rect, {"kind": "routine_builder_close"}),
        (script_rect, {"kind": "routine_builder_script"}),
        (run_rect, {"kind": "routine_builder_run"}),
    ]

    nodes = routine_command_entries(routine_text)
    kind_cols = {
        "pose": ((35, 60, 45), (0, 255, 127)),
        "move": ((35, 54, 64), (0, 220, 255)),
        "advance": ((62, 56, 34), (255, 210, 90)),
        "wash": ((56, 42, 66), (210, 150, 255)),
        "plunger": ((66, 46, 35), (255, 170, 90)),
        "custom": ((46, 46, 58), (160, 180, 190)),
    }

    prev_clip = screen.get_clip()
    screen.set_clip(chain_rect.inflate(-2, -2))
    chain_y = chain_rect.y + 14 - int(scroll)
    center_x = chain_rect.x + 28
    if not nodes:
        empty_lines = wrap_lines(font, "No nodes yet. Add existing poses, save current robot position as a pose, or add an action from the right.", chain_rect.w - 28)
        ey = chain_y
        for ln in empty_lines:
            surf = font.render(ln, True, (205, 215, 215))
            screen.blit(surf, (chain_rect.x + 14, ey))
            ey += surf.get_height() + 4
        chain_y += 86
    else:
        for idx, node in enumerate(nodes):
            line = node["line"]
            line_index = node["line_index"]
            kind = node.get("kind") or routine_node_kind(line)
            fill, edge = kind_cols.get(kind, kind_cols["custom"])
            node_h = 58
            node_rect = pygame.Rect(chain_rect.x + 52, chain_y, chain_rect.w - 72, node_h)
            body_rect = pygame.Rect(node_rect.x, node_rect.y, max(80, node_rect.w - 118), node_rect.h)
            dot = (center_x, chain_y + node_h // 2)
            if idx > 0:
                pygame.draw.line(screen, (80, 115, 115), (center_x, chain_y - 12), dot, 2)
            if node_rect.bottom >= chain_rect.y and node_rect.y <= chain_rect.bottom:
                pygame.draw.circle(screen, edge, dot, 7)
                pygame.draw.rect(screen, fill, node_rect)
                pygame.draw.rect(screen, edge, node_rect, 2)
                clickables.append((body_rect, {
                    "kind": "routine_builder_edit_node",
                    "line_index": line_index,
                    "line": line,
                }))
                tag = f"{idx + 1}. {kind.upper()}"
                tag_surf = small.render(tag, True, edge)
                screen.blit(tag_surf, (node_rect.x + 10, node_rect.y + 6))
                primary, secondary = routine_node_display(line)
                lines = wrap_lines(font, primary, body_rect.w - 20)
                ly = node_rect.y + 22
                for ln in lines[:1]:
                    surf = font.render(ln, True, (235, 240, 240))
                    screen.blit(surf, (node_rect.x + 10, ly))
                    ly += surf.get_height() + 2
                detail_text = wrap_lines(small, secondary, body_rect.w - 20)[0]
                detail = small.render(detail_text, True, (175, 195, 195))
                screen.blit(detail, (node_rect.x + 10, node_rect.bottom - detail.get_height() - 5))
                btn_w, btn_h, gap = 30, 22, 5
                bx = node_rect.right - (btn_w * 3 + gap * 2 + 8)
                by = node_rect.y + (node_rect.h - btn_h) // 2
                up_rect = pygame.Rect(bx, by, btn_w, btn_h)
                down_rect = pygame.Rect(up_rect.right + gap, by, btn_w, btn_h)
                del_rect = pygame.Rect(down_rect.right + gap, by, btn_w, btn_h)

                def node_button(rect, label, enabled, action, border):
                    fill_col = (42, 44, 54) if enabled else (32, 34, 40)
                    txt_col = (235, 238, 238) if enabled else (125, 135, 135)
                    edge_col = border if enabled else (70, 78, 82)
                    pygame.draw.rect(screen, fill_col, rect)
                    pygame.draw.rect(screen, edge_col, rect, 1)
                    surf = small.render(label, True, txt_col)
                    screen.blit(surf, (rect.centerx - surf.get_width() // 2, rect.centery - surf.get_height() // 2))
                    if enabled:
                        clickables.append((rect, action))

                node_button(up_rect, "Up", idx > 0, {
                    "kind": "routine_builder_move_node",
                    "line_index": line_index,
                    "delta": -1,
                }, (0, 220, 200))
                node_button(down_rect, "Dn", idx < len(nodes) - 1, {
                    "kind": "routine_builder_move_node",
                    "line_index": line_index,
                    "delta": 1,
                }, (0, 220, 200))
                node_button(del_rect, "Del", True, {
                    "kind": "routine_builder_delete_node",
                    "line_index": line_index,
                }, (255, 120, 120))
            chain_y += node_h + 18
    screen.set_clip(prev_clip)

    chain_content_h = chain_y - (chain_rect.y + 14 - int(scroll)) + 30

    pal_y = palette_rect.y + 12
    section = small.render("Actions", True, (170, 220, 210))
    screen.blit(section, (palette_rect.x + 12, pal_y))
    pal_y += section.get_height() + 8

    def palette_button(label, action, border=(0, 220, 200)):
        nonlocal pal_y
        rect = pygame.Rect(palette_rect.x + 12, pal_y, palette_rect.w - 24, 34)
        pygame.draw.rect(screen, (38, 40, 52), rect)
        pygame.draw.rect(screen, border, rect, 1)
        lines = wrap_lines(font, label, rect.w - 14)
        ly = rect.y + max(3, (rect.h - len(lines) * font.get_height()) // 2)
        for ln in lines[:2]:
            surf = font.render(ln, True, (225, 230, 230))
            screen.blit(surf, (rect.x + 8, ly))
            ly += font.get_height()
        clickables.append((rect, action))
        pal_y += rect.h + 7

    palette_button("Save current robot pose + add", {"kind": "routine_builder_save_pose"}, border=(0, 255, 127))
    palette_button("Add wait-for-A step", {"kind": "routine_builder_append", "snippet": "ADVANCE_WAIT", "note": "wait step"})
    palette_button("Add auto-advance step", {"kind": "routine_builder_append", "snippet": "ADVANCE_AUTO", "note": "auto-advance step"})
    palette_button("Add tool-axis move", {"kind": "routine_builder_append", "snippet": "MOVE_TOOL_Z -10.000", "note": "tool-axis move"})
    palette_button("Add WASH", {"kind": "routine_builder_append", "snippet": "WASH", "note": "WASH"})
    palette_button("Add PLUNGER FWD", {"kind": "routine_builder_append", "snippet": "PLUNGER FWD", "note": "PLUNGER FWD"})
    palette_button("Add PLUNGER BWD", {"kind": "routine_builder_append", "snippet": "PLUNGER BWD", "note": "PLUNGER BWD"})
    palette_button("Custom command...", {"kind": "routine_builder_custom"}, border=(160, 180, 220))
    palette_button("Clear routine", {"kind": "routine_builder_clear"}, border=(255, 100, 100))

    pal_y += 6
    section = small.render("Existing Poses", True, (170, 220, 210))
    screen.blit(section, (palette_rect.x + 12, pal_y))
    pal_y += section.get_height() + 8
    pose_area_bottom = palette_rect.bottom - 10
    prev_clip = screen.get_clip()
    screen.set_clip(palette_rect)
    for name in pose_names[:18]:
        if pal_y + 30 > pose_area_bottom:
            more = small.render("More poses available in Settings > Poses", True, (190, 200, 200))
            screen.blit(more, (palette_rect.x + 12, pal_y + 4))
            break
        palette_button(f"POSE {name}", {"kind": "routine_builder_add_pose", "pose": name}, border=(0, 255, 127))
    screen.set_clip(prev_clip)

    max_scroll = max(0, chain_content_h - chain_rect.h + 24)
    if max_scroll > 0:
        track = pygame.Rect(chain_rect.right - 10, chain_rect.y + 8, 4, chain_rect.h - 16)
        pygame.draw.rect(screen, (55, 65, 70), track)
        knob_h = max(24, int(track.h * chain_rect.h / max(chain_rect.h, chain_content_h)))
        knob_y = track.y + int((track.h - knob_h) * (max(0, min(scroll, max_scroll)) / max_scroll))
        pygame.draw.rect(screen, (0, 220, 200), pygame.Rect(track.x, knob_y, track.w, knob_h))

    return clickables, close_rect, panel, max_scroll


# ─────────────────────────────────────────────────────────────────────────────
#                                   main()
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    startup_settings = load_startup_settings()
    args = _resolve_startup_args(args, startup_settings)
    if getattr(args, "launcher_subprocess", False):
        if args.launcher:
            show_startup_launcher(args, result_path=getattr(args, "launcher_result_path", None))
        else:
            result_path = getattr(args, "launcher_result_path", None)
            if result_path:
                try:
                    with open(result_path, "w", encoding="utf-8") as f:
                        json.dump(_launcher_result_payload(args, ok=True), f, indent=2)
                except Exception as e:
                    print(f"[Launcher] Could not write launcher result file: {e}")
                    raise
        return
    if args.launcher:
        payload = run_startup_launcher_subprocess()
        if payload is None:
            return
        args = _apply_launcher_result(args, payload)
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
    os.environ.setdefault("SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR", "1")
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
    pygame.init()
    pygame.font.init()
    pygame.joystick.init()
    audio_cues = init_audio_cues()
    if pygame.joystick.get_count() == 0:
        print("❌ No joystick detected.")
        return
    state = SharedState()
    joy = pygame.joystick.Joystick(0)
    joy.init()
    joystick_thread = JoystickPollThread(joy, state, poll_hz=500.0)
    joystick_thread.start()

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
                    t.join(timeout=0.8)
                    if t.is_alive():
                        print(f"[UVC] {t.name} did not stop promptly; continuing startup without blocking UI.")
            uvc_threads.clear()
            old_uvc1_alive = bool(uvc1 and uvc1.is_alive())
            old_uvc2_alive = bool(uvc2 and uvc2.is_alive())
            if uvc1_path and not old_uvc1_alive:
                uvc1 = UvcThread(
                    uvc1_path, args.uvc_width, args.uvc_height, 15, False,
                    name="UVC#1", try_index1_fallback=args.uvc_try_index1
                )
                uvc1.start()
                uvc_threads.append(uvc1)
            elif old_uvc1_alive:
                uvc_threads.append(uvc1)
            if uvc2_path and not old_uvc2_alive:
                uvc2 = UvcThread(
                    uvc2_path, args.uvc_width, args.uvc_height, 15, False,
                    name="UVC#2", try_index1_fallback=args.uvc_try_index1
                )
                uvc2.start()
                uvc_threads.append(uvc2)
            elif old_uvc2_alive:
                uvc_threads.append(uvc2)

    governors = _read_cpu_governors()
    if governors and any(gov != "performance" for gov in governors):
        print(
            "[LinuxSched] CPU governor "
            f"{'/'.join(governors)}; performance governor is recommended for jog testing."
        )

    def set_camera_thermal_throttle(enabled: bool):
        for cam_thread in [rs_thread] + list(uvc_threads):
            try:
                cam_thread.set_thermal_throttle(enabled)
            except Exception:
                pass
        print(
            "[Thermal] Camera robot-priority mode "
            f"{'enabled' if enabled else 'disabled'}."
        )

    # ─────────────────────────────────────────────────────────────
    # Window / fonts
    # ─────────────────────────────────────────────────────────────
    screen, flags, is_fullscreen = create_display_surface(bool(args.fullscreen))
    pygame.display.set_caption(
        "Robotic Injector — RS+UVC (routine + syringe calibration + logging)"
    )
    clock = pygame.time.Clock()
    frame_blit_cache = FrameBlitCache()

    label_font       = pygame.font.SysFont("Consolas", 14, bold=True)
    value_font       = pygame.font.SysFont("Consolas", 12)
    placeholder_font = pygame.font.SysFont("Consolas", 12)
    big_font         = pygame.font.SysFont("Consolas", 24, bold=True)
    small_font       = pygame.font.SysFont("Consolas", 12)

    # ─────────────────────────────────────────────────────────────
    # Shared state + Robot thread
    # ─────────────────────────────────────────────────────────────
    with state.lock:
        threshold, duration = _clamp_alarm_pid(
            getattr(args, "alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN),
            getattr(args, "alarm_duration", DEFAULT_PID_DURATION_MAIN),
        )
        state.settings["alarm_threshold"] = threshold
        state.settings["alarm_duration"] = duration
        state.alarm_threshold = threshold
        state.alarm_duration = duration
        if getattr(args, "tool_center_demo_radius", None) is not None:
            state.settings["tool_center_demo_radius_mm"] = max(
                0.0,
                min(50.0, float(args.tool_center_demo_radius)),
            )
    save_poses(state.poses)
    save_settings(state.settings)

    rt = RobotThread(
        state,
        host=args.host,
        port=args.port,
        clear_alarm_on_launch=bool(getattr(args, "clear_alarm_startup", True)),
        apply_halt_settings_startup=bool(getattr(args, "apply_halt_settings_startup", True)),
        auto_tune_halt_startup=bool(getattr(args, "auto_tune_halt_startup", False)),
        jog_profile=getattr(args, "jog_profile", DEFAULT_JOG_PROFILE),
        startup_calibration_test=getattr(args, "startup_calibration_test", DEFAULT_STARTUP_CALIBRATION_TEST),
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
    tunnel_offset_input_active = False
    tunnel_retract_input_active = False
    tunnel_return_input_active = False
    high_angle_input_active = False
    yaw_input_active      = False
    yaw_text              = ""
    angle_input_active    = False
    angle_text            = ""
    pitch_input_active    = False
    pitch_text            = ""
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
    settings_modal_active = False
    settings_modal_tab = "tool"
    settings_modal_scroll = 0
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
    routine_builder_active = False
    routine_builder_scroll = 0
    routine_pose_name_input_active = False
    routine_pose_name_text = ""
    routine_custom_cmd_input_active = False
    routine_custom_cmd_text = ""
    routine_step_edit_active = False
    routine_step_edit_line_index = -1
    routine_step_edit_text = ""

    syringe_calib_active  = False
    syringe_field_active  = "vol"

    injection_info_active  = False
    injection_field_active = "study"
    post_step_prompt_active = False
    subject_prompt_active   = False
    material_loaded_warning_active = False
    post_outcome_choice_active = False
    post_outcome_choice_just_opened = False
    injection_reload_active = False
    injection_reload_started = False
    injection_reload_before_ul = 0.0
    injection_reload_stage = "precheck"
    injection_tunnel_active = False
    injection_tunnel_target_mode = "tunnel"
    injection_tunnel_stage = "confirm"
    injection_tunnel_phase_active = False
    injection_exit_active = False
    outcome_detail_scroll = 0

    # text fields
    toollen_text          = f"{state.tool_lz:.1f}"
    approach_text         = f"{state.approach_mm:.1f}"
    toolcx_text           = f"{getattr(state, 'tool_cx', DEFAULT_TOOL_CX):.3f}"
    toolcy_text           = f"{getattr(state, 'tool_cy', DEFAULT_TOOL_CY):.3f}"
    tunnel_offset_text    = f"{state.settings.get('tunnel_tcp_y_mm', DEFAULT_TUNNEL_TCP_Y_MM):.3f}"
    tunnel_retract_text   = f"{state.settings.get('tunnel_retract_mm', DEFAULT_TUNNEL_RETRACT_MM):.1f}"
    tunnel_return_text    = f"{state.settings.get('tunnel_return_mm', DEFAULT_TUNNEL_RETURN_MM):.1f}"
    high_angle_text       = f"{state.settings.get('high_angle_pitch_warning_deg', DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG):.1f}"
    alarm_threshold_val = float(state.settings.get("alarm_threshold", DEFAULT_PID_THRESHOLD_MAIN))
    alarm_duration_val = float(state.settings.get("alarm_duration", DEFAULT_PID_DURATION_MAIN))
    settings_path_text    = SETTINGS_PATH
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
    injection_date_text    = inj_date or current_injection_timestamp_text()

    injection_record_telemetry = True
    injection_record_video     = True
    injection_record_ui        = True
    injection_use_tunnel_workflow = bool(
        state.settings.get("injection_use_tunnel_workflow", False)
    )
    injection_material_loaded = bool(
        state.settings.get("injection_material_loaded", False)
    )

    # Injection step state
    injection_step_in_progress = False
    injection_step_done        = False
    injection_step_distance    = 0.0
    injection_step_target      = 0.0
    injection_step_volume_ul   = 0.0
    injection_last_step_small  = False
    injection_current_rate     = 0.0
    injection_trigger_attenuation = 0.0
    current_step_outcome_marked = False
    last_right_trigger_down    = False

    # Track the upcoming step index once the subject is confirmed
    pending_step_index = None

    # Final small-step warning
    final_small_warning_active   = False
    final_small_step_volume_ul   = 0.0
    final_small_step_target      = 0.0

    # Subject popup text (per-step)
    subject_field_text, subject_eye_choice = split_subject_eye_text(injection_subject_text)
    subject_tunnel_workflow = bool(state.settings.get("injection_use_tunnel_workflow", False))

    # UI scrolling on right panel
    ui_scroll = 0
    ui_content_height = 0

    syringe_calibrator   = None
    current_step_recorder = None
    subject_injection_counts = {}
    subject_injection_volumes = {}
    with state.lock:
        initial_tcp_mode_alert = str(
            state.settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)
            or DEFAULT_TUNNEL_TCP_MODE
        ).strip().lower()
        initial_injection_active_alert = bool(state.settings.get("injection_active", False))
    if initial_tcp_mode_alert not in ("injector", "tunnel"):
        initial_tcp_mode_alert = DEFAULT_TUNNEL_TCP_MODE
    aural_alert_prev = {
        "alarm": False,
        "thermal_hot": False,
        "thermal_critical": False,
        "post_step": False,
        "final_small": False,
        "high_angle": False,
        "midway": False,
        "tcp_mode": initial_tcp_mode_alert,
        "injection_active": initial_injection_active_alert,
    }

    # On-screen “Stop recording” button over RS view
    rec_stop_rect = None
    injection_outcome_rect = None
    show_status_overlay = True

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
                starred = list(state.settings.get("starred_poses", []))
                if name in starred:
                    state.settings["starred_poses"] = [x for x in starred if x != name]
                    save_settings(state.settings)
                # Clear current selection if it was deleted
                if state.current_pose_name in (name, mid):
                    state.current_pose_name = ""
            return removed_any
    quit_confirm_active = False

    # Short-lived UI notices
    ui_notice_text = ""
    ui_notice_until = 0.0
    thermal_status_text = ""
    thermal_hot = False
    thermal_throttle_active = False
    thermal_last_poll = 0.0
    thermal_critical = False
    thermal_warning_dismissed = False
    thermal_warning_dismiss_rect = None
    aural_alert_last = {}
    voice_demo_state = {"thread": None}
    muted_warning_keys = set()
    warning_clickables = []

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

    def warning_key_for_cue(name: str):
        return {
            "gpws_controller_alarm": "alarm",
            "gpws_thermal_warning": "thermal",
            "gpws_thermal_critical": "thermal",
            "gpws_high_angle": "angle",
            "gpws_final_dose": "dose",
            "gpws_record_outcome": "outcome",
            "gpws_retract_before_workflow": "clearance",
            "gpws_midway_hold": "midway",
        }.get(str(name or ""))

    def _tts_command(speech_cmd: str, phrase: str, voice_type: str, tts_rate: int):
        voice_type = _normalize_tts_voice_type(voice_type)
        tts_rate = max(-100, min(100, int(tts_rate)))
        if speech_cmd == "spd-say":
            cmd = [speech_cmd, "-r", str(tts_rate)]
            if voice_type != "default":
                cmd += ["-t", voice_type]
            cmd.append(str(phrase))
            return cmd
        if speech_cmd in ("espeak-ng", "espeak"):
            cmd = [speech_cmd]
            if voice_type != "default":
                espeak_voice = {
                    "female1": "en+f1",
                    "female2": "en+f2",
                    "female3": "en+f3",
                    "male1": "en+m1",
                    "male2": "en+m2",
                    "male3": "en+m3",
                    "child_female": "en+f4",
                    "child_male": "en+m4",
                }.get(voice_type)
                if espeak_voice:
                    cmd += ["-v", espeak_voice]
            cmd.append(str(phrase))
            return cmd
        return [speech_cmd, str(phrase)]

    def _play_audio_tone(cue, wait=False):
        wav_path = cue.get("wav") if isinstance(cue, dict) else None
        if wav_path and shutil.which("aplay"):
            try:
                proc = subprocess.Popen(
                    ["aplay", "-q", wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if wait:
                    proc.wait(timeout=1.0)
                return True
            except Exception:
                pass
        snd = cue.get("sound") if isinstance(cue, dict) else cue
        if snd is not None:
            try:
                snd.play()
                if wait:
                    time.sleep(0.20)
                return True
            except Exception:
                pass
        return False

    def play_audio_cue(name: str):
        cue = audio_cues.get(str(name or ""))
        if cue is None:
            return
        phrase = cue.get("phrase") if isinstance(cue, dict) else None
        speech_cmd = cue.get("speech_cmd") if isinstance(cue, dict) else None
        if phrase and speech_cmd:
            try:
                with state.lock:
                    voice_type = _normalize_tts_voice_type(
                        state.settings.get("tts_voice_type", DEFAULT_TTS_VOICE_TYPE)
                    )
                    tts_rate = int(state.settings.get("tts_rate", DEFAULT_TTS_RATE))
                tts_rate = max(-100, min(100, tts_rate))
                cmd = _tts_command(speech_cmd, str(phrase), voice_type, tts_rate)
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        _play_audio_tone(cue)

    def play_all_warning_demos():
        thread = voice_demo_state.get("thread")
        if thread is not None and thread.is_alive():
            with state.lock:
                state.calibration_status = "Warning demo is already running."
            return

        def worker():
            demo_cues = [
                ("gpws_controller_alarm", "Robot alarm"),
                ("gpws_thermal_warning", "Thermal warning"),
                ("gpws_thermal_critical", "Thermal critical"),
                ("gpws_high_angle", "High angle"),
                ("gpws_final_dose", "Final dose"),
                ("gpws_record_outcome", "Record outcome"),
                ("gpws_retract_before_workflow", "Clearance workflow"),
                ("gpws_midway_hold", "Midway hold"),
                ("tunnel_mode_start", "Tunnel mode start"),
                ("tunnel_complete", "Tunnel complete"),
                ("gpws_tunnel_tcp", "Tunnel TCP"),
                ("gpws_injector_tcp", "Injector TCP"),
                ("gpws_injection_armed", "Injection armed"),
            ]
            with state.lock:
                voice_type = _normalize_tts_voice_type(
                    state.settings.get("tts_voice_type", DEFAULT_TTS_VOICE_TYPE)
                )
                try:
                    tts_rate = int(state.settings.get("tts_rate", DEFAULT_TTS_RATE))
                except Exception:
                    tts_rate = DEFAULT_TTS_RATE
            any_speech = any(
                isinstance(audio_cues.get(cue_name), dict)
                and audio_cues.get(cue_name, {}).get("speech_cmd")
                for cue_name, _ in demo_cues
            )
            if not any_speech:
                with state.lock:
                    state.calibration_status = "No speech backend found; playing warning tones only."
            for cue_name, label in demo_cues:
                cue = audio_cues.get(cue_name, {})
                if not cue:
                    continue
                with state.lock:
                    state.calibration_status = f"Warning demo: {label}"
                _play_audio_tone(cue, wait=True)
                speech_cmd = cue.get("speech_cmd") if isinstance(cue, dict) else None
                phrase = cue.get("phrase") if isinstance(cue, dict) else None
                if phrase and speech_cmd:
                    try:
                        proc = subprocess.Popen(
                            _tts_command(speech_cmd, str(phrase), voice_type, tts_rate),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        proc.wait(timeout=6.0)
                    except subprocess.TimeoutExpired:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    except Exception:
                        pass
                time.sleep(0.25)
            with state.lock:
                state.calibration_status = "Warning demo complete."

        voice_demo_state["thread"] = threading.Thread(
            target=worker,
            name="WarningCueDemo",
            daemon=True,
        )
        voice_demo_state["thread"].start()

    def queue_aural_alert(name: str, min_interval_s: float = 10.0, force: bool = False):
        # Advisory GPWS/TCAS-style callouts. They never command motion; they only
        # describe state transitions that the operator must notice immediately.
        cue_name = str(name or "").strip()
        if not cue_name:
            return
        warning_key = warning_key_for_cue(cue_name)
        if warning_key and warning_key in muted_warning_keys:
            return
        now_alert = time.time()
        if not force:
            last = float(aural_alert_last.get(cue_name, 0.0) or 0.0)
            if (now_alert - last) < float(min_interval_s):
                return
        with state.lock:
            enabled = bool(
                state.settings.get(
                    "aircraft_callouts_enabled",
                    DEFAULT_AIRCRAFT_CALLOUTS_ENABLED,
                )
            )
            if not enabled:
                return
            pending_name = str(state.audio_cue or "")
            priority = {
                "gpws_controller_alarm": 100,
                "gpws_thermal_critical": 90,
                "gpws_thermal_warning": 70,
                "gpws_high_angle": 65,
                "gpws_final_dose": 60,
                "gpws_record_outcome": 55,
                "gpws_retract_before_workflow": 50,
                "gpws_tunnel_tcp": 45,
                "gpws_injector_tcp": 45,
                "tunnel_mode_start": 42,
                "gpws_midway_hold": 40,
                "gpws_injection_armed": 35,
                "gpws_test": 10,
            }
            if pending_name and force and priority.get(pending_name, 0) > priority.get(cue_name, 0):
                return
            if force or state.audio_cue is None:
                state.audio_cue = cue_name
                aural_alert_last[cue_name] = now_alert

    def build_warning_items(
        alarm_is_active,
        thermal_is_hot,
        thermal_is_critical,
        high_angle_is_active,
        dose_warning_active,
        outcome_warning_active,
        midway_warning_active,
        clearance_warning_active,
    ):
        items = []
        if alarm_is_active:
            items.append({
                "key": "alarm",
                "label": "ALARM",
                "color": (255, 70, 70),
                "action": "X RESET",
            })
        if thermal_is_hot:
            items.append({
                "key": "thermal",
                "label": "THERM",
                "color": (255, 70, 70) if thermal_is_critical else (255, 170, 80),
                "action": "B HIDE",
            })
        if high_angle_is_active:
            items.append({
                "key": "angle",
                "label": "ANGLE",
                "color": (255, 210, 90),
                "action": "X MUTE",
            })
        if dose_warning_active:
            items.append({
                "key": "dose",
                "label": "DOSE",
                "color": (255, 210, 90),
                "action": "X MUTE",
            })
        if outcome_warning_active:
            items.append({
                "key": "outcome",
                "label": "OUTCOME",
                "color": (255, 170, 80),
                "action": "X MUTE",
            })
        if midway_warning_active:
            items.append({
                "key": "midway",
                "label": "MIDWAY",
                "color": (120, 220, 255),
                "action": "X MUTE",
            })
        if clearance_warning_active:
            items.append({
                "key": "clearance",
                "label": "CLEAR",
                "color": (255, 210, 90),
                "action": "X MUTE",
            })
        return items

    def toggle_warning_mute_by_key(key, label=None):
        key = str(key or "").strip()
        if not key:
            return False
        label = str(label or key).strip()
        if key in muted_warning_keys:
            muted_warning_keys.remove(key)
            status = f"Audio restored for {label} warning."
        else:
            muted_warning_keys.add(key)
            status = f"Audio muted for {label} warning; visual warning remains active."
        with state.lock:
            state.calibration_status = status
        return True

    def adjust_numeric_text(text: str, delta: float, min_value: float = 0.001):
        try:
            current = float(str(text or "0").strip())
        except Exception:
            current = 0.0
        value = max(float(min_value), current + float(delta))
        return f"{value:.3f}"

    def make_buttons(current_tool_lz, current_approach, poses, settings, routines, current_routine_name, calibration_active, alarm_threshold, alarm_duration):
        syringe_volume_ul = settings.get("syringe_volume_ul", 10.0)
        syringe_step_ul = settings.get("syringe_step_ul", 1.0)
        syringe_remaining_ul = settings.get("syringe_remaining_ul", syringe_volume_ul)
        ammo_steps_exact = (
            max(0.0, float(syringe_remaining_ul) / float(syringe_step_ul))
            if syringe_step_ul > 0
            else 0.0
        )
        reserved_names = set(normalize_reserved_list(settings.get("reserved_poses", [])))
        alarm_threshold = float(alarm_threshold)
        alarm_duration = float(alarm_duration)
        tunnel_mode = str(settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)).strip().lower()
        if tunnel_mode not in ("injector", "tunnel"):
            tunnel_mode = DEFAULT_TUNNEL_TCP_MODE
        tunnel_label = "Tunnel needle" if tunnel_mode == "tunnel" else "Injector"
        tunnel_offset = float(settings.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM))
        tunnel_retract = float(settings.get("tunnel_retract_mm", DEFAULT_TUNNEL_RETRACT_MM))
        tunnel_return = float(settings.get("tunnel_return_mm", DEFAULT_TUNNEL_RETURN_MM))

        items = [
            {"label": "Settings / Quit", "kind": "top_actions"},
            {"label": "Workflow", "kind": "header"},
            {"label": "Stop Routine (halt motion only)", "kind": "routine_stop"},
            {"label": "Saved Routines", "kind": "header"},
            {"label": "Manage / create routines in Settings", "kind": "settings_routines"},
            # NOTE: swap_endstops still supported in settings.json, but UI toggle removed
        ]

        for rname in sorted(routines.keys(), key=str.lower):
            items.append({
                "label": f"Run: {rname}",
                "kind": "routine_row",
                "routine": rname,
                "current": (rname == current_routine_name),
            })

        items += [{"label": "Injection", "kind": "header"}]
        if settings.get("injection_active", False):
            use_tunnel_session = bool(settings.get("injection_use_tunnel_workflow", False))
            items += [
                {"label": f"Session active: remaining ≈ {ammo_steps_exact:.2f} steps", "kind": "header"},
                {"label": f"TCP mode: {tunnel_label} (Y offset {tunnel_offset:.3f} mm)", "kind": "header"},
            ]
            if use_tunnel_session:
                current_tunnel_phase = bool(injection_tunnel_phase_active)
                if current_tunnel_phase and tunnel_mode == "tunnel":
                    items.append({
                        "label": "Tunnel Complete -> Injector (safe TCP switch)",
                        "kind": "injection_tunnel_complete",
                    })
                elif current_tunnel_phase:
                    items.append({
                        "label": "Tunnel phase: safe switch to tunnel needle is queued/running",
                        "kind": "header",
                    })
                items += [
                    {"label": "Prepare tunnel needle (safe retract + TCP shift)", "kind": "injection_tunnel_prepare"},
                    {"label": "Prepare injector (requires calibrated + loaded syringe)", "kind": "injection_tunnel_injector"},
                ]
            else:
                items.append(
                    {"label": "Tunnel workflow disabled for this session", "kind": "header"}
                )
            items += [
                {"label": "Reload / refill syringe (keep animal)", "kind": "injection_reload"},
                {"label": "Stop current injection / record outcome", "kind": "injection_record_outcome"},
                {"label": "Next animal / sample", "kind": "injection_new_subject"},
                {"label": "Exit Injection Mode / Syringe Options", "kind": "stop_injection"},
            ]
        else:
            items.append(
                {"label": f"Start Injection (remaining ≈ {ammo_steps_exact:.2f} steps)", "kind": "start_injection"}
            )

        items += [
            {"label": "Starred Poses", "kind": "header"},
        ]

        starred_names = set(normalize_reserved_list(settings.get("starred_poses", ["Default", "Reload"])))
        starred_pose_list = [
            name for name in sorted_pose_names(poses)
            if name in starred_names
        ]
        if not starred_pose_list:
            items.append({"label": "No starred poses. Star poses in Settings > Poses.", "kind": "settings_poses"})
        else:
            items.append({"label": "Manage / save poses in Settings", "kind": "settings_poses"})

        for name in starred_pose_list:
            items.append({
                "label": f"Go: {name}",
                "kind": "pose_row",
                "pose": name,
                "main_only": True,
                "reserved": (
                    (name in RESERVED_POSES)
                    or (name in reserved_names)
                ),
            })

        return items

    def make_settings_items(tab, current_tool_lz, current_approach, poses, settings, routines, current_routine_name, calibration_active, alarm_threshold, alarm_duration):
        syringe_volume_ul = float(settings.get("syringe_volume_ul", 10.0) or 0.0)
        syringe_step_ul = float(settings.get("syringe_step_ul", 1.0) or 0.0)
        syringe_remaining_ul = float(settings.get("syringe_remaining_ul", syringe_volume_ul) or 0.0)
        reserved_names = set(normalize_reserved_list(settings.get("reserved_poses", [])))
        starred_names = set(normalize_reserved_list(settings.get("starred_poses", ["Default", "Reload"])))
        tunnel_mode = str(settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)).strip().lower()
        if tunnel_mode not in ("injector", "tunnel"):
            tunnel_mode = DEFAULT_TUNNEL_TCP_MODE
        tunnel_label = "Tunnel needle" if tunnel_mode == "tunnel" else "Injector"
        tunnel_offset = float(settings.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM) or 0.0)
        tunnel_retract = float(settings.get("tunnel_retract_mm", DEFAULT_TUNNEL_RETRACT_MM) or 0.0)
        tunnel_return = float(settings.get("tunnel_return_mm", DEFAULT_TUNNEL_RETURN_MM) or 0.0)

        if tab == "tool":
            items = [
                {"label": "— Tool Geometry —", "kind": "header"},
                {"label": f"Tool length (mm): {current_tool_lz:.1f}  [Edit]", "kind": "edit_tool"},
                {"label": f"Tool center X (mm): {settings.get('tool_cx', DEFAULT_TOOL_CX):.3f}  [Edit]", "kind": "edit_toolcx"},
                {"label": f"Tool center Y (mm): {settings.get('tool_cy', DEFAULT_TOOL_CY):.3f}  [Edit]", "kind": "edit_toolcy"},
                {"label": f"Approach distance (mm): {current_approach:.1f}  [Edit]", "kind": "edit_approach"},
                {"label": "Save current tool setup as preset", "kind": "tool_preset_save"},
                {"label": "— Tunnel Needle TCP —", "kind": "header"},
                {"label": f"Active TCP: {tunnel_label}", "kind": "header"},
                {"label": f"Side needle Y offset (mm): {tunnel_offset:.3f}  [Edit]", "kind": "edit_tunnel_offset"},
                {"label": f"Safety retract (mm): {tunnel_retract:.1f}  [Edit]", "kind": "edit_tunnel_retract"},
                {"label": f"Guided return (mm): {tunnel_return:.1f}  [Edit]", "kind": "edit_tunnel_return"},
                {"label": "Switch to tunnel needle TCP", "kind": "switch_tunnel_tcp"},
                {"label": "Switch back to injector TCP", "kind": "switch_injector_tcp"},
            ]
            items.append(
                {"label": "Stop current tool demo/test", "kind": "startup_calibration_stop"}
                if calibration_active else
                {"label": "Fixed-tip cone demo (~50 mm dia)", "kind": "fixed_tip_cone_demo"}
            )
            tool_presets = settings.get("tool_presets", []) or []
            if tool_presets:
                items.append({"label": "— Tool Presets —", "kind": "header"})
                for p in tool_presets:
                    nm = str(p.get("name", "")).strip()
                    if not nm:
                        continue
                    try:
                        lz_p = float(p.get("lz", current_tool_lz))
                        cx_p = float(p.get("cx", settings.get("tool_cx", DEFAULT_TOOL_CX)))
                        cy_p = float(p.get("cy", settings.get("tool_cy", DEFAULT_TOOL_CY)))
                    except Exception:
                        lz_p, cx_p, cy_p = current_tool_lz, DEFAULT_TOOL_CX, DEFAULT_TOOL_CY
                    items.append({
                        "label": f"Apply: {nm}  (LZ={lz_p:.2f}, CX={cx_p:.3f}, CY={cy_p:.3f})",
                        "kind": "tool_preset_manage_row",
                        "preset": nm,
                    })
            return items

        if tab == "poses":
            items = [
                {"label": "— Pose Library —", "kind": "header"},
                {"label": "Save current robot position as a new pose", "kind": "new_pose"},
                {"label": "Star a pose to keep it visible in the main panel.", "kind": "header"},
            ]
            names = sorted_pose_names(poses)
            if not names:
                items.append({"label": "No saved poses found.", "kind": "header"})
            for name in names:
                items.append({
                    "label": name,
                    "kind": "pose_manage_row",
                    "pose": name,
                    "reserved": (name in RESERVED_POSES) or (name in reserved_names),
                    "starred": name in starred_names,
                })
            return items

        if tab == "injection":
            steps_left = syringe_remaining_ul / syringe_step_ul if syringe_step_ul > 0 else 0.0
            return [
                {"label": "— Syringe —", "kind": "header"},
                {"label": (
                    f"Syringe calibration: stroke={settings.get('syringe_full_travel_time_s', 0.0):.2f}s, "
                    f"volume={syringe_volume_ul:.3f} uL, step={syringe_step_ul:.3f} uL"
                ), "kind": "syringe_calib"},
                {"label": f"Remaining syringe volume: {syringe_remaining_ul:.3f} uL ({steps_left:.2f} configured step(s))", "kind": "header"},
                {"label": f"Retract distance after step (mm): {settings.get('injection_retract_mm', 10.0):.1f}  [Edit]", "kind": "edit_retract"},
                {"label": "Start injection metadata and logging are configured from the Start Injection workflow.", "kind": "header"},
            ]

        if tab == "routine":
            items = [
                {"label": "— Plunger / Routine Parameters —", "kind": "header"},
                {"label": f"Wash FWD speed (steps/s): {settings.get('plunger_fwd_rate', MAX_RATE)}  [Edit]", "kind": "edit_fwd_rate"},
                {"label": f"Wash BWD speed (steps/s): {settings.get('plunger_bwd_rate', MAX_RATE)}  [Edit]", "kind": "edit_bwd_rate"},
                {"label": f"Failsafe timeout (ms) [0 = infinite]: {settings.get('endstop_timeout_ms', 0)}  [Edit]", "kind": "edit_endstop_timeout"},
                {"label": f"Final pose dwell (s) [0 = none]: {settings.get('pose_final_wait_s', 0.0):.1f}  [Edit]", "kind": "edit_pose_wait"},
                {"label": "— Routine Builder —", "kind": "header"},
                {"label": f"Current routine: {current_routine_name}", "kind": "header"},
                {"label": "Create empty routine", "kind": "routine_new"},
                {"label": "Create routine from starred poses", "kind": "routine_new_from_starred"},
                {"label": "Create routine from all saved poses", "kind": "routine_new_from_all"},
                {"label": "Open visual routine builder", "kind": "routine_builder_current"},
                {"label": "Edit current routine script", "kind": "routine_edit_current"},
                {"label": "Run current routine", "kind": "routine_run_current"},
                {"label": "Use the visual builder to add, edit, reorder, or delete routine steps.", "kind": "header"},
            ]
            if routines:
                items.append({"label": "— Saved Routines —", "kind": "header"})
                for rname in sorted(routines.keys(), key=str.lower):
                    items.append({
                        "label": rname,
                        "kind": "routine_manage_row",
                        "routine": rname,
                        "current": (rname == current_routine_name),
                    })
            return items

        if tab == "safety":
            live_eps = _clamp_live_linear_epsilon_mm(
                settings.get("live_linear_epsilon_mm", DEFAULT_LIVE_LINEAR_EPSILON_MM)
            )
            items = [
                {"label": "— Halt / Alarm —", "kind": "header"},
                {"label": "Halt settings apply automatically on startup.", "kind": "alarm_status"},
                {"label": "Apply Halt Settings Now", "kind": "alarm_arm_now"},
                {"label": "Auto-Tune Halt at Default", "kind": "alarm_auto_tune"},
                {"label": "Clear Latched Alarm", "kind": "alarm_disarm_now"},
                {"label": f"Halt threshold: {int(round(alarm_threshold))}", "kind": "alarm_threshold"},
                {"label": f"Halt duration: {int(round(alarm_duration))}", "kind": "alarm_duration"},
                {"label": "— Injection Angle Warning —", "kind": "header"},
                {
                    "label": (
                        "High-angle warning pitch (deg): "
                        f"{float(settings.get('high_angle_pitch_warning_deg', DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG)):.1f}  [Edit]"
                    ),
                    "kind": "edit_high_angle_warning",
                },
                {"label": "— Manual Jog Precision —", "kind": "header"},
                {
                    "label": (
                        "Fine-speed reference adds sub-1% speed levels. The "
                        "Dorna live command floor remains fixed at the stable "
                        f"{STABLE_LIVE_LINEAR_EPSILON_MM * 1000.0:.0f} um threshold."
                    ),
                    "kind": "header",
                },
            ]
            for eps in LIVE_LINEAR_EPSILON_PRESETS_MM:
                is_current = abs(live_eps - eps) < 1e-9
                items.append({
                    "label": (
                        f"{'Current' if is_current else 'Set'} fine speed reference: "
                        f"{_live_linear_epsilon_label(eps)}"
                    ),
                    "kind": "jog_epsilon_preset",
                    "value": eps,
                    "current": is_current,
                })
            items.append({
                "label": "Lower references no longer lower the robot command threshold; this avoids the back-and-forth motion seen with sub-10 um live targets.",
                "kind": "header",
            })
            return items

        if tab == "audio":
            backend = _speech_backend_name() or "none"
            voice_type = _normalize_tts_voice_type(settings.get("tts_voice_type", DEFAULT_TTS_VOICE_TYPE))
            try:
                tts_rate = int(settings.get("tts_rate", DEFAULT_TTS_RATE))
            except Exception:
                tts_rate = DEFAULT_TTS_RATE
            callouts_enabled = bool(
                settings.get(
                    "aircraft_callouts_enabled",
                    DEFAULT_AIRCRAFT_CALLOUTS_ENABLED,
                )
            )
            return [
                {"label": "— Voice Cues —", "kind": "header"},
                {"label": f"Speech backend: {backend}", "kind": "header"},
                {"label": f"Voice type: {_tts_voice_label(voice_type)}  [Next]", "kind": "tts_voice_cycle"},
                {"label": f"Speech rate: {tts_rate:+d}  [Next]", "kind": "tts_rate_cycle"},
                {"label": "Play selected voice cue", "kind": "tts_test"},
                {"label": "Play all warning demos", "kind": "tts_all_test"},
                {"label": "— Safety Callouts —", "kind": "header"},
                {
                    "label": (
                        "Aircraft-style safety callouts: "
                        f"{'ON' if callouts_enabled else 'OFF'}  [Toggle]"
                    ),
                    "kind": "aircraft_callouts_toggle",
                },
                {"label": "Play safety callout test", "kind": "gpws_test"},
                {"label": "Callouts are advisory only; robot motion and halt behavior are unchanged.", "kind": "header"},
                {"label": "Voice type applies to spd-say / Speech Dispatcher. Tone cues still play if speech is unavailable.", "kind": "header"},
            ]

        return [
            {"label": "— Settings File —", "kind": "header"},
            {"label": "Load settings.json", "kind": "settings_load"},
            {"label": "Save settings.json", "kind": "settings_save"},
            {"label": "Application exit remains available as a dedicated Quit button in the main panel.", "kind": "header"},
        ]

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
                state.audio_cue = "syringe_calibration_failed"
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

    def routine_template_empty():
        return ROUTINE_BLANK_TEMPLATE.strip()

    def routine_template_from_poses(pose_names):
        names = [str(n).strip() for n in pose_names if str(n).strip()]
        lines = [
            "# Routine generated from saved poses",
            "# Add ADVANCE_WAIT or ADVANCE_AUTO explicitly where a pose should continue past __midway.",
        ]
        for name in names:
            lines.append(f"POSE {name}")
        if not names:
            lines.append("# No poses were available when this routine was created.")
        return "\n".join(lines).strip()

    def starred_pose_names_snapshot():
        with state.lock:
            poses_now = dict(state.poses)
            starred = set(normalize_reserved_list(
                state.settings.get("starred_poses", ["Default", "Reload"])
            ))
        return [name for name in sorted_pose_names(poses_now) if name in starred]

    def all_pose_names_snapshot():
        with state.lock:
            poses_now = dict(state.poses)
        return sorted_pose_names(poses_now)

    def set_current_routine(name: str):
        nonlocal current_routine_name, routine_editor
        if not name or name not in routines:
            return
        persist_current_routine()
        current_routine_name = name
        txt = routines.get(name, ROUTINE_SAMPLE.strip())
        routine_editor = EditorState(text=txt, caret=len(txt))
        with state.lock:
            state.routine_status_text = f"Selected routine '{name}'"

    def append_to_current_routine(snippet: str, note: str):
        nonlocal routine_editor
        snippet = str(snippet or "").strip()
        if not snippet:
            return
        text = routine_editor.text.rstrip()
        if text:
            text += "\n"
        text += snippet + "\n"
        routine_editor = EditorState(text=text, caret=len(text))
        persist_current_routine()
        with state.lock:
            state.routine_status_text = f"Appended {note} to '{current_routine_name}'"

    def _commit_routine_lines(lines, status_text):
        nonlocal routine_editor
        text = "\n".join(lines).rstrip()
        if text:
            text += "\n"
        routine_editor = EditorState(text=text, caret=len(text))
        persist_current_routine()
        with state.lock:
            state.routine_status_text = status_text

    def set_routine_line(line_index: int, new_line: str):
        new_line = str(new_line or "").strip()
        if not new_line:
            with state.lock:
                state.routine_status_text = "Routine step edit cancelled: empty command."
            return
        lines = routine_editor.text.splitlines()
        if 0 <= int(line_index) < len(lines):
            old = lines[int(line_index)].strip()
            lines[int(line_index)] = new_line
            _commit_routine_lines(lines, f"Updated routine step '{old}' -> '{new_line}'")
        else:
            with state.lock:
                state.routine_status_text = "Could not update routine step; index is stale."

    def delete_routine_line(line_index: int):
        lines = routine_editor.text.splitlines()
        if 0 <= int(line_index) < len(lines):
            removed = lines.pop(int(line_index)).strip()
            _commit_routine_lines(lines, f"Deleted routine step '{removed}'")
        else:
            with state.lock:
                state.routine_status_text = "Could not delete routine step; index is stale."

    def move_routine_line(line_index: int, delta: int):
        lines = routine_editor.text.splitlines()
        command_indices = [
            idx for idx, raw in enumerate(lines)
            if raw.strip() and not raw.strip().startswith("#")
        ]
        try:
            pos = command_indices.index(int(line_index))
            target_pos = pos + int(delta)
        except ValueError:
            with state.lock:
                state.routine_status_text = "Could not move routine step; index is stale."
            return
        if target_pos < 0 or target_pos >= len(command_indices):
            return
        src_idx = command_indices[pos]
        dst_idx = command_indices[target_pos]
        lines[src_idx], lines[dst_idx] = lines[dst_idx], lines[src_idx]
        direction = "up" if delta < 0 else "down"
        _commit_routine_lines(lines, f"Moved routine step {direction}.")

    def clear_current_routine_to_blank():
        nonlocal routine_editor
        text = routine_template_empty()
        routine_editor = EditorState(text=text, caret=len(text))
        persist_current_routine()
        with state.lock:
            state.routine_status_text = f"Cleared routine '{current_routine_name}'"

    def apply_settings_dict(new_s: dict):
        """Apply loaded settings into state and refresh key cached fields."""
        with state.lock:
            state.settings = new_s.copy()
            state.tool_lz = float(new_s.get("tool_lz", DEFAULT_TOOL_LZ))
            state.tool_cx = float(new_s.get("tool_cx", DEFAULT_TOOL_CX))
            state.tool_cy = float(new_s.get("tool_cy", DEFAULT_TOOL_CY))
            state.approach_mm = float(new_s.get("approach_mm", DEFAULT_APPROACH_MM))
            state.tunnel_tcp_y_mm = float(new_s.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM))
            state.tunnel_retract_mm = float(new_s.get("tunnel_retract_mm", DEFAULT_TUNNEL_RETRACT_MM))
            state.tunnel_return_mm = float(new_s.get("tunnel_return_mm", DEFAULT_TUNNEL_RETURN_MM))
            state.tunnel_tcp_mode = str(new_s.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE))
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
            state.audio_cue = "syringe_calibration_complete"

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
        nonlocal current_step_recorder, current_step_outcome_marked
        current_step_outcome_marked = False
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
            try:
                log_injection_session_event(
                    "recording_started",
                    "Per-attempt recording started after animal/eye/sample confirmation.",
                    {
                        "recording_dir": current_step_recorder.step_dir,
                        "record_video": bool(rec_vid),
                        "record_ui": bool(rec_ui),
                        "record_telemetry": bool(rec_tel),
                    },
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[Recorder] Failed to create recorder: {e}")
            current_step_recorder = None

    def current_step_fraction():
        if injection_step_target <= 0.0:
            return 0.0
        return max(0.0, min(1.0, injection_step_distance / injection_step_target))

    def current_step_delivered_ul():
        return max(0.0, injection_step_volume_ul * current_step_fraction())

    def current_subject_key():
        with state.lock:
            s = state.settings
            subject = str(s.get("injection_subject", "") or "").strip()
            study = str(s.get("injection_study", "") or "").strip()
            material = str(s.get("injection_material", "") or "").strip()
            date_s = str(s.get("injection_date", "") or "").strip()
        return "|".join((study, subject, material, date_s)), subject

    def current_subject_volume_summary(include_current=False):
        subject_key, subject_name = current_subject_key()
        volumes = list(subject_injection_volumes.get(subject_key, []))
        total = float(sum(volumes))
        count = int(subject_injection_counts.get(subject_key, len(volumes)))
        current = float(current_step_delivered_ul() or 0.0)
        if include_current:
            total += current
            count += 1
        return subject_name, count, total, volumes, current

    def append_injection_event(payload):
        try:
            os.makedirs(DATA_ROOT_DIR, exist_ok=True)
            with open(os.path.join(DATA_ROOT_DIR, "injection_events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            return True
        except Exception as e:
            print(f"[Recorder] Failed to write injection event log: {e}")
            return False

    def log_injection_session_event(event_name, note="", extra=None):
        subject_key, subject_name = current_subject_key()
        _, subject_count, subject_total, volumes, _ = current_subject_volume_summary(include_current=False)
        with state.lock:
            s = state.settings
            remaining_ul = float(s.get("syringe_remaining_ul", 0.0) or 0.0)
            volume_ul = float(s.get("syringe_volume_ul", 0.0) or 0.0)
            tcp_mode = str(s.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE) or DEFAULT_TUNNEL_TCP_MODE)
            tunnel_tcp_y = float(s.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM) or 0.0)
            use_tunnel = bool(s.get("injection_use_tunnel_workflow", False))
            material_loaded = bool(s.get("injection_material_loaded", False))
            tool_cx_now = float(getattr(state, "tool_cx", s.get("tool_cx", DEFAULT_TOOL_CX)))
            tool_cy_now = float(getattr(state, "tool_cy", s.get("tool_cy", DEFAULT_TOOL_CY)))
            tool_lz_now = float(getattr(state, "tool_lz", s.get("tool_lz", DEFAULT_TOOL_LZ)))
        payload = {
            "timestamp": time.time(),
            "event": str(event_name or ""),
            "subject": subject_name,
            "subject_key": subject_key,
            "step_index": int(getattr(state, "current_injection_step", 0) or 0),
            "subject_total_delivered_ul": float(subject_total),
            "subject_injection_count": int(subject_count),
            "subject_injection_volumes_ul": list(volumes),
            "syringe_remaining_ul": remaining_ul,
            "syringe_volume_ul": volume_ul,
            "tcp_mode": tcp_mode,
            "tunnel_tcp_y_mm": tunnel_tcp_y,
            "injection_use_tunnel_workflow": use_tunnel,
            "injection_material_loaded": material_loaded,
            "tool_cx": tool_cx_now,
            "tool_cy": tool_cy_now,
            "tool_lz": tool_lz_now,
            "note": str(note or ""),
        }
        if isinstance(extra, dict):
            payload.update(extra)
        append_injection_event(payload)
        return payload

    def stop_injection_plunger():
        nonlocal last_rate, injection_current_rate
        if ser:
            try:
                ser.write(b"V0\n")
                if ser.in_waiting:
                    ser.read(ser.in_waiting)
            except Exception:
                pass
        last_rate = 0
        injection_current_rate = 0.0

    def mark_current_step_outcome(outcome, note="", stop_recorder=True):
        nonlocal current_step_recorder, current_step_outcome_marked
        delivered = current_step_delivered_ul()
        if current_step_outcome_marked:
            if stop_recorder and current_step_recorder:
                try:
                    current_step_recorder.stop()
                except Exception:
                    pass
                current_step_recorder = None
            return delivered
        subject_key, subject_name = current_subject_key()
        volumes = subject_injection_volumes.setdefault(subject_key, [])
        volumes.append(round(float(delivered or 0.0), 6))
        subject_injection_counts[subject_key] = subject_injection_counts.get(subject_key, 0) + 1
        subject_total = float(sum(volumes))
        subject_count = int(subject_injection_counts[subject_key])
        with state.lock:
            s = state.settings
            tcp_mode = str(s.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE) or DEFAULT_TUNNEL_TCP_MODE)
            tunnel_tcp_y = float(s.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM) or 0.0)
            use_tunnel = bool(s.get("injection_use_tunnel_workflow", False))
            material_loaded = bool(s.get("injection_material_loaded", False))
            tool_cx_now = float(getattr(state, "tool_cx", s.get("tool_cx", DEFAULT_TOOL_CX)))
            tool_cy_now = float(getattr(state, "tool_cy", s.get("tool_cy", DEFAULT_TOOL_CY)))
            tool_lz_now = float(getattr(state, "tool_lz", s.get("tool_lz", DEFAULT_TOOL_LZ)))
        event = {
            "timestamp": time.time(),
            "outcome": str(outcome or ""),
            "subject": subject_name,
            "subject_key": subject_key,
            "step_index": int(getattr(state, "current_injection_step", 0) or 0),
            "delivered_volume_ul": float(delivered or 0.0),
            "target_volume_ul": float(injection_step_volume_ul or 0.0),
            "subject_total_delivered_ul": subject_total,
            "subject_injection_count": subject_count,
            "subject_injection_volumes_ul": list(volumes),
            "tcp_mode": tcp_mode,
            "tunnel_tcp_y_mm": tunnel_tcp_y,
            "injection_use_tunnel_workflow": use_tunnel,
            "injection_material_loaded": material_loaded,
            "tool_cx": tool_cx_now,
            "tool_cy": tool_cy_now,
            "tool_lz": tool_lz_now,
            "note": str(note or ""),
        }
        append_injection_event(event)
        if current_step_recorder and not current_step_outcome_marked:
            try:
                current_step_recorder.mark_outcome(
                    outcome,
                    delivered_volume=delivered,
                    target_volume=injection_step_volume_ul,
                    distance=injection_step_distance,
                    target_distance=injection_step_target,
                    subject=subject_name,
                    subject_total_delivered=subject_total,
                    subject_injection_count=subject_count,
                    subject_injection_volumes=list(volumes),
                    note=note,
                )
            except Exception as e:
                print(f"[Recorder] Failed to mark outcome: {e}")
            current_step_outcome_marked = True
        if stop_recorder and current_step_recorder:
            try:
                current_step_recorder.stop()
            except Exception:
                pass
            current_step_recorder = None
        return delivered

    def clear_post_step_decision(open_new_subject=False):
        nonlocal post_step_prompt_active, injection_step_done, injection_step_in_progress
        nonlocal final_small_warning_active, pending_step_index, subject_prompt_active
        nonlocal subject_field_text, subject_eye_choice, subject_tunnel_workflow
        post_step_prompt_active = False
        injection_step_done = False
        injection_step_in_progress = False
        final_small_warning_active = False
        pending_step_index = None
        if open_new_subject:
            with state.lock:
                vol_left = float(state.settings.get("syringe_remaining_ul", 0.0) or 0.0)
                inj_active = bool(state.settings.get("injection_active", False))
                use_tunnel = bool(state.settings.get("injection_use_tunnel_workflow", False))
            if inj_active and vol_left > 1e-9:
                subject_field_text = ""
                subject_eye_choice = "OD"
                subject_tunnel_workflow = use_tunnel
                subject_prompt_active = True

    def stop_current_injection_for_outcome():
        nonlocal injection_step_in_progress, injection_step_done, post_step_prompt_active
        nonlocal injection_last_step_small, pending_step_index
        if not injection_step_in_progress and not current_step_recorder:
            return
        stop_injection_plunger()
        delivered = current_step_delivered_ul()
        with state.lock:
            s = state.settings
            rem = float(s.get("syringe_remaining_ul", 0.0) or 0.0)
            s["syringe_remaining_ul"] = max(0.0, rem - delivered)
            state.calibration_status = (
                f"Injection stopped for outcome entry after {delivered:.3f} µL delivered. "
                f"Remaining {s['syringe_remaining_ul']:.3f} µL."
            )
        save_settings(state.settings)
        injection_step_in_progress = False
        injection_step_done = True
        injection_last_step_small = injection_step_volume_ul > delivered + 1e-6
        pending_step_index = None
        post_step_prompt_active = True

    def open_injection_reload_modal():
        nonlocal injection_reload_active, injection_reload_started, injection_reload_before_ul
        nonlocal injection_reload_stage
        nonlocal injection_tunnel_active, injection_tunnel_stage
        nonlocal injection_tunnel_phase_active
        if injection_step_in_progress or current_step_recorder:
            stop_current_injection_for_outcome()
            with state.lock:
                state.calibration_status = "Record the current injection outcome before reload."
            return
        with state.lock:
            injection_reload_before_ul = float(state.settings.get("syringe_remaining_ul", 0.0) or 0.0)
            state.calibration_status = "Injection reload/refill paused. Current animal logging is preserved."
        injection_reload_active = True
        injection_reload_started = False
        injection_reload_stage = "precheck"
        injection_tunnel_active = False
        injection_tunnel_stage = "confirm"
        injection_tunnel_phase_active = False
        queue_aural_alert("gpws_retract_before_workflow", min_interval_s=10.0)
        log_injection_session_event(
            "reload_opened",
            "User opened reload/refill workflow.",
            {"remaining_before_reload_ul": injection_reload_before_ul},
        )

    def syringe_calibration_ready():
        with state.lock:
            s = state.settings
            step_time = float(s.get("syringe_step_time_s", 0.0) or 0.0)
            vol_ul = float(s.get("syringe_volume_ul", 0.0) or 0.0)
        return step_time > 0.0 and vol_ul > 0.0

    def injector_readiness_status():
        with state.lock:
            s = state.settings
            step_time = float(s.get("syringe_step_time_s", 0.0) or 0.0)
            vol_ul = float(s.get("syringe_volume_ul", 0.0) or 0.0)
            loaded = bool(s.get("injection_material_loaded", False))
        calibrated = step_time > 0.0 and vol_ul > 0.0
        return calibrated, loaded

    def open_injection_tunnel_modal(target_mode="tunnel"):
        nonlocal injection_tunnel_active, injection_tunnel_target_mode, injection_tunnel_stage
        nonlocal injection_tunnel_phase_active
        nonlocal injection_reload_active, injection_reload_started, injection_reload_before_ul
        nonlocal injection_reload_stage
        target_mode = str(target_mode or "tunnel").strip().lower()
        if target_mode not in ("injector", "tunnel"):
            target_mode = "tunnel"
        if injection_step_in_progress:
            stop_current_injection_for_outcome()
            with state.lock:
                state.calibration_status = "Record the current injection outcome before TCP workflow."
            return
        if target_mode == "injector":
            calibrated, material_loaded = injector_readiness_status()
            if not calibrated or not material_loaded:
                missing = []
                if not calibrated:
                    missing.append("syringe calibration")
                if not material_loaded:
                    missing.append("loaded material confirmation")
                with state.lock:
                    state.calibration_status = (
                        "Prepare injector blocked: " + " and ".join(missing) + " required."
                    )
                log_injection_session_event(
                    "prepare_injector_blocked",
                    "Prepare injector workflow was blocked by missing prerequisites.",
                    {
                        "syringe_calibrated": bool(calibrated),
                        "material_loaded": bool(material_loaded),
                    },
                )
                return
        with state.lock:
            current_mode = str(
                state.settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)
                or DEFAULT_TUNNEL_TCP_MODE
            ).strip().lower()
            if current_mode not in ("injector", "tunnel"):
                current_mode = DEFAULT_TUNNEL_TCP_MODE
            offset_y = float(state.settings.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM) or 0.0)
            state.calibration_status = "Tunnel/injector TCP workflow opened."
        injection_tunnel_target_mode = target_mode
        injection_tunnel_stage = "confirm"
        injection_tunnel_active = True
        if target_mode == "tunnel":
            injection_tunnel_phase_active = True
        injection_reload_active = False
        injection_reload_started = False
        injection_reload_before_ul = 0.0
        injection_reload_stage = "precheck"
        queue_aural_alert("gpws_retract_before_workflow", min_interval_s=10.0)
        log_injection_session_event(
            "tunnel_workflow_opened",
            f"User opened TCP workflow to switch from {current_mode} to {target_mode}.",
            {
                "current_tcp_mode": current_mode,
                "target_tcp_mode": target_mode,
                "tunnel_tcp_y_mm": offset_y,
            },
        )

    def queue_injection_tunnel_switch(target_mode):
        nonlocal injection_tunnel_stage, injection_tunnel_target_mode
        target_mode = str(target_mode or "tunnel").strip().lower()
        if target_mode not in ("injector", "tunnel"):
            return
        if target_mode == "injector":
            calibrated, material_loaded = injector_readiness_status()
            if not calibrated or not material_loaded:
                with state.lock:
                    state.calibration_status = (
                        "Prepare injector blocked: syringe calibration and loaded material are required."
                    )
                return
        injection_tunnel_target_mode = target_mode
        with state.lock:
            current_mode = str(
                state.settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)
                or DEFAULT_TUNNEL_TCP_MODE
            ).strip().lower()
            if current_mode not in ("injector", "tunnel"):
                current_mode = DEFAULT_TUNNEL_TCP_MODE
            offset_y = float(state.settings.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM) or 0.0)
            retract_mm = float(state.settings.get("tunnel_retract_mm", DEFAULT_TUNNEL_RETRACT_MM) or 0.0)
            return_mm = float(state.settings.get("tunnel_return_mm", DEFAULT_TUNNEL_RETURN_MM) or 0.0)
        if current_mode == target_mode:
            with state.lock:
                state.calibration_status = f"TCP is already in {target_mode} mode."
            injection_tunnel_stage = "queued"
            return
        if abs(offset_y) < 0.001:
            with state.lock:
                state.calibration_status = "Set tunnel side needle Y offset before running TCP workflow."
            return
        queue_aural_alert("gpws_retract_before_workflow", min_interval_s=4.0)
        rt.enqueue({"type": "safe_tunnel_tcp_shift", "target_mode": target_mode})
        injection_tunnel_stage = "queued"
        log_injection_session_event(
            "tunnel_tcp_switch_queued",
            f"Safe retract/TCP-switch/return queued for {target_mode} mode.",
            {
                "current_tcp_mode": current_mode,
                "target_tcp_mode": target_mode,
                "tunnel_tcp_y_mm": offset_y,
                "tunnel_retract_mm": retract_mm,
                "tunnel_return_mm": return_mm,
            },
        )
        with state.lock:
            state.calibration_status = (
                f"Safe TCP switch to {target_mode} queued: retract {retract_mm:.1f} mm, "
                f"return {return_mm:.1f} mm."
            )

    def begin_tunnel_phase_for_current_subject():
        nonlocal injection_tunnel_phase_active, injection_tunnel_active
        nonlocal injection_tunnel_stage, injection_tunnel_target_mode
        injection_tunnel_phase_active = True
        injection_tunnel_target_mode = "tunnel"
        injection_tunnel_active = False
        with state.lock:
            current_mode = str(
                state.settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)
                or DEFAULT_TUNNEL_TCP_MODE
            ).strip().lower()
            if current_mode not in ("injector", "tunnel"):
                current_mode = DEFAULT_TUNNEL_TCP_MODE
            state.audio_cue = "tunnel_mode_start"
            state.calibration_status = (
                "Tunnel mode started. Free control enabled; use Tunnel Complete -> Injector after making the tunnel."
            )
        log_injection_session_event(
            "tunnel_mode_started",
            "Tunnel-first workflow started for the current animal/sample.",
            {"current_tcp_mode": current_mode, "target_tcp_mode": "tunnel"},
        )
        if current_mode != "tunnel":
            queue_injection_tunnel_switch("tunnel")
        else:
            injection_tunnel_stage = "confirm"

    def complete_tunnel_phase_to_injector():
        nonlocal injection_tunnel_phase_active
        with state.lock:
            current_mode = str(
                state.settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)
                or DEFAULT_TUNNEL_TCP_MODE
            ).strip().lower()
            if current_mode not in ("injector", "tunnel"):
                current_mode = DEFAULT_TUNNEL_TCP_MODE
        if current_mode != "tunnel":
            with state.lock:
                state.calibration_status = (
                    "Tunnel needle TCP is not active yet. Wait for the tunnel switch before completing tunnel."
                )
            return
        with state.lock:
            state.audio_cue = "tunnel_complete"
            state.calibration_status = "Tunnel complete; switching safely to injector TCP."
        log_injection_session_event(
            "tunnel_complete",
            "User marked tunnel complete and requested injector TCP workflow.",
            {"current_tcp_mode": current_mode, "target_tcp_mode": "injector"},
        )
        injection_tunnel_phase_active = False
        queue_injection_tunnel_switch("injector")

    def complete_injection_reload(return_to_retracted=False):
        nonlocal injection_reload_active, injection_reload_started, injection_reload_before_ul
        nonlocal injection_reload_stage
        with state.lock:
            s = state.settings
            before = float(s.get("syringe_remaining_ul", injection_reload_before_ul) or 0.0)
            full = float(s.get("syringe_volume_ul", 0.0) or 0.0)
            s["syringe_remaining_ul"] = max(0.0, full)
            s["injection_active"] = True
            s["injection_material_loaded"] = True
            state.calibration_status = (
                f"Syringe refill complete. Remaining reset from {before:.3f} to {full:.3f} µL; "
                "current animal logging preserved."
            )
        save_settings(state.settings)
        log_injection_session_event(
            "reload_complete",
            "Syringe refilled without ending the current animal/session.",
            {
                "remaining_before_reload_ul": float(before),
                "remaining_after_reload_ul": float(full),
                "return_to_retracted_pose": bool(return_to_retracted),
            },
        )
        if return_to_retracted:
            rt.enqueue({"type": "injection_reload_return_retracted"})
        injection_reload_active = False
        injection_reload_started = False
        injection_reload_before_ul = 0.0
        injection_reload_stage = "precheck"

    def open_injection_exit_modal():
        nonlocal injection_exit_active
        nonlocal injection_reload_active, injection_tunnel_active
        nonlocal injection_tunnel_phase_active
        if injection_step_in_progress or current_step_recorder:
            stop_current_injection_for_outcome()
            with state.lock:
                state.calibration_status = "Record the current injection outcome before exiting injection mode."
            return
        injection_exit_active = True
        injection_reload_active = False
        injection_tunnel_active = False
        injection_tunnel_phase_active = False
        log_injection_session_event(
            "injection_exit_opened",
            "User opened injection mode exit choices.",
        )

    def finish_injection_mode_exit(exit_mode):
        nonlocal injection_exit_active, injection_step_in_progress, injection_step_done
        nonlocal post_step_prompt_active, post_outcome_choice_active, post_outcome_choice_just_opened
        nonlocal final_small_warning_active, subject_prompt_active, pending_step_index
        nonlocal injection_reload_active, injection_tunnel_active, injection_tunnel_stage
        nonlocal injection_tunnel_phase_active
        stop_injection_plunger()
        exit_mode = str(exit_mode or "keep_remaining")
        with state.lock:
            s = state.settings
            before = float(s.get("syringe_remaining_ul", 0.0) or 0.0)
            after = before
            if exit_mode == "eject_remaining":
                after = 0.0
                s["syringe_remaining_ul"] = 0.0
                s["injection_material_loaded"] = False
            s["injection_active"] = False
            state.current_injection_step = 0
            state.audio_cue = "injection_stop"
            if exit_mode == "eject_remaining":
                state.calibration_status = (
                    f"Injection mode exited; remaining {before:.3f} µL marked ejected/returned."
                )
            else:
                state.calibration_status = (
                    f"Injection mode exited; remaining {after:.3f} µL preserved for manual handling."
                )
        save_settings(state.settings)
        log_injection_session_event(
            "injection_mode_exit",
            "Injection mode exited without forcing syringe depletion.",
            {
                "exit_mode": exit_mode,
                "remaining_before_exit_ul": float(before),
                "remaining_after_exit_ul": float(after),
            },
        )
        injection_exit_active = False
        injection_step_in_progress = False
        injection_step_done = False
        post_step_prompt_active = False
        post_outcome_choice_active = False
        post_outcome_choice_just_opened = False
        final_small_warning_active = False
        subject_prompt_active = False
        pending_step_index = None
        injection_reload_active = False
        injection_tunnel_active = False
        injection_tunnel_stage = "confirm"
        injection_tunnel_phase_active = False

    def start_injection_session(confirm_unloaded=False):
        nonlocal injection_info_active, injection_step_in_progress
        nonlocal injection_step_done, post_step_prompt_active, post_outcome_choice_active
        nonlocal post_outcome_choice_just_opened
        nonlocal injection_reload_active, injection_reload_started, injection_reload_before_ul
        nonlocal injection_reload_stage
        nonlocal injection_tunnel_active, injection_tunnel_stage
        nonlocal injection_tunnel_phase_active
        nonlocal injection_exit_active
        nonlocal injection_last_step_small
        nonlocal injection_record_telemetry, injection_record_video, injection_record_ui
        nonlocal injection_use_tunnel_workflow, injection_material_loaded
        nonlocal subject_prompt_active, subject_field_text, subject_eye_choice, subject_tunnel_workflow, injection_subject_text
        nonlocal pending_step_index, current_step_recorder
        nonlocal subject_injection_counts, subject_injection_volumes
        nonlocal material_loaded_warning_active

        with state.lock:
            s = state.settings
            s["injection_study"]    = injection_study_text
            s["injection_material"] = injection_material_text
            s["injection_date"]     = injection_date_text
            s["injection_use_tunnel_workflow"] = bool(injection_use_tunnel_workflow)
            s["injection_material_loaded"] = bool(injection_material_loaded)

            step_time = float(s.get("syringe_step_time_s", 0.0) or 0.0)
            vol_ul = float(s.get("syringe_volume_ul", 0.0) or 0.0)

            if step_time <= 0 or vol_ul <= 0:
                state.calibration_status = "Cannot start injection: syringe not calibrated."
                s["injection_active"] = False
                injection_active_now = False
                print("⚠️ Cannot start injection: syringe not calibrated.")
            elif not injection_material_loaded and not confirm_unloaded:
                state.calibration_status = "Confirm loaded material before starting injection mode."
                s["injection_active"] = False
                injection_active_now = False
                material_loaded_warning_active = True
                print("⚠️ Start injection requested without loaded-material confirmation.")
            else:
                if confirm_unloaded and not injection_material_loaded:
                    injection_material_loaded = True
                    s["injection_material_loaded"] = True
                s["injection_active"] = True
                s["syringe_remaining_ul"] = vol_ul  # reset to full
                state.record_telemetry = injection_record_telemetry
                state.record_video     = injection_record_video
                state.record_ui        = injection_record_ui
                state.current_injection_step = 0
                injection_last_step_small = False
                state.calibration_status = "Injection session active."
                state.audio_cue = "injection_start"
                injection_active_now = True
                print("[Injection] Session started.")

        save_settings(state.settings)
        if not injection_active_now:
            return

        pending_step_index = None
        subject_injection_counts = {}
        subject_injection_volumes = {}
        if current_step_recorder:
            try:
                current_step_recorder.stop()
            except Exception:
                pass
            current_step_recorder = None
        injection_step_in_progress = False
        injection_step_done = False
        post_step_prompt_active = False
        post_outcome_choice_active = False
        post_outcome_choice_just_opened = False
        injection_reload_active = False
        injection_reload_started = False
        injection_reload_before_ul = 0.0
        injection_tunnel_active = False
        injection_tunnel_stage = "confirm"
        injection_tunnel_phase_active = False
        injection_exit_active = False
        injection_info_active = False

        if injection_active_now:
            # Immediately ask Subject/Animal for FIRST step
            with state.lock:
                prev_subj = state.settings.get("injection_subject", "")
                use_tunnel = bool(state.settings.get("injection_use_tunnel_workflow", False))
            subject_field_text, subject_eye_choice = split_subject_eye_text(
                injection_subject_text or prev_subj or ""
            )
            subject_tunnel_workflow = use_tunnel
            subject_prompt_active = True

    # ─────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────
    running = True
    quit_confirm_active = False
    loop_prev_time = time.time()

    prev_a = prev_b = prev_x = prev_y = False
    prev_menu = False
    ui_nav_active = False
    ui_nav_focus_index = 0
    ui_nav_items = []
    ui_nav_needs_initial_focus = False
    ui_nav_last_move_time = 0.0
    roll_visual_angle = None
    roll_visual_mode = None
    last_sig_axis_time = 0.0
    prev_lt = 0.0
    prev_rt = 0.0

    click_pos = None

    def request_quit_confirmation():
        nonlocal quit_confirm_active
        quit_confirm_active = True
        with state.lock:
            state.calibration_status = "Exit requested. Confirm to halt motion and close cleanly."

    def perform_graceful_exit(reason="user"):
        nonlocal running, quit_confirm_active, current_step_recorder
        nonlocal injection_step_in_progress, injection_step_done, post_step_prompt_active
        nonlocal post_outcome_choice_active, post_outcome_choice_just_opened
        nonlocal final_small_warning_active, subject_prompt_active, injection_info_active
        nonlocal injection_reload_active, injection_reload_started, injection_reload_before_ul
        nonlocal injection_reload_stage, injection_tunnel_active, injection_tunnel_stage
        nonlocal injection_exit_active
        nonlocal pending_step_index

        try:
            log_injection_session_event(
                "app_exit_requested",
                "Application exit requested through graceful shutdown.",
                {
                    "reason": str(reason or "user"),
                    "injection_step_in_progress": bool(injection_step_in_progress),
                    "recording_active": bool(current_step_recorder),
                },
            )
        except Exception:
            pass

        if injection_step_in_progress:
            stop_current_injection_for_outcome()
            try:
                mark_current_step_outcome(
                    "app_exit",
                    "Application exited during an active injection step.",
                )
            except Exception:
                pass
        elif current_step_recorder:
            try:
                current_step_recorder.stop()
            except Exception:
                pass
            current_step_recorder = None

        stop_injection_plunger()

        with state.lock:
            state.settings["injection_active"] = False
            state.injection_step_in_progress = False
            state.await_confirm = False
            state.await_target = None
            state.confirm_msg = ""
            state.calibration_status = "Application exiting cleanly."
        save_settings(state.settings)

        injection_step_in_progress = False
        injection_step_done = False
        post_step_prompt_active = False
        post_outcome_choice_active = False
        post_outcome_choice_just_opened = False
        final_small_warning_active = False
        subject_prompt_active = False
        injection_info_active = False
        injection_reload_active = False
        injection_reload_started = False
        injection_reload_before_ul = 0.0
        injection_reload_stage = "precheck"
        injection_tunnel_active = False
        injection_tunnel_stage = "confirm"
        injection_exit_active = False
        pending_step_index = None

        try:
            rt.enqueue({"type": "stop_motors"})
            time.sleep(0.03)
        except Exception:
            pass
        rt.keep_motors_on_exit = True
        rt.stop_event.set()
        quit_confirm_active = False
        running = False

    while running:
        click_pos = None
        frame_nav_items = []
        alarm_threshold_rect = None
        alarm_duration_rect = None
        tool_center_demo_mode = getattr(state, "tool_center_demo_mode", "circle")
        now = time.time()

        if (now - thermal_last_poll) >= THERMAL_POLL_INTERVAL_S:
            thermal_last_poll = now
            hotspot = _pick_thermal_hotspot()
            if hotspot is not None:
                temp_c = float(hotspot.get("temp_c", 0.0))
                zone_name = str(hotspot.get("type", "thermal")).strip() or "thermal"
                if temp_c >= THERMAL_WARN_TEMP_C:
                    if not thermal_hot:
                        thermal_warning_dismissed = False
                        print(
                            f"[Thermal] High temperature detected "
                            f"({zone_name} {temp_c:.1f} C). Control lag may occur."
                        )
                        if not thermal_throttle_active:
                            set_camera_thermal_throttle(True)
                            thermal_throttle_active = True
                        ui_notice_text = (
                            f"Thermal warning: {zone_name} {temp_c:.0f} C; "
                            "camera load reduced for robot control"
                        )
                        ui_notice_until = now + 5.0
                    elif not thermal_throttle_active:
                        set_camera_thermal_throttle(True)
                        thermal_throttle_active = True
                    if temp_c >= THERMAL_CRITICAL_TEMP_C:
                        warning_suffix = "critical; camera load reduced"
                    else:
                        warning_suffix = "camera load reduced"
                    thermal_critical = temp_c >= THERMAL_CRITICAL_TEMP_C
                    thermal_status_text = (
                        f"Thermal warning: {zone_name} {temp_c:.0f} C; {warning_suffix}"
                    )
                    thermal_hot = True
                elif thermal_hot and temp_c <= THERMAL_CLEAR_TEMP_C:
                    thermal_hot = False
                    thermal_critical = False
                    thermal_status_text = ""
                    thermal_warning_dismissed = False
                    if thermal_throttle_active:
                        set_camera_thermal_throttle(False)
                        thermal_throttle_active = False
                    print(
                        f"[Thermal] Temperature recovered "
                        f"({zone_name} {temp_c:.1f} C)."
                    )
            elif thermal_hot:
                thermal_hot = False
                thermal_critical = False
                thermal_status_text = ""
                thermal_warning_dismissed = False
                if thermal_throttle_active:
                    set_camera_thermal_throttle(False)
                    thermal_throttle_active = False

        # ───────────── Event pump ─────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                perform_graceful_exit("window_close")
                break

            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                is_fullscreen = not is_fullscreen
                screen, flags, is_fullscreen = create_display_surface(is_fullscreen)
                continue
            text_or_modal_key_focus = (
                save_mode_modal_active or
                toollen_input_active or approach_input_active or
                toolcx_input_active or toolcy_input_active or pitch_input_active or angle_input_active or yaw_input_active or
                tunnel_offset_input_active or tunnel_retract_input_active or tunnel_return_input_active or high_angle_input_active or
                naming_input_active or rename_input_active or settings_path_input_active or
                fwd_rate_input_active or bwd_rate_input_active or endstop_timeout_input_active or pose_wait_input_active or
                retract_input_active or routine_name_input_active or
                routine_pose_name_input_active or routine_custom_cmd_input_active or routine_step_edit_active or
                routine_edit_active or syringe_calib_active or injection_info_active or subject_prompt_active
            )
            if ev.type == pygame.KEYDOWN and not text_or_modal_key_focus:
                if ev.key == pygame.K_h:
                    show_status_overlay = not show_status_overlay
                    continue
                if ev.key == pygame.K_r:
                    rs_thread.cycle_view_mode()
                    continue

            if ev.type == pygame.KEYDOWN and save_mode_modal_active:
                if ev.key == pygame.K_ESCAPE:
                    save_mode_modal_active = False
                    pending_save_pose_name = ""
                elif ev.key == pygame.K_RETURN:
                    if pending_save_pose_name:
                        rt.enqueue({"type": "save_named_with_midway", "name": pending_save_pose_name, "demo": False, "star": True})
                    save_mode_modal_active = False
                    pending_save_pose_name = ""
                continue

            if ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                _, tr_rect, _, _ = compute_layout(
                    *pygame.display.get_surface().get_size(),
                    args.ui_frac, args.ui_min_width
                )
                if routine_builder_active:
                    routine_builder_scroll = max(0, routine_builder_scroll - ev.y * 44)
                elif settings_modal_active:
                    settings_modal_scroll = max(0, settings_modal_scroll - ev.y * 42)
                elif post_step_prompt_active:
                    outcome_detail_scroll = max(0, outcome_detail_scroll - ev.y * 28)
                elif tr_rect.collidepoint(mx, my):
                    ui_scroll -= ev.y * 40
                    ui_scroll = max(
                        0,
                        min(ui_scroll, max(0, ui_content_height - tr_rect.h + 12))
                    )

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
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
                alarm_threshold_drag = False
                alarm_duration_drag = False

            if ev.type == pygame.MOUSEMOTION:
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

                # Tunnel needle Y TCP offset
                if tunnel_offset_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            new_offset = float(tunnel_offset_text.strip())
                            with state.lock:
                                state.settings["tunnel_tcp_y_mm"] = new_offset
                                state.tunnel_tcp_y_mm = new_offset
                            save_settings(state.settings)
                        except ValueError:
                            print("⚠️ Invalid tunnel TCP Y offset.")
                        tunnel_offset_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        tunnel_offset_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        tunnel_offset_text = tunnel_offset_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            tunnel_offset_text += ch
                    continue

                # Tunnel safety retract distance
                if tunnel_retract_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            new_retract = max(1.0, float(tunnel_retract_text.strip()))
                            with state.lock:
                                current_return = max(
                                    0.0,
                                    float(state.settings.get("tunnel_return_mm", DEFAULT_TUNNEL_RETURN_MM)),
                                )
                                if current_return > new_retract:
                                    current_return = new_retract
                                    state.settings["tunnel_return_mm"] = current_return
                                    state.tunnel_return_mm = current_return
                                state.settings["tunnel_retract_mm"] = new_retract
                                state.tunnel_retract_mm = new_retract
                            save_settings(state.settings)
                        except ValueError:
                            print("⚠️ Invalid tunnel safety retract distance.")
                        tunnel_retract_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        tunnel_retract_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        tunnel_retract_text = tunnel_retract_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            tunnel_retract_text += ch
                    continue

                # Tunnel guided return distance
                if tunnel_return_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            with state.lock:
                                retract_limit = max(
                                    1.0,
                                    float(state.settings.get("tunnel_retract_mm", DEFAULT_TUNNEL_RETRACT_MM)),
                                )
                            new_return = max(0.0, min(retract_limit, float(tunnel_return_text.strip())))
                            with state.lock:
                                state.settings["tunnel_return_mm"] = new_return
                                state.tunnel_return_mm = new_return
                            save_settings(state.settings)
                        except ValueError:
                            print("⚠️ Invalid tunnel guided return distance.")
                        tunnel_return_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        tunnel_return_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        tunnel_return_text = tunnel_return_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            tunnel_return_text += ch
                    continue

                # Injection high-angle warning threshold
                if high_angle_input_active:
                    if ev.key == pygame.K_RETURN:
                        try:
                            val = _clamp_high_angle_pitch_warning_deg(high_angle_text.strip())
                            with state.lock:
                                state.settings["high_angle_pitch_warning_deg"] = val
                                state.calibration_status = (
                                    f"High-angle warning set to +{val:.1f} degrees."
                                )
                            high_angle_text = f"{val:.1f}"
                            save_settings(state.settings)
                        except ValueError:
                            print("⚠️ Invalid high-angle warning threshold.")
                        high_angle_input_active = False
                    elif ev.key == pygame.K_ESCAPE:
                        high_angle_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        high_angle_text = high_angle_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch in "0123456789.-":
                            high_angle_text += ch
                    continue

                # Roll input (joint 5 / tool-axis roll)
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
                            print("⚠️ Invalid roll angle value.")
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
                            entered_yaw = float(yaw_text.strip())
                            with state.lock:
                                yaw_base_now = getattr(state, "yaw_base", None)
                            if yaw_base_now is not None:
                                target_yaw = _norm_angle_deg(float(yaw_base_now) + entered_yaw)
                            else:
                                target_yaw = entered_yaw
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
                    elif ev.key == pygame.K_b and (mods & pygame.KMOD_CTRL):
                        persist_current_routine()
                        routine_edit_active = False
                        routine_builder_active = True
                        routine_builder_scroll = 0
                    elif ev.key == pygame.K_ESCAPE:
                        routine_edit_active = False
                    else:
                        editor_handle_key(routine_editor, ev)
                    continue

                # Routine builder: save current pose node
                if routine_pose_name_input_active:
                    if ev.key == pygame.K_RETURN:
                        name = routine_pose_name_text.strip()
                        routine_pose_name_input_active = False
                        if name:
                            rt.enqueue({"type": "save_named_with_midway", "name": name, "demo": False, "star": True})
                            append_to_current_routine(
                                f"POSE {name}",
                                f"new pose {name}",
                            )
                    elif ev.key == pygame.K_ESCAPE:
                        routine_pose_name_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        routine_pose_name_text = routine_pose_name_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and (ch.isalnum() or ch in "_- "):
                            routine_pose_name_text += ch
                    continue

                # Routine builder: custom command node
                if routine_custom_cmd_input_active:
                    if ev.key == pygame.K_RETURN:
                        cmd_text = routine_custom_cmd_text.strip()
                        routine_custom_cmd_input_active = False
                        if cmd_text:
                            append_to_current_routine(cmd_text, "custom command")
                    elif ev.key == pygame.K_ESCAPE:
                        routine_custom_cmd_input_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        routine_custom_cmd_text = routine_custom_cmd_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and ch.isprintable() and ch not in "\r\n\t":
                            routine_custom_cmd_text += ch
                    continue

                # Routine builder: edit one existing node
                if routine_step_edit_active:
                    if ev.key == pygame.K_RETURN:
                        routine_step_edit_active = False
                        set_routine_line(routine_step_edit_line_index, routine_step_edit_text)
                    elif ev.key == pygame.K_ESCAPE:
                        routine_step_edit_active = False
                    elif ev.key == pygame.K_BACKSPACE:
                        routine_step_edit_text = routine_step_edit_text[:-1]
                    else:
                        ch = ev.unicode
                        if ch and ch.isprintable() and ch not in "\r\n\t":
                            routine_step_edit_text += ch
                    continue

                if routine_builder_active:
                    if ev.key == pygame.K_ESCAPE:
                        routine_builder_active = False
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
                            elif routine_name_mode in ("create", "create_starred", "create_all"):
                                if exists:
                                    with state.lock:
                                        state.routine_status_text = "Name already exists; choose another."
                                else:
                                    if routine_name_mode == "create_starred":
                                        txt = routine_template_from_poses(starred_pose_names_snapshot())
                                    elif routine_name_mode == "create_all":
                                        txt = routine_template_from_poses(all_pose_names_snapshot())
                                    else:
                                        txt = routine_template_empty()
                                    routines[name] = txt
                                    current_routine_name = name
                                    routine_editor = EditorState(text=txt, caret=len(txt))
                                    persist_current_routine()
                                    with state.lock:
                                        state.routine_status_text = f"Created routine '{name}'"
                                    routine_builder_active = True
                                    routine_builder_scroll = 0
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
                        perform_graceful_exit("keyboard_confirm")
                    continue

                if settings_modal_active:
                    if ev.key == pygame.K_ESCAPE:
                        settings_modal_active = False
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
                    elif ev.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        if syringe_field_active == "vol":
                            syringe_vol_text = adjust_numeric_text(syringe_vol_text, 0.5)
                        else:
                            syringe_step_ul_txt = adjust_numeric_text(syringe_step_ul_txt, 0.1)
                    elif ev.key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
                        if syringe_field_active == "vol":
                            syringe_vol_text = adjust_numeric_text(syringe_vol_text, -0.5)
                        else:
                            syringe_step_ul_txt = adjust_numeric_text(syringe_step_ul_txt, -0.1)
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
                    elif ev.key == pygame.K_TAB:
                        subject_eye_choice = "OS" if subject_eye_choice == "OD" else "OD"
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
                if material_loaded_warning_active:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_b):
                        material_loaded_warning_active = False
                    elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        material_loaded_warning_active = False
                        start_injection_session(confirm_unloaded=True)
                    continue

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
        with state.lock:
            lx, ly = state.lx, state.ly
            rx, ry = state.rx, state.ry
            raw_hx, raw_hy = state.hx, state.hy
            lb, rb = state.lb, state.rb
            cur_lt = state.lt
            cur_rt = state.rt
            alarm_active = bool(state.alarm_latched)
            alarm_message = str(state.last_alarm_message or "")
            pending_audio_cue = state.audio_cue
            state.audio_cue = None
        pending_audio_key = warning_key_for_cue(pending_audio_cue)
        if pending_audio_cue and not (pending_audio_key and pending_audio_key in muted_warning_keys):
            play_audio_cue(pending_audio_cue)
        if abs(cur_lt - prev_lt) > AXIS_GUARD_DELTA or abs(cur_rt - prev_rt) > AXIS_GUARD_DELTA:
            last_sig_axis_time = time.time()
        prev_lt, prev_rt = cur_lt, cur_rt
        right_trigger_norm = normalize(cur_rt)
        right_down = right_trigger_norm > DEADZONE
        injection_trigger_attenuation = injection_trigger_curve(right_trigger_norm)
        injection_trigger_down = injection_trigger_attenuation > INJECTION_TRIGGER_START

        # Buttons (A,B,X,Y)
        a_now = joy_button_pressed(joy, A_BUTTON)
        b_now = joy_button_pressed(joy, B_BUTTON)
        x_now = joy_button_pressed(joy, X_BUTTON)
        y_now = joy_button_pressed(joy, Y_BUTTON)
        menu_now = joy_any_button_pressed(joy, MENU_BUTTON_CANDIDATES)

        a_edge_raw = a_now and not prev_a
        b_edge_raw = b_now and not prev_b
        x_edge_raw = x_now and not prev_x
        y_edge_raw = y_now and not prev_y
        menu_edge = menu_now and not prev_menu

        a_edge = a_edge_raw
        b_edge = b_edge_raw
        x_edge = x_edge_raw
        y_edge = y_edge_raw

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
            pitch_for_alert = float(getattr(state, "pitch", 0.0) or 0.0)
            high_angle_threshold_for_alert = _clamp_high_angle_pitch_warning_deg(
                state.settings.get(
                    "high_angle_pitch_warning_deg",
                    DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG,
                )
            )
            tcp_mode_for_alert = str(
                state.settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)
                or DEFAULT_TUNNEL_TCP_MODE
            ).strip().lower()
        if tcp_mode_for_alert not in ("injector", "tunnel"):
            tcp_mode_for_alert = DEFAULT_TUNNEL_TCP_MODE

        if menu_edge:
            ui_nav_active = not ui_nav_active
            ui_nav_needs_initial_focus = bool(ui_nav_active)
            with state.lock:
                state.ui_nav_active = bool(ui_nav_active)
                state.calibration_status = (
                    "Controller UI mode active: right stick selects, A activates, Menu exits."
                    if ui_nav_active else
                    "Controller UI mode closed. Robot jog controls restored."
                )
            try:
                joy.rumble(0.35, 0.35, 100)
            except Exception:
                pass

        if ui_nav_active:
            if not ui_nav_items:
                ui_nav_focus_index = 0
            else:
                ui_nav_focus_index %= len(ui_nav_items)
                nav_rx = _apply_deadzone(rx, UI_NAV_DEADZONE)
                nav_ry = _apply_deadzone(ry, UI_NAV_DEADZONE)
                if (
                    max(abs(nav_rx), abs(nav_ry)) > 0.0
                    and (now_loop - ui_nav_last_move_time) >= UI_NAV_REPEAT_S
                ):
                    if abs(nav_rx) >= abs(nav_ry):
                        step = 1 if nav_rx > 0 else -1
                    else:
                        step = 1 if nav_ry > 0 else -1
                    scrolled_menu = False
                    current_nav_item = ui_nav_items[ui_nav_focus_index]
                    if current_nav_item.get("group") == "panel":
                        try:
                            _, nav_tr_rect, _, _ = compute_layout(
                                *pygame.display.get_surface().get_size(),
                                args.ui_frac,
                                args.ui_min_width,
                            )
                        except Exception:
                            nav_tr_rect = None
                        panel_indices = [
                            idx for idx, item in enumerate(ui_nav_items)
                            if item.get("group") == "panel"
                        ]
                        if nav_tr_rect is not None and panel_indices:
                            max_scroll = max(0, ui_content_height - nav_tr_rect.h + 12)
                            rect = pygame.Rect(current_nav_item.get("rect") or (0, 0, 0, 0))
                            scroll_step = max(38, min(96, rect.h + 20))
                            if (
                                step > 0
                                and ui_nav_focus_index == panel_indices[-1]
                                and ui_scroll < max_scroll
                            ):
                                ui_scroll = min(max_scroll, ui_scroll + scroll_step)
                                scrolled_menu = True
                            elif (
                                step < 0
                                and ui_nav_focus_index == panel_indices[0]
                                and ui_scroll > 0
                            ):
                                ui_scroll = max(0, ui_scroll - scroll_step)
                                scrolled_menu = True
                    if not scrolled_menu:
                        ui_nav_focus_index = (ui_nav_focus_index + step) % len(ui_nav_items)
                    ui_nav_last_move_time = now_loop
                    try:
                        joy.rumble(0.08, 0.08, 45)
                    except Exception:
                        pass
                if a_edge_raw:
                    selected = ui_nav_items[ui_nav_focus_index]
                    rect = selected.get("rect")
                    if rect is not None:
                        click_pos = pygame.Rect(rect).center
                        try:
                            joy.rumble(0.2, 0.2, 80)
                        except Exception:
                            pass
            # Suppress direct controller shortcuts while UI focus is active.
            # A is converted to click_pos above; Menu exits focus mode.
            a_edge = False
            b_edge = False
            x_edge = x_edge_raw if alarm_active else False
            y_edge = False

        alarm_rising = alarm_active and not bool(aural_alert_prev.get("alarm", False))
        if alarm_active:
            queue_aural_alert("gpws_controller_alarm", min_interval_s=18.0, force=alarm_rising)
        aural_alert_prev["alarm"] = bool(alarm_active)

        thermal_critical_rising = thermal_critical and not bool(
            aural_alert_prev.get("thermal_critical", False)
        )
        thermal_hot_rising = thermal_hot and not bool(aural_alert_prev.get("thermal_hot", False))
        if thermal_critical:
            queue_aural_alert(
                "gpws_thermal_critical",
                min_interval_s=20.0,
                force=thermal_critical_rising,
            )
        elif thermal_hot:
            queue_aural_alert(
                "gpws_thermal_warning",
                min_interval_s=20.0,
                force=thermal_hot_rising,
            )
        aural_alert_prev["thermal_hot"] = bool(thermal_hot)
        aural_alert_prev["thermal_critical"] = bool(thermal_critical)

        if post_step_prompt_active and not bool(aural_alert_prev.get("post_step", False)):
            queue_aural_alert("gpws_record_outcome", min_interval_s=6.0, force=True)
        aural_alert_prev["post_step"] = bool(post_step_prompt_active)

        if final_small_warning_active and not bool(aural_alert_prev.get("final_small", False)):
            queue_aural_alert("gpws_final_dose", min_interval_s=8.0, force=True)
        aural_alert_prev["final_small"] = bool(final_small_warning_active)

        injection_warning_paused = bool(injection_reload_active)
        high_angle_active = bool(
            injection_active_flag
            and not injection_warning_paused
            and pitch_for_alert > high_angle_threshold_for_alert
        )
        high_angle_rising = high_angle_active and not bool(
            aural_alert_prev.get("high_angle", False)
        )
        if high_angle_active:
            queue_aural_alert(
                "gpws_high_angle",
                min_interval_s=4.0,
                force=high_angle_rising,
            )
        aural_alert_prev["high_angle"] = high_angle_active

        midway_alert_active = bool(waiting and at_midway_now)
        if midway_alert_active and not bool(aural_alert_prev.get("midway", False)):
            queue_aural_alert("gpws_midway_hold", min_interval_s=10.0, force=True)
        aural_alert_prev["midway"] = midway_alert_active

        if (
            injection_active_flag
            and tcp_mode_for_alert != aural_alert_prev.get("tcp_mode", DEFAULT_TUNNEL_TCP_MODE)
        ):
            queue_aural_alert(
                "gpws_tunnel_tcp" if tcp_mode_for_alert == "tunnel" else "gpws_injector_tcp",
                min_interval_s=3.0,
                force=True,
            )
        aural_alert_prev["tcp_mode"] = tcp_mode_for_alert

        if injection_active_flag and not bool(aural_alert_prev.get("injection_active", False)):
            queue_aural_alert("gpws_injection_armed", min_interval_s=8.0, force=True)
        aural_alert_prev["injection_active"] = bool(injection_active_flag)

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
                        steps_left = max(0.0, new_remaining / step_nom)
                    else:
                        steps_left = 0.0
                    if small:
                        state.calibration_status = (
                            f"Inject step complete (small step {injection_step_volume_ul:.3f} µL). "
                            f"Remaining {new_remaining:.3f} µL (≈{steps_left:.2f} step(s))."
                        )
                    else:
                        state.calibration_status = (
                            f"Inject step complete ({injection_step_volume_ul:.3f} µL). "
                            f"Remaining {new_remaining:.3f} µL (≈{steps_left:.2f} step(s))."
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
            routine_pose_name_input_active or routine_custom_cmd_input_active or routine_step_edit_active or
            toolcx_input_active or toolcy_input_active or tunnel_offset_input_active or
            tunnel_retract_input_active or tunnel_return_input_active or high_angle_input_active or settings_path_input_active
        )
        any_modal_active = (
            syringe_calib_active or injection_info_active or subject_prompt_active or
            material_loaded_warning_active or
            final_small_warning_active or post_step_prompt_active or
            post_outcome_choice_active or
            injection_reload_active or injection_tunnel_active or injection_exit_active or
            settings_modal_active or
            routine_edit_active or routine_builder_active or routine_action_modal_active or routine_edit_modal_active or
            edit_menu_active or routine_name_input_active or
            delete_confirm_active or overwrite_confirm_active or alarm_active
        )

        warning_items_for_controls = build_warning_items(
            alarm_active,
            thermal_hot,
            thermal_critical,
            high_angle_active,
            final_small_warning_active,
            post_step_prompt_active,
            midway_alert_active,
            injection_reload_active or injection_tunnel_active,
        )
        a_consumed = False
        b_consumed = False
        x_consumed = False
        y_consumed = False

        if alarm_active and x_edge:
            rt.enqueue({"type": "disarm_alarm"})
            try:
                joy.rumble(0.8, 0.8, 180)
            except Exception:
                pass
            x_consumed = True

        if (
            b_edge
            and thermal_hot
            and thermal_status_text
            and not thermal_warning_dismissed
            and not any_text_input_active
            and not any_modal_active
        ):
            thermal_warning_dismissed = True
            with state.lock:
                state.calibration_status = "Thermal popup hidden; warning badge remains visible while hot."
            b_consumed = True

        if (
            a_edge
            and injection_tunnel_phase_active
            and tcp_mode_for_alert == "tunnel"
            and not any_text_input_active
            and not any_modal_active
        ):
            complete_tunnel_phase_to_injector()
            try:
                joy.rumble(0.7, 0.7, 160)
            except Exception:
                pass
            a_consumed = True

        if a_edge and not a_consumed and not any_text_input_active and not any_modal_active:
            rt.enqueue({"type": "confirm_or_advance"})
            try:
                joy.rumble(0.7, 0.7, 160)
            except Exception:
                pass

        if b_edge and not b_consumed and not any_text_input_active and not any_modal_active:
            rt.enqueue({"type": "midway_free_control"})
            try:
                joy.rumble(0.3, 0.3, 120)
            except Exception:
                pass

        if (
            x_edge
            and not x_consumed
            and not alarm_active
            and warning_items_for_controls
            and not any_text_input_active
            and not any_modal_active
        ):
            item_to_mute = next(
                (
                    item
                    for item in warning_items_for_controls
                    if str(item.get("key", "")) not in ("alarm", "thermal")
                ),
                None,
            )
            if item_to_mute is not None:
                toggle_warning_mute_by_key(
                    item_to_mute.get("key", ""),
                    item_to_mute.get("label", "warning"),
                )
                try:
                    joy.rumble(0.2, 0.2, 90)
                except Exception:
                    pass
                x_consumed = True

        # Y during an active injection is the controller shortcut for the
        # on-screen "Record outcome" action.
        if y_edge and injection_active_flag and not any_modal_active:
            if injection_step_in_progress or current_step_recorder:
                stop_current_injection_for_outcome()
                try:
                    joy.rumble(0.9, 0.2, 220)
                except Exception:
                    pass
                y_consumed = True

        prev_a, prev_b = a_now, b_now
        prev_x, prev_y = x_now, y_now
        prev_menu = menu_now

        # Plunger bridge when NOT in injection nor calibration
        if ser:
            plunger_bridge_allowed = (
                not waiting and not at_midway_now and time.time() >= motion_until
            )
            if (
                plunger_bridge_allowed
                and not alarm_active
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

            if (
                joy.get_button(RESET_BUTTON)
                and not any_text_input_active
                and not any_modal_active
                and not ui_nav_active
                and not warning_items_for_controls
                and not injection_active_flag
                and not calibration_running_flag
            ):
                try:
                    ser.write(b"R\n")
                    if ser.in_waiting:
                        ser.read(ser.in_waiting)
                except Exception:
                    pass
                last_rate = 0
                time.sleep(0.1)

        if alarm_active and ser and last_rate not in (0, None):
            try:
                ser.write(b"V0\n")
                if ser.in_waiting:
                    ser.read(ser.in_waiting)
            except Exception:
                pass
            last_rate = 0
            injection_current_rate = 0.0

        # Injection-specific plunger control. Keep the session/logging active
        # during reload/TCP/exit workflows, but pause trigger-driven injection.
        injection_control_paused = bool(
            injection_reload_active or injection_tunnel_active or injection_exit_active
        )
        if injection_active_flag and ser and not calibration_running_flag and not alarm_active:
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
                    state.audio_cue = "injection_stop"
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
                    and not injection_control_paused
                ):
                    if injection_trigger_down and not last_right_trigger_down:
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
                            current_step_outcome_marked = False

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
                if (
                    injection_step_in_progress
                    and not post_step_prompt_active
                    and not final_small_warning_active
                    and not injection_control_paused
                ):
                    with state.lock:
                        plunger_fwd_rate = int(
                            state.settings.get("plunger_fwd_rate", MAX_RATE) or MAX_RATE
                        )
                        dir_sign = int(
                            state.settings.get("plunger_dir_sign", 1) or 1
                        )

                    desired_rate = 0
                    if injection_trigger_down:
                        rate_mag = int(abs(plunger_fwd_rate) * injection_trigger_attenuation)
                        if rate_mag <= 0:
                            rate_mag = 1
                        desired_rate = -rate_mag * dir_sign

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
                    or injection_control_paused
                ) and last_rate not in (0, None):
                    try:
                        ser.write(b"V0\n")
                        if ser.in_waiting:
                            ser.read(ser.in_waiting)
                    except Exception:
                        pass
                    last_rate = 0
                    injection_current_rate = 0.0

        last_right_trigger_down = injection_trigger_down
        with state.lock:
            state.injection_step_in_progress = bool(
                injection_active_flag and injection_step_in_progress
            )

        # ───────────── Draw layout ─────────────
        screen_w, screen_h = pygame.display.get_surface().get_size()
        tl_rect, tr_rect, bl_rect, br_rect = compute_layout(
            screen_w, screen_h, args.ui_frac, args.ui_min_width
        )

        rs_frame = rs_thread.latest()
        rs_mode_now = rs_thread.view_mode()
        blit_frame_cover(screen, rs_frame, tl_rect, cache=frame_blit_cache, cache_key="rs")

        # Recording indicator & RS “Stop recording” button
        rec_stop_rect = None
        if current_step_recorder and current_step_recorder.active:
            blink_on = (pygame.time.get_ticks() // 400) % 2 == 0
            banner_font = pygame.font.SysFont("Consolas", 14, bold=True)
            banner_small = pygame.font.SysFont("Consolas", 12)
            elapsed_rec = max(0.0, time.time() - current_step_recorder.t0)
            modes = []
            if current_step_recorder.record_telemetry:
                modes.append("TEL")
            if current_step_recorder.record_video:
                modes.append("VIDEO")
            if current_step_recorder.record_ui:
                modes.append("UI")
            modes_text = "+".join(modes) if modes else "NO STREAMS"
            banner_w = min(tl_rect.w - 24, 560)
            banner_h = 44
            banner = pygame.Rect(tl_rect.x + 12, tl_rect.y + 38, banner_w, banner_h)
            pygame.draw.rect(screen, (40, 12, 12), banner)
            pygame.draw.rect(screen, (255, 40, 40), banner, 2)
            dot_r = 8
            dot_x = banner.x + 15
            dot_y = banner.y + banner_h // 2

            if blink_on:
                pygame.draw.circle(screen, (255, 0, 0), (dot_x, dot_y), dot_r)

            rec_label = banner_font.render(
                f"RECORDING  step {current_step_recorder.step_index:02d}  {modes_text}  {elapsed_rec:05.1f}s",
                True,
                (255, 235, 235),
            )
            screen.blit(rec_label, (dot_x + dot_r + 8, banner.y + 6))
            rec_hint = banner_small.render("Video timestamps are burned in; Y opens outcome entry.", True, (245, 210, 210))
            screen.blit(rec_hint, (dot_x + dot_r + 8, banner.y + 25))

            btn_w = 132
            btn_h = 26
            btn_rect = pygame.Rect(
                banner.right - btn_w - 8,
                banner.y + (banner_h - btn_h) // 2,
                btn_w,
                btn_h,
            )
            pygame.draw.rect(screen, (70, 35, 35), btn_rect)
            pygame.draw.rect(screen, (255, 80, 80), btn_rect, 2)
            btn_txt = small_font.render("Stop recording", True, (255, 255, 255))
            screen.blit(
                btn_txt,
                (btn_rect.x + (btn_w - btn_txt.get_width()) // 2,
                 btn_rect.y + (btn_h - btn_txt.get_height()) // 2),
            )
            rec_stop_rect = btn_rect

        # Tool pitch / yaw / roll status overlay on RS view
        tl_w, tl_h = tl_rect.w, tl_rect.h
        m = max(8, int(min(tl_w, tl_h) * 0.012))

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
            high_angle_threshold_now = _clamp_high_angle_pitch_warning_deg(
                state.settings.get(
                    "high_angle_pitch_warning_deg",
                    DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG,
                )
            )
            live_eps_now = _clamp_live_linear_epsilon_mm(
                state.settings.get("live_linear_epsilon_mm", DEFAULT_LIVE_LINEAR_EPSILON_MM)
            )
            tcp_mode_now = str(
                state.settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)
                or DEFAULT_TUNNEL_TCP_MODE
            ).strip().lower()
            if tcp_mode_now not in ("injector", "tunnel"):
                tcp_mode_now = DEFAULT_TUNNEL_TCP_MODE
            tcp_label_now = "Tunnel" if tcp_mode_now == "tunnel" else "Injector"
            tunnel_y_now = float(state.settings.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM) or 0.0)
            calibration_status     = state.calibration_status or ""
            inj_retract_mm_val     = float(state.settings.get("injection_retract_mm", 10.0) or 0.0)
            j5_test_active_now     = bool(getattr(state, "j5_test_active", False))
            j5_waiting_flag        = bool(getattr(state, "j5_test_waiting", False))
            tool_center_demo_on    = bool(getattr(state, "tool_center_demo", False))
            tool_center_demo_mode  = getattr(state, "tool_center_demo_mode", "circle")
            tool_center_demo_wait  = bool(getattr(state, "tool_center_demo_waiting", False))
            tool_center_demo_mode  = getattr(state, "tool_center_demo_mode", "circle")

        if syringe_step_ul > 0:
            ammo_steps_exact = max(0.0, float(syringe_remaining_ul) / float(syringe_step_ul))
            ammo_steps = max(0, int(syringe_remaining_ul // syringe_step_ul))
            remainder_ul_for_overlay = max(0.0, syringe_remaining_ul - ammo_steps * syringe_step_ul)
        else:
            ammo_steps_exact = 0.0
            ammo_steps = 0
            remainder_ul_for_overlay = 0.0

        cx, cy = tl_rect.x + tl_w // 2, tl_rect.y + tl_h // 2
        if rs_mode_now == "depth_crop":
            guide_size = int(min(tl_w, tl_h) * 0.34)
            guide = pygame.Rect(0, 0, guide_size, guide_size)
            guide.center = (cx, cy)
            pygame.draw.rect(screen, (0, 255, 127), guide, 2)
            tick = max(14, guide_size // 10)
            pygame.draw.line(screen, (0, 255, 127), (cx - tick, cy), (cx + tick, cy), 2)
            pygame.draw.line(screen, (0, 255, 127), (cx, cy - tick), (cx, cy + tick), 2)
            pygame.draw.line(screen, (0, 180, 160), (guide.left, cy), (guide.right, cy), 1)
            pygame.draw.line(screen, (0, 180, 160), (cx, guide.top), (cx, guide.bottom), 1)
        else:
            pygame.draw.circle(
                screen,
                (0, 255, 127),
                (cx, cy),
                int(min(tl_w, tl_h) * 0.09),
                2,
            )

        rs_clickables = []
        ctrl_font = pygame.font.SysFont("Consolas", 12, bold=True)
        ctrl_pad = 6
        ctrl_h = 24
        ctrl_y = tl_rect.y + 8
        hud_label = f"HUD {'ON' if show_status_overlay else 'OFF'}"
        hud_surf = ctrl_font.render(hud_label, True, (235, 245, 245))
        hud_rect = pygame.Rect(tl_rect.x + 8, ctrl_y, hud_surf.get_width() + ctrl_pad * 2, ctrl_h)
        pygame.draw.rect(screen, (20, 40, 45), hud_rect)
        pygame.draw.rect(screen, (0, 220, 200) if show_status_overlay else (120, 140, 140), hud_rect, 2)
        screen.blit(hud_surf, (hud_rect.x + ctrl_pad, hud_rect.y + (ctrl_h - hud_surf.get_height()) // 2))
        rs_clickables.append((hud_rect, {"kind": "rs_hud_toggle"}))

        mode_label = f"RS {rs_mode_now.replace('_', ' ').upper()}"
        mode_surf = ctrl_font.render(mode_label, True, (235, 245, 245))
        mode_rect = pygame.Rect(hud_rect.right + 8, ctrl_y, mode_surf.get_width() + ctrl_pad * 2, ctrl_h)
        pygame.draw.rect(screen, (20, 40, 45), mode_rect)
        pygame.draw.rect(screen, (0, 220, 200), mode_rect, 2)
        screen.blit(mode_surf, (mode_rect.x + ctrl_pad, mode_rect.y + (ctrl_h - mode_surf.get_height()) // 2))
        rs_clickables.append((mode_rect, {"kind": "rs_mode_cycle"}))

        lf = pygame.font.SysFont("Consolas", 13, bold=True)
        vf_ = pygame.font.SysFont("Consolas", 11)
        yaw_ref = yaw_base if yaw_base is not None else 0.0
        yaw_offset = _norm_angle_deg(yaw_now - yaw_ref)
        base_ref = j5_base_ref if j5_base_ref is not None else j5v
        angle_val = j5v if j5_angle_mode == "actual" else (j5v - base_ref)
        roll_mode_key = (
            str(j5_angle_mode),
            round(float(base_ref), 4) if j5_angle_mode != "actual" else 0.0,
        )
        if roll_visual_angle is None or roll_visual_mode != roll_mode_key:
            roll_visual_angle = float(angle_val)
            roll_visual_mode = roll_mode_key
        else:
            roll_visual_angle += _norm_angle_deg(float(angle_val) - roll_visual_angle)
        angle_draw = roll_visual_angle

        # Compact attitude row: pitch/yaw use the tool-Z vector; roll is joint 5.
        row_gap = max(6, int(tl_w * 0.012))
        card_w = min(160, max(108, (tl_w - 2 * m - 2 * row_gap) // 3))
        card_h = min(88, max(68, int(tl_h * 0.16)))
        row_x = tl_rect.x + m
        row_y = tl_rect.bottom - card_h - m
        card_fill = (5, 18, 24, 190)
        card_edge = (0, 220, 180)
        bright = (0, 255, 127)
        high_angle_visual = bool(
            injection_active_flag
            and not injection_reload_active
            and float(p) > high_angle_threshold_now
        )

        pitch_rect = pygame.Rect(row_x, row_y, card_w, card_h)
        angle_rect = pygame.Rect(pitch_rect.right + row_gap, row_y, card_w, card_h)
        yaw_rect = pygame.Rect(angle_rect.right + row_gap, row_y, card_w, card_h)

        def draw_card_shell(rect, title, value, warn=False):
            panel = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            panel.fill((55, 12, 12, 215) if warn else card_fill)
            screen.blit(panel, rect.topleft)
            draw_edge = (255, 80, 70) if warn else card_edge
            draw_bright = (255, 210, 120) if warn else bright
            pygame.draw.rect(screen, draw_edge, rect, 2 if warn else 1)
            title_surf = lf.render(title, True, draw_bright)
            val_surf = vf_.render(value, True, draw_bright)
            screen.blit(title_surf, (rect.x + 8, rect.y + 5))
            if title_surf.get_width() + val_surf.get_width() + 24 <= rect.w:
                val_pos = (rect.right - val_surf.get_width() - 8, rect.y + 6)
            else:
                val_pos = (rect.x + 8, rect.y + 22)
            screen.blit(val_surf, val_pos)
            return draw_bright

        def draw_action_pill(rect, label, align="right", warn=False):
            txt = vf_.render(label, True, (220, 255, 220))
            pill = pygame.Rect(0, rect.bottom - 24, txt.get_width() + 10, 18)
            if align == "left":
                pill.x = rect.x + 8
            else:
                pill.x = rect.right - pill.w - 8
            pygame.draw.rect(screen, (70, 32, 24) if warn else (25, 48, 44), pill)
            pygame.draw.rect(screen, (255, 210, 120) if warn else bright, pill, 1)
            screen.blit(txt, (pill.x + 5, pill.y + (pill.h - txt.get_height()) // 2))
            return pill

        pitch_bright = draw_card_shell(
            pitch_rect,
            "Pitch",
            f"{p:+.2f}°",
            warn=high_angle_visual,
        )
        pitch_c = (pitch_rect.x + 40, pitch_rect.y + card_h // 2 + 8)
        pitch_len = max(24, min(44, card_w // 3))
        pitch_ang = np.radians(-p)
        pitch_tail = (
            pitch_c[0] - int(pitch_len * 0.45 * np.cos(pitch_ang)),
            pitch_c[1] - int(pitch_len * 0.45 * np.sin(pitch_ang)),
        )
        pitch_tip = (
            pitch_c[0] + int(pitch_len * np.cos(pitch_ang)),
            pitch_c[1] + int(pitch_len * np.sin(pitch_ang)),
        )
        pygame.draw.line(screen, (65, 105, 110), (pitch_rect.x + 14, pitch_c[1]), (pitch_rect.right - 14, pitch_c[1]), 1)
        pygame.draw.circle(screen, (65, 105, 110), pitch_c, 3, 1)
        pygame.draw.line(screen, pitch_bright, pitch_tail, pitch_tip, 2)
        pygame.draw.circle(screen, pitch_bright, pitch_tip, 3)
        draw_action_pill(pitch_rect, "ANGLE" if high_angle_visual else "SET", warn=high_angle_visual)
        rs_clickables.append((pitch_rect, {"kind": "pitch_input"}))

        angle_label = "Roll abs" if j5_angle_mode == "actual" else "Roll rel"
        draw_card_shell(angle_rect, angle_label, f"{angle_val:+.2f}°")
        j5_c = (angle_rect.x + 36, angle_rect.y + card_h // 2 + 6)
        j5_r = max(15, min(22, card_h // 4))
        pygame.draw.circle(screen, (65, 105, 110), j5_c, j5_r, 1)
        pygame.draw.line(screen, (65, 105, 110), j5_c, (j5_c[0] + j5_r, j5_c[1]), 1)
        j5_ang = np.radians(-angle_draw)
        j5_tip = (
            j5_c[0] + int(j5_r * 0.86 * np.cos(j5_ang)),
            j5_c[1] + int(j5_r * 0.86 * np.sin(j5_ang)),
        )
        pygame.draw.line(screen, bright, j5_c, j5_tip, 2)
        pygame.draw.circle(screen, bright, j5_tip, 3)
        mode_rect = draw_action_pill(angle_rect, "ABS/REL", align="left")
        set_rect = draw_action_pill(angle_rect, "SET")
        rs_clickables.append((set_rect, {"kind": "angle_set"}))
        rs_clickables.append((mode_rect, {"kind": "angle_toggle"}))

        yaw_title = "Yaw rel" if yaw_base is not None else "Yaw"
        draw_card_shell(yaw_rect, yaw_title, f"{yaw_offset:+.2f}°")
        compass_c = (yaw_rect.x + 36, yaw_rect.y + card_h // 2 + 6)
        compass_r = max(15, min(22, card_h // 4))
        pygame.draw.circle(screen, (65, 105, 110), compass_c, compass_r, 1)
        pygame.draw.line(screen, (65, 105, 110), compass_c, (compass_c[0] + compass_r, compass_c[1]), 1)
        yaw_ang = np.radians(-yaw_offset)
        yaw_tip = (
            compass_c[0] + int(compass_r * 0.82 * np.cos(yaw_ang)),
            compass_c[1] + int(compass_r * 0.82 * np.sin(yaw_ang)),
        )
        pygame.draw.line(screen, bright, compass_c, yaw_tip, 2)
        pygame.draw.circle(screen, bright, yaw_tip, 3)
        pygame.draw.line(screen, (65, 105, 110), (yaw_rect.x + 70, compass_c[1]), (yaw_rect.right - 12, compass_c[1]), 1)
        yaw_zero_rect = draw_action_pill(yaw_rect, "ZERO", align="left")
        yaw_set_rect = draw_action_pill(yaw_rect, "SET")
        rs_clickables.append((yaw_zero_rect, {"kind": "yaw_base_set"}))
        rs_clickables.append((yaw_set_rect, {"kind": "yaw_set"}))
        rs_clickables.append((yaw_rect, {"kind": "yaw_set"}))

        inj_label = (
            f"Injection: {'ACTIVE' if injection_active_flag else 'idle'}; "
            f"TCP={tcp_label_now}; ammo={ammo_steps_exact:.2f} steps "
            f"(Yoff={tunnel_y_now:+.2f}, retract A={inj_retract_mm_val:.1f} mm)"
        )

        if show_status_overlay:
            max_overlay_w = min(380, max(220, tl_w // 2))
            text_specs = [
                (f"Status: {routine_status_text}", (230, 230, 230)),
                (f"Routine phase: {routine_phase}", (220, 220, 220)),
                (inj_label, (255, 220, 180)),
                (f"Plunger: {last_plunger_note}", (220, 220, 220)),
                (f"Syringe calib: {calibration_status}", (200, 230, 200)),
                (f"Approach: {approach_mm:.2f} mm", (200, 255, 200)),
                (f"Manual speed scale: {_speed_scale_percent_label(sens)}", (200, 230, 255)),
                (
                    f"Fine speed ref: {live_eps_now * 1000.0:.0f} um; command floor: "
                    f"{STABLE_LIVE_LINEAR_EPSILON_MM * 1000.0:.0f} um",
                    (200, 230, 255),
                ),
                (f"Tool translate / d-pad: ±{(5.0 * sens):.3f} mm/s", (0, 255, 127)),
                (f"Roll: ±{(5.0 * sens):.3f} deg/s; yaw: ±{(1.0 * sens):.3f} deg/s", (0, 255, 127)),
                (f"Pitch: ±{(0.5 * sens):.3f} deg/s", (0, 255, 127)),
            ]
            texts = []
            for msg, color in text_specs:
                for line in wrap_text(str(msg), vf_, max_overlay_w):
                    texts.append(vf_.render(line, True, color))
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
            bool(getattr(state, "tool_center_demo", False) or getattr(state, "j5_test_active", False)),
            alarm_threshold_val,
            alarm_duration_val,
        )

        panel_warning_items = build_warning_items(
            alarm_active,
            thermal_hot,
            thermal_critical,
            high_angle_visual,
            final_small_warning_active,
            post_step_prompt_active,
            bool(waiting and at_midway_now),
            injection_reload_active or injection_tunnel_active,
        )
        controller_actions = []
        if ui_nav_active:
            controller_actions.append(("A", "Select focused UI", (200, 95, 255)))
            controller_actions.append(("M", "Exit UI mode", (200, 95, 255)))
        else:
            controller_actions.append(("M", "UI select mode", (200, 95, 255)))
        if alarm_active:
            controller_actions.append(("X", "Clear alarm", (255, 80, 80)))
        elif any(
            str(item.get("key", "")) not in ("alarm", "thermal")
            for item in panel_warning_items
        ):
            controller_actions.append(("X", "Mute warning", (255, 210, 90)))
        if thermal_hot and thermal_status_text and not thermal_warning_dismissed:
            controller_actions.append(("B", "Hide thermal popup", (255, 170, 80)))
        elif waiting:
            controller_actions.append(("B", "Free control", (255, 210, 90)))
        elif not injection_reload_active and not injection_tunnel_active:
            controller_actions.append(("B", "Free control", (180, 210, 220)))

        if injection_active_flag:
            if injection_tunnel_phase_active and tcp_mode_now == "tunnel":
                controller_actions.insert(0, ("A", "Tunnel complete", (0, 255, 127)))
            elif waiting:
                controller_actions.insert(0, ("A", "Advance", (0, 255, 127)))
            if injection_step_in_progress or current_step_recorder:
                controller_actions.append(("Y", "Record outcome", (0, 220, 255)))
        else:
            if waiting:
                controller_actions.insert(0, ("A", "Advance", (0, 255, 127)))

        controller_bar_rect = draw_controller_action_bar(
            screen,
            tr_rect,
            controller_actions,
            position="top",
        )
        controller_bar_reserved = (
            controller_bar_rect.h + 8 if controller_bar_rect is not None else 0
        )
        button_view_rect = pygame.Rect(
            tr_rect.x,
            tr_rect.y + controller_bar_reserved,
            tr_rect.w,
            max(1, tr_rect.h - controller_bar_reserved),
        )

        if ui_nav_active and ui_nav_needs_initial_focus:
            preferred_scroll = estimate_right_panel_scroll_for_kinds(
                buttons,
                (
                    "start_injection",
                    "stop_injection",
                    "injection_record_outcome",
                    "injection_tunnel_complete",
                    "injection_reload",
                    "injection_new_subject",
                    "injection_tunnel_prepare",
                    "injection_tunnel_injector",
                ),
                button_view_rect,
                m_ui,
                btn_w,
                btn_gap,
                label_font,
                placeholder_font,
            )
            if preferred_scroll is not None:
                ui_scroll = preferred_scroll

        prev_clip = screen.get_clip()
        screen.set_clip(tr_rect)

        ui_clickables = []
        y_off_btn = button_view_rect.y + m_ui - ui_scroll
        total_h = 0

        for b in buttons:
            kind = b["kind"]
            if kind == "header":
                header_h = label_font.get_height() + 10
                surf_txt = label_font.render(str(b["label"]).upper(), True, (170, 220, 210))
                screen.blit(surf_txt, (tr_rect.x + m_ui, y_off_btn + 3))
                line_y = y_off_btn + header_h - 3
                pygame.draw.line(
                    screen,
                    (50, 95, 95),
                    (tr_rect.x + m_ui, line_y),
                    (tr_rect.x + tr_rect.w - m_ui, line_y),
                    1,
                )
                total_h += header_h + btn_gap
                y_off_btn += header_h + btn_gap
                continue
            if kind == "top_actions":
                action_h = 38
                gap = 8
                half_w = (btn_w - gap) // 2
                settings_rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, half_w, action_h)
                quit_rect = pygame.Rect(settings_rect.right + gap, y_off_btn, btn_w - half_w - gap, action_h)
                for rect, label, border, fill, action_kind in [
                    (settings_rect, "Settings", (0, 220, 200), (35, 55, 62), "settings_modal"),
                    (quit_rect, "Quit", (255, 100, 100), (65, 40, 45), "quit_app"),
                ]:
                    pygame.draw.rect(screen, fill, rect)
                    pygame.draw.rect(screen, border, rect, 2)
                    txt = label_font.render(label, True, (235, 245, 245))
                    screen.blit(
                        txt,
                        (
                            rect.x + (rect.w - txt.get_width()) // 2,
                            rect.y + (rect.h - txt.get_height()) // 2,
                        ),
                    )
                    ui_clickables.append((rect, {"kind": action_kind}))
                y_off_btn += action_h + btn_gap + 2
                total_h += action_h + btn_gap + 2
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
                    f"{'Current / Run' if is_current else 'Run'}: {rname}",
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
                if bool(b.get("main_only", False)):
                    rect = pygame.Rect(tr_rect.x + m_ui, y_off_btn, btn_w, 34)
                    pygame.draw.rect(screen, (40, 40, 60), rect)
                    pygame.draw.rect(screen, (255, 220, 90), rect, 2)
                    txt_color = (255, 220, 120) if reserved_pose else (225, 230, 230)
                    go_txt_lines = wrap_text(f"Go: {name}", placeholder_font, rect.w - 10)
                    ly2 = rect.y + 4
                    for ln in go_txt_lines:
                        go_txt = placeholder_font.render(ln, True, txt_color)
                        screen.blit(go_txt, (rect.x + 8, ly2))
                        ly2 += placeholder_font.get_height()
                    ui_clickables.append((rect, {"kind": "goto", "pose": name}))
                    y_off_btn += rect.h + btn_gap
                    total_h += rect.h + btn_gap
                    continue
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
            if kind == "start_injection":
                fill_col = (42, 68, 52)
                border_col = (0, 255, 127)
            elif kind in ("stop_injection", "injection_record_outcome"):
                fill_col = (62, 50, 45)
                border_col = (255, 170, 80)
            elif kind == "injection_tunnel_complete":
                fill_col = (48, 66, 42)
                border_col = (130, 255, 90)
            elif kind in ("injection_reload", "injection_tunnel_prepare", "injection_tunnel_injector"):
                fill_col = (42, 52, 66)
                border_col = (0, 220, 200)
            else:
                fill_col = (38, 40, 52)
                border_col = (80, 125, 125)
            pygame.draw.rect(screen, fill_col, rect)
            pygame.draw.rect(screen, border_col, rect, 1)
            ly2 = y_off_btn + 5
            for ln in lines:
                surf_txt = placeholder_font.render(ln, True, (220, 224, 226))
                tx = rect.x + 8
                screen.blit(surf_txt, (tx, ly2))
                ly2 += placeholder_font.get_height() + 2
            ui_clickables.append((rect, b))
            y_off_btn += bh + btn_gap
            total_h   += bh + btn_gap

        ui_content_height = total_h + m_ui

        # Scrollbar
        if ui_content_height > button_view_rect.h:
            view_h = max(1, button_view_rect.h - 2 * m_ui)
            content_h = max(1, ui_content_height - 2 * m_ui)
            bar_h = max(24, int(view_h * (view_h / content_h)))
            max_scroll = max(1, ui_content_height - button_view_rect.h + 12)
            bar_y = button_view_rect.y + m_ui + int(
                (ui_scroll / max_scroll) * (view_h - bar_h)
            )
            bar_x = tr_rect.right - 10
            pygame.draw.rect(
                screen,
                (60, 60, 60),
                (bar_x, button_view_rect.y + m_ui, 6, view_h),
            )
            pygame.draw.rect(screen, (0, 255, 127), (bar_x, bar_y, 6, bar_h))

        screen.set_clip(prev_clip)

        # ── UVC bottom views ──
        uvc_clickables = list(rs_clickables)

        f2_raw = uvc2.latest() if uvc2 else None
        f2 = transform_uvc_frame(f2_raw, u2_rot, u2_hf, u2_vf) if f2_raw is not None else None

        f1_raw = uvc1.latest() if uvc1 else None
        f1 = transform_uvc_frame(f1_raw, u1_rot, u1_hf, u1_vf) if f1_raw is not None else None

        need_demo_tip_tracking = bool(tool_center_demo_on)
        # Needle-tip extraction is relatively expensive and only matters for the
        # tool-center demo traces. Skip it during normal manual control.
        tip_h_norm = (
            detect_needle_tip(f2, entry_side="bottom")
            if (need_demo_tip_tracking and f2 is not None)
            else None
        )
        tip_v_norm = (
            detect_needle_tip(f1, entry_side="auto")
            if (need_demo_tip_tracking and f1 is not None)
            else None
        )

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
        injection_outcome_rect = None
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

            rec_state_text = (
                "REC ON"
                if (current_step_recorder and current_step_recorder.active)
                else "REC OFF"
            )
            txt = small_font.render(
                f"Step {state.current_injection_step}: {frac*100:4.1f}% of {injection_step_volume_ul:.3f} µL   {rec_state_text}",
                True,
                (255, 210, 210) if rec_state_text == "REC ON" else (230, 230, 230),
            )
            screen.blit(txt, (bar_x, bar_y - txt.get_height() - 2))

            btn_w = 176
            btn_h = 28
            btn_x = bar_x + bar_w - btn_w
            btn_y = max(bl_rect.y + 8, bar_y - btn_h - 8)
            injection_outcome_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
            pygame.draw.rect(screen, (45, 55, 70), injection_outcome_rect)
            pygame.draw.rect(screen, (0, 220, 200), injection_outcome_rect, 2)
            btn_txt = small_font.render("Y  Record outcome", True, (235, 245, 245))
            screen.blit(
                btn_txt,
                (
                    injection_outcome_rect.x + (btn_w - btn_txt.get_width()) // 2,
                    injection_outcome_rect.y + (btn_h - btn_txt.get_height()) // 2,
                ),
            )

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
                tcp_mode_tel = str(
                    state.settings.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE)
                    or DEFAULT_TUNNEL_TCP_MODE
                )
                tunnel_tcp_y_tel = float(state.settings.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM) or 0.0)
                tool_cx_tel = float(getattr(state, "tool_cx", state.settings.get("tool_cx", DEFAULT_TOOL_CX)))
                tool_cy_tel = float(getattr(state, "tool_cy", state.settings.get("tool_cy", DEFAULT_TOOL_CY)))
                tool_lz_tel = float(getattr(state, "tool_lz", state.settings.get("tool_lz", DEFAULT_TOOL_LZ)))
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
                frac = current_step_fraction()
                current_step_recorder.log_telemetry(
                    t_rel,
                    joints_snapshot,
                    joystick_tuple,
                    {
                        "distance": injection_step_distance,
                        "step_target": injection_step_target,
                        "step_fraction": frac,
                        "step_volume": injection_step_volume_ul,
                        "delivered_volume": injection_step_volume_ul * frac,
                        "trigger_attenuation": injection_trigger_attenuation,
                        "injection_rate": injection_current_rate,
                        "tcp_mode": tcp_mode_tel,
                        "tunnel_tcp_y_mm": tunnel_tcp_y_tel,
                        "tool_cx": tool_cx_tel,
                        "tool_cy": tool_cy_tel,
                        "tool_lz": tool_lz_tel,
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
            or tunnel_offset_input_active or tunnel_retract_input_active or tunnel_return_input_active or high_angle_input_active
            or naming_input_active or rename_input_active
            or toolpreset_input_active or toolpreset_rename_input_active
            or fwd_rate_input_active or bwd_rate_input_active
            or endstop_timeout_input_active or pose_wait_input_active
            or retract_input_active or routine_name_input_active or settings_path_input_active
            or routine_pose_name_input_active or routine_custom_cmd_input_active or routine_step_edit_active
            or routine_edit_active or routine_builder_active or routine_action_modal_active or routine_edit_modal_active
            or syringe_calib_active or quit_confirm_active
            or injection_info_active or subject_prompt_active
            or material_loaded_warning_active
            or post_step_prompt_active or final_small_warning_active
            or post_outcome_choice_active
            or injection_reload_active or injection_tunnel_active or injection_exit_active
            or settings_modal_active
            or tool_preset_menu_active
            or save_mode_modal_active
        )

        if click_pos is not None and warning_clickables:
            mx_warn, my_warn = click_pos
            for rect, item in warning_clickables:
                if rect.collidepoint(mx_warn, my_warn):
                    key = str(item.get("key", ""))
                    label = str(item.get("label", "warning"))
                    if key:
                        toggle_warning_mute_by_key(key, label)
                    click_pos = None
                    break

        if click_pos is not None and thermal_warning_dismiss_rect is not None:
            if thermal_warning_dismiss_rect.collidepoint(click_pos):
                thermal_warning_dismissed = True
                click_pos = None

        if click_pos is not None and not modals_open:
            mx, my = click_pos

            # RS stop recording
            if rec_stop_rect is not None and rec_stop_rect.collidepoint(mx, my):
                if current_step_recorder:
                    try:
                        log_injection_session_event(
                            "recording_stopped_manual",
                            "Per-attempt recording stopped from the on-screen button.",
                            {"recording_dir": current_step_recorder.step_dir},
                        )
                    except Exception:
                        pass
                    current_step_recorder.stop()
                    current_step_recorder = None
                print("[Recording] Stopped manually via on-screen button.")
                click_pos = None
            elif injection_outcome_rect is not None and injection_outcome_rect.collidepoint(mx, my):
                stop_current_injection_for_outcome()
                print("[Injection] Current attempt stopped via on-screen outcome button.")
                click_pos = None
            else:
                handled = False
                for rect, act in uvc_clickables:
                    if rect.collidepoint(mx, my):
                        if act.get("kind") == "rs_hud_toggle":
                            show_status_overlay = not show_status_overlay
                            handled = True
                            break
                        if act.get("kind") == "rs_mode_cycle":
                            new_mode = rs_thread.cycle_view_mode()
                            print(f"[RS] View mode: {new_mode}")
                            handled = True
                            break
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
                                yaw_now_for_input = float(getattr(state, "yaw", 0.0))
                                yaw_base_for_input = getattr(state, "yaw_base", None)
                            if yaw_base_for_input is not None:
                                yaw_text = f"{_norm_angle_deg(yaw_now_for_input - float(yaw_base_for_input)):.3f}"
                            else:
                                yaw_text = f"{yaw_now_for_input:.3f}"
                            yaw_input_active = True
                            handled = True
                            break
                        if act.get("kind") == "yaw_base_set":
                            with state.lock:
                                state.yaw_base = float(getattr(state, "yaw", 0.0))
                                state.calibration_status = "Yaw zero reference set."
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
                                elif kind == "edit_tunnel_offset":
                                    tunnel_offset_input_active = True
                                    tunnel_offset_text = f"{state.settings.get('tunnel_tcp_y_mm', DEFAULT_TUNNEL_TCP_Y_MM):.3f}"
                                elif kind == "edit_tunnel_retract":
                                    tunnel_retract_input_active = True
                                    tunnel_retract_text = f"{state.settings.get('tunnel_retract_mm', DEFAULT_TUNNEL_RETRACT_MM):.1f}"
                                elif kind == "edit_tunnel_return":
                                    tunnel_return_input_active = True
                                    tunnel_return_text = f"{state.settings.get('tunnel_return_mm', DEFAULT_TUNNEL_RETURN_MM):.1f}"
                                elif kind == "edit_high_angle_warning":
                                    high_angle_input_active = True
                                    high_angle_text = f"{state.settings.get('high_angle_pitch_warning_deg', DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG):.1f}"
                                elif kind == "switch_tunnel_tcp":
                                    rt.enqueue({"type": "safe_tunnel_tcp_shift", "target_mode": "tunnel"})
                                    log_injection_session_event(
                                        "tunnel_tcp_switch_requested",
                                        "Switch to side tunnel needle TCP",
                                        {"target_tcp_mode": "tunnel"},
                                    )
                                    try:
                                        joy.rumble(0.45, 0.45, 180)
                                    except Exception:
                                        pass
                                    click_pos = None
                                elif kind == "switch_injector_tcp":
                                    rt.enqueue({"type": "safe_tunnel_tcp_shift", "target_mode": "injector"})
                                    log_injection_session_event(
                                        "tunnel_tcp_switch_requested",
                                        "Switch back to injector TCP",
                                        {"target_tcp_mode": "injector"},
                                    )
                                    try:
                                        joy.rumble(0.45, 0.45, 180)
                                    except Exception:
                                        pass
                                    click_pos = None
                                elif kind == "startup_calibration_stop":
                                    rt.tool_center_demo_stop.set()
                                    rt.enqueue({"type": "tool_center_demo_stop"})
                                    rt.enqueue({"type": "j5_test_toggle", "start": False})
                                    with state.lock:
                                        state.tool_center_demo_waiting = False
                                        state.tool_center_demo = False
                                        state.j5_test_active = False
                                        state.j5_test_waiting = False
                                    needle_trace_h.clear()
                                    needle_trace_v.clear()
                                    click_pos = None
                                elif kind == "fixed_tip_cone_demo":
                                    rt.enqueue({"type": "fixed_tip_cone_demo", "diameter_mm": 50.0, "rotations": 5})
                                    try:
                                        joy.rumble(0.35, 0.35, 120)
                                    except Exception:
                                        pass
                                    click_pos = None
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
                                elif kind == "settings_poses":
                                    settings_modal_active = True
                                    settings_modal_tab = "poses"
                                    settings_modal_scroll = 0
                                    click_pos = None
                                elif kind == "settings_routines":
                                    settings_modal_active = True
                                    settings_modal_tab = "routine"
                                    settings_modal_scroll = 0
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
                                    request_quit_confirmation()
                                    click_pos = None
                                elif kind == "settings_modal":
                                    settings_modal_active = True
                                    settings_modal_tab = "tool"
                                    settings_modal_scroll = 0
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
                                    post_step_prompt_active = False
                                    post_outcome_choice_active = False
                                    post_outcome_choice_just_opened = False
                                    injection_reload_active = False
                                    injection_reload_started = False
                                    injection_reload_before_ul = 0.0
                                    injection_reload_stage = "precheck"
                                    material_loaded_warning_active = False
                                    injection_tunnel_active = False
                                    injection_tunnel_stage = "confirm"
                                    injection_exit_active = False
                                    final_small_warning_active = False
                                    subject_prompt_active = False
                                    syringe_calib_active = True
                                    syringe_field_active = "vol"
                                    click_pos = None
                                elif kind == "edit_retract":
                                    retract_input_active = True
                                    retract_text = f"{state.settings.get('injection_retract_mm', 10.0):.1f}"
                                elif kind == "start_injection":
                                    post_step_prompt_active = False
                                    post_outcome_choice_active = False
                                    post_outcome_choice_just_opened = False
                                    injection_reload_active = False
                                    injection_reload_started = False
                                    injection_reload_before_ul = 0.0
                                    injection_reload_stage = "precheck"
                                    injection_tunnel_active = False
                                    injection_tunnel_stage = "confirm"
                                    final_small_warning_active = False
                                    subject_prompt_active = False
                                    injection_step_in_progress = False
                                    injection_step_done = False
                                    current_step_outcome_marked = False
                                    injection_info_active = True
                                    injection_field_active = "study"
                                    injection_date_text = current_injection_timestamp_text()
                                    injection_record_telemetry = True
                                    injection_record_video = True
                                    injection_record_ui = True
                                    injection_use_tunnel_workflow = bool(
                                        state.settings.get("injection_use_tunnel_workflow", False)
                                    )
                                    injection_material_loaded = bool(
                                        state.settings.get("injection_material_loaded", False)
                                    )
                                    click_pos = None
                                elif kind == "stop_injection":
                                    open_injection_exit_modal()
                                    click_pos = None
                                elif kind == "injection_record_outcome":
                                    if injection_step_in_progress or current_step_recorder:
                                        stop_current_injection_for_outcome()
                                    else:
                                        with state.lock:
                                            state.calibration_status = "No active injection step to stop."
                                    click_pos = None
                                elif kind == "injection_reload":
                                    open_injection_reload_modal()
                                    click_pos = None
                                elif kind == "injection_tunnel_prepare":
                                    open_injection_tunnel_modal("tunnel")
                                    click_pos = None
                                elif kind == "injection_tunnel_injector":
                                    open_injection_tunnel_modal("injector")
                                    click_pos = None
                                elif kind == "injection_tunnel_complete":
                                    complete_tunnel_phase_to_injector()
                                    click_pos = None
                                elif kind == "injection_new_subject":
                                    if injection_step_in_progress or current_step_recorder:
                                        stop_current_injection_for_outcome()
                                    else:
                                        subject_field_text = ""
                                        subject_eye_choice = "OD"
                                        with state.lock:
                                            subject_tunnel_workflow = bool(
                                                state.settings.get("injection_use_tunnel_workflow", False)
                                            )
                                        subject_prompt_active = True
                                    click_pos = None
                                break

        if settings_modal_active:
            with state.lock:
                settings_for_modal = state.settings.copy()
                poses_for_modal = dict(state.poses)
                tool_lz_modal = state.tool_lz
                approach_modal = state.approach_mm
                calibration_active_modal = bool(
                    getattr(state, "tool_center_demo", False)
                    or getattr(state, "j5_test_active", False)
                )
            modal_items = make_settings_items(
                settings_modal_tab,
                tool_lz_modal,
                approach_modal,
                poses_for_modal,
                settings_for_modal,
                routines,
                current_routine_name,
                calibration_active_modal,
                alarm_threshold_val,
                alarm_duration_val,
            )
            tab_rects, settings_clickables, settings_close_rect, settings_panel, settings_max_scroll = draw_settings_modal(
                screen,
                settings_modal_tab,
                modal_items,
                alarm_threshold_val,
                alarm_duration_val,
                settings_modal_scroll,
            )
            settings_modal_scroll = max(0, min(settings_modal_scroll, settings_max_scroll))
            if b_edge:
                settings_modal_active = False
                click_pos = None
            if click_pos is not None:
                mx, my = click_pos
                handled_settings = False
                if settings_close_rect.collidepoint(mx, my) or not settings_panel.collidepoint(mx, my):
                    settings_modal_active = False
                    handled_settings = True
                else:
                    for tab_key, tab_rect in tab_rects.items():
                        if tab_rect.collidepoint(mx, my):
                            settings_modal_tab = tab_key
                            settings_modal_scroll = 0
                            handled_settings = True
                            break
                if not handled_settings:
                    for rect, act in settings_clickables:
                        if not rect.collidepoint(mx, my):
                            continue
                        kind = act.get("kind")
                        if kind == "edit_tool":
                            toollen_input_active = True
                            toollen_text = f"{state.tool_lz:.1f}"
                            settings_modal_active = False
                        elif kind == "edit_toolcx":
                            toolcx_input_active = True
                            toolcx_text = f"{getattr(state, 'tool_cx', DEFAULT_TOOL_CX):.3f}"
                            settings_modal_active = False
                        elif kind == "edit_toolcy":
                            toolcy_input_active = True
                            toolcy_text = f"{getattr(state, 'tool_cy', DEFAULT_TOOL_CY):.3f}"
                            settings_modal_active = False
                        elif kind == "edit_approach":
                            approach_input_active = True
                            approach_text = f"{state.approach_mm:.1f}"
                            settings_modal_active = False
                        elif kind == "edit_tunnel_offset":
                            tunnel_offset_input_active = True
                            tunnel_offset_text = f"{state.settings.get('tunnel_tcp_y_mm', DEFAULT_TUNNEL_TCP_Y_MM):.3f}"
                            settings_modal_active = False
                        elif kind == "edit_tunnel_retract":
                            tunnel_retract_input_active = True
                            tunnel_retract_text = f"{state.settings.get('tunnel_retract_mm', DEFAULT_TUNNEL_RETRACT_MM):.1f}"
                            settings_modal_active = False
                        elif kind == "edit_tunnel_return":
                            tunnel_return_input_active = True
                            tunnel_return_text = f"{state.settings.get('tunnel_return_mm', DEFAULT_TUNNEL_RETURN_MM):.1f}"
                            settings_modal_active = False
                        elif kind == "edit_high_angle_warning":
                            high_angle_input_active = True
                            high_angle_text = f"{state.settings.get('high_angle_pitch_warning_deg', DEFAULT_HIGH_ANGLE_PITCH_WARNING_DEG):.1f}"
                            settings_modal_active = False
                        elif kind == "tool_preset_save":
                            toolpreset_input_active = True
                            toolpreset_name_text = ""
                            settings_modal_active = False
                        elif kind == "tool_preset_apply":
                            preset_name = str(act.get("preset", "")).strip()
                            if preset_name:
                                rt.enqueue({"type": "apply_tool_preset", "name": preset_name})
                            settings_modal_active = False
                        elif kind == "tool_preset_menu":
                            preset_name = str(act.get("preset", "")).strip()
                            if preset_name:
                                tool_preset_menu_active = True
                                tool_preset_menu_target = preset_name
                            settings_modal_active = False
                        elif kind == "new_pose":
                            naming_input_active = True
                            naming_text = ""
                            settings_modal_active = False
                        elif kind == "goto":
                            pose_name = str(act.get("pose", "")).strip()
                            if pose_name:
                                rt.enqueue({"type": "goto", "pose_name": pose_name})
                                try:
                                    joy.rumble(0.4, 0.4, 120)
                                except Exception:
                                    pass
                            settings_modal_active = False
                        elif kind == "edit_menu":
                            edit_menu_active = True
                            edit_menu_target = str(act.get("pose", "")).strip()
                            edit_menu_target_reserved = bool(act.get("reserved", False))
                            settings_modal_active = False
                        elif kind == "pose_star_toggle":
                            pose_name = str(act.get("pose", "")).strip()
                            if pose_name:
                                with state.lock:
                                    starred = list(state.settings.get("starred_poses", []))
                                    if pose_name in starred:
                                        starred = [x for x in starred if x != pose_name]
                                        state.routine_status_text = f"Unstarred pose '{pose_name}'"
                                    else:
                                        starred.append(pose_name)
                                        state.routine_status_text = f"Starred pose '{pose_name}'"
                                    state.settings["starred_poses"] = starred
                                save_settings(state.settings)
                        elif kind == "routine_new":
                            routine_name_mode = "create"
                            routine_name_text = ""
                            routine_rename_target = ""
                            routine_name_input_active = True
                            settings_modal_active = False
                        elif kind == "routine_new_from_starred":
                            routine_name_mode = "create_starred"
                            routine_name_text = ""
                            routine_rename_target = ""
                            routine_name_input_active = True
                            settings_modal_active = False
                        elif kind == "routine_new_from_all":
                            routine_name_mode = "create_all"
                            routine_name_text = ""
                            routine_rename_target = ""
                            routine_name_input_active = True
                            settings_modal_active = False
                        elif kind == "routine_builder_current":
                            persist_current_routine()
                            routine_builder_active = True
                            routine_builder_scroll = 0
                            settings_modal_active = False
                        elif kind == "routine_edit_current":
                            persist_current_routine()
                            routine_edit_active = True
                            settings_modal_active = False
                        elif kind == "routine_run_current":
                            routine_action_target = current_routine_name
                            routine_action_modal_active = True
                            settings_modal_active = False
                        elif kind == "routine_append":
                            append_to_current_routine(
                                act.get("snippet", ""),
                                act.get("note", "routine step"),
                            )
                        elif kind == "routine_append_pose":
                            pose_name = str(act.get("pose", "")).strip()
                            if pose_name:
                                append_to_current_routine(
                                    f"POSE {pose_name}",
                                    f"POSE {pose_name}",
                                )
                        elif kind == "routine_make_current":
                            set_current_routine(str(act.get("routine", "")).strip())
                        elif kind == "routine_run":
                            rname = str(act.get("routine", "")).strip()
                            if rname:
                                routine_action_target = rname
                                routine_action_modal_active = True
                                settings_modal_active = False
                        elif kind == "routine_edit_row":
                            rname = str(act.get("routine", "")).strip()
                            if rname:
                                routine_edit_modal_target = rname
                                routine_edit_modal_active = True
                                settings_modal_active = False
                        elif kind == "switch_tunnel_tcp":
                            rt.enqueue({"type": "safe_tunnel_tcp_shift", "target_mode": "tunnel"})
                            log_injection_session_event(
                                "tunnel_tcp_switch_requested",
                                "Switch to side tunnel needle TCP from settings.",
                                {"target_tcp_mode": "tunnel"},
                            )
                            settings_modal_active = False
                        elif kind == "switch_injector_tcp":
                            rt.enqueue({"type": "safe_tunnel_tcp_shift", "target_mode": "injector"})
                            log_injection_session_event(
                                "tunnel_tcp_switch_requested",
                                "Switch back to injector TCP from settings.",
                                {"target_tcp_mode": "injector"},
                            )
                            settings_modal_active = False
                        elif kind == "startup_calibration_stop":
                            rt.tool_center_demo_stop.set()
                            rt.enqueue({"type": "tool_center_demo_stop"})
                            rt.enqueue({"type": "j5_test_toggle", "start": False})
                            with state.lock:
                                state.tool_center_demo_waiting = False
                                state.tool_center_demo = False
                                state.j5_test_active = False
                                state.j5_test_waiting = False
                            needle_trace_h.clear()
                            needle_trace_v.clear()
                            settings_modal_active = False
                        elif kind == "fixed_tip_cone_demo":
                            rt.enqueue({"type": "fixed_tip_cone_demo", "diameter_mm": 50.0, "rotations": 5})
                            settings_modal_active = False
                        elif kind == "syringe_calib":
                            post_step_prompt_active = False
                            post_outcome_choice_active = False
                            post_outcome_choice_just_opened = False
                            injection_reload_active = False
                            injection_reload_started = False
                            injection_reload_before_ul = 0.0
                            injection_reload_stage = "precheck"
                            injection_tunnel_active = False
                            injection_tunnel_stage = "confirm"
                            injection_exit_active = False
                            final_small_warning_active = False
                            subject_prompt_active = False
                            syringe_calib_active = True
                            syringe_field_active = "vol"
                            settings_modal_active = False
                        elif kind == "edit_retract":
                            retract_input_active = True
                            retract_text = f"{state.settings.get('injection_retract_mm', 10.0):.1f}"
                            settings_modal_active = False
                        elif kind == "edit_fwd_rate":
                            fwd_rate_input_active = True
                            fwd_rate_text = str(state.settings.get("plunger_fwd_rate", MAX_RATE))
                            settings_modal_active = False
                        elif kind == "edit_bwd_rate":
                            bwd_rate_input_active = True
                            bwd_rate_text = str(state.settings.get("plunger_bwd_rate", MAX_RATE))
                            settings_modal_active = False
                        elif kind == "edit_endstop_timeout":
                            endstop_timeout_input_active = True
                            endstop_timeout_text = str(state.settings.get("endstop_timeout_ms", 0))
                            settings_modal_active = False
                        elif kind == "edit_pose_wait":
                            pose_wait_input_active = True
                            pose_wait_text = f"{state.settings.get('pose_final_wait_s', 0.0):.1f}"
                            settings_modal_active = False
                        elif kind == "alarm_arm_now":
                            rt.enqueue({
                                "type": "set_alarm_pid",
                                "threshold": alarm_threshold_val,
                                "duration": alarm_duration_val,
                            })
                        elif kind == "alarm_auto_tune":
                            rt.enqueue({
                                "type": "auto_tune_alarm_pid",
                                "threshold": alarm_threshold_val,
                                "duration": alarm_duration_val,
                            })
                        elif kind == "alarm_disarm_now":
                            rt.enqueue({"type": "disarm_alarm"})
                        elif kind in ("alarm_threshold", "alarm_duration"):
                            bar = act.get("rect")
                            if bar and bar.w > 0:
                                rel = max(0.0, min(1.0, (mx - bar.x) / bar.w))
                                if kind == "alarm_threshold":
                                    alarm_threshold_val = DEFAULT_PID_THRESHOLD_MIN + rel * (
                                        DEFAULT_PID_THRESHOLD_MAX - DEFAULT_PID_THRESHOLD_MIN
                                    )
                                else:
                                    alarm_duration_val = DEFAULT_PID_DURATION_MIN + rel * (
                                        DEFAULT_PID_DURATION_MAX - DEFAULT_PID_DURATION_MIN
                                    )
                                alarm_threshold_val, alarm_duration_val = _clamp_alarm_pid(
                                    alarm_threshold_val,
                                    alarm_duration_val,
                                )
                                with state.lock:
                                    state.settings["alarm_threshold"] = alarm_threshold_val
                                    state.settings["alarm_duration"] = alarm_duration_val
                                save_settings(state.settings)
                                rt.enqueue({
                                    "type": "set_alarm_pid",
                                    "threshold": alarm_threshold_val,
                                    "duration": alarm_duration_val,
                                })
                        elif kind == "jog_epsilon_preset":
                            value = _clamp_live_linear_epsilon_mm(
                                act.get("value", DEFAULT_LIVE_LINEAR_EPSILON_MM)
                            )
                            with state.lock:
                                state.settings["live_linear_epsilon_mm"] = value
                                _apply_manual_speed_levels_for_epsilon_locked(state, value)
                                state.calibration_status = (
                                    f"Fine speed reference set to {_live_linear_epsilon_label(value)}; "
                                    f"live command floor held at {STABLE_LIVE_LINEAR_EPSILON_MM * 1000.0:.0f} um."
                                )
                            save_settings(state.settings)
                            rt.enqueue({"type": "set_live_linear_epsilon", "value": value})
                        elif kind == "tts_voice_cycle":
                            with state.lock:
                                next_voice = _next_tts_voice_type(
                                    state.settings.get("tts_voice_type", DEFAULT_TTS_VOICE_TYPE)
                                )
                                state.settings["tts_voice_type"] = next_voice
                                state.calibration_status = f"Voice type set to {_tts_voice_label(next_voice)}."
                                state.audio_cue = "tts_test"
                            save_settings(state.settings)
                        elif kind == "tts_rate_cycle":
                            with state.lock:
                                current_rate = int(state.settings.get("tts_rate", DEFAULT_TTS_RATE))
                                rate_choices = [-30, -10, 0, 15, 30]
                                try:
                                    idx_rate = rate_choices.index(current_rate)
                                except ValueError:
                                    idx_rate = 1
                                next_rate = rate_choices[(idx_rate + 1) % len(rate_choices)]
                                state.settings["tts_rate"] = next_rate
                                state.calibration_status = f"Voice rate set to {next_rate:+d}."
                                state.audio_cue = "tts_test"
                            save_settings(state.settings)
                        elif kind == "tts_test":
                            with state.lock:
                                state.audio_cue = "tts_test"
                        elif kind == "tts_all_test":
                            play_all_warning_demos()
                        elif kind == "aircraft_callouts_toggle":
                            with state.lock:
                                current = bool(
                                    state.settings.get(
                                        "aircraft_callouts_enabled",
                                        DEFAULT_AIRCRAFT_CALLOUTS_ENABLED,
                                    )
                                )
                                next_enabled = not current
                                state.settings["aircraft_callouts_enabled"] = next_enabled
                                state.calibration_status = (
                                    "Aircraft-style safety callouts enabled."
                                    if next_enabled
                                    else "Aircraft-style safety callouts disabled."
                                )
                                if next_enabled:
                                    state.audio_cue = "gpws_test"
                            save_settings(state.settings)
                        elif kind == "gpws_test":
                            with state.lock:
                                state.audio_cue = "gpws_test"
                        elif kind == "settings_load":
                            settings_path_mode = "load"
                            settings_path_text = SETTINGS_PATH
                            settings_path_input_active = True
                            settings_modal_active = False
                        elif kind == "settings_save":
                            settings_path_mode = "save"
                            settings_path_text = SETTINGS_PATH
                            settings_path_input_active = True
                            settings_modal_active = False
                        handled_settings = True
                        break
                click_pos = None

        if routine_edit_active:
            script_panel, _, _, script_clickables = draw_routine_editor(screen, routine_editor)
            if click_pos is not None:
                mx, my = click_pos
                for rect, act in script_clickables:
                    if not rect.collidepoint(mx, my):
                        continue
                    kind = act.get("kind")
                    if kind == "routine_script_builder":
                        persist_current_routine()
                        routine_edit_active = False
                        routine_builder_active = True
                        routine_builder_scroll = 0
                    elif kind == "routine_script_apply":
                        persist_current_routine()
                        routine_edit_active = False
                    elif kind == "routine_script_close":
                        routine_edit_active = False
                    break
                click_pos = None

        if routine_builder_active:
            with state.lock:
                builder_poses = dict(state.poses)
            builder_pose_names = sorted_pose_names(builder_poses)
            builder_clickables, builder_close_rect, builder_panel, builder_max_scroll = (
                draw_routine_builder_modal(
                    screen,
                    current_routine_name,
                    routine_editor.text,
                    builder_pose_names,
                    routine_builder_scroll,
                )
            )
            routine_builder_scroll = max(0, min(routine_builder_scroll, builder_max_scroll))
            builder_text_input_open = routine_pose_name_input_active or routine_custom_cmd_input_active or routine_step_edit_active
            if b_edge and not builder_text_input_open:
                routine_builder_active = False
                click_pos = None
            if click_pos is not None and not builder_text_input_open:
                mx, my = click_pos
                if builder_close_rect.collidepoint(mx, my) or not builder_panel.collidepoint(mx, my):
                    routine_builder_active = False
                else:
                    for rect, act in builder_clickables:
                        if not rect.collidepoint(mx, my):
                            continue
                        kind = act.get("kind")
                        if kind == "routine_builder_close":
                            routine_builder_active = False
                        elif kind == "routine_builder_script":
                            persist_current_routine()
                            routine_edit_active = True
                            routine_builder_active = False
                        elif kind == "routine_builder_run":
                            routine_action_target = current_routine_name
                            routine_action_modal_active = True
                            routine_builder_active = False
                        elif kind == "routine_builder_add_pose":
                            pose_name = str(act.get("pose", "")).strip()
                            if pose_name:
                                append_to_current_routine(
                                    f"POSE {pose_name}",
                                    f"POSE {pose_name}",
                                )
                        elif kind == "routine_builder_append":
                            append_to_current_routine(
                                act.get("snippet", ""),
                                act.get("note", "routine step"),
                            )
                        elif kind == "routine_builder_save_pose":
                            routine_pose_name_input_active = True
                            routine_pose_name_text = ""
                        elif kind == "routine_builder_custom":
                            routine_custom_cmd_input_active = True
                            routine_custom_cmd_text = ""
                        elif kind == "routine_builder_edit_node":
                            routine_step_edit_active = True
                            routine_step_edit_line_index = int(act.get("line_index", -1))
                            routine_step_edit_text = str(act.get("line", ""))
                        elif kind == "routine_builder_delete_node":
                            delete_routine_line(int(act.get("line_index", -1)))
                        elif kind == "routine_builder_move_node":
                            move_routine_line(
                                int(act.get("line_index", -1)),
                                int(act.get("delta", 0)),
                            )
                        elif kind == "routine_builder_clear":
                            clear_current_routine_to_blank()
                        break
                click_pos = None

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
        if tunnel_offset_input_active:
            draw_center_input_box(
                screen, "Tunnel side needle Y offset (mm)", tunnel_offset_text
            )
        if tunnel_retract_input_active:
            draw_center_input_box(
                screen, "Tunnel TCP safety retract (mm)", tunnel_retract_text
            )
        if tunnel_return_input_active:
            draw_center_input_box(
                screen, "Tunnel TCP guided return (mm)", tunnel_return_text
            )
        if high_angle_input_active:
            draw_center_input_box(
                screen, "High-angle warning pitch (deg)", high_angle_text
            )
        if yaw_input_active:
            with state.lock:
                yaw_base_for_label = getattr(state, "yaw_base", None)
            draw_center_input_box(
                screen,
                "Set yaw relative to ZERO (deg)" if yaw_base_for_label is not None else "Set yaw heading (deg)",
                yaw_text,
            )
        if angle_input_active:
            draw_center_input_box(
                screen, "Set Roll angle (deg)", angle_text
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
                 "Create routine from starred poses" if routine_name_mode == "create_starred" else
                 "Create routine from all saved poses" if routine_name_mode == "create_all" else
                 "Create new routine" if routine_name_mode == "create" else
                 "Save routine as…"),
                routine_name_text,
            )
        if routine_pose_name_input_active:
            draw_center_input_box(
                screen,
                "Save current robot pose and add to routine",
                routine_pose_name_text,
            )
        if routine_custom_cmd_input_active:
            draw_center_input_box(
                screen,
                "Add custom routine command",
                routine_custom_cmd_text,
            )
        if routine_step_edit_active:
            draw_center_input_box(
                screen,
                "Edit routine step command",
                routine_step_edit_text,
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
            for key, rect in field_rects.items():
                add_nav_item(frame_nav_items, rect, f"Syringe calibration field: {key}", group="modal")
            for key, rect in button_rects.items():
                add_nav_item(frame_nav_items, rect, f"Syringe calibration: {key}", group="modal")
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
                    elif button_rects["vol_minus"].collidepoint(mx, my):
                        syringe_field_active = "vol"
                        syringe_vol_text = adjust_numeric_text(syringe_vol_text, -0.5)
                    elif button_rects["vol_plus"].collidepoint(mx, my):
                        syringe_field_active = "vol"
                        syringe_vol_text = adjust_numeric_text(syringe_vol_text, 0.5)
                    elif button_rects["step_minus"].collidepoint(mx, my):
                        syringe_field_active = "step"
                        syringe_step_ul_txt = adjust_numeric_text(syringe_step_ul_txt, -0.1)
                    elif button_rects["step_plus"].collidepoint(mx, my):
                        syringe_field_active = "step"
                        syringe_step_ul_txt = adjust_numeric_text(syringe_step_ul_txt, 0.1)
                click_pos = None

        # Injection metadata modal
        if injection_info_active:
            with state.lock:
                s_info = state.settings
                info_step_time = float(s_info.get("syringe_step_time_s", 0.0) or 0.0)
                info_vol_ul = float(s_info.get("syringe_volume_ul", 0.0) or 0.0)
            info_syringe_calibrated = info_step_time > 0.0 and info_vol_ul > 0.0
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
                    injection_use_tunnel_workflow,
                    injection_material_loaded,
                    info_syringe_calibrated,
                )
            )
            for key, rect in field_rects2.items():
                add_nav_item(frame_nav_items, rect, f"Injection setup field: {key}", group="modal")
            for key, rect in checkbox_rects2.items():
                add_nav_item(frame_nav_items, rect, f"Injection setup toggle: {key}", group="modal")
            for key, rect in button_rects2.items():
                add_nav_item(frame_nav_items, rect, f"Injection setup: {key}", group="modal")
            if click_pos is not None and not material_loaded_warning_active:
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
                    elif checkbox_rects2["material_loaded"].collidepoint(mx, my):
                        injection_material_loaded = not injection_material_loaded
                    elif checkbox_rects2["tunnel"].collidepoint(mx, my):
                        injection_use_tunnel_workflow = not injection_use_tunnel_workflow
                    elif button_rects2["cancel"].collidepoint(mx, my):
                        injection_info_active = False
                    elif button_rects2["start"].collidepoint(mx, my):
                        start_injection_session()
                click_pos = None
            if not material_loaded_warning_active:
                if x_edge and not x_consumed:
                    injection_material_loaded = not injection_material_loaded
                    with state.lock:
                        state.calibration_status = (
                            "Material loaded confirmed."
                            if injection_material_loaded
                            else "Material loaded confirmation cleared."
                        )
                elif y_edge and not y_consumed:
                    injection_use_tunnel_workflow = not injection_use_tunnel_workflow
                    with state.lock:
                        state.calibration_status = (
                            "Tunnel workflow enabled for this session."
                            if injection_use_tunnel_workflow
                            else "Tunnel workflow disabled for this session."
                        )
                elif a_edge and not a_consumed:
                    start_injection_session()
                elif b_edge and not b_consumed:
                    injection_info_active = False

        if material_loaded_warning_active:
            warning_buttons, warning_panel = draw_material_loaded_warning_modal(screen)
            for key, rect in warning_buttons.items():
                add_nav_item(frame_nav_items, rect, f"Loaded material warning: {key}", group="modal")
            warning_choice = None
            if click_pos is not None:
                mx, my = click_pos
                if warning_buttons["proceed"].collidepoint(mx, my):
                    warning_choice = "proceed"
                elif warning_buttons["cancel"].collidepoint(mx, my) or not warning_panel.collidepoint(mx, my):
                    warning_choice = "cancel"
                click_pos = None
            if a_edge:
                warning_choice = "proceed"
            elif b_edge:
                warning_choice = "cancel"

            if warning_choice == "proceed":
                material_loaded_warning_active = False
                start_injection_session(confirm_unloaded=True)
            elif warning_choice == "cancel":
                material_loaded_warning_active = False

        # Animal / eye / sample popup (next step; includes Confirm & Advance)
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
            allow_advance = (
                prev_step_idx > 0
                and advance_mm > 0.0
                and not subject_tunnel_workflow
            )

            box_s, btns_s, controls_s, panel_s = draw_subject_modal(
                screen,
                subject_field_text,
                subject_eye_choice,
                subject_tunnel_workflow,
                step_idx,
                allow_advance,
            )
            add_nav_item(frame_nav_items, box_s, "Animal/sample ID field", group="modal")
            for key, rect in controls_s.items():
                add_nav_item(frame_nav_items, rect, f"Animal/sample control: {key}", group="modal")
            for key, rect in btns_s.items():
                if rect is not None:
                    add_nav_item(frame_nav_items, rect, f"Animal/sample: {key}", group="modal")

            choice = None

            if click_pos is not None:
                mx, my = click_pos
                if not panel_s.collidepoint(mx, my):
                    subject_prompt_active = False
                else:
                    if box_s.collidepoint(mx, my):
                        pass
                    elif controls_s["od"].collidepoint(mx, my):
                        subject_eye_choice = "OD"
                    elif controls_s["os"].collidepoint(mx, my):
                        subject_eye_choice = "OS"
                    elif controls_s["tunnel"].collidepoint(mx, my):
                        subject_tunnel_workflow = not subject_tunnel_workflow
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
            elif x_edge:
                subject_eye_choice = "OS" if subject_eye_choice == "OD" else "OD"
            elif y_edge:
                subject_tunnel_workflow = not subject_tunnel_workflow

            if choice in ("confirm", "confirm_adv"):
                subj = compose_subject_eye_text(subject_field_text, subject_eye_choice)
                injection_subject_text = subj
                with state.lock:
                    state.settings["injection_subject"] = subj
                    state.settings["injection_subject_id"] = subject_field_text.strip()
                    state.settings["injection_eye"] = subject_eye_choice
                    state.settings["injection_use_tunnel_workflow"] = bool(subject_tunnel_workflow)
                    state.current_injection_step = step_idx
                save_settings(state.settings)

                pending_step_index = step_idx
                injection_step_in_progress = False
                injection_step_done = False
                injection_step_distance = 0.0
                injection_step_target = 0.0
                injection_step_volume_ul = 0.0
                current_step_outcome_marked = False
                start_step_recording(step_idx)

                if choice == "confirm_adv" and advance_mm > 0.0:
                    rt.enqueue({"type": "inject_advance", "dist": advance_mm})

                subject_prompt_active = False
                if subject_tunnel_workflow:
                    begin_tunnel_phase_for_current_subject()

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
                ("Visual builder", "builder"),
                ("Edit script", "edit"),
                ("Rename", "rename"),
                ("Delete", "delete"),
                ("Close", "close"),
            ]
            btn_w = pan_w - 40
            btn_h = 32
            gap = 8
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
                    if buttons_modal["builder"].collidepoint(mx, my):
                        txt = routines.get(routine_edit_modal_target, ROUTINE_SAMPLE.strip())
                        current_routine_name = routine_edit_modal_target
                        routine_editor = EditorState(text=txt, caret=len(txt))
                        routine_builder_active = True
                        routine_builder_scroll = 0
                        routine_edit_modal_active = False
                    elif buttons_modal["edit"].collidepoint(mx, my):
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
            pan_w = min(660, max(520, screen_w - 80))
            pan_h = 292
            px = (screen_w - pan_w) // 2
            py = (screen_h - pan_h) // 2
            panel = pygame.Rect(px, py, pan_w, pan_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (255, 80, 80), panel, 2)
            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            small_modal = pygame.font.SysFont("Consolas", 14)
            title = font_modal.render("Exit Robot Interface?", True, (245, 235, 235))
            screen.blit(title, (px + 16, py + 16))

            info_rect = pygame.Rect(px + 16, py + 54, pan_w - 32, 132)
            pygame.draw.rect(screen, (24, 24, 32), info_rect)
            pygame.draw.rect(screen, (110, 80, 80), info_rect, 1)
            info_lines = [
                "This performs a clean shutdown of the interface.",
                "Motion is halted, active recordings are closed, and injection mode is cleared for a clean restart.",
                "Motors are not powered off by this button.",
            ]
            dy = info_rect.y + 10
            for msg in info_lines:
                for ln in wrap_lines(small_modal, msg, info_rect.w - 20):
                    surf = small_modal.render(ln, True, (225, 225, 220))
                    if dy + surf.get_height() <= info_rect.bottom - 6:
                        screen.blit(surf, (info_rect.x + 10, dy))
                    dy += surf.get_height() + 4
                dy += 2

            hint = small_modal.render("Esc/Q or B keeps the app running. Enter or A exits cleanly.", True, (210, 210, 210))
            screen.blit(hint, (px + 16, info_rect.bottom + 12))

            btn_w, btn_h, gap = min(250, (pan_w - 48) // 2), 46, 14
            bx = px + (pan_w - (btn_w * 2 + gap)) // 2
            by = py + pan_h - btn_h - 16
            cancel_rect = pygame.Rect(bx, by, btn_w, btn_h)
            quit_rect = pygame.Rect(cancel_rect.right + gap, by, btn_w, btn_h)
            draw_modal_button(
                screen,
                cancel_rect,
                "B  Keep Running",
                font_modal,
                border=(0, 255, 127),
                fill=(45, 70, 55),
            )
            draw_modal_button(
                screen,
                quit_rect,
                "A  Exit Cleanly",
                font_modal,
                border=(255, 80, 80),
                fill=(75, 45, 50),
            )
            add_nav_item(frame_nav_items, cancel_rect, "Keep application running", group="modal")
            add_nav_item(frame_nav_items, quit_rect, "Exit application cleanly", group="modal")

            if click_pos is not None:
                mx, my = click_pos
                if cancel_rect.collidepoint(mx, my) or not panel.collidepoint(mx, my):
                    quit_confirm_active = False
                elif quit_rect.collidepoint(mx, my):
                    perform_graceful_exit("quit_modal")
                click_pos = None
            if b_edge:
                quit_confirm_active = False
            elif a_edge:
                perform_graceful_exit("controller_confirm")

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
                    current_step_outcome_marked = False
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
                    state.audio_cue = "injection_stop"
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

        # Post-step outcome overlay
        syringe_remaining_ul_overlay = 0.0
        with state.lock:
            syringe_remaining_ul_overlay = state.settings.get("syringe_remaining_ul", 0.0)
        if post_step_prompt_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            with state.lock:
                s = state.settings
                syringe_step_ul_overlay    = s.get("syringe_step_ul", 1.0)
                syringe_remaining_ul_overlay = s.get("syringe_remaining_ul", 0.0)
                retract_mm_cfg = float(
                    s.get("injection_retract_mm", 10.0) or 0.0
                )

            steps_left_exact = 0.0
            full_steps_left = 0
            remainder_ul = syringe_remaining_ul_overlay
            if syringe_step_ul_overlay > 0:
                steps_left_exact = max(
                    0.0,
                    float(syringe_remaining_ul_overlay) / float(syringe_step_ul_overlay),
                )
                full_steps_left = int(syringe_remaining_ul_overlay // syringe_step_ul_overlay)
                remainder_ul = (
                    syringe_remaining_ul_overlay
                    - full_steps_left * syringe_step_ul_overlay
                )

            panel_w = min(760, max(580, screen_w - 80))
            panel_h = 390
            px = (screen_w - panel_w) // 2
            py = (screen_h - panel_h) // 2
            panel = pygame.Rect(px, py, panel_w, panel_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (0, 255, 127), panel, 2)
            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            small_modal = pygame.font.SysFont("Consolas", 14)
            title = font_modal.render("Record Injection Outcome", True, (230, 230, 230))
            screen.blit(title, (px + 16, py + 16))

            subj_name, subj_count_after, subj_total_after, _, delivered_now = (
                current_subject_volume_summary(include_current=True)
            )
            detail_lines = []
            detail_lines.append(f"Animal/sample: {subj_name or '(blank)'}")
            detail_lines.append(
                f"Animal total after this attempt: {subj_total_after:.3f} µL across {subj_count_after} injection(s)"
            )
            detail_lines.append(f"Last target dose: {injection_step_volume_ul:.3f} µL")
            detail_lines.append(f"Estimated delivered this attempt: {delivered_now:.3f} µL")
            if syringe_step_ul_overlay > 0:
                detail_lines.append(
                    f"Remaining syringe: {steps_left_exact:.2f} configured step(s)"
                )
                detail_lines.append(
                    f"Full steps available: {full_steps_left} × {syringe_step_ul_overlay:.3f} µL"
                )
            detail_lines.append(f"Remaining volume: {syringe_remaining_ul_overlay:.3f} µL")
            if remainder_ul > 1e-3 and syringe_step_ul_overlay > 0:
                detail_lines.append(f"Final smaller step ≈ {remainder_ul:.3f} µL")
            if injection_last_step_small:
                detail_lines.append(
                    f"Last step was smaller than configured step ({injection_step_volume_ul:.3f} µL)"
                )
            if syringe_remaining_ul_overlay <= 1e-9:
                detail_lines.append("Injection session complete.")
            detail_lines.append(f"Optional toolhead retract distance: {retract_mm_cfg:.1f} mm")

            detail_rect = pygame.Rect(px + 16, py + 56, panel_w - 32, 148)
            pygame.draw.rect(screen, (24, 24, 32), detail_rect)
            pygame.draw.rect(screen, (80, 105, 105), detail_rect, 1)
            rendered_detail_lines = []
            for msg in detail_lines:
                for ln in wrap_lines(small_modal, msg, detail_rect.w - 20):
                    rendered_detail_lines.append(ln)
            line_h = small_modal.get_height() + 3
            content_h = len(rendered_detail_lines) * line_h + 20
            max_detail_scroll = max(0, content_h - detail_rect.h)
            outcome_detail_scroll = max(0, min(outcome_detail_scroll, max_detail_scroll))
            prev_clip_detail = screen.get_clip()
            screen.set_clip(detail_rect.inflate(-2, -2))
            dy = detail_rect.y + 10 - outcome_detail_scroll
            for ln in rendered_detail_lines:
                if dy + small_modal.get_height() >= detail_rect.y and dy <= detail_rect.bottom:
                    t = small_modal.render(ln, True, (220, 220, 220))
                    screen.blit(t, (detail_rect.x + 10, dy))
                dy += line_h
            screen.set_clip(prev_clip_detail)
            if max_detail_scroll > 0:
                track = pygame.Rect(detail_rect.right - 8, detail_rect.y + 4, 4, detail_rect.h - 8)
                pygame.draw.rect(screen, (55, 65, 70), track)
                knob_h = max(18, int(track.h * detail_rect.h / max(detail_rect.h, content_h)))
                knob_y = track.y + int((track.h - knob_h) * (outcome_detail_scroll / max_detail_scroll))
                pygame.draw.rect(screen, (0, 220, 200), pygame.Rect(track.x, knob_y, track.w, knob_h))

            btn_w = min(330, (panel_w - 48) // 2)
            btn_h = 48
            gap = 14
            bx = px + (panel_w - (btn_w * 2 + gap)) // 2
            by1 = detail_rect.bottom + 20
            by2 = by1 + btn_h + 12
            pass_rect = pygame.Rect(bx, by1, btn_w, btn_h)
            fail_rect = pygame.Rect(bx + btn_w + gap, by1, btn_w, btn_h)
            retry_rect = pygame.Rect(bx, by2, btn_w, btn_h)
            reload_rect = pygame.Rect(bx + btn_w + gap, by2, btn_w, btn_h)
            draw_modal_button(screen, pass_rect, "A  Injection Passed", font_modal, border=(0, 255, 127), fill=(45, 70, 55))
            draw_modal_button(screen, fail_rect, "B  Injection Failed", font_modal, border=(255, 170, 80), fill=(75, 55, 45))
            draw_modal_button(screen, retry_rect, "X  Retry Same Animal", font_modal, border=(0, 220, 200), fill=(45, 55, 70))
            draw_modal_button(screen, reload_rect, "Y  Reload / Refill", font_modal, border=(170, 170, 255), fill=(50, 50, 75))
            add_nav_item(frame_nav_items, pass_rect, "Mark injection passed", group="modal")
            add_nav_item(frame_nav_items, fail_rect, "Mark injection failed", group="modal")
            add_nav_item(frame_nav_items, retry_rect, "Retry same animal", group="modal")
            add_nav_item(frame_nav_items, reload_rect, "Reload/refill syringe", group="modal")
            hint = small_modal.render("Mouse wheel scrolls log. Reload records this attempt, then opens refill workflow.", True, (200, 200, 200))
            screen.blit(hint, (px + 16, py + panel_h - hint.get_height() - 16))

            post_choice = None
            if click_pos is not None:
                mx, my = click_pos
                if pass_rect.collidepoint(mx, my):
                    post_choice = "passed"
                elif fail_rect.collidepoint(mx, my):
                    post_choice = "failed"
                elif retry_rect.collidepoint(mx, my):
                    post_choice = "retry"
                elif reload_rect.collidepoint(mx, my):
                    post_choice = "reload"
                click_pos = None
            if a_edge:
                post_choice = "passed"
            elif b_edge:
                post_choice = "failed"
            elif x_edge:
                post_choice = "retry"
            elif y_edge:
                post_choice = "reload"

            if post_choice == "passed":
                with state.lock:
                    state.calibration_status = "Injection marked passed."
                mark_current_step_outcome("passed", "User marked injection passed.")
                clear_post_step_decision(open_new_subject=False)
                post_outcome_choice_active = True
                post_outcome_choice_just_opened = True
                print("[Injection] Step marked passed.")

            elif post_choice == "failed":
                with state.lock:
                    state.calibration_status = "Injection marked failed."
                mark_current_step_outcome("failed", "User marked injection failed after step.")
                clear_post_step_decision(open_new_subject=False)
                post_outcome_choice_active = True
                post_outcome_choice_just_opened = True
                print("[Injection] Step marked failed.")

            elif post_choice == "retry":
                with state.lock:
                    state.calibration_status = "Retry recorded. Same animal/sample remains active."
                mark_current_step_outcome("retry", "User selected retry on the same animal/sample.")
                clear_post_step_decision(open_new_subject=False)
                print("[Injection] Retry recorded; same animal/sample remains active.")

            elif post_choice == "reload":
                with state.lock:
                    state.calibration_status = "Reload/refill selected after injection attempt."
                mark_current_step_outcome("reload", "User selected reload/refill after this attempt.")
                clear_post_step_decision(open_new_subject=False)
                open_injection_reload_modal()
                print("[Injection] Reload/refill selected after outcome.")

        if post_outcome_choice_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            screen.blit(overlay, (0, 0))

            panel_w = min(660, max(520, screen_w - 80))
            panel_h = 250
            px = (screen_w - panel_w) // 2
            py = (screen_h - panel_h) // 2
            panel = pygame.Rect(px, py, panel_w, panel_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (0, 255, 127), panel, 2)
            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            small_modal = pygame.font.SysFont("Consolas", 14)
            title = font_modal.render("Outcome Recorded", True, (230, 230, 230))
            screen.blit(title, (px + 16, py + 16))

            with state.lock:
                vol_left = float(state.settings.get("syringe_remaining_ul", 0.0) or 0.0)
                subject_now = str(state.settings.get("injection_subject", "") or "")
            _, subject_count, subject_total, _, _ = current_subject_volume_summary(include_current=False)
            lines = [
                f"Current animal/eye/sample: {subject_now or '(blank)'}",
                f"Recorded total: {subject_total:.3f} µL across {subject_count} injection(s)",
                f"Remaining syringe volume: {vol_left:.3f} µL",
                "Injection outcome is complete. Enter the next animal/eye/sample ID.",
            ]
            dy = py + 58
            for msg in lines:
                for ln in wrap_lines(small_modal, msg, panel_w - 32):
                    t = small_modal.render(ln, True, (220, 220, 220))
                    screen.blit(t, (px + 16, dy))
                    dy += t.get_height() + 4

            btn_w = min(260, (panel_w - 48) // 2)
            btn_h = 48
            gap = 14
            bx = px + (panel_w - (btn_w * 2 + gap)) // 2
            by = py + panel_h - btn_h - 24
            reload_rect = pygame.Rect(bx, by, btn_w, btn_h)
            next_rect = pygame.Rect(reload_rect.right + gap, by, btn_w, btn_h)
            draw_modal_button(screen, reload_rect, "X  Reload / Refill", font_modal, border=(0, 220, 200), fill=(45, 55, 70))
            draw_modal_button(screen, next_rect, "Y  Next ID", font_modal, border=(170, 170, 255), fill=(50, 50, 75))
            add_nav_item(frame_nav_items, reload_rect, "Reload/refill before next ID", group="modal")
            add_nav_item(frame_nav_items, next_rect, "Next animal/sample ID", group="modal")

            subject_choice = None
            if post_outcome_choice_just_opened:
                post_outcome_choice_just_opened = False
                click_pos = None
            elif click_pos is not None:
                mx, my = click_pos
                if reload_rect.collidepoint(mx, my):
                    subject_choice = "reload"
                elif next_rect.collidepoint(mx, my):
                    subject_choice = "next"
                click_pos = None
                if x_edge:
                    subject_choice = "reload"
                elif y_edge:
                    subject_choice = "next"
            elif x_edge:
                subject_choice = "reload"
            elif y_edge:
                subject_choice = "next"

            if subject_choice == "reload":
                post_outcome_choice_active = False
                open_injection_reload_modal()
            elif subject_choice == "next":
                post_outcome_choice_active = False
                clear_post_step_decision(open_new_subject=True)
                with state.lock:
                    state.calibration_status = "Ready for next animal/eye/sample."
                print("[Injection] Next animal/eye/sample selected after outcome.")

        if injection_tunnel_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 175))
            screen.blit(overlay, (0, 0))

            panel_w = min(760, max(600, screen_w - 80))
            panel_h = 350
            px = (screen_w - panel_w) // 2
            py = (screen_h - panel_h) // 2
            panel = pygame.Rect(px, py, panel_w, panel_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (0, 255, 127), panel, 2)
            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            small_modal = pygame.font.SysFont("Consolas", 14)

            target_mode = str(injection_tunnel_target_mode or "tunnel").lower()
            target_label = "Tunnel needle" if target_mode == "tunnel" else "Injector"
            with state.lock:
                s = state.settings
                current_mode = str(s.get("tunnel_tcp_mode", DEFAULT_TUNNEL_TCP_MODE) or DEFAULT_TUNNEL_TCP_MODE).lower()
                if current_mode not in ("injector", "tunnel"):
                    current_mode = DEFAULT_TUNNEL_TCP_MODE
                current_label = "Tunnel needle" if current_mode == "tunnel" else "Injector"
                offset_y = float(s.get("tunnel_tcp_y_mm", DEFAULT_TUNNEL_TCP_Y_MM) or 0.0)
                retract_mm = float(s.get("tunnel_retract_mm", DEFAULT_TUNNEL_RETRACT_MM) or 0.0)
                return_mm = float(s.get("tunnel_return_mm", DEFAULT_TUNNEL_RETURN_MM) or 0.0)
                subj_now = str(s.get("injection_subject", "") or "")
                status_now = str(state.calibration_status or "")
            _, subj_count, subj_total, _, _ = current_subject_volume_summary(include_current=False)
            clearance_mm = max(0.0, retract_mm - return_mm)
            tunnel_ready_for_injector = (target_mode == "tunnel" and current_mode == "tunnel")

            title = font_modal.render("Tunnel / Injector TCP Workflow", True, (230, 230, 230))
            screen.blit(title, (px + 16, py + 16))
            detail_lines = [
                f"Animal/sample: {subj_now or '(blank)'}",
                f"Recorded total: {subj_total:.3f} µL across {subj_count} injection(s)",
                f"Current TCP: {current_label}; target TCP: {target_label}",
                f"Tunnel Y offset: {offset_y:+.3f} mm",
                f"Automatic safety path: retract {retract_mm:.1f} mm, switch TCP, return {return_mm:.1f} mm",
                f"Manual clearance after return: {clearance_mm:.1f} mm",
            ]
            if tunnel_ready_for_injector:
                detail_lines.append("Tunnel TCP is active. After making the tunnel, press A to switch safely to the injector.")
                if status_now:
                    detail_lines.append(f"Status: {status_now}")
            elif injection_tunnel_stage == "queued":
                detail_lines.append("Safe TCP switch is queued/running. Keep the field clear and watch motion.")
                if status_now:
                    detail_lines.append(f"Status: {status_now}")
            else:
                detail_lines.append("Do not switch TCP while inside tissue. Use A to run the safety path.")
                detail_lines.append("Use Y only if you want to leave this modal and jog manually first.")

            detail_rect = pygame.Rect(px + 16, py + 56, panel_w - 32, 160)
            pygame.draw.rect(screen, (24, 24, 32), detail_rect)
            pygame.draw.rect(screen, (80, 105, 105), detail_rect, 1)
            dy = detail_rect.y + 10
            for msg in detail_lines:
                for ln in wrap_lines(small_modal, msg, detail_rect.w - 20):
                    t = small_modal.render(ln, True, (220, 220, 220))
                    if dy + t.get_height() <= detail_rect.bottom - 4:
                        screen.blit(t, (detail_rect.x + 10, dy))
                    dy += t.get_height() + 3

            btn_h = 46
            gap = 12
            btn_w = (panel_w - 44 - gap) // 2
            bx = px + 22
            by = py + panel_h - btn_h * 2 - gap - 22
            left_rect = pygame.Rect(bx, by, btn_w, btn_h)
            mid_rect = pygame.Rect(left_rect.right + gap, by, btn_w, btn_h)
            free_rect = pygame.Rect(bx, left_rect.bottom + gap, btn_w, btn_h)
            cancel_rect = pygame.Rect(free_rect.right + gap, free_rect.y, btn_w, btn_h)

            if tunnel_ready_for_injector:
                left_label = "A  Tunnel Complete -> Injector"
                mid_label = "X  Halt Motion"
                cancel_label = "B  Close"
            elif injection_tunnel_stage == "queued":
                left_label = "A  Close"
                mid_label = "X  Halt Motion"
                cancel_label = "B  Close"
            else:
                left_label = f"A  Safe Switch to {target_label}"
                mid_label = "X  Halt Motion"
                cancel_label = "B  Cancel"
            draw_modal_button(screen, left_rect, left_label, font_modal, border=(0, 255, 127), fill=(45, 70, 55))
            draw_modal_button(screen, mid_rect, mid_label, font_modal, border=(0, 220, 200), fill=(45, 55, 70))
            draw_modal_button(screen, free_rect, "Y  Free Control", font_modal, border=(255, 210, 90), fill=(65, 58, 42))
            draw_modal_button(screen, cancel_rect, cancel_label, font_modal, border=(255, 100, 100), fill=(70, 45, 50))
            add_nav_item(frame_nav_items, left_rect, left_label.replace("A  ", ""), group="modal")
            add_nav_item(frame_nav_items, mid_rect, mid_label.replace("X  ", ""), group="modal")
            add_nav_item(frame_nav_items, free_rect, "Free control", group="modal")
            add_nav_item(frame_nav_items, cancel_rect, cancel_label.replace("B  ", ""), group="modal")

            tunnel_choice = None
            if click_pos is not None:
                mx, my = click_pos
                if left_rect.collidepoint(mx, my):
                    tunnel_choice = "left"
                elif mid_rect.collidepoint(mx, my):
                    tunnel_choice = "middle"
                elif free_rect.collidepoint(mx, my):
                    tunnel_choice = "free"
                elif cancel_rect.collidepoint(mx, my):
                    tunnel_choice = "cancel"
                click_pos = None
            if a_edge:
                tunnel_choice = "left"
            elif x_edge:
                tunnel_choice = "middle"
            elif y_edge:
                tunnel_choice = "free"
            elif b_edge:
                tunnel_choice = "cancel"

            if tunnel_ready_for_injector:
                if tunnel_choice == "left":
                    complete_tunnel_phase_to_injector()
                elif tunnel_choice == "middle":
                    rt.enqueue({"type": "stop_motors"})
                    log_injection_session_event(
                        "tunnel_workflow_halt_requested",
                        "User requested halt from tunnel TCP workflow modal.",
                        {"target_tcp_mode": target_mode},
                    )
                    with state.lock:
                        state.calibration_status = "Motion halt requested from tunnel TCP workflow."
                elif tunnel_choice == "free":
                    log_injection_session_event(
                        "tunnel_workflow_free_control",
                        "User left tunnel TCP workflow for manual free control.",
                        {"target_tcp_mode": target_mode},
                    )
                    injection_tunnel_active = False
                    injection_tunnel_stage = "confirm"
                    with state.lock:
                        state.calibration_status = "Free control enabled. Injection session logging is preserved."
                elif tunnel_choice == "cancel":
                    injection_tunnel_active = False
                    injection_tunnel_stage = "confirm"
            elif injection_tunnel_stage == "queued":
                if tunnel_choice == "middle":
                    rt.enqueue({"type": "stop_motors"})
                    log_injection_session_event(
                        "tunnel_tcp_switch_halt_requested",
                        "User requested halt during tunnel TCP workflow.",
                        {"target_tcp_mode": target_mode},
                    )
                    with state.lock:
                        state.calibration_status = "Tunnel TCP workflow halted by user request."
                    injection_tunnel_active = False
                    injection_tunnel_stage = "confirm"
                elif tunnel_choice in ("left", "cancel", "free"):
                    if tunnel_choice == "free":
                        log_injection_session_event(
                            "tunnel_workflow_free_control",
                            "User left queued tunnel TCP workflow for manual free control.",
                            {"target_tcp_mode": target_mode},
                        )
                    injection_tunnel_active = False
                    injection_tunnel_stage = "confirm"
            else:
                if tunnel_choice == "left":
                    queue_injection_tunnel_switch(target_mode)
                elif tunnel_choice == "middle":
                    rt.enqueue({"type": "stop_motors"})
                    log_injection_session_event(
                        "tunnel_workflow_halt_requested",
                        "User requested halt from tunnel TCP workflow modal.",
                        {"target_tcp_mode": target_mode},
                    )
                    with state.lock:
                        state.calibration_status = "Motion halt requested from tunnel TCP workflow."
                elif tunnel_choice == "free":
                    log_injection_session_event(
                        "tunnel_workflow_free_control",
                        "User left tunnel TCP workflow for manual free control.",
                        {"target_tcp_mode": target_mode},
                    )
                    injection_tunnel_active = False
                    injection_tunnel_stage = "confirm"
                    with state.lock:
                        state.calibration_status = "Free control enabled. Injection session logging is preserved."
                elif tunnel_choice == "cancel":
                    log_injection_session_event(
                        "tunnel_workflow_cancelled",
                        "User cancelled tunnel TCP workflow.",
                        {"target_tcp_mode": target_mode},
                    )
                    injection_tunnel_active = False
                    injection_tunnel_stage = "confirm"

        if injection_reload_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 175))
            screen.blit(overlay, (0, 0))

            panel_w = min(760, max(600, screen_w - 80))
            panel_h = 390
            px = (screen_w - panel_w) // 2
            py = (screen_h - panel_h) // 2
            panel = pygame.Rect(px, py, panel_w, panel_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (0, 255, 127), panel, 2)
            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            small_modal = pygame.font.SysFont("Consolas", 14)
            title = font_modal.render("Reload / Refill Syringe", True, (230, 230, 230))
            screen.blit(title, (px + 16, py + 16))

            with state.lock:
                vol_left = float(state.settings.get("syringe_remaining_ul", 0.0) or 0.0)
                vol_full = float(state.settings.get("syringe_volume_ul", 0.0) or 0.0)
                subj_now = str(state.settings.get("injection_subject", "") or "")
                retracted_pose_known = state.injection_reload_retracted_pose is not None
                reload_motion_until = float(getattr(state, "pose_motion_until", 0.0) or 0.0)
            reload_motion_busy = time.time() < reload_motion_until
            _, subj_count, subj_total, _, _ = current_subject_volume_summary(include_current=False)
            stage_label = {
                "precheck": "Safety retraction check",
                "retracted": "Retracted pose captured",
                "at_reload": "At reload pose",
                "reload_down": "Reload station lowered",
                "reload_up": "Reload station raised",
            }.get(injection_reload_stage, "Reload workflow")
            detail_lines = [
                f"Animal/sample: {subj_now or '(blank)'}",
                f"Recorded total: {subj_total:.3f} µL across {subj_count} injection(s)",
                f"Syringe now: {vol_left:.3f}/{vol_full:.3f} µL",
                f"Stage: {stage_label}",
                "This keeps the same animal/session log active.",
            ]
            if reload_motion_busy:
                detail_lines.append("Robot motion is still running; wait before the next automatic step.")
            if injection_reload_stage == "precheck":
                detail_lines.append("Is the needle clear of the sample/animal?")
                detail_lines.append(f"If not, use X to retract {RELOAD_SAFE_RETRACT_MM:.1f} mm along tool-Z before moving to Reload.")
                detail_lines.append("Use Y for free control without any automatic motion.")
            elif injection_reload_stage == "retracted":
                detail_lines.append("Retracted pose is captured." if retracted_pose_known else "Retracted pose capture is pending.")
                detail_lines.append("Press A to move to the saved Reload pose, or Y for free control.")
            elif injection_reload_stage == "at_reload":
                detail_lines.append(
                    f"At Reload: press A for the {RELOAD_SAFE_RETRACT_MM:.1f} mm refill motion, "
                    "X only if refill was done manually and you want to return, or Y for free control."
                )
            elif injection_reload_stage == "reload_down":
                detail_lines.append(f"Reload station lowered by {RELOAD_SAFE_RETRACT_MM:.1f} mm. Press A to move upwards again, or Y for free control.")
            elif injection_reload_stage == "reload_up":
                detail_lines.append("Reload station raised. Press A to mark refill complete and return, or Y for free control.")

            detail_rect = pygame.Rect(px + 16, py + 56, panel_w - 32, 150)
            pygame.draw.rect(screen, (24, 24, 32), detail_rect)
            pygame.draw.rect(screen, (80, 105, 105), detail_rect, 1)
            prev_reload_clip = screen.get_clip()
            screen.set_clip(detail_rect.inflate(-2, -2))
            dy = detail_rect.y + 10
            for msg in detail_lines:
                for ln in wrap_lines(small_modal, msg, detail_rect.w - 20):
                    t = small_modal.render(ln, True, (220, 220, 220))
                    if dy + t.get_height() >= detail_rect.y and dy <= detail_rect.bottom:
                        screen.blit(t, (detail_rect.x + 10, dy))
                    dy += t.get_height() + 3
            screen.set_clip(prev_reload_clip)

            btn_h = 46
            gap = 12
            btn_w = (panel_w - 44 - gap) // 2
            bx = px + 22
            by = py + panel_h - btn_h * 2 - gap - 22
            left_rect = pygame.Rect(bx, by, btn_w, btn_h)
            mid_rect = pygame.Rect(left_rect.right + gap, by, btn_w, btn_h)
            free_rect = pygame.Rect(bx, left_rect.bottom + gap, btn_w, btn_h)
            cancel_rect = pygame.Rect(free_rect.right + gap, free_rect.y, btn_w, btn_h)

            if injection_reload_stage == "precheck":
                left_label = "A  Needle Clear"
                mid_label = f"X  Retract {int(RELOAD_SAFE_RETRACT_MM)}mm"
            elif injection_reload_stage == "retracted":
                left_label = "A  Go Reload" if retracted_pose_known else "A  Wait Capture"
                mid_label = f"X  Retract {int(RELOAD_SAFE_RETRACT_MM)}mm"
            elif injection_reload_stage == "at_reload":
                left_label = f"A  Down {int(RELOAD_SAFE_RETRACT_MM)}mm"
                mid_label = "X  Mark Refilled + Return"
            elif injection_reload_stage == "reload_down":
                left_label = f"A  Up {int(RELOAD_SAFE_RETRACT_MM)}mm"
                mid_label = "X  Up + Refill + Return"
            else:
                left_label = "A  Mark Refilled + Return"
                mid_label = "X  Mark Refilled + Return"
            free_label = "Y  Free Control"
            cancel_label = "B  Cancel"
            if injection_reload_stage == "reload_down":
                cancel_label = f"B  Up {int(RELOAD_SAFE_RETRACT_MM)}mm + Cancel"
            if reload_motion_busy:
                left_label = "A  Wait for Motion"
                mid_label = "X  Halt Motion"
                free_label = "Y  Wait"
                cancel_label = "B  Wait"

            draw_modal_button(screen, left_rect, left_label, font_modal, border=(0, 220, 200), fill=(45, 55, 70))
            draw_modal_button(screen, mid_rect, mid_label, font_modal, border=(0, 255, 127), fill=(45, 70, 55))
            draw_modal_button(screen, free_rect, free_label, font_modal, border=(255, 210, 90), fill=(65, 58, 42))
            draw_modal_button(screen, cancel_rect, cancel_label, font_modal, border=(255, 100, 100), fill=(70, 45, 50))
            add_nav_item(frame_nav_items, left_rect, left_label.replace("A  ", ""), group="modal")
            add_nav_item(frame_nav_items, mid_rect, mid_label.replace("X  ", ""), group="modal")
            add_nav_item(frame_nav_items, free_rect, free_label.replace("Y  ", ""), group="modal")
            add_nav_item(frame_nav_items, cancel_rect, cancel_label.replace("B  ", ""), group="modal")

            reload_choice = None
            if click_pos is not None:
                mx, my = click_pos
                if left_rect.collidepoint(mx, my):
                    reload_choice = "left"
                elif mid_rect.collidepoint(mx, my):
                    reload_choice = "middle"
                elif free_rect.collidepoint(mx, my):
                    reload_choice = "free"
                elif cancel_rect.collidepoint(mx, my):
                    reload_choice = "cancel"
                click_pos = None
            if a_edge:
                reload_choice = "left"
            elif x_edge:
                reload_choice = "middle"
            elif y_edge:
                reload_choice = "free"
            elif b_edge:
                reload_choice = "cancel"

            if reload_motion_busy and reload_choice:
                if reload_choice == "middle":
                    rt.enqueue({"type": "stop_motors"})
                    log_injection_session_event(
                        "reload_halt_requested",
                        "User requested halt while reload/refill motion was running.",
                        {"reload_stage": str(injection_reload_stage)},
                    )
                    with state.lock:
                        state.calibration_status = "Reload/refill motion halt requested."
                else:
                    with state.lock:
                        state.calibration_status = "Wait for reload/refill motion to finish, or press X to halt."
                reload_choice = None

            if reload_choice == "left":
                if injection_reload_stage == "precheck":
                    rt.enqueue({"type": "injection_reload_capture_retracted"})
                    injection_reload_stage = "retracted"
                    with state.lock:
                        state.calibration_status = "Retracted pose capture queued. Press A to go Reload when ready."
                    log_injection_session_event("reload_retracted_confirmed", "User confirmed already retracted before reload.")
                elif injection_reload_stage == "retracted":
                    if not retracted_pose_known:
                        with state.lock:
                            state.calibration_status = "Waiting for retracted pose capture before moving to Reload."
                    else:
                        rt.enqueue({"type": "goto_direct", "pose_name": "Reload"})
                        injection_reload_started = True
                        injection_reload_stage = "at_reload"
                        with state.lock:
                            state.calibration_status = "Reload pose queued. Use A for downwards reload motion when ready."
                        log_injection_session_event(
                            "reload_pose_queued",
                            "Robot queued direct move to saved Reload pose during injection reload workflow.",
                            {"remaining_before_reload_ul": float(injection_reload_before_ul)},
                        )
                        print("[Injection] Reload pose queued directly; session preserved.")
                elif injection_reload_stage == "at_reload":
                    # At the reload pose, +tool-Z is the configured down motion
                    # into the refill station; the paired up motion is -tool-Z.
                    rt.enqueue({"type": "inject_advance", "dist": RELOAD_SAFE_RETRACT_MM})
                    injection_reload_stage = "reload_down"
                    with state.lock:
                        state.calibration_status = "Reload downwards motion queued. Press A to move upwards again."
                    log_injection_session_event("reload_down_queued", "Reload station downward motion queued.")
                elif injection_reload_stage == "reload_down":
                    rt.enqueue({"type": "inject_retract", "dist": RELOAD_SAFE_RETRACT_MM})
                    injection_reload_stage = "reload_up"
                    with state.lock:
                        state.calibration_status = "Reload upwards motion queued. Press A to return to the retracted animal pose."
                    log_injection_session_event("reload_up_queued", "Reload station upward motion queued.")
                else:
                    complete_injection_reload(return_to_retracted=True)
                    print("[Injection] Reload/refill complete; returning to retracted pose.")
            elif reload_choice == "middle":
                if injection_reload_stage in ("precheck", "retracted"):
                    rt.enqueue({"type": "injection_reload_retract_capture", "dist": RELOAD_SAFE_RETRACT_MM})
                    injection_reload_stage = "retracted"
                    with state.lock:
                        state.calibration_status = f"Safety retract {RELOAD_SAFE_RETRACT_MM:.1f} mm queued. Press A to go Reload when ready."
                    log_injection_session_event(
                        "reload_safety_retract_queued",
                        "Safety retract before reload was queued.",
                        {"retract_mm": float(RELOAD_SAFE_RETRACT_MM)},
                    )
                elif injection_reload_stage in ("at_reload", "reload_up"):
                    complete_injection_reload(return_to_retracted=True)
                    print("[Injection] Reload/refill complete; returning to retracted pose.")
                elif injection_reload_stage == "reload_down":
                    rt.enqueue({"type": "inject_retract", "dist": RELOAD_SAFE_RETRACT_MM})
                    log_injection_session_event(
                        "reload_up_before_complete_queued",
                        "Reload station upward motion queued before marking refill complete.",
                        {"retract_mm": float(RELOAD_SAFE_RETRACT_MM)},
                    )
                    complete_injection_reload(return_to_retracted=True)
                    print("[Injection] Reload/refill complete; raising then returning to retracted pose.")
            elif reload_choice == "free":
                if injection_reload_stage == "reload_down":
                    log_injection_session_event(
                        "reload_free_control_lowered",
                        "User left reload/refill workflow for manual free control while reload station was lowered.",
                        {
                            "remaining_before_reload_ul": float(injection_reload_before_ul),
                            "reload_pose_was_queued": bool(injection_reload_started),
                            "reload_stage": str(injection_reload_stage),
                        },
                    )
                    with state.lock:
                        state.calibration_status = "Free control enabled while lowered. Raise before leaving reload station."
                else:
                    log_injection_session_event(
                        "reload_free_control",
                        "User left reload/refill workflow for manual free control.",
                        {
                            "remaining_before_reload_ul": float(injection_reload_before_ul),
                            "reload_pose_was_queued": bool(injection_reload_started),
                            "reload_stage": str(injection_reload_stage),
                        },
                    )
                    with state.lock:
                        state.calibration_status = "Free control enabled. Current animal/session logging is preserved."
                injection_reload_active = False
                injection_reload_started = False
                injection_reload_before_ul = 0.0
                injection_reload_stage = "precheck"
            elif reload_choice == "cancel":
                if injection_reload_stage == "reload_down":
                    rt.enqueue({"type": "inject_retract", "dist": RELOAD_SAFE_RETRACT_MM})
                    log_injection_session_event(
                        "reload_cancel_raise_queued",
                        "User cancelled reload/refill workflow while lowered; upward motion was queued first.",
                        {
                            "remaining_before_reload_ul": float(injection_reload_before_ul),
                            "reload_pose_was_queued": bool(injection_reload_started),
                            "reload_stage": str(injection_reload_stage),
                            "retract_mm": float(RELOAD_SAFE_RETRACT_MM),
                        },
                    )
                    injection_reload_active = False
                    injection_reload_started = False
                    injection_reload_before_ul = 0.0
                    injection_reload_stage = "precheck"
                    with state.lock:
                        state.calibration_status = "Reload/refill cancelled; upward motion queued before closing workflow."
                else:
                    log_injection_session_event(
                        "reload_cancelled",
                        "User cancelled reload/refill workflow.",
                        {
                            "remaining_before_reload_ul": float(injection_reload_before_ul),
                            "reload_pose_was_queued": bool(injection_reload_started),
                        },
                    )
                    injection_reload_active = False
                    injection_reload_started = False
                    injection_reload_before_ul = 0.0
                    injection_reload_stage = "precheck"
                    with state.lock:
                        state.calibration_status = "Reload/refill cancelled. Injection session preserved."

        if injection_exit_active:
            overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))

            panel_w = min(760, max(600, screen_w - 80))
            panel_h = 350
            px = (screen_w - panel_w) // 2
            py = (screen_h - panel_h) // 2
            panel = pygame.Rect(px, py, panel_w, panel_h)
            pygame.draw.rect(screen, (30, 30, 40), panel)
            pygame.draw.rect(screen, (255, 170, 80), panel, 2)
            font_modal = pygame.font.SysFont("Consolas", 18, bold=True)
            small_modal = pygame.font.SysFont("Consolas", 14)
            title = font_modal.render("Exit Injection Mode / Syringe Options", True, (245, 235, 220))
            screen.blit(title, (px + 16, py + 16))

            with state.lock:
                s = state.settings
                vol_left = float(s.get("syringe_remaining_ul", 0.0) or 0.0)
                vol_full = float(s.get("syringe_volume_ul", 0.0) or 0.0)
                subj_now = str(s.get("injection_subject", "") or "")
            _, subj_count, subj_total, _, _ = current_subject_volume_summary(include_current=False)
            detail_lines = [
                f"Current animal/eye/sample: {subj_now or '(blank)'}",
                f"Recorded total for this ID: {subj_total:.3f} µL across {subj_count} injection(s)",
                f"Syringe remaining: {vol_left:.3f}/{vol_full:.3f} µL",
                "Exiting injection mode does not power off motors.",
                "Use A to leave injection mode and keep the remaining volume in the log.",
                "Use X only if the remaining material is ejected/returned and should be logged as depleted.",
                "Use Y for the reload/obtain-more workflow without ending the current session.",
            ]
            detail_rect = pygame.Rect(px + 16, py + 56, panel_w - 32, 152)
            pygame.draw.rect(screen, (24, 24, 32), detail_rect)
            pygame.draw.rect(screen, (110, 95, 70), detail_rect, 1)
            dy = detail_rect.y + 10
            for msg in detail_lines:
                for ln in wrap_lines(small_modal, msg, detail_rect.w - 20):
                    t = small_modal.render(ln, True, (225, 225, 220))
                    if dy + t.get_height() <= detail_rect.bottom - 4:
                        screen.blit(t, (detail_rect.x + 10, dy))
                    dy += t.get_height() + 3

            btn_h = 46
            gap = 12
            btn_w = (panel_w - 44 - gap) // 2
            bx = px + 22
            by = py + panel_h - btn_h * 2 - gap - 22
            keep_rect = pygame.Rect(bx, by, btn_w, btn_h)
            eject_rect = pygame.Rect(keep_rect.right + gap, by, btn_w, btn_h)
            reload_rect = pygame.Rect(bx, keep_rect.bottom + gap, btn_w, btn_h)
            cancel_rect = pygame.Rect(reload_rect.right + gap, reload_rect.y, btn_w, btn_h)

            draw_modal_button(screen, keep_rect, "A  Exit, Keep Volume", font_modal, border=(0, 255, 127), fill=(45, 70, 55))
            draw_modal_button(screen, eject_rect, "X  Exit, Mark Ejected", font_modal, border=(255, 170, 80), fill=(75, 55, 45))
            draw_modal_button(screen, reload_rect, "Y  Reload / Obtain More", font_modal, border=(0, 220, 200), fill=(45, 55, 70))
            draw_modal_button(screen, cancel_rect, "B  Cancel", font_modal, border=(255, 100, 100), fill=(70, 45, 50))
            add_nav_item(frame_nav_items, keep_rect, "Exit injection, keep volume", group="modal")
            add_nav_item(frame_nav_items, eject_rect, "Exit injection, mark ejected", group="modal")
            add_nav_item(frame_nav_items, reload_rect, "Reload/obtain more", group="modal")
            add_nav_item(frame_nav_items, cancel_rect, "Cancel exit injection", group="modal")

            exit_choice = None
            if click_pos is not None:
                mx, my = click_pos
                if keep_rect.collidepoint(mx, my):
                    exit_choice = "keep"
                elif eject_rect.collidepoint(mx, my):
                    exit_choice = "eject"
                elif reload_rect.collidepoint(mx, my):
                    exit_choice = "reload"
                elif cancel_rect.collidepoint(mx, my):
                    exit_choice = "cancel"
                click_pos = None
            if a_edge:
                exit_choice = "keep"
            elif x_edge:
                exit_choice = "eject"
            elif y_edge:
                exit_choice = "reload"
            elif b_edge:
                exit_choice = "cancel"

            if exit_choice == "keep":
                finish_injection_mode_exit("keep_remaining")
            elif exit_choice == "eject":
                finish_injection_mode_exit("eject_remaining")
            elif exit_choice == "reload":
                injection_exit_active = False
                open_injection_reload_modal()
            elif exit_choice == "cancel":
                injection_exit_active = False
                with state.lock:
                    state.calibration_status = "Injection mode exit cancelled."

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
                    rt.enqueue({"type": "save_named_with_midway", "name": pending_save_pose_name, "demo": False, "star": True})
                    save_mode_modal_active = False
                    pending_save_pose_name = ""
                elif demo_rect.collidepoint(mx, my):
                    rt.enqueue({"type": "save_named_with_midway", "name": pending_save_pose_name, "demo": True, "star": True})
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

        if alarm_active:
            draw_alarm_popup(screen, alarm_message)

        if thermal_hot and thermal_status_text and not thermal_warning_dismissed:
            thermal_warning_dismiss_rect = draw_thermal_warning_popup(screen, thermal_status_text)
        else:
            thermal_warning_dismiss_rect = None

        active_warning_items = build_warning_items(
            alarm_active,
            thermal_hot,
            thermal_critical,
            bool(
                injection_active_flag
                and not injection_reload_active
                and float(p) > float(high_angle_threshold_now)
            ),
            final_small_warning_active,
            post_step_prompt_active,
            bool(waiting and at_midway_now),
            injection_reload_active or injection_tunnel_active,
        )
        warn_anchor_x = hud_rect.x
        warn_anchor_y = hud_rect.bottom + 8
        warning_clickables = draw_warning_strip(
            screen,
            active_warning_items,
            muted_warning_keys,
            anchor_x=warn_anchor_x,
            anchor_y=warn_anchor_y,
        )

        screen_rect = screen.get_rect()
        for rect, item in warning_clickables:
            add_nav_item(
                frame_nav_items,
                rect,
                f"Warning: {item.get('label', 'warning')}",
                group="warning",
                screen_rect=screen_rect,
            )
        if thermal_warning_dismiss_rect is not None:
            add_nav_item(
                frame_nav_items,
                thermal_warning_dismiss_rect,
                "Dismiss thermal popup",
                group="warning",
                screen_rect=screen_rect,
            )
        if not modals_open:
            if rec_stop_rect is not None:
                add_nav_item(
                    frame_nav_items,
                    rec_stop_rect,
                    "Stop recording",
                    group="recording",
                    screen_rect=screen_rect,
                )
            if injection_outcome_rect is not None:
                add_nav_item(
                    frame_nav_items,
                    injection_outcome_rect,
                    "Record injection outcome",
                    group="recording",
                    screen_rect=screen_rect,
                )
            for rect, act in uvc_clickables:
                add_nav_item(
                    frame_nav_items,
                    rect,
                    nav_label_from_action(act),
                    group="camera",
                    screen_rect=screen_rect,
                )
            for rect, act in ui_clickables:
                add_nav_item(
                    frame_nav_items,
                    rect,
                    nav_label_from_action(act),
                    group="panel",
                    screen_rect=tr_rect,
                    kind=str(act.get("kind", "") or ""),
                )
        ui_nav_items = sort_nav_items(frame_nav_items)
        if ui_nav_items:
            if ui_nav_active and ui_nav_needs_initial_focus:
                ui_nav_focus_index = preferred_nav_focus_index(ui_nav_items)
                ui_nav_needs_initial_focus = False
            else:
                ui_nav_focus_index = max(0, min(ui_nav_focus_index, len(ui_nav_items) - 1))
        else:
            ui_nav_focus_index = 0
            ui_nav_needs_initial_focus = False
        if ui_nav_active:
            draw_ui_nav_focus(screen, ui_nav_items, ui_nav_focus_index, show_hint=False)

        pygame.display.flip()
        clock.tick(DEFAULT_UI_FPS)

    # ─────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────
    try:
        rt.stop()
        rt.join(timeout=2.0)
    except Exception:
        pass
    try:
        joystick_thread.stop()
        joystick_thread.join(timeout=1.0)
    except Exception:
        pass
    try:
        rs_thread.stop()
        rs_thread.join(timeout=2.0)
    except Exception:
        pass
    for t in uvc_threads:
        try:
            t.stop()
            t.join(timeout=1.0)
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
