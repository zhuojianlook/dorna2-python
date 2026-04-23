#!/usr/bin/env python3
"""
Headless tool-center calibration using dual UVC microscopes with per-camera
scale mismatch handling.

It:
1) Calibrates a pixel→world Jacobian for each camera via small XYZ translations.
2) Measures tip drift under roll (Z), pitch (X), and pitch (Y) rotations.
3) Solves a least-squares pivot for the true tool center r_tool (mm, tool frame).

It prints the suggested tool center values and can optionally write them back to
settings.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from dorna2 import Dorna

SETTINGS_PATH = "settings.json"

# ──────────────────────────────────────────────────────────────────────────────
# Utility: settings I/O
# ──────────────────────────────────────────────────────────────────────────────


def load_settings(path: str = SETTINGS_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path))
    except Exception:
        return {}


def save_settings(data: dict, path: str = SETTINGS_PATH) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────


def axis_angle_to_R(a_deg: float, b_deg: float, c_deg: float) -> np.ndarray:
    vec = np.radians([a_deg, b_deg, c_deg])
    th = np.linalg.norm(vec)
    if th < 1e-9:
        return np.eye(3)
    u = vec / th
    ux, uy, uz = u
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def R_to_axis_angle(R: np.ndarray) -> Tuple[float, float, float]:
    tr = np.trace(R)
    th = math.acos(max(min((tr - 1) / 2, 1), -1))
    if abs(th) < 1e-9:
        return (0.0, 0.0, 0.0)
    ux = (R[2, 1] - R[1, 2]) / (2 * math.sin(th))
    uy = (R[0, 2] - R[2, 0]) / (2 * math.sin(th))
    uz = (R[1, 0] - R[0, 1]) / (2 * math.sin(th))
    ang = np.degrees(th)
    return (float(ux * ang), float(uy * ang), float(uz * ang))


def rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ──────────────────────────────────────────────────────────────────────────────
# UVC helpers (lightweight thread)
# ──────────────────────────────────────────────────────────────────────────────


def _read_text(path: str) -> str:
    try:
        return open(path, "r").read().strip()
    except Exception:
        return ""


def discover_uvc_index0(limit: int = 2) -> list[str]:
    import glob
    import re

    def _is_realsense_name(name: str) -> bool:
        n = name.lower()
        return ("realsense" in n) or ("depth" in n) or ("infrared" in n)

    links = sorted(glob.glob("/dev/v4l/by-path/*video-index0"))
    out = []
    for link in links:
        node = os.path.realpath(link)
        if not node.startswith("/dev/video"):
            continue
        name = _read_text(f"/sys/class/video4linux/{os.path.basename(node)}/name")
        if _is_realsense_name(name):
            continue
        out.append(link)
        if len(out) >= limit:
            break
    return out


class UvcThread(threading.Thread):
    def __init__(self, device: str, width: int = 640, height: int = 480, fps: int = 30, rotate180: bool = False):
        super().__init__(daemon=True)
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.rotate180 = rotate180

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self.status = "init"

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        cap = None
        try:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

            # Warmup reads
            for _ in range(30):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.01)

            if not cap or not cap.isOpened():
                self.status = f"{self.device}: failed to open"
                return

            self.status = f"{self.device}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}@{int(cap.get(cv2.CAP_PROP_FPS))}"

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                if self.rotate180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                with self._lock:
                    self._frame = frame
                time.sleep(0.005)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.status += " (stopped)"


# ──────────────────────────────────────────────────────────────────────────────
# Vision helpers
# ──────────────────────────────────────────────────────────────────────────────


def detect_needle_tip(frame_bgr: np.ndarray, entry_side: str = "auto", prefer: str = "auto") -> Optional[Tuple[float, float]]:
    """
    Return normalized (x,y) of the bright needle tip or None.
    Logic mirrors dorna_joy_control's tip detector to stay consistent.
    """
    if frame_bgr is None:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return None

    # Ignore top/bottom margins so a tip exiting the frame doesn't trigger
    ymin, ymax = int(h * 0.05), int(h * 0.95)
    gray[:ymin, :] = 0
    gray[ymax:, :] = 0

    # Bright threshold on blurred image
    g_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = max(10, int(np.percentile(g_blur, 98)))
    _, mask = cv2.threshold(g_blur, thr, 255, cv2.THRESH_BINARY)

    # Clean small flecks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    pts = np.vstack([c.reshape(-1, 2) for c in cnts])
    x_left = pts[np.argmin(pts[:, 0])]
    x_right = pts[np.argmax(pts[:, 0])]
    y_top = pts[np.argmin(pts[:, 1])]
    y_bot = pts[np.argmax(pts[:, 1])]

    if entry_side == "right":
        tip = x_left
    elif entry_side == "left":
        tip = x_right
    elif entry_side == "bottom":
        tip = y_top
    elif entry_side == "top":
        tip = y_bot
    elif prefer == "left":
        tip = x_left
    elif prefer == "right":
        tip = x_right
    else:
        # pick point farthest from image borders to bias toward the true tip in-frame
        d_border = [min(p[0], w - 1 - p[0], p[1], h - 1 - p[1]) for p in pts]
        tip = pts[int(np.argmax(d_border))]
    x, y = float(tip[0]), float(tip[1])
    if w <= 0 or h <= 0:
        return None
    return (x / w, y / h)


def detect_tip_px(frame_bgr: np.ndarray, entry_side: str = "auto") -> Optional[Tuple[float, float]]:
    pt = detect_needle_tip(frame_bgr, entry_side=entry_side)
    if pt is None:
        return None
    h, w = frame_bgr.shape[:2]
    return (pt[0] * w, pt[1] * h)


def capture_tip_average(cams: Dict[str, UvcThread], samples: int = 6, delay_s: float = 0.03) -> Dict[str, Tuple[float, float]]:
    acc: Dict[str, list] = {}
    for _ in range(samples):
        for name, cam in cams.items():
            frame = cam.latest()
            if frame is None:
                continue
            tip = detect_tip_px(frame)
            if tip:
                acc.setdefault(name, []).append(tip)
        time.sleep(delay_s)
    avg: Dict[str, Tuple[float, float]] = {}
    for name, pts in acc.items():
        arr = np.array(pts, dtype=float)
        avg[name] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return avg


# ──────────────────────────────────────────────────────────────────────────────
# Robot helpers
# ──────────────────────────────────────────────────────────────────────────────


def wait_for_motion(dist_mm: float, vel: float, extra: float = 0.25) -> None:
    t = abs(dist_mm) / max(1e-6, vel) + extra
    time.sleep(t)


def wait_for_angle(ang_deg: float, vel: float, extra: float = 0.35) -> None:
    t = abs(ang_deg) / max(1e-6, vel) + extra
    time.sleep(t)


def get_pose(robot: Dorna) -> Optional[Tuple[float, float, float, float, float, float]]:
    try:
        pose = robot.get_all_pose()
        if pose and len(pose) >= 6:
            return tuple(float(x) for x in pose[:6])
    except Exception:
        return None
    return None


def move_rel(robot: Dorna, dx: float, dy: float, dz: float, vel: float = 5.0) -> None:
    robot.play_dict({"cmd": "lmove", "rel": 1, "x": dx, "y": dy, "z": dz, "vel": vel})
    wait_for_motion(max(abs(dx), abs(dy), abs(dz)), vel)


def move_to_pose(robot: Dorna, pose: Tuple[float, float, float, float, float, float], vel: float = 6.0) -> None:
    x, y, z, a, b, c = pose
    robot.play_dict({"cmd": "lmove", "rel": 0, "x": x, "y": y, "z": z, "a": a, "b": b, "c": c, "vel": vel})
    wait_for_motion(0.0, vel)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration steps
# ──────────────────────────────────────────────────────────────────────────────


def calibrate_pixel_jacobians(robot: Dorna, cams: Dict[str, UvcThread], move_mm: float, samples: int, base_pose: Tuple[float, float, float, float, float, float]) -> Dict[str, np.ndarray]:
    """
    Returns per-camera 2x3 Jacobians (du,dv per mm of world XYZ).
    """
    jac: Dict[str, np.ndarray] = {}
    axes = {"x": (move_mm, 0.0, 0.0), "y": (0.0, move_mm, 0.0), "z": (0.0, 0.0, move_mm)}
    base_tips = capture_tip_average(cams, samples=samples)
    if not base_tips:
        raise RuntimeError("No tip detected during Jacobian calibration.")

    for name in cams.keys():
        jac[name] = np.zeros((2, 3), dtype=float)

    for idx, (axis, vec) in enumerate(axes.items()):
        dx, dy, dz = vec
        move_rel(robot, dx, dy, dz, vel=5.0)
        tips_after = capture_tip_average(cams, samples=samples)
        move_rel(robot, -dx, -dy, -dz, vel=5.0)
        # return to base pose in case controller drifted
        move_to_pose(robot, base_pose, vel=6.0)
        for name, tip0 in base_tips.items():
            tip1 = tips_after.get(name)
            if tip1 is None:
                continue
            du = tip1[0] - tip0[0]
            dv = tip1[1] - tip0[1]
            denom = move_mm if move_mm != 0 else 1e-6
            col = 0 if axis == "x" else (1 if axis == "y" else 2)
            jac[name][0, col] = du / denom
            jac[name][1, col] = dv / denom
    return jac


def estimate_world_shift(duv_by_cam: Dict[str, Tuple[float, float]], jac_by_cam: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    rows = []
    b = []
    for name, duv in duv_by_cam.items():
        J = jac_by_cam.get(name)
        if J is None or J.shape != (2, 3):
            continue
        rows.append(J)
        b.append(np.array(duv, dtype=float))
    if not rows:
        return None
    A = np.vstack(rows)
    bvec = np.concatenate(b)
    sol, _, _, _ = np.linalg.lstsq(A, bvec, rcond=None)
    return sol


def measure_rotation(robot: Dorna, cams: Dict[str, UvcThread], jac_by_cam: Dict[str, np.ndarray], base_pose: Tuple[float, float, float, float, float, float], axis: str, ang_deg: float, samples: int, rot_vel: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, z0, a0, b0, c0 = base_pose
    R_base = axis_angle_to_R(a0, b0, c0)

    if axis == "z":
        R_motion = rot_z(ang_deg)
    elif axis == "x":
        R_motion = rot_x(ang_deg)
    else:
        R_motion = rot_y(ang_deg)

    # Capture baseline tip
    tip_before = capture_tip_average(cams, samples=samples)
    # Rotate TCP about its origin (absolute pose)
    R_target = R_base @ R_motion
    a_t, b_t, c_t = R_to_axis_angle(R_target)
    move_to_pose(robot, (x0, y0, z0, a_t, b_t, c_t), vel=rot_vel)
    wait_for_angle(ang_deg, rot_vel)
    tip_after = capture_tip_average(cams, samples=samples)
    # Return to base orientation
    move_to_pose(robot, base_pose, vel=rot_vel)
    wait_for_angle(ang_deg, rot_vel)

    duv = {}
    for name, pt0 in tip_before.items():
        pt1 = tip_after.get(name)
        if pt1 is None:
            continue
        duv[name] = (pt1[0] - pt0[0], pt1[1] - pt0[1])

    dp_world = estimate_world_shift(duv, jac_by_cam)
    if dp_world is None:
        raise RuntimeError(f"No tip shift detected for axis {axis}.")

    A = R_base @ (R_motion - np.eye(3))
    return A, dp_world


def solve_tool_offset(A_list: list[np.ndarray], dp_list: list[np.ndarray]) -> np.ndarray:
    A_mat = np.vstack(A_list)
    b_vec = np.concatenate(dp_list)
    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    return sol


# ──────────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Dual-camera tool-center calibration (headless).")
    p.add_argument("--host", "-H", default="10.42.0.11", help="Dorna controller IP")
    p.add_argument("--port", "-P", type=int, default=443, help="Dorna controller port")
    p.add_argument("--uvc1", help="Path to UVC camera #1 (defaults to first index0 device)")
    p.add_argument("--uvc2", help="Path to UVC camera #2 (defaults to second index0 device)")
    p.add_argument("--move-mm", type=float, default=0.8, help="Translation used to calibrate pixel/mm")
    p.add_argument("--rot-deg", type=float, default=30.0, help="Rotation used for each axis (deg)")
    p.add_argument("--axes", type=str, default="z,x,y", help="Comma list of axes to use (subset of z,x,y); use to skip risky pitches")
    p.add_argument("--lift-mm", type=float, default=5.0, help="Raise along +Z before each rotation to avoid table contact")
    p.add_argument("--samples", type=int, default=8, help="Frames to average per measurement")
    p.add_argument("--write-settings", action="store_true", help="Persist new tool center into settings.json")
    return p.parse_args()


def main():
    args = parse_args()
    settings = load_settings()

    tool_cx = float(settings.get("tool_cx", 0.0))
    tool_cy = float(settings.get("tool_cy", 0.0))
    tool_lz = float(settings.get("tool_lz", 205.0))

    # Camera setup
    if args.uvc1 and args.uvc2:
        uvc_paths = [args.uvc1, args.uvc2]
    else:
        picks = discover_uvc_index0(limit=2)
        if len(picks) < 2:
            raise SystemExit("Need two UVC cameras; supply with --uvc1/--uvc2.")
        uvc_paths = picks

    cams = {
        "uvc1": UvcThread(uvc_paths[0]),
        "uvc2": UvcThread(uvc_paths[1]),
    }
    for cam in cams.values():
        cam.start()

    t0 = time.time()
    while time.time() - t0 < 3.0 and any(c.latest() is None for c in cams.values()):
        time.sleep(0.05)
    for name, cam in cams.items():
        print(f"[Cam] {name}: {cam.status}")

    # Robot setup
    robot = Dorna(model="dorna_ta")
    if not robot.connect(host=args.host, port=args.port):
        raise SystemExit(f"Failed to connect to Dorna at {args.host}:{args.port}")
    robot.set_motor(1)
    robot.play_dict({"cmd": "tool", "id": 1, "lx": tool_cx, "ly": tool_cy, "lz": tool_lz})
    base_pose = get_pose(robot)
    if base_pose is None:
        raise SystemExit("Could not read robot pose; aborting.")
    print(f"[Robot] Base pose: x={base_pose[0]:.3f} y={base_pose[1]:.3f} z={base_pose[2]:.3f} a={base_pose[3]:.3f} b={base_pose[4]:.3f} c={base_pose[5]:.3f}")

    try:
        jac_by_cam = calibrate_pixel_jacobians(robot, cams, move_mm=args.move_mm, samples=args.samples, base_pose=base_pose)
        for name, J in jac_by_cam.items():
            print(f"[Calib] {name} Jacobian (px/mm):\n{J}")

        A_list = []
        dp_list = []
        axes = [ax.strip().lower() for ax in (args.axes or "").split(",") if ax.strip()]
        allowed = {"z", "x", "y"}
        axes = [ax for ax in axes if ax in allowed]
        if not axes:
            axes = ["z"]
        for axis in axes:
            print(f"[Calib] Measuring rotation about {axis}-axis by {args.rot_deg} deg…")
            pose_for_rot = base_pose
            lifted = False
            if args.lift_mm > 1e-6:
                lifted = True
                x0, y0, z0, a0, b0, c0 = base_pose
                pose_for_rot = (x0, y0, z0 + args.lift_mm, a0, b0, c0)
                move_to_pose(robot, pose_for_rot, vel=6.0)
                wait_for_motion(args.lift_mm, 6.0, extra=0.1)
            A_i, dp_i = measure_rotation(robot, cams, jac_by_cam, pose_for_rot, axis, args.rot_deg, samples=args.samples)
            if lifted:
                move_to_pose(robot, base_pose, vel=6.0)
                wait_for_motion(args.lift_mm, 6.0, extra=0.1)
            A_list.append(A_i)
            dp_list.append(dp_i)
            print(f"[Calib]  Δp_world = [{dp_i[0]:+.4f}, {dp_i[1]:+.4f}, {dp_i[2]:+.4f}] mm")

        r_tool = solve_tool_offset(A_list, dp_list)
        print(f"\nEstimated tool center offset (tool frame, mm):")
        print(f"  dx = {r_tool[0]:+.3f} mm")
        print(f"  dy = {r_tool[1]:+.3f} mm")
        print(f"  dz = {r_tool[2]:+.3f} mm")

        suggested = {
            "tool_cx": tool_cx + r_tool[0],
            "tool_cy": tool_cy + r_tool[1],
            "tool_lz": tool_lz + r_tool[2],
        }
        print("\nSuggested new tool center:")
        for k, v in suggested.items():
            print(f"  {k}: {v:.3f}")

        if args.write_settings:
            settings.update(suggested)
            save_settings(settings)
            print(f"[Save] Updated {SETTINGS_PATH} with new tool center.")

        # Log results for reference
        log = {
            "r_tool_mm": r_tool.tolist(),
            "jacobians_px_per_mm": {k: J.tolist() for k, J in jac_by_cam.items()},
            "base_pose": list(base_pose),
            "suggested": suggested,
            "rot_deg": args.rot_deg,
            "move_mm": args.move_mm,
        }
        with open("tcp_calibration_result.json", "w") as f:
            json.dump(log, f, indent=2)
            print("[Save] Wrote tcp_calibration_result.json")
    finally:
        for cam in cams.values():
            cam.stop()
        time.sleep(0.2)
        try:
            robot.set_motor(0)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
