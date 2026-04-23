#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Two-step pre-home (bounded J5) + Joystick teleop + Arduino trigger bridge + HUD
# AUTO homing:
#   Phase 1 = Y/Z centroid alignment (smoothed, coarse/fine)
#   Phase 2 = Pivot-locked plane alignment about the camera optical center:
#             (a) align camera Z to plane normal (pitch/yaw),
#             (b) align heading/roll so the top edge is horizontal (mod 180°).
#
# ESC to quit; A toggles AUTO. Motors remain enabled on exit.
#
import sys
import argparse
import threading
import time

import pygame
import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import serial
from dorna2 import Dorna

# ─────────────────────────────── USER CONFIG ────────────────────────────────
DEFAULT_HOST = "10.42.0.11"
DEFAULT_PORT = 443

# Two-step pre-home (absolute, degrees)
POSE_STEP1_DEG = {"j0": 73.59, "j1": 67.43, "j2": -107.20, "j3": -11.10, "j4": -38.76, "j5": -4.00}
POSE_STEP2_DEG = {"j0": 20.81, "j1": 28.74, "j2": -114.30, "j3": 29.97, "j4": 95.76,  "j5": 27.00}

MAX_J5_DELTA_PREHOME = 45.0   # deg bound per pre-home step
J5_LIMITS = None              # e.g., (-90.0, 90.0) or None

VEL_PREHOME = 45.0            # deg/s (joint pre-home)
VEL_YZ      = 120.0           # mm/s for Y/Z in Phase 1
VEL_PIVOT   = 60.0            # mm/s for pivoted orientation updates in Phase 2

# AUTO Phase‑1 (centroid → Y/Z)
YZ_DISABLE_THR = 2.0          # mm threshold to switch to Phase 2
HYSTERESIS_MM  = 5.0          # coarse/fine boundary
MAX_STEP_MM    = 20.0         # max step per loop
FINE_STEP_MM   = 1.0          # fine step near zero
SMOOTH_FACTOR  = 0.4          # 0..1 exponential smoothing for dy,dz

# AUTO Phase‑2 (pivot orientation)
ORIENT_THRESH       = 1.0     # deg, stop threshold (for both steps)
ROLL_ENABLE_VIEW_THR= 8.0     # deg, only do roll after view-axis is mostly aligned
Kp_ORI              = 0.5     # proportional gain (view-axis align)
Kp_ROLL             = 0.8     # proportional gain (heading/roll align)
MAX_COARSE_DEG      = 5.0     # max step when |err| large
MAX_FINE_DEG        = 1.0     # max step when |err| small
COARSE_THRESH_DEG   = 10.0    # coarse/fine boundary
NORM_SMOOTH         = 0.4     # smoothing for plane normals (0..1)
MAX_DEVIATION_DEG   = 45.0    # NEVER exceed this from reference orientation

ARROW_LEN_M         = 0.05    # normal overlay length (m)
DEBUG_PRINTS        = True

# Camera mount relative to TCP (meters, in TCP coordinates):
# Example: camera 10 cm above TCP origin along +Y:
CAM_OFFSET_TCP_M = np.array([0.0, +0.10, 0.0], dtype=float)

# Camera→TCP rotation (camera pitched ~14° down about +X)
def axis_angle_to_R(a, b, c):
    # a,b,c are components of an axis-angle vector in DEGREES (axis * angle_deg)
    v = np.radians([a, b, c]); th = np.linalg.norm(v)
    if th < 1e-9: return np.eye(3)
    u = v / th; ux,uy,uz = u
    K = np.array([[0, -uz,  uy],[uz, 0, -ux],[-uy, ux,  0]])
    return np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*(K @ K)

def R_to_axis_angle(R):
    tr = np.trace(R)
    val = (tr - 1.0) / 2.0
    val = 1.0 if val > 1.0 else -1.0 if val < -1.0 else val
    th = np.arccos(val)
    if abs(th) < 1e-9:
        return (0.0, 0.0, 0.0)
    ux = (R[2,1] - R[1,2])/(2*np.sin(th))
    uy = (R[0,2] - R[2,0])/(2*np.sin(th))
    uz = (R[1,0] - R[0,1])/(2*np.sin(th))
    return tuple(np.degrees([ux*th, uy*th, uz*th]))

def rot_angle_deg(R):
    tr = np.trace(R)
    val = (tr - 1.0) / 2.0
    val = 1.0 if val > 1.0 else -1.0 if val < -1.0 else val
    return np.degrees(np.arccos(val))

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

R_cam2tcp = axis_angle_to_R(14.0, 0.0, 0.0)
cam_z_tcp = R_cam2tcp[:, 2]  # camera forward axis in TCP frame
cam_y_tcp = R_cam2tcp[:, 1]  # camera up axis in TCP frame

# ArUco
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
AR_DET     = aruco.ArucoDetector(ARUCO_DICT, aruco.DetectorParameters())

# ───────────────────── Trigger → Serial (Arduino) config ───────────────────
SERIAL_PORT  = '/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_44236313735351100201-if00'
BAUDRATE     = 115200
MAX_RATE     = 800
DEADZONE     = 0.02
RIGHT_AXIS   = 5
LEFT_AXIS    = 2
RESET_BUTTON = 2

def normalize(v):
    return (v + 1.0) / 2.0 if v < -0.2 or v > 1.0 else v

# ─────────────────────────── Utility helpers ───────────────────────────────
def clamp(val, lo, hi): return float(lo) if val < lo else float(hi) if val > hi else float(val)
def wrap_diff_deg(a, b): return (b - a + 180.0) % 360.0 - 180.0

def avg_distance(depth, cx, cy, sz=2):
    h, w = depth.get_height(), depth.get_width()
    vals=[]
    for dx in range(-sz, sz+1):
        for dy in range(-sz, sz+1):
            x = int(np.clip(cx+dx,0,w-1))
            y = int(np.clip(cy+dy,0,h-1))
            d = depth.get_distance(x,y)
            if d>0: vals.append(d)
    return float(np.mean(vals)) if vals else 0.0

def draw_center(corners):
    pts = np.vstack([c[0] for c in corners[:4]])
    return int(pts[:,0].mean()), int(pts[:,1].mean())

def compute_YZ_step(err_mm):
    if abs(err_mm) < HYSTERESIS_MM: step = np.sign(err_mm) * FINE_STEP_MM
    else:                            step = 0.6 * err_mm
    return clamp(step, -MAX_STEP_MM, MAX_STEP_MM)

def compute_board_normal_and_top_edge(corners, depth, intr):
    Pimg = [c[0].mean(axis=0) for c in corners[:3]]
    pts3d=[]
    for u,v in Pimg:
        d = avg_distance(depth, int(u), int(v))
        pts3d.append(np.array([((u-intr.ppx)/intr.fx)*d, ((v-intr.ppy)/intr.fy)*d, d]))
    n = unit(np.cross(pts3d[1]-pts3d[0], pts3d[2]-pts3d[0]))
    if n[2] > 0: n = -n  # point toward camera
    centers = [(c[0].mean(axis=0)[0], c[0].mean(axis=0)[1]) for c in corners[:4]]
    top2 = sorted(centers, key=lambda p: p[1])[:2]  # two smallest v (image y)
    return n, top2

# rotation clamp helper: ensure result stays within ±MAX_DEVIATION_DEG of reference
def clamp_step_to_window(R_cur, R_ref, axis_w, step_deg, max_dev_deg):
    if abs(step_deg) < 1e-6:
        return 0.0, R_cur
    R_try = axis_angle_to_R(*(axis_w * step_deg)) @ R_cur
    dev = rot_angle_deg(R_try @ R_ref.T)
    if dev <= max_dev_deg + 1e-6:
        return step_deg, R_try
    # binary search scale
    lo, hi = 0.0, 1.0
    R_ok = R_cur; s_ok = 0.0
    for _ in range(10):
        mid = 0.5*(lo+hi)
        R_mid = axis_angle_to_R(*(axis_w * (step_deg * mid))) @ R_cur
        dmid  = rot_angle_deg(R_mid @ R_ref.T)
        if dmid <= max_dev_deg:
            lo = mid; R_ok = R_mid; s_ok = mid
        else:
            hi = mid
    return step_deg * s_ok, R_ok

# ───────────────────────── Pre-home (bounded J5) ───────────────────────────
def read_current_joints(robot):
    try:
        vals = robot.get_all_joint()
        if isinstance(vals,(list,tuple)) and len(vals) >= 6:
            return {"j0":float(vals[0]), "j1":float(vals[1]), "j2":float(vals[2]),
                    "j3":float(vals[3]), "j4":float(vals[4]), "j5":float(vals[5])}
    except Exception as e:
        print(f"[WARN] get_all_joint() failed: {e}")
    out, ok = {}, True
    for i, name in enumerate(["j0","j1","j2","j3","j4","j5"]):
        try: out[name] = float(robot.get_joint(i))
        except Exception as e:
            print(f"[WARN] get_joint({i}) failed: {e}")
            ok = False; break
    return out if ok else None

def clamp_j5_absolute(target_j5):
    if J5_LIMITS is None: return float(target_j5)
    return clamp(float(target_j5), J5_LIMITS[0], J5_LIMITS[1])

def move_to_pose_bounded(robot, target_pose_deg, max_j5_delta=MAX_J5_DELTA_PREHOME, vel_deg_per_s=VEL_PREHOME, label="pre-home"):
    cur_j = read_current_joints(robot)
    if cur_j is None:
        cmd = {"cmd":"jmove","rel":0,"vel":float(vel_deg_per_s),"cont":0}
        for j in ["j0","j1","j2","j3","j4"]:
            cmd[j] = float(target_pose_deg[j])
        print(f"⚠️  Joints not readable → {label}: will NOT rotate J5.")
        robot.play_dict(cmd)
        print(f"✅ {label}: reached (J5 unchanged).")
        return

    cur_j5 = cur_j["j5"]; des_j5 = float(target_pose_deg["j5"])
    d_short = wrap_diff_deg(cur_j5, des_j5)
    if abs(d_short) > max_j5_delta:
        bounded_j5 = cur_j5 + np.sign(d_short)*max_j5_delta
        print(f"🔒 {label}: J5 Δ short={d_short:+.2f}° → bound target {bounded_j5:.2f}°")
    else:
        bounded_j5 = des_j5
        print(f"✅ {label}: J5 within ±{max_j5_delta}° (Δ={d_short:+.2f}°).")
    bounded_j5 = clamp_j5_absolute(bounded_j5)

    cmd = {"cmd":"jmove","rel":0,"vel":float(vel_deg_per_s),"cont":0}
    cmd.update({"j0":float(target_pose_deg["j0"]), "j1":float(target_pose_deg["j1"]),
                "j2":float(target_pose_deg["j2"]), "j3":float(target_pose_deg["j3"]),
                "j4":float(target_pose_deg["j4"]), "j5":float(bounded_j5)})
    print(f"➡️  {label}: moving @ {vel_deg_per_s}°/s")
    robot.play_dict(cmd)
    print(f"✅ {label}: done.")

def two_step_prehome_on(robot):
    print("✅ Motors ON. Two-step pre-home…")
    move_to_pose_bounded(robot, POSE_STEP1_DEG, label="Pre-home Step 1")
    move_to_pose_bounded(robot, POSE_STEP2_DEG, label="Pre-home Step 2")
    print("✅ Two-step pre-home complete (motors remain enabled).")

# ───────────────────── RealSense init with Color→IR fallback ───────────────
IR_INDEX = 1

def start_realsense_with_fallback():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    try:
        profile = pipeline.start(cfg)
        aligner = rs.align(rs.stream.color)
        vs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = vs.get_intrinsics()
        print("📷 RealSense mode: COLOR+DEPTH")
        return pipeline, aligner, intr, 'color'
    except RuntimeError as e:
        print(f"⚠️ COLOR start failed: {str(e).strip()} → IR+DEPTH fallback…")
        try: pipeline.stop()
        except Exception: pass

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, IR_INDEX, 640, 480, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.depth,                 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    aligner = rs.align(rs.stream.infrared)
    vs = profile.get_stream(rs.stream.infrared, IR_INDEX).as_video_stream_profile()
    intr = vs.get_intrinsics()
    print("📷 RealSense mode: IR+DEPTH")
    return pipeline, aligner, intr, 'ir'

# ───────────────────────── Shared state & Teleop thread ────────────────────
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.lx = self.ly = self.rx = self.ry = 0.0
        self.hx = self.hy = 0
        self.lb = self.rb = False
        self.levels = [0.01,0.05,0.1,0.2,0.4,0.8,1.0,2.0]
        self.idx = self.levels.index(1.0)
        self.pitch = 0.0
        self.j5    = 0.0
        self.auto_active = True  # AUTO starts ON

class RobotThread(threading.Thread):
    def __init__(self, state: SharedState, host: str, port: int, robot: Dorna | None):
        super().__init__(daemon=True)
        self.state = state
        self.host  = host
        self.port  = port
        self.stop_event = threading.Event()
        self.robot_lock = threading.Lock()
        self.robot = robot
        self.BSX,self.BSJ5 = 5.0,5.0
        self.BSB,self.BSC  = 0.5,0.5
        self.BSH = 5.0
        self.VT,self.VR = 10.0,5.0
        self.DZ,self.CD = 0.1,0.2

    def run(self):
        r = self.robot
        if r is None:
            r = Dorna()
            for attempt in range(10):
                if r.connect(host=self.host, port=self.port): break
                print(f"⚠️  Teleop connect attempt {attempt+1}/10 failed; retrying…", flush=True)
                time.sleep(1.5)
            else:
                print("❌ Teleop connect failed", flush=True); return
            r.set_motor(1)
            with self.robot_lock: self.robot = r
            print("✅ Teleop thread connected.", flush=True)
        else:
            print("✅ Teleop thread using existing Dorna connection.", flush=True)

        a0,b0,c0 = r.get_all_pose()[3:6]
        R = axis_angle_to_R(a0,b0,c0)
        j5v = 0.0
        last_sync_t = time.time()

        next_t = time.time()
        while not self.stop_event.is_set():
            now = time.time()
            if now < next_t: time.sleep(next_t - now)
            next_t += 1/200.0

            with self.state.lock:
                auto_on = self.state.auto_active
            if auto_on:
                if time.time() - last_sync_t > 0.5:
                    pose = r.get_all_pose()
                    a,b,c = pose[3:6]
                    R = axis_angle_to_R(a,b,c)
                    last_sync_t = time.time()
                continue

            with self.state.lock:
                lx,ly = self.state.lx, self.state.ly
                rx,ry = self.state.rx, self.state.ry
                hx,hy = self.state.hx, self.state.hy
                sens  = self.state.levels[self.state.idx]

            sx,sj5,sb,sc,sh = self.BSX*sens, self.BSJ5*sens, self.BSB*sens, self.BSC*sens, self.BSH*sens

            if abs(ly) > self.DZ and abs(ly) > abs(lx) + self.CD:
                d = -ly * sx
                zx,zy,zz = R[:,2]
                with self.robot_lock:
                    r.play_dict({"cmd":"lmove","rel":1, "x":zx*d,"y":zy*d,"z":zz*d, "vel":self.VT,"cont":1})
            elif abs(lx) > self.DZ and abs(lx) > abs(ly) + self.CD:
                delta = lx * sj5
                j5v += delta
                R = R @ axis_angle_to_R(0,0,delta)
                with self.robot_lock:
                    r.play_dict({"cmd":"jmove","rel":1,"j5":delta,"vel":self.VR})
                with self.state.lock:
                    self.state.j5 = j5v

            moved=False
            if abs(ry) > self.DZ and abs(ry) > abs(rx) + self.CD:
                moved=True; R = R @ axis_angle_to_R(0, ry*sb, 0)
            elif abs(rx) > self.DZ and abs(rx) > abs(ry) + self.CD:
                moved=True; R = axis_angle_to_R(0, 0, rx*sc) @ R

            if moved:
                pose = r.get_all_pose()[:6]
                x0,y0,z0 = pose[0:3]
                a1,b1,c1 = (R_to_axis_angle(R))
                with self.robot_lock:
                    r.play_dict({"cmd":"lmove","rel":0, "x":x0,"y":y0,"z":z0, "a":a1,"b":b1,"c":c1, "vel":self.VR})
                tz = R[:,2]
                pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
                with self.state.lock:
                    self.state.pitch = pitch

            if hx != 0 or hy != 0:
                if hx != 0:
                    tz = R[:,2]
                    proj = np.array([tz[0], tz[1], 0.0]); n = np.linalg.norm(proj)
                    perp = ((np.array([-proj[1], proj[0], 0.0]) / n) if n>1e-3 else np.array([0.0,-hx,0.0])) * hx
                    dx,dy = perp[0]*self.BSH, perp[1]*self.BSH
                    with self.robot_lock:
                        r.play_dict({"cmd":"lmove","rel":1,"x":dx,"y":dy,"z":0.0,"vel":self.VT,"cont":1})
                if hy != 0:
                    dz = hy * self.BSH
                    with self.robot_lock:
                        r.play_dict({"cmd":"lmove","rel":1,"x":0,"y":0,"z":dz,"vel":self.VT,"cont":1})

    def stop(self):
        self.stop_event.set()

# ──────────────────────────────── MAIN APP ─────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Two-step pre-home + Joystick teleop + AUTO homing (pivot plane+heading)"
    )
    p.add_argument("--host", "-H", default=DEFAULT_HOST, help="Dorna IP")
    p.add_argument("--port", "-P", type=int, default=DEFAULT_PORT, help="Dorna port (443)")
    return p.parse_args()

def start_ui_caption(mode):
    return f"Pre-home + Teleop + AUTO [{mode.upper()}]  |  A=toggle AUTO"

def main():
    args = parse_args()

    # Single shared Dorna session
    robot = Dorna()
    print(f"🔌 Connecting to Dorna at {args.host}:{args.port}…", flush=True)
    if not robot.connect(host=args.host, port=args.port):
        print("❌ Could not connect to Dorna.", file=sys.stderr); return
    robot.set_motor(1)
    print("✅ Connected and motors ON.", flush=True)

    # Two-step pre-home
    two_step_prehome_on(robot)

    # Save reference orientation (for ±45° heading/pitch/roll limit)
    a_ref, b_ref, c_ref = robot.get_all_pose()[3:6]
    R_ref = axis_angle_to_R(a_ref, b_ref, c_ref)

    # Serial bridge
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1); time.sleep(2)
        print(f"[Bridge] Serial open {SERIAL_PORT} @ {BAUDRATE}")
    except Exception as e:
        print(f"⚠️ Could not open serial port: {e}", file=sys.stderr); ser = None
    last_rate = None

    # Pygame & joystick
    pygame.init(); pygame.font.init()
    if pygame.joystick.get_count() == 0:
        print("❌ No joystick detected."); return
    joy = pygame.joystick.Joystick(0); joy.init()

    # RealSense
    try:
        pipeline, aligner, intr, rs_mode = start_realsense_with_fallback()
    except Exception as e:
        print("❌ Failed to start RealSense pipeline:", e, file=sys.stderr); return

    # Init frame
    init_frames = pipeline.wait_for_frames(timeout_ms=1000)
    aligned     = aligner.process(init_frames)
    if rs_mode == 'color':
        vf = aligned.get_color_frame(); img0 = np.asanyarray(vf.get_data()) if vf else None
    else:
        vf = aligned.get_infrared_frame(IR_INDEX); img0 = cv2.cvtColor(np.asanyarray(vf.get_data()), cv2.COLOR_GRAY2BGR) if vf else None
    if img0 is None:
        print("❌ Empty initial frame"); pipeline.stop(); return

    h, w = img0.shape[:2]; ui_w = int(w * 0.25)
    screen = pygame.display.set_mode((w + ui_w, h))
    pygame.display.set_caption(start_ui_caption(rs_mode))
    clock = pygame.time.Clock()

    # Teleop thread (shared robot)
    state = SharedState()
    rt    = RobotThread(state, host=args.host, port=args.port, robot=robot)
    rt.start()
    print("✅ Teleop thread started (shared connection).", flush=True)

    # AUTO state (starts ON)
    with state.lock: state.auto_active = True
    auto_on = True
    prev_dy = prev_dz = 0.0
    auto_status = "AUTO: ON (Phase 1)"
    n_filt = None  # filtered board normal

    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: raise KeyboardInterrupt
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE: raise KeyboardInterrupt
                    if ev.key == pygame.K_a:
                        auto_on = not auto_on
                        with state.lock: state.auto_active = auto_on
                        auto_status = "AUTO: ON (Phase 1)" if auto_on else "AUTO: OFF"
                        print(f"▶ {auto_status}")

            # Joystick → shared state
            raw_hx, raw_hy = joy.get_hat(0)
            lb, rb = joy.get_button(4), joy.get_button(5)
            with state.lock:
                state.lx, state.ly = joy.get_axis(0), joy.get_axis(1)
                state.rx, state.ry = -joy.get_axis(3), joy.get_axis(4)
                state.hx, state.hy = -raw_hx, raw_hy
                state.lb, state.rb = lb, rb

            # Triggers → Arduino
            rt_val = normalize(joy.get_axis(RIGHT_AXIS))
            lt_val = normalize(joy.get_axis(LEFT_AXIS))
            rate = int(rt_val*MAX_RATE) if rt_val > DEADZONE else -int(lt_val*MAX_RATE) if lt_val > DEADZONE else 0
            if ser:
                if rate != last_rate:
                    ser.write(f"V{rate}\n".encode()); last_rate = rate
                    if ser.in_waiting: ser.read(ser.in_waiting)
                if joy.get_button(RESET_BUTTON):
                    ser.write(b"R\n"); 
                    if ser.in_waiting: ser.read(ser.in_waiting)
                    last_rate = 0; time.sleep(0.1)

            # Frames
            try:
                frames  = pipeline.wait_for_frames(timeout_ms=500)
                aligned = aligner.process(frames)
                df      = aligned.get_depth_frame()
                if rs_mode == 'color':
                    vf = aligned.get_color_frame()
                    if not vf or not df: continue
                    img_bgr = np.asanyarray(vf.get_data())
                    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                else:
                    vf = aligned.get_infrared_frame(IR_INDEX)
                    if not vf or not df: continue
                    ir     = np.asanyarray(vf.get_data())
                    img_bgr = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
                    gray    = ir
            except RuntimeError:
                continue

            overlay = img_bgr.copy()

            # AUTO
            if auto_on:
                corners, ids, _ = AR_DET.detectMarkers(gray)
                if ids is not None and len(ids) >= 4:
                    aruco.drawDetectedMarkers(overlay, corners, ids)
                    cx, cy = draw_center(corners)
                    d  = avg_distance(df, cx, cy)
                    Xc = ((cx - intr.ppx) / intr.fx) * d  # right +X
                    Yc = ((cy - intr.ppy) / intr.fy) * d  # down  +Y
                    errY_mm = -Xc * 1000.0
                    errZ_mm = -Yc * 1000.0

                    # Phase 1: Y/Z centroid
                    if abs(errY_mm) > YZ_DISABLE_THR or abs(errZ_mm) > YZ_DISABLE_THR:
                        raw_dy = compute_YZ_step(errY_mm)
                        raw_dz = compute_YZ_step(errZ_mm)
                        dy = SMOOTH_FACTOR*raw_dy + (1.0 - SMOOTH_FACTOR)*prev_dy
                        dz = SMOOTH_FACTOR*raw_dz + (1.0 - SMOOTH_FACTOR)*prev_dz
                        prev_dy, prev_dz = dy, dz
                        with rt.robot_lock:
                            robot.play_dict({"cmd":"lmove","rel":1,"x":0.0,"y":float(dy),"z":float(dz),
                                             "vel":VEL_YZ,"cont":1})
                        auto_status = f"AUTO: Phase 1 YZ  dY={dy:+.1f}mm  dZ={dz:+.1f}mm"

                    # Phase 2: Plane + Heading (both pivot-locked)
                    else:
                        # --- normal and top-edge ---
                        n_cam, top2 = compute_board_normal_and_top_edge(corners, df, intr)
                        if n_filt is None: n_filt = n_cam
                        else: n_filt = unit(NORM_SMOOTH*n_cam + (1.0 - NORM_SMOOTH)*n_filt)
                        n_cam = n_filt

                        # Draw top edge (blue) and normal (red)
                        pt1 = (int(top2[0][0]), int(top2[0][1]))
                        pt2 = (int(top2[1][0]), int(top2[1][1]))
                        cv2.line(overlay, pt1, pt2, (255,0,0), 2)
                        nz = n_cam[2] if abs(n_cam[2]) > 1e-9 else 1e-9
                        du = intr.fx*(n_cam[0]/nz)*ARROW_LEN_M
                        dv = intr.fy*(n_cam[1]/nz)*ARROW_LEN_M
                        cv2.arrowedLine(overlay, (cx,cy), (int(cx+du), int(cy+dv)), (0,0,255), 2, tipLength=0.2)

                        # Current pose/rotation
                        with rt.robot_lock:
                            pose = robot.get_all_pose()[:6]
                        x0,y0,z0 = pose[0:3]
                        a0,b0,c0 = pose[3:6]
                        R_tool   = axis_angle_to_R(a0,b0,c0)

                        # World vectors
                        camZ_w   = R_tool.dot(cam_z_tcp)
                        world_n  = R_tool.dot(R_cam2tcp.dot(n_cam))  # plane normal in world
                        target   = -world_n                           # we want camZ_w → target

                        # ===== Step (a): view-axis alignment (pitch/yaw) =====
                        dot      = np.clip(camZ_w.dot(target), -1.0, 1.0)
                        err_ang1 = np.degrees(np.arccos(dot))
                        step1 = 0.0
                        R_mid = R_tool
                        dx1 = dy1 = dz1 = 0.0
                        if err_ang1 > ORIENT_THRESH:
                            axis1_w = unit(np.cross(camZ_w, target))
                            raw_step1 = err_ang1 * Kp_ORI
                            max_step1 = MAX_COARSE_DEG if err_ang1 > COARSE_THRESH_DEG else MAX_FINE_DEG
                            step1 = clamp(raw_step1, -max_step1, +max_step1)  # magnitude; sign from axis
                            # clamp to ±45° window
                            step1, R_mid = clamp_step_to_window(R_tool, R_ref, axis1_w, step1, MAX_DEVIATION_DEG)
                            # pivot translation to keep camera fixed
                            delta_world_m1 = (R_tool - R_mid).dot(CAM_OFFSET_TCP_M)
                            dx1,dy1,dz1 = (delta_world_m1 * 1000.0).tolist()

                        # Update intermediates for step (b)
                        camZ_mid = R_mid.dot(cam_z_tcp)

                        # ===== Step (b): roll/heading alignment (top edge horizontal, mod 180°) =====
                        step2 = 0.0
                        R_new = R_mid
                        dx2 = dy2 = dz2 = 0.0

                        if err_ang1 < ROLL_ENABLE_VIEW_THR:
                            v_img = np.array([pt2[0]-pt1[0], pt2[1]-pt1[1]], dtype=float)
                            ang_deg = np.degrees(np.arctan2(v_img[1], v_img[0]))  # [-180, 180]
                            # deviation to nearest horizontal (0 or 180) → [-90, +90]
                            roll_err_deg = ((ang_deg + 90.0) % 180.0) - 90.0

                            if abs(roll_err_deg) > ORIENT_THRESH:
                                # rotate around current camera Z (in world)
                                axis2_w = unit(camZ_mid)
                                raw_step2 = Kp_ROLL * roll_err_deg      # ← sign FIXED (positive)
                                max_step2 = MAX_COARSE_DEG if abs(roll_err_deg) > COARSE_THRESH_DEG else MAX_FINE_DEG
                                step2 = clamp(raw_step2, -max_step2, +max_step2)
                                step2, R_new = clamp_step_to_window(R_mid, R_ref, axis2_w, step2, MAX_DEVIATION_DEG)
                                # pivot translation for step (b)
                                delta_world_m2 = (R_mid - R_new).dot(CAM_OFFSET_TCP_M)
                                dx2,dy2,dz2 = (delta_world_m2 * 1000.0).tolist()
                        else:
                            R_new = R_mid  # skip roll this cycle

                        # ----- Apply combined absolute pose -----
                        if (abs(step1) > 1e-3) or (abs(step2) > 1e-3):
                            a1,b1,c1 = R_to_axis_angle(R_new)
                            Xabs = x0 + dx1 + dx2
                            Yabs = y0 + dy1 + dy2
                            Zabs = z0 + dz1 + dz2
                            if DEBUG_PRINTS:
                                dev = rot_angle_deg(R_new @ R_ref.T)
                                # For logs, show roll_ang only when computed
                                if err_ang1 < ROLL_ENABLE_VIEW_THR:
                                    print(f"[PIVOT] view_err={err_ang1:5.1f}°, step1={step1:+4.1f}° | "
                                          f"roll_ang={ang_deg:+6.1f}°, roll_err={roll_err_deg:+6.1f}°, step2={step2:+4.1f}° | "
                                          f"Δmm=({dx1+dx2:+.1f},{dy1+dy2:+.1f},{dz1+dz2:+.1f}) | dev→ref={dev:4.1f}°")
                                else:
                                    print(f"[PIVOT] view_err={err_ang1:5.1f}°, step1={step1:+4.1f}° | "
                                          f"(roll gated) Δmm=({dx1:+.1f},{dy1:+.1f},{dz1:+.1f}) | dev→ref={dev:4.1f}°")
                            with rt.robot_lock:
                                robot.play_dict({"cmd":"lmove","rel":0,
                                                 "x":float(Xabs), "y":float(Yabs), "z":float(Zabs),
                                                 "a":float(a1),   "b":float(b1),   "c":float(c1),
                                                 "vel":VEL_PIVOT, "cont":0})
                            auto_status = "AUTO: Phase 2 PIVOT (view+roll)"
                        else:
                            auto_status = "AUTO: Phase 2 PIVOT aligned (≤ tol)"
                else:
                    auto_status = "AUTO: waiting for ≥4 markers…"
            else:
                auto_status = "AUTO: OFF"

            # Camera blit
            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
            screen.blit(surf, (0, 0))

            # Minimal HUD
            txt = pygame.font.SysFont("Consolas", 12).render(auto_status, True, (255,255,0))
            screen.blit(txt, (10, 10))
            pygame.display.set_caption(start_ui_caption(rs_mode))
            pygame.display.flip()
            clock.tick(60)

    except KeyboardInterrupt:
        pass
    finally:
        try: pipeline.stop()
        except Exception: pass
        rt.stop(); rt.join(timeout=2.0)
        try: robot.close()
        except Exception: pass
        if ser: ser.close()
        pygame.quit()
        print("✅ Exit. Motors remain enabled.")

if __name__ == "__main__":
    main()
