#!/usr/bin/env python3
import sys
import argparse
import threading
import time

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
RIGHT_AXIS     = 5
LEFT_AXIS      = 2
RESET_BUTTON   = 2

def normalize(v):
    # normalize [-1..1] to [0..1]
    return (v + 1.0) / 2.0 if v < -0.2 or v > 1.0 else v

# ─── Argument parsing ────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Joystick control for Dorna + RealSense D405 + Stepper trigger bridge"
    )
    p.add_argument(
        "--host", "-H",
        default="10.42.0.11",
        help="IP address of your Dorna controller"
    )
    p.add_argument(
        "--port", "-P",
        type=int, default=443,
        help="Port for Dorna control (usually 443)"
    )
    return p.parse_args()

# ─── Helpers ─────────────────────────────────────────────────────────────
def axis_angle_to_R(a, b, c):
    vec = np.radians([a, b, c])
    th = np.linalg.norm(vec)
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

def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines, current = [], ''
    for w in words:
        test = f"{current} {w}".strip()
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines

# ─── Shared state ────────────────────────────────────────────────────────
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

# ─── Robot control thread (200 Hz) ───────────────────────────────────────
class RobotThread(threading.Thread):
    def __init__(self, state: SharedState, host: str, port: int):
        super().__init__(daemon=True)
        self.state      = state
        self.host       = host
        self.port       = port
        self.stop_event = threading.Event()
        # tuning constants
        self.BSX, self.BSJ5 = 5.0, 5.0
        self.BSB, self.BSC   = 0.5, 0.5
        self.BSH            = 5.0
        self.VT, self.VR    = 10.0, 5.0
        self.DZ, self.CD    = 0.1, 0.2

    def run(self):
        robot = Dorna()
        # retry connection up to 5 times
        for attempt in range(5):
            if robot.connect(host=self.host, port=self.port):
                break
            print(f"⚠️  Connection attempt {attempt+1}/5 to {self.host}:{self.port} failed. Retrying…")
            time.sleep(2)
        else:
            print(f"❌ All connection attempts to {self.host}:{self.port} failed.")
            sys.exit(1)

        robot.set_motor(1)
        robot.play_dict(cmd={"cmd":"tool","id":1,"lz":205})
        robot.play_dict(cmd={"cmd":"jmove","rel":0,"j5":0,"vel":5.0})

        x0,y0,z0,a0,b0,c0 = robot.get_all_pose()[:6]
        R = axis_angle_to_R(a0, b0, c0)
        j5v = 0.0

        next_t = time.time()
        while not self.stop_event.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += 1/200

            with self.state.lock:
                lx,ly = self.state.lx, self.state.ly
                rx,ry = self.state.rx, self.state.ry
                hx,hy = self.state.hx, self.state.hy
                sens  = self.state.levels[self.state.idx]

            sx, sj5, sb, sc, sh = (
                self.BSX*sens,
                self.BSJ5*sens,
                self.BSB*sens,
                self.BSC*sens,
                self.BSH*sens,
            )

            # LEFT stick: Z vs J5
            if abs(ly) > self.DZ and abs(ly) > abs(lx) + self.CD:
                d = -ly * sx
                zx,zy,zz = R[:,2]
                robot.play_dict({"cmd":"lmove","rel":1,
                                 "x":zx*d,"y":zy*d,"z":zz*d,
                                 "vel":self.VT,"cont":1})
                x0 += zx*d; y0 += zy*d; z0 += zz*d
            elif abs(lx) > self.DZ and abs(lx) > abs(ly) + self.CD:
                delta = lx * sj5
                j5v += delta
                R = R @ axis_angle_to_R(0, 0, delta)
                robot.play_dict({"cmd":"jmove","rel":1,
                                 "j5":delta,"vel":self.VR})
                with self.state.lock:
                    self.state.j5 = j5v

            # RIGHT stick: pitch & heading
            moved = False
            if abs(ry) > self.DZ and abs(ry) > abs(rx) + self.CD:
                moved = True
                R = R @ axis_angle_to_R(0, ry*sb, 0)
            elif abs(rx) > self.DZ and abs(rx) > abs(ry) + self.CD:
                moved = True
                R = axis_angle_to_R(0, 0, rx*sc) @ R

            if moved:
                a1,b1,c1 = R_to_axis_angle(R)
                robot.play_dict({"cmd":"lmove","rel":0,
                                 "x":x0,"y":y0,"z":z0,
                                 "a":a1,"b":b1,"c":c1,
                                 "vel":self.VR})
                tz = R[:,2]
                pitch = -np.degrees(np.arcsin(np.clip(tz[2], -1, 1)))
                with self.state.lock:
                    self.state.pitch = pitch

            # D-pad movement
            if hx != 0:
                tz = R[:,2]; proj = np.array([tz[0], tz[1], 0.0])
                n = np.linalg.norm(proj)
                perp = ((np.array([-proj[1], proj[0], 0.0]) / n) if n>1e-3
                        else np.array([0.0,-hx,0.0])) * hx
                dx,dy = perp[0]*sh, perp[1]*sh
                robot.play_dict({"cmd":"lmove","rel":1,
                                 "x":dx,"y":dy,"z":0.0,
                                 "vel":self.VT,"cont":1})
                x0 += dx; y0 += dy
            if hy != 0:
                dz = hy * sh
                robot.play_dict({"cmd":"lmove","rel":1,
                                 "x":0,"y":0,"z":dz,
                                 "vel":self.VT,"cont":1})
                z0 += dz

        robot.set_motor(0)
        robot.close()

    def stop(self):
        self.stop_event.set()

# ─── Main HUD + Side UI + Trigger Bridge (60 FPS) ─────────────────────────
def main():
    args = parse_args()

    # ─── Open serial port for trigger bridge ─────────────────────────────
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        time.sleep(2)  # allow Arduino reset
        print(f"[Bridge] Serial open {SERIAL_PORT} @ {BAUDRATE}")
    except Exception as e:
        print(f"⚠️ Could not open serial port: {e}", file=sys.stderr)
        ser = None
    last_rate = None

    pygame.init()
    pygame.font.init()
    if pygame.joystick.get_count() == 0:
        print("❌ No joystick detected.")
        return
    joy = pygame.joystick.Joystick(0); joy.init()

    # ─── Setup RealSense D405 ────────────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        pipeline.start(config)
    except Exception as e:
        print("❌ Failed to start RealSense pipeline:", e, file=sys.stderr)
        return

    # grab one frame to get dimensions
    try:
        init_frames = pipeline.wait_for_frames(timeout_ms=1000)
        init_color  = init_frames.get_color_frame()
        if not init_color:
            raise RuntimeError("Empty initial frame")
    except Exception as e:
        print("❌ Couldn't initialize RealSense:", e, file=sys.stderr)
        pipeline.stop()
        return

    frame = np.asanyarray(init_color.get_data())
    h, w   = frame.shape[:2]
    ui_w    = int(w * 0.25)

    screen = pygame.display.set_mode((w + ui_w, h))
    pygame.display.set_caption("Robotic Injector [SERI]")
    clock = pygame.time.Clock()

    # Fonts & layout params
    label_font       = pygame.font.SysFont("Consolas", 14, bold=True)
    value_font       = pygame.font.SysFont("Consolas", 12)
    placeholder_font = pygame.font.SysFont("Consolas", 12)
    m           = 10
    panel_size  = int(h * 0.12)
    diag_gap    = int(w * 0.06)
    padding     = 4
    label_h     = label_font.get_height()
    value_h     = value_font.get_height()
    H_panel     = panel_size + padding + label_h + padding + value_h + padding
    btn_w   = ui_w - 2*m
    btn_gap = 8
    buttons = [
        "Return to Start Pose",
        "Reload",
        "Wash",
        "Stop Motors and Engage Brakes",
        "Kill all Motors"
    ]

    state = SharedState()
    rt    = RobotThread(state, host=args.host, port=args.port)
    rt.start()

    prev_lb  = prev_rb  = False
    prev_idx = -1

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                rt.stop(); rt.join()
                pipeline.stop()
                if ser:
                    ser.close()
                pygame.quit()
                return

        # ─── Read joystick into shared state ──────────────────────────────
        lb, rb = joy.get_button(4), joy.get_button(5)
        with state.lock:
            state.lx, state.ly = joy.get_axis(0), joy.get_axis(1)
            state.rx, state.ry = joy.get_axis(3), joy.get_axis(4)
            state.hx, state.hy = joy.get_hat(0)
            state.lb, state.rb = lb, rb

        # ─── Sensitivity cycling on LB/RB ─────────────────────────────────
        if lb and not prev_lb and state.idx < len(state.levels)-1:
            state.idx += 1
        if rb and not prev_rb and state.idx > 0:
            state.idx -= 1
        prev_lb, prev_rb = lb, rb
        if state.idx != prev_idx:
            joy.rumble(0.3, 0.3, 200)
            prev_idx = state.idx

        # ─── Trigger handling → Arduino ────────────────────────────────────
        rt_val = normalize(joy.get_axis(RIGHT_AXIS))
        lt_val = normalize(joy.get_axis(LEFT_AXIS))
        if rt_val > DEADZONE:
            rate = int(rt_val * MAX_RATE)
        elif lt_val > DEADZONE:
            rate = -int(lt_val * MAX_RATE)
        else:
            rate = 0

        if ser:
            if rate != last_rate:
                ser.write(f"V{rate}\n".encode())
                last_rate = rate
                if ser.in_waiting:
                    ser.read(ser.in_waiting)
            if joy.get_button(RESET_BUTTON):
                ser.write(b"R\n")
                if ser.in_waiting:
                    ser.read(ser.in_waiting)
                last_rate = 0
                time.sleep(0.1)

        # ─── Grab RealSense frame (timeout) ───────────────────────────────
        try:
            frames      = pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
        except RuntimeError:
            continue

        # ─── Draw camera feed ─────────────────────────────────────────────
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        screen.blit(surf, (0, 0))

        # ─── All your existing HUD drawing code follows unmodified ─────────
        # central reticle
        cx, cy = w//2, h//2
        col = (0, 255, 127)
        pygame.draw.circle(screen, col, (cx, cy), 50, 2)
        for dx, dy in [(-70,0),(-20,0),(20,0),(70,0),(0,-70),(0,-20),(0,20),(0,70)]:
            ex, ey = cx+dx, cy+dy
            ix = cx + (np.sign(dx)*20 if dx else 0)
            iy = cy + (np.sign(dy)*20 if dy else 0)
            pygame.draw.line(screen, col, (ex, ey), (ix, iy), 2)

        with state.lock:
            p    = state.pitch
            j5v  = state.j5
            sens = state.levels[state.idx]
        sx, sj5, sb = rt.BSX*sens, rt.BSJ5*sens, rt.BSB*sens

        # Pitch panel
        y1 = h - H_panel - m
        x1 = m
        pan1 = pygame.Surface((panel_size, panel_size), pygame.SRCALPHA)
        pan1.fill((10,10,30,180))
        pivot = (5, panel_size//2)
        r_len = panel_size - pivot[0] - 5
        ang = np.radians(-p)
        ex = pivot[0] + r_len*np.cos(ang)
        ey = pivot[1] + r_len*np.sin(ang)
        pygame.draw.line(pan1, col, pivot, (ex, ey), 2)
        pygame.draw.circle(pan1, col, (int(ex), int(ey)), 4)
        screen.blit(pan1, (x1, y1))
        screen.blit(label_font.render("Pitch", True, col),
                    (x1, y1 + panel_size + padding))
        screen.blit(value_font.render(f"{p:+.5f}°", True, col),
                    (x1, y1 + panel_size + padding + label_h + padding))

        # Angle panel (compass)
        x2 = x1 + panel_size + diag_gap
        y2 = y1
        pan2 = pygame.Surface((panel_size, panel_size), pygame.SRCALPHA)
        pan2.fill((10,10,30,180))
        center = (panel_size//2, panel_size//2)
        radius = panel_size//2 - 5
        pygame.draw.circle(pan2, col, center, radius, 2)
        ang2 = np.radians(j5v - 90)
        ex2 = center[0] + radius*np.cos(ang2)
        ey2 = center[1] + radius*np.sin(ang2)
        pygame.draw.line(pan2, col, center, (ex2, ey2), 2)
        pygame.draw.polygon(pan2, col, [
            (ex2 + 4*np.cos(ang2),    ey2 + 4*np.sin(ang2)),
            (ex2 + 4*np.cos(ang2+0.3), ey2 + 4*np.sin(ang2+0.3)),
            (ex2 + 4*np.cos(ang2-0.3), ey2 + 4*np.sin(ang2-0.3))
        ])
        screen.blit(pan2, (x2, y2))
        screen.blit(label_font.render("Angle", True, col),
                    (x2, y2 + panel_size + padding))
        screen.blit(value_font.render(f"{-j5v:+.5f}°", True, col),
                    (x2, y2 + panel_size + padding + label_h + padding))

        # Sensitivity box
        heading = f"Sensitivity ({sens*100:.0f}%)"
        hdr = label_font.render(heading, True, (200,230,255))
        l1  = value_font.render(f"Move step:  ±{sx:.5f} mm", True, col)
        l2  = value_font.render(f"Angle step: ±{sj5:.5f}°", True, col)
        l3  = value_font.render(f"Pitch step: ±{sb:.5f}°", True, col)
        texts = [hdr, l1, l2, l3]
        sb_w = max(t.get_width() for t in texts) + 2*padding
        sb_h = sum(t.get_height() + padding for t in texts) + padding
        x3 = w - sb_w - m
        y3 = m
        sbpan = pygame.Surface((sb_w, sb_h), pygame.SRCALPHA)
        sbpan.fill((30,30,10,180))
        y_off = padding
        for t in texts:
            sbpan.blit(t, (padding, y_off))
            y_off += t.get_height() + padding
        screen.blit(sbpan, (x3, y3))

        # Side UI background
        pygame.draw.rect(screen, (20,20,20), (w, 0, ui_w, h))

        # Placeholder buttons
        y_off_btn = m
        for label in buttons:
            lines = wrap_text(label, placeholder_font, btn_w - 10)
            bh    = len(lines)*(placeholder_font.get_height()+2) + 10
            rect  = pygame.Rect(w + m, y_off_btn, btn_w, bh)
            pygame.draw.rect(screen, (40,40,60), rect)
            pygame.draw.rect(screen, col, rect, 2)
            ly2 = y_off_btn + 5
            for ln in lines:
                surf_txt = placeholder_font.render(ln, True, (200,200,200))
                tx = w + m + (btn_w - surf_txt.get_width())//2
                screen.blit(surf_txt, (tx, ly2))
                ly2 += placeholder_font.get_height() + 2
            y_off_btn += bh + btn_gap

        # Controller input overlays
        threshold = rt.DZ
        ov_sz     = 36
        ov_m      = 8
        sp        = ov_sz + ov_m
        start_x   = m
        start_y   = m

        ls_x, ls_y = state.lx, state.ly
        rs_x, rs_y = state.rx, state.ry
        dp_x, dp_y = state.hx, state.hy
        lb_btn     = state.lb
        rb_btn     = state.rb
        lt_btn     = joy.get_button(6)
        rt_btn     = joy.get_button(7)

        flags = {
            "LS↑": ls_y < -threshold,
            "LS↓": ls_y >  threshold,
            "LS←": ls_x < -threshold,
            "LS→": ls_x >  threshold,
            "RS↑": rs_y < -threshold,
            "RS↓": rs_y >  threshold,
            "RS←": rs_x < -threshold,
            "RS→": rs_x >  threshold,
            "DP↑": dp_y ==  1,
            "DP↓": dp_y == -1,
            "DP←": dp_x == -1,
            "DP→": dp_x ==  1,
             "LB": lb_btn,
             "RB": rb_btn,
             "LT": lt_btn,
             "RT": rt_btn,
        }

        def draw_btn(lbl, active, x, y):
            bg_col = (0,255,0) if active else (60,60,60)
            cx, cy = x+ov_sz//2, y+ov_sz//2
            pygame.draw.circle(screen, bg_col, (cx, cy), ov_sz//2)
            pygame.draw.circle(screen, col,    (cx, cy), ov_sz//2, 2)
            txt = placeholder_font.render(lbl, True, (255,255,255))
            tx = x + (ov_sz - txt.get_width())//2
            ty = y + (ov_sz - txt.get_height())//2
            screen.blit(txt, (tx, ty))

        for i, prefix in enumerate(["LS", "RS", "DP"]):
            base_x = start_x + i*3*sp
            base_y = start_y
            draw_btn(f"{prefix}↑", flags[f"{prefix}↑"], base_x+sp,   base_y)
            draw_btn(f"{prefix}←", flags[f"{prefix}←"], base_x,      base_y+sp)
            draw_btn(f"{prefix}→", flags[f"{prefix}→"], base_x+2*sp, base_y+sp)
            draw_btn(f"{prefix}↓", flags[f"{prefix}↓"], base_x+sp,   base_y+2*sp)

        ty = start_y + 3*sp + ov_m
        for j, lbl in enumerate(["LB", "RB", "LT", "RT"]):
            draw_btn(lbl, flags[lbl], start_x + j*sp, ty)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
