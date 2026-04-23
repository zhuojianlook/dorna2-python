#!/usr/bin/env python3
import cv2
import os
import sys
import time
import glob
import re
import argparse
from datetime import datetime

# Quiet OpenCV warnings; keep errors
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    try:
        cv2.utils.logging.setLogLevel(2)
    except Exception:
        pass

# ------------------------------- Discovery helpers -------------------------------

def _read_text(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return ""

def _video_nodes_from_by_path_index0_and_1():
    """
    Return sorted list of (/dev/videoN) that are targets of by-path/*video-index0 or index1,
    but prioritize index0. We keep both because some devices only stream on index1.
    """
    nodes = set()
    for symlink in glob.glob("/dev/v4l/by-path/*video-index[01]"):
        try:
            real = os.path.realpath(symlink)
            if real.startswith("/dev/video"):
                nodes.add(real)
        except Exception:
            pass
    return sorted(
        nodes,
        key=lambda p: int(p.replace("/dev/video", "")) if p.replace("/dev/video", "").isdigit() else 9999
    )

def _symlink_map_for_node(root_glob, node):
    """Return list of symlinks under root_glob (e.g., /dev/v4l/by-id/*) that point to node."""
    out = []
    for symlink in glob.glob(root_glob):
        try:
            if os.path.realpath(symlink) == node:
                out.append(symlink)
        except Exception:
            pass
    return sorted(out)

def _v4l_name_for_node(node):
    """Read /sys/class/video4linux/videoN/name."""
    base = os.path.basename(node)  # videoN
    return _read_text(f"/sys/class/video4linux/{base}/name") or "unknown"

def _is_realsense_name(name):
    n = name.lower()
    return ("realsense" in n) or ("depth" in n) or ("infrared" in n)

def _looks_like_uvc(name):
    n = name.lower()
    # loose heuristics for UVC webcams
    return ("usb" in n and "camera" in n) or ("uvc" in n) or ("webcam" in n)

def _canon_base(path):
    """Collapse usbv2-/usbv3- to usb- so we dedupe multiple aliases of same physical port."""
    return path.replace("/usbv2-", "/usb-").replace("/usbv3-", "/usb-")

def discover_candidates():
    """
    Return (uvc_candidates, all_devices)
    each item: {node, name, by_id[], by_path[]}
    We dedupe by by-path base (usb/usbv2/usbv3) and prefer index0 symlink.
    """
    # collect all video-index[01] by-path symlinks
    bypath = sorted(glob.glob("/dev/v4l/by-path/*video-index[01]"))
    # bucket by canonical base (without -video-indexX and with usbv2/v3 collapsed)
    buckets = {}
    for link in bypath:
        base = re.sub(r"-video-index[01]$", "", link)
        base = _canon_base(base)
        buckets.setdefault(base, []).append(link)

    devices = []
    for base, links in buckets.items():
        # prefer index0, but keep index1 as an alternate
        links0 = [l for l in links if l.endswith("video-index0")]
        links1 = [l for l in links if l.endswith("video-index1")]
        pick = links0[0] if links0 else (links1[0] if links1 else None)
        if not pick:
            continue
        node = os.path.realpath(pick)
        name = _v4l_name_for_node(node)
        # get all aliases for the chosen /dev/videoN
        by_id = _symlink_map_for_node("/dev/v4l/by-id/*", node)
        by_path_all = _symlink_map_for_node("/dev/v4l/by-path/*", node)
        devices.append({"node": node, "name": name, "by_id": by_id, "by_path": by_path_all})

    # Filter obvious RealSense nodes; prefer ones that look like UVC
    uvc = [d for d in devices if not _is_realsense_name(d["name"]) and _looks_like_uvc(d["name"])]
    if not uvc:
        uvc = [d for d in devices if not _is_realsense_name(d["name"])]
    return uvc, devices

def print_inventory(uvc, all_devs):
    print("\n=== All by-path → /dev/videoN (index0/1 targets) ===")
    for d in all_devs:
        print(f"- {d['node']}: {d['name']}")
        if d["by_id"]:
            print(f"    by-id:   {d['by_id'][0]}")
            for extra in d["by_id"][1:]:
                print(f"             {extra}")
        if d["by_path"]:
            print(f"    by-path: {d['by_path'][0]}")
            for extra in d["by_path"][1:]:
                print(f"             {extra}")
    if not all_devs:
        print("  (none)")

    print("\n=== UVC candidates (preferred order) ===")
    if not uvc:
        print("  (none matched; you may need to pass --uvc1/--uvc2 explicitly)")
    for i, d in enumerate(uvc, 1):
        bid = d["by_id"][0] if d["by_id"] else "(none)"
        bpt = d["by_path"][0] if d["by_path"] else "(none)"
        print(f"  [{i}] {d['name']} -> {d['node']}")
        print(f"      by-id:   {bid}")
        print(f"      by-path: {bpt}")
    print()

# ------------------------------- OpenCV helpers ---------------------------------

def _decode_fourcc(v):
    try:
        v = int(v)
        return "".join([chr((v >> (8*i)) & 0xFF) for i in range(4)])
    except Exception:
        return "----"

def _dev_arg_for_backend(dev, backend):
    """
    For V4L2 we pass a numeric index (from /dev/videoN) when possible.
    For ANY we pass the string path (by-path/by-id or /dev/videoN).
    """
    if backend == "v4l2" and isinstance(dev, str):
        real = os.path.realpath(dev)
        m = re.match(r"^/dev/video(\d+)$", real)
        if m:
            return int(m.group(1))
    return dev

def try_open_once(dev, w, h, fps, backend, fourcc, warmup_reads=40):
    """Attempt a single open with given parameters. Returns (cap, frame, meta) or (None, None, meta)."""
    api_pref = cv2.CAP_V4L2 if backend == "v4l2" else cv2.CAP_ANY
    dev_arg = _dev_arg_for_backend(dev, backend)
    cap = cv2.VideoCapture(dev_arg, api_pref)
    if not cap.isOpened():
        return None, None, {"ok": False, "why": "open_failed", "backend": backend, "fourcc": fourcc, "fps": fps}

    # Apply settings
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_CONVERT_RGB,  1)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Warm up: read a few frames; some devices need a moment to deliver
    frame = None
    ok = False
    t0 = time.time()
    for _ in range(warmup_reads):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.01)
    if not ok or frame is None:
        cap.release()
        return None, None, {"ok": False, "why": "no_frames", "backend": backend, "fourcc": fourcc, "fps": fps}

    w_eff = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or frame.shape[1]
    h_eff = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
    fourcc_eff = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    fps_eff = cap.get(cv2.CAP_PROP_FPS) or fps
    return cap, frame, {"ok": True, "backend": backend, "fourcc": fourcc_eff or fourcc or "?", "fps": fps_eff, "w": w_eff, "h": h_eff}

def open_capture_resilient(dev, w, h, fps, backend, try_index1=True, verbose=True):
    """
    Try multiple combinations to get frames out of stubborn UVC devices.
    Also tries sibling by-path video-index1 if dev ends with index0 and fails.
    Returns (cap, meta_dict, used_path) or raises RuntimeError with diagnostics.
    """
    # Ordered attempts
    fourccs = ["YUYV", "MJPG", None]
    fps_try = [fps, 15, 10]
    backends = ["any", "v4l2"] if backend == "any" else ["v4l2", "any"]

    candidates = [dev]
    if try_index1 and isinstance(dev, str) and dev.endswith("video-index0"):
        sib = re.sub(r"video-index0$", "video-index1", dev)
        if os.path.exists(sib):
            candidates.append(sib)

    errors = []
    for path in candidates:
        for b in backends:
            for fc in fourccs:
                for f in fps_try:
                    cap, frame, meta = try_open_once(path, w, h, f, b, fc, warmup_reads=50)
                    if meta["ok"]:
                        if verbose:
                            print(f"  ✓ Opened {path} as {meta['fourcc']} {int(meta['w'])}x{int(meta['h'])}@{int(meta['fps'])} via {b.upper()}")
                        return cap, meta, path
                    else:
                        errors.append((path, meta))

    # If we get here, we failed every attempt
    lines = [f"Could not get frames from {dev}. Tried:"]
    for path, e in errors:
        lines.append(f"  - {e['why']} on {os.path.basename(path)} with backend={e['backend']}, fourcc={e['fourcc']}, fps={e['fps']}")
    lines.append("Hints: ensure 'uvcvideo' has quirks=0, no other app holds the device (use 'fuser /dev/videoN'),")
    lines.append("       and stick to index0 unless your device truly streams on index1.")
    raise RuntimeError("\n".join(lines))

def draw_tag(img, text, y=22):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x2 = 10 + tw + 10
    cv2.rectangle(img, (5, 5), (x2, 5 + 26), (0, 0, 0), -1)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

def make_panel(frame, label, flip=False, color=(0,255,0)):
    if frame is None:
        return None
    if flip:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    overlay = frame.copy()
    draw_tag(overlay, label)
    # border
    cv2.rectangle(overlay, (0,0), (overlay.shape[1]-1, overlay.shape[0]-1), color, 2)
    return overlay

def hstack_resize(a, b):
    """Resize b to a's height and hstack (handles None)."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if hb != ha:
        scale = ha / float(hb)
        b = cv2.resize(b, (int(wb*scale), ha), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([a, b])

# --------------------------------- Main script ----------------------------------

def build_args():
    p = argparse.ArgumentParser(
        description="Identify two UVC cameras (side-by-side), map /dev/videoN ↔ by-id/by-path, and preview.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--uvc1", default="", help="Explicit device path for left camera (by-path/by-id or /dev/videoN)")
    p.add_argument("--uvc2", default="", help="Explicit device path for right camera")
    p.add_argument("--width", type=int, default=640, help="Capture width")
    p.add_argument("--height", type=int, default=480, help="Capture height")
    p.add_argument("--fps", type=int, default=30, help="Requested FPS")
    p.add_argument("--no-flip", action="store_true", help="Disable default 180° rotation for both")
    p.add_argument("--backend", type=str, default="any", choices=["v4l2", "any"], help="OpenCV backend preference")
    p.add_argument("--list-only", action="store_true", help="List candidates and exit (no preview)")
    p.add_argument("--no-index1-fallback", action="store_true", help="Do not try sibling video-index1 on failure")
    return p.parse_args()

def main():
    args = build_args()

    # Discover devices
    uvc, all_devs = discover_candidates()
    print_inventory(uvc, all_devs)

    if args.list_only:
        return

    # Choose two devices
    devL = args.uvc1.strip()
    devR = args.uvc2.strip()

    if not devL or not devR:
        # Auto-pick from candidates
        if len(uvc) == 0:
            print("❌ No UVC candidates found. Pass --uvc1/--uvc2 explicitly (prefer by-path *video-index0).")
            return
        elif len(uvc) == 1:
            if not devL:
                devL = uvc[0]["by_path"][0] if uvc[0]["by_path"] else uvc[0]["node"]
            print("ℹ️ Only one UVC candidate detected; right pane will be empty unless you pass --uvc2.")
        else:
            if not devL:
                devL = uvc[0]["by_path"][0] if uvc[0]["by_path"] else uvc[0]["node"]
            if not devR:
                devR = uvc[1]["by_path"][0] if uvc[1]["by_path"] else uvc[1]["node"]

    print(f"[Open] Left : {devL or '(none)'}")
    print(f"[Open] Right: {devR or '(none)'}")

    # Open with resilient fallback
    capL = capR = None
    metaL = metaR = None
    usedL = usedR = None
    try:
        if devL:
            capL, metaL, usedL = open_capture_resilient(
                devL, args.width, args.height, args.fps, args.backend,
                try_index1=not args.no_index1_fallback, verbose=True
            )
        if devR:
            capR, metaR, usedR = open_capture_resilient(
                devR, args.width, args.height, args.fps, args.backend,
                try_index1=not args.no_index1_fallback, verbose=True
            )
    except RuntimeError as e:
        print("⚠️", e)

    if capL is None and capR is None:
        print("❌ Could not open any camera. Exiting.")
        return

    flipL = flipR = (not args.no_flip)
    swap = False
    win = "UVC Identify"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Status strings for panels
    labL = f"L | {os.path.basename(usedL) if usedL else os.path.basename(devL) if devL else '-'}"
    labR = f"R | {os.path.basename(usedR) if usedR else os.path.basename(devR) if devR else '-'}"
    if metaL:
        labL += f"  [{metaL['w']}x{metaL['h']}@{int(metaL['fps'])} {metaL['fourcc']} {metaL['backend'].upper()}]"
    if metaR:
        labR += f"  [{metaR['w']}x{metaR['h']}@{int(metaR['fps'])} {metaR['fourcc']} {metaR['backend'].upper()}]"

    print("\nPreview controls: q/ESC quit | 1 flip-left | 2 flip-right | space swap | s snapshot\n")

    try:
        while True:
            frameL = None
            frameR = None

            if capL is not None:
                ok, frameL = capL.read()
                if not ok or frameL is None:
                    frameL = None
            if capR is not None:
                ok, frameR = capR.read()
                if not ok or frameR is None:
                    frameR = None

            # Build panels
            panelL = make_panel(frameL, labL, flip=flipL, color=(0,255,0)) if frameL is not None else None
            panelR = make_panel(frameR, labR, flip=flipR, color=(255,200,0)) if frameR is not None else None

            out = hstack_resize(panelL if not swap else panelR, panelR if not swap else panelL)
            if out is None:
                time.sleep(0.01)
                continue

            # Instructions bar
            instr = "q/ESC=quit | 1=flip left | 2=flip right | space=swap | s=snapshot"
            (tw, th), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (5, out.shape[0]-th-14), (5+tw+10, out.shape[0]-4), (0,0,0), -1)
            cv2.putText(out, instr, (10, out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow(win, out)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):
                break
            elif key == ord('1'):
                flipL = not flipL
            elif key == ord('2'):
                flipR = not flipR
            elif key == ord(' '):
                swap = not swap
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                if panelL is not None:
                    cv2.imwrite(f"uvc_left_{ts}.jpg", panelL)
                if panelR is not None:
                    cv2.imwrite(f"uvc_right_{ts}.jpg", panelR)
                print(f"[Saved] snapshots at {ts}")

    finally:
        if capL is not None:
            capL.release()
        if capR is not None:
            capR.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
