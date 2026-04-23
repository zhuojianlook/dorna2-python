#!/usr/bin/env python3
import os, re, glob, time, shlex, argparse, subprocess
from datetime import datetime
import cv2

# Quiet OpenCV warnings; keep errors
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    try: cv2.utils.logging.setLogLevel(2)
    except Exception: pass

def read(p):
    try: return open(p).read().strip()
    except: return ""

def list_video_nodes():
    nodes = sorted(glob.glob("/dev/video[0-9]*"),
                   key=lambda p: int(re.sub(r"^/dev/video", "", p)))
    out=[]
    for n in nodes:
        base=os.path.basename(n)
        name=read(f"/sys/class/video4linux/{base}/name") or "unknown"
        byp=[l for l in glob.glob("/dev/v4l/by-path/*") if os.path.realpath(l)==n]
        out.append({"node":n,"name":name,"by_path":sorted(byp)})
    return out

def which(cmd):
    for d in os.environ.get("PATH","").split(os.pathsep):
        p=os.path.join(d,cmd)
        if os.path.exists(p) and os.access(p,os.X_OK):
            return p
    return None

def v4l2_pair(devA, devB, W,H,F, frames=120, timeout=12):
    def cmd(dev): 
        return shlex.split(
            f'v4l2-ctl -d {shlex.quote(dev)} --set-fmt-video=width={W},height={H},pixelformat=YUYV '
            f'--set-parm={F} --stream-mmap --stream-count={frames} --stream-to=/dev/null'
        )
    pA = subprocess.Popen(cmd(devA), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.15)  # slight stagger
    pB = subprocess.Popen(cmd(devB), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0=time.time()
    while True:
        if pA.poll() is not None and pB.poll() is not None: break
        if time.time()-t0 > timeout:
            for p in (pA,pB):
                try: p.kill()
                except: pass
            break
        time.sleep(0.05)
    outA,errA = pA.communicate(timeout=1) if pA.poll() is not None else ("","Timed out")
    outB,errB = pB.communicate(timeout=1) if pB.poll() is not None else ("","Timed out")
    ok = (pA.returncode==0 and pB.returncode==0)
    return ok, errA.strip(), errB.strip()

def decode_fourcc(v):
    try:
        v=int(v); return "".join([chr((v>>(8*i))&0xFF) for i in range(4)])
    except Exception: return "----"

def open_cap_capany(dev_path, w, h, fps):
    """Open by string path with CAP_ANY only (no index conversion)."""
    cap = cv2.VideoCapture(dev_path, cv2.CAP_ANY)
    if not cap.isOpened():
        return None, "open_failed"
    # Prefer YUYV negotiaton; CAP_ANY often honors size/fps directly
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_CONVERT_RGB,  1)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    # Warm up a few frames
    ok, frame = False, None
    for _ in range(40):
        ok, frame = cap.read()
        if ok and frame is not None: break
        time.sleep(0.01)
    if not ok or frame is None:
        cap.release()
        return None, "no_frames"

    meta = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or frame.shape[1],
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0],
        "fps": int(cap.get(cv2.CAP_PROP_FPS) or fps),
        "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)) or "?"
    }
    return (cap, meta), None

def tag(img, txt):
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img,(5,5),(5+tw+10,5+26),(0,0,0),-1)
    cv2.putText(img,txt,(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return img

def hstack_resize(a,b):
    if a is None and b is None: return None
    if a is None: return b
    if b is None: return a
    ha,wa=a.shape[:2]; hb,wb=b.shape[:2]
    if hb!=ha:
        scale=ha/float(hb)
        b=cv2.resize(b,(int(wb*scale),ha),interpolation=cv2.INTER_AREA)
    return cv2.hconcat([a,b])

def main():
    ap=argparse.ArgumentParser(description="Kernel+OpenCV two-cam simultaneous tester (CAP_ANY only).")
    ap.add_argument("--left",  default="", help="Left device (e.g., /dev/v4l/by-path/...index0 or /dev/videoN)")
    ap.add_argument("--right", default="", help="Right device")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height",type=int, default=480)
    ap.add_argument("--fps",   type=int, default=15)
    ap.add_argument("--frames",type=int, default=150, help="Kernel test frames per stream")
    args=ap.parse_args()

    devs=list_video_nodes()
    print("\n=== Detected /dev/video* ===")
    for i,d in enumerate(devs):
        print(f"[{i}] {d['node']:>10}  {d['name']}")
        for p in d["by_path"]:
            print(f"     by-path: {p.replace('/dev/v4l/by-path/','')}")

    def pick(prompt):
        while True:
            s=input(prompt).strip()
            if s.startswith("/dev/"): return s
            if s.isdigit():
                idx=int(s)
                if 0<=idx<len(devs): return devs[idx]["node"]
                print("Index out of range.")
            else:
                print("Enter /dev/videoN or a listed index.")

    left  = args.left  or pick("\nSelect LEFT  camera (index or /dev/videoN): ")
    right = args.right or pick("Select RIGHT camera (index or /dev/videoN): ")
    if left == right:
        print("Left and Right cannot be the same."); return

    print(f"\n[Kernel test] {left} + {right} @ {args.width}x{args.height}@{args.fps} ...")
    if which("v4l2-ctl") is None:
        print("ERROR: v4l2-ctl not found (sudo apt install v4l-utils)."); return
    ok,ea,eb = v4l2_pair(left, right, args.width, args.height, args.fps, frames=args.frames)
    if ok:
        print("PASS: kernel could stream both simultaneously.")
    else:
        print("FAIL: kernel could NOT stream both simultaneously.")
        if ea: print("  A:", ea)
        if eb: print("  B:", eb)
        print("No point testing OpenCV until kernel PASS; try different ports or lower settings.")
        return

    print("\n[OpenCV CAP_ANY] opening both by string path…")
    LR = []
    for dev in (left, right):
        res, err = open_cap_capany(dev, args.width, args.height, args.fps)
        if err:
            print(f"  {dev}: {err}")
            LR.append((None, None))
        else:
            cap, meta = res
            print(f"  {dev}: {meta['w']}x{meta['h']}@{meta['fps']} {meta['fourcc']} via CAP_ANY")
            LR.append((cap, meta))

    capL, metaL = LR[0]
    capR, metaR = LR[1]
    if not capL or not capR:
        print("One of the OpenCV streams failed to open. Consider switching that one to FFmpeg/PyAV for capture.")
        return

    print("\nPreview controls: q/ESC quit | s snapshot")
    win="Two Cams (CAP_ANY)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            okL,fL = capL.read(); okR,fR = capR.read()
            if okL and fL is not None:
                fL=tag(fL, f"{os.path.basename(left)} | {metaL['w']}x{metaL['h']}@{metaL['fps']} {metaL['fourcc']}")
            else:
                fL=None
            if okR and fR is not None:
                fR=tag(fR, f"{os.path.basename(right)} | {metaR['w']}x{metaR['h']}@{metaR['fps']} {metaR['fourcc']}")
            else:
                fR=None
            out=hstack_resize(fL,fR)
            if out is None:
                time.sleep(0.01); continue
            cv2.imshow(win,out)
            k=cv2.waitKey(1)&0xFF
            if k in (27, ord('q')): break
            elif k==ord('s'):
                ts=datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"two_cams_{ts}.jpg", out)
                print(f"[Saved] two_cams_{ts}.jpg")
    finally:
        if capL: capL.release()
        if capR: capR.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: pass
