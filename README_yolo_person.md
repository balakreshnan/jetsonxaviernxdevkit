# YOLO Vision on Jetson Xavier NX

Two scripts in `~/projects/led`, both reading a USB camera and running on the Jetson GPU (CUDA) every 5 seconds:

- **`yolo_person.py`** — person detection only. Small, fast, a good first test.
- **`yolo_scene.py`** — everything: all 80 COCO objects, per-person tracking with a persistent ID and how long each person has been visible (first seen, last seen, total), pose (skeleton, torso angle, keypoints, movement), activity (active / still / lying-sleeping) and facial expression, plus GPU utilisation, temperature and CUDA memory on every log line.

USB cameras get renumbered after a reboot or re-plug, so always check `ls /dev/video*` first. `yolo_person.py` defaults to `/dev/video1`; `yolo_scene.py` defaults to `/dev/video0` and also accepts a stable `/dev/v4l/by-id/...` path.

Both print what they found and write the annotated frame to `latest.jpg` (or show a live window when a display is available).

---

# Part 1 — `yolo_person.py`

`yolo_person.py` reads a USB camera, runs a small YOLO model (YOLO11n) on the Jetson GPU every 5 seconds, prints the objects it found with confidence and bounding-box coordinates, and writes the annotated frame to `latest.jpg` (or shows a live window when a display is available).

Project folder: `~/projects/led`
Camera: `/dev/video1`

## Requirements

- Jetson Xavier NX, JetPack 5.x, Python 3.8
- PyTorch with CUDA (NVIDIA's Jetson wheel, not the PyPI one) and matching torchvision
- `ultralytics` 8.3.x (YOLO11 support, Python 3.8 compatible)
- OpenCV (`cv2`) — included with JetPack
- USB camera on `/dev/video1`

## Setup (once, over SSH)

Connect:

```bash
ssh aioffice@aioffice-desktop
```

Install PyTorch with CUDA for JetPack 5.1.2 (check your version with `cat /etc/nv_tegra_release`; other 5.1.x wheels are at https://developer.download.nvidia.com/compute/redist/jp/):

```bash
sudo apt-get update && sudo apt-get install -y libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev
pip3 install --upgrade pip
pip3 install https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

Build matching torchvision (~15 min; add `MAX_JOBS=2` before `python3` if the board runs out of RAM):

```bash
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
cd torchvision && export BUILD_VERSION=0.16.1
python3 setup.py install --user && cd ..
```

Install Ultralytics and verify the GPU is visible:

```bash
pip3 install "ultralytics>=8.3,<8.4"
python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # must print True
```

Download the model:

```bash
mkdir -p ~/projects/led && cd ~/projects/led
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

If that fails to download, fetch it directly:

```bash
wget https://github.com/ultralytics/assets/releases/latest/download/yolo11n.pt
```

## Create the script

Paste this block into the SSH terminal:

```bash
mkdir -p ~/projects/led && cd ~/projects/led

cat > yolo_person.py << 'EOF2'
"""
Detect people from a USB camera on Jetson Xavier NX using a small YOLO model.
Runs one inference every INTERVAL seconds on the GPU, draws bounding boxes,
prints what was found, and shows the frame (or saves it when headless).
"""
import argparse
import sys
import time
import cv2
import torch
from ultralytics import YOLO

PERSON_CLASS = 0   # COCO class id for "person"

def log(msg):
    print('[{}] {}'.format(time.strftime('%H:%M:%S'), msg), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=1, help='USB camera index (/dev/videoN)')
    ap.add_argument('--model', default='yolo11n.pt', help='model file (.pt or .engine)')
    ap.add_argument('--interval', type=float, default=5.0, help='seconds between detections')
    ap.add_argument('--conf', type=float, default=0.4, help='confidence threshold')
    ap.add_argument('--save', action='store_true', help='save latest.jpg instead of showing a window')
    ap.add_argument('--all', action='store_true', help='detect all classes, not only person')
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else 'cpu'
    log('Torch {}  device: {}'.format(torch.__version__,
        'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'))

    log('Loading model {} ...'.format(args.model))
    t0 = time.time()
    model = YOLO(args.model)
    log('Model loaded in {:.1f} s, {} classes'.format(time.time() - t0, len(model.names)))
    classes = None if args.all else [PERSON_CLASS]

    log('Opening camera /dev/video{} ...'.format(args.cam))
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit('Cannot open camera {} - check ls /dev/video*'.format(args.cam))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit('Camera opened but returned no frame - try another --cam index')
    log('Camera OK, frame {}x{}'.format(frame.shape[1], frame.shape[0]))

    log('Warming up GPU (first inference can take 30-90 s on Jetson) ...')
    t0 = time.time()
    model.predict(frame, device=device, verbose=False, imgsz=640)
    log('Warm-up done in {:.1f} s'.format(time.time() - t0))

    log('Detecting every {} s. Press CTRL+C to stop.'.format(args.interval))
    next_run = time.time()
    try:
        while True:
            ok, frame = cap.read()      # keep draining frames so the buffer stays fresh
            if not ok:
                log('Frame grab failed, retrying...')
                time.sleep(0.5)
                continue

            if time.time() < next_run:
                continue
            next_run = time.time() + args.interval

            t0 = time.time()
            result = model.predict(frame, device=device, conf=args.conf,
                                   classes=classes, imgsz=640, verbose=False)[0]
            ms = (time.time() - t0) * 1000

            found = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                found.append('{} {:.0%} at [{},{},{},{}]'.format(label, conf, x1, y1, x2, y2))

            if found:
                log('{:.0f} ms  {} object(s): {}'.format(ms, len(found), '; '.join(found)))
            else:
                log('{:.0f} ms  nothing detected'.format(ms))

            annotated = result.plot()   # frame with boxes + labels drawn
            header = '{} objects'.format(len(found)) if args.all else '{} persons'.format(len(found))
            cv2.putText(annotated, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if args.save:
                cv2.imwrite('latest.jpg', annotated)
            else:
                cv2.imshow('YOLO person detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        log('Stopped by user')
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
EOF2
```

## Run

```bash
cd ~/projects/led
ls /dev/video*                              # confirm video1 is present
python3 -u yolo_person.py --cam 1 --save
```

`--cam 1` is the default, so `python3 -u yolo_person.py --save` does the same thing.

Expected output:

```
[15:41:52] Torch 2.1.0a0+41361538.nv23.06  device: GPU Xavier
[15:41:52] Loading model yolo11n.pt ...
[15:41:55] Model loaded in 2.8 s, 80 classes
[15:41:55] Opening camera /dev/video1 ...
[15:41:56] Camera OK, frame 640x480
[15:41:56] Warming up GPU (first inference can take 30-90 s on Jetson) ...
[15:42:31] Warm-up done in 35.2 s
[15:42:31] Detecting every 5.0 s. Press CTRL+C to stop.
[15:42:36] 41 ms  1 object(s): person 91% at [212,88,431,478]
[15:42:41] 39 ms  nothing detected
```

Each detection line shows inference time, the label, confidence, and the box as `[x1,y1,x2,y2]` in pixels. Press **Ctrl+C** to stop; the camera is released cleanly.

## Viewing the result

With `--save`, the annotated frame is overwritten in `~/projects/led/latest.jpg` every 5 seconds. From your laptop:

```bash
scp aioffice@aioffice-desktop:~/projects/led/latest.jpg .
```

With a monitor attached to the Jetson, or an `ssh -X aioffice@aioffice-desktop` session, drop `--save` for a live window (press `q` to quit):

```bash
python3 -u yolo_person.py --cam 1
```

## Options

| Flag           | Default      | Meaning                                              |
|----------------|--------------|------------------------------------------------------|
| `--cam N`      | 1            | Camera index, i.e. `/dev/videoN`                     |
| `--model FILE` | `yolo11n.pt` | Model weights (`.pt`) or TensorRT engine (`.engine`) |
| `--interval S` | 5.0          | Seconds between detections                           |
| `--conf C`     | 0.4          | Minimum confidence (0–1)                             |
| `--save`       | off          | Write `latest.jpg` instead of opening a window       |
| `--all`        | off          | Detect all 80 COCO classes instead of only person    |

Examples:

```bash
python3 -u yolo_person.py --interval 2 --save          # every 2 seconds
python3 -u yolo_person.py --all --conf 0.5 --save      # every class, stricter confidence
python3 -u yolo_person.py --cam 0 --save               # different camera
```

---

# Part 2 — `yolo_scene.py` (objects + person tracking/timing + pose + activity + expression + GPU stats)

## What it detects

| Output | Model | Notes |
|--------|-------|-------|
| All objects (chair, laptop, cup, dog, …) | YOLO11n | 80 COCO classes, counted per class |
| Person tracking: persistent `ID N`, first seen, last seen, total visible time | YOLO11n-pose + ByteTrack | Tracking runs every 0.5 s; time accumulates only while continuously detected; `left the frame` after `--gap` seconds unseen |
| Person pose: skeleton on image, torso angle, visible keypoints /17, movement in px | YOLO11n-pose | 17 COCO keypoints; angle measured from horizontal (≈90° = upright) |
| Person activity: `active`, `still / resting`, `lying / sleeping?` | YOLO11n-pose | From torso angle (< 35° = lying) and movement between checks |
| Facial expression: neutral, happy, surprised, sad, angry, disgusted, fearful, contempt | emotion-ferplus (ONNX, 64×64, run with OpenCV DNN) | Face located from pose keypoints; needs a roughly frontal face ≥ 40 px |
| Peak GPU %, GPU temp, CUDA memory, RAM % | jtop + torch | On every summary line; GPU % is the peak over the whole interval |

How states are decided:

- **lying / sleeping?** — shoulder-to-hip line within 35° of horizontal (or box much wider than tall). Posture only — eyes can't be checked at 640×480, hence the question mark.
- **still / resting** — upright and the person's center moved less than `--move` (default 6 % of frame width) since the previous check.
- **active** — upright and moved more than that, or seen for the first time.

How timing works: the pose model runs with ByteTrack every `--track-interval` (0.5 s) and assigns each person an ID that persists while they stay in view. Each update adds the elapsed time to that ID's total, but only if the gap since the previous sighting is under `--gap` (10 s) — so brief dropouts (turning away, occlusion) don't count and don't end the session. When an ID is unseen for more than `--gap` seconds the script logs `ID N left the frame` with the total; if the tracker re-acquires the same ID later it logs a re-entry. On Ctrl+C a per-ID session summary is printed. The heavier full analysis (objects, expression, log line, `latest.jpg`) still runs every `--interval` (5 s).

## Extra setup

No extra Python packages beyond Part 1 and `jetson-stats` — the expression model runs through OpenCV's built-in DNN module (onnxruntime is deliberately not used; see Troubleshooting).

```bash
pip3 install "jetson-stats==4.3.2" lap      # lap = matching library used by ByteTrack
cd ~/projects/led
wget -O emotion-ferplus-8.onnx https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"      # downloads the pose model
```

If the `wget` URL has moved, search GitHub for `emotion-ferplus-8.onnx` in the ONNX Model Zoo. The script runs without it and simply skips the expression column.

## Create the script

Paste into the SSH terminal:

```bash
mkdir -p ~/projects/led && cd ~/projects/led

cat > yolo_scene.py << 'EOF2'
"""
Scene understanding from a USB camera on Jetson Xavier NX:
  - every TRACK_INTERVAL (0.5 s): pose model with ByteTrack -> persistent person IDs, timers
  - every INTERVAL (5 s): all objects (YOLO11n), per-person activity, pose details,
    facial expression (emotion-ferplus via OpenCV DNN), visible-time per person,
    peak GPU utilisation / temperature / CUDA memory (jtop)
Prints a summary and writes latest.jpg (or shows a window).
"""
import argparse
import math
import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from jtop import jtop

EMOTIONS = ['neutral', 'happy', 'surprised', 'sad', 'angry', 'disgusted', 'fearful', 'contempt']
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12

def log(msg):
    print('[{}] {}'.format(time.strftime('%H:%M:%S'), msg), flush=True)

def fmt_dur(sec):
    sec = int(round(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h: return '{}h {:02d}m {:02d}s'.format(h, m, s)
    if m: return '{}m {:02d}s'.format(m, s)
    return '{}s'.format(s)

def fmt_clock(t):
    return time.strftime('%H:%M:%S', time.localtime(t))

def gpu_load(jetson):
    try:
        g = next(iter(jetson.gpu.values()))
        return float(g['status']['load'])
    except Exception:
        v = jetson.stats.get('GPU', 0) or 0
        return v * 100 if v <= 1 else v

class PersonTracker:
    """Accumulates visible time per tracker ID."""
    def __init__(self, gap_limit):
        self.people = {}          # id -> dict(first, last, total, present, center)
        self.gap_limit = gap_limit

    def update(self, ids, centers, now):
        seen = set()
        for tid, c in zip(ids, centers):
            seen.add(tid)
            p = self.people.get(tid)
            if p is None:
                self.people[tid] = {'first': now, 'last': now, 'total': 0.0,
                                    'present': True, 'center': c, 'prev_center': None}
                log('ID {} entered the frame'.format(tid))
            else:
                gap = now - p['last']
                if gap <= self.gap_limit:
                    p['total'] += gap
                elif not p['present']:
                    log('ID {} re-entered the frame (was away {})'.format(tid, fmt_dur(gap)))
                p['last'] = now
                p['present'] = True
                p['center'] = c
        for tid, p in self.people.items():
            if tid not in seen and p['present'] and now - p['last'] > self.gap_limit:
                p['present'] = False
                log('ID {} left the frame - visible {} (first {}, last {})'.format(
                    tid, fmt_dur(p['total']), fmt_clock(p['first']), fmt_clock(p['last'])))

    def present_ids(self):
        return [t for t, p in self.people.items() if p['present']]

def load_emotion_model(path):
    if not os.path.exists(path):
        log('Emotion model {} not found - sentiment disabled'.format(path))
        return None
    try:
        net = cv2.dnn.readNetFromONNX(path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        log('Emotion model loaded via OpenCV DNN ({})'.format(path))
        return net
    except Exception as e:
        log('Could not load emotion model ({}) - sentiment disabled'.format(e))
        return None

def face_crop(frame, kp, kc, box):
    h, w = frame.shape[:2]
    pts = [kp[i] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kc[i] > 0.3]
    if len(pts) >= 2:
        cx = np.mean([p[0] for p in pts]); cy = np.mean([p[1] for p in pts])
        spread = max(np.ptp([p[0] for p in pts]), np.ptp([p[1] for p in pts]))
        size = max(40, spread * 2.4)
    else:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, y1 + (y2 - y1) * 0.15
        size = max(40, (x2 - x1) * 0.5)
    half = size / 2
    xa, ya = int(max(0, cx - half)), int(max(0, cy - half))
    xb, yb = int(min(w, cx + half)), int(min(h, cy + half))
    if xb - xa < 24 or yb - ya < 24:
        return None, None
    return frame[ya:yb, xa:xb], (xa, ya, xb, yb)

def emotion(net, face):
    gray = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (64, 64))
    net.setInput(cv2.dnn.blobFromImage(gray, scalefactor=1.0, size=(64, 64)))
    logits = net.forward().flatten()
    p = np.exp(logits - logits.max()); p /= p.sum()
    i = int(p.argmax())
    return EMOTIONS[i], float(p[i])

def posture(kp, kc, box):
    x1, y1, x2, y2 = box
    if min(kc[L_SHO], kc[R_SHO], kc[L_HIP], kc[R_HIP]) > 0.3:
        sho = (kp[L_SHO] + kp[R_SHO]) / 2
        hip = (kp[L_HIP] + kp[R_HIP]) / 2
        dx, dy = hip[0] - sho[0], hip[1] - sho[1]
        angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
        return ('lying' if angle < 35 else 'upright'), angle
    return ('lying' if (x2 - x1) > 1.3 * (y2 - y1) else 'upright'), None

def track_people(frame, pose, device, args, tracker, now):
    """Run pose+ByteTrack, update timers. Returns the pose result."""
    pres = pose.track(frame, persist=True, device=device, conf=args.conf, imgsz=640,
                      verbose=False, tracker='bytetrack.yaml')[0]
    ids, centers = [], []
    if pres.boxes.id is not None:
        for b, tid in zip(pres.boxes, pres.boxes.id.int().tolist()):
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            ids.append(tid); centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    tracker.update(ids, centers, now)
    return pres

def analyse(frame, pres, det, emo, device, args, tracker, W, now):
    dres = det.predict(frame, device=device, conf=args.conf, imgsz=640, verbose=False)[0]
    objects = {}
    for b in dres.boxes:
        name = det.names[int(b.cls[0])]
        if name != 'person':
            objects[name] = objects.get(name, 0) + 1

    annotated = dres.plot()
    if len(pres.boxes):
        annotated = pres.plot(img=annotated, boxes=False, labels=False, kpt_radius=4, kpt_line=True)

    people = []
    ids = pres.boxes.id.int().tolist() if pres.boxes.id is not None else [None] * len(pres.boxes)
    for i, (b, tid) in enumerate(zip(pres.boxes, ids)):
        box = tuple(map(int, b.xyxy[0]))
        pconf = float(b.conf[0])
        kp = pres.keypoints.xy[i].cpu().numpy()
        kc = pres.keypoints.conf[i].cpu().numpy() if pres.keypoints.conf is not None else np.ones(17)

        post, angle = posture(kp, kc, box)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        rec = tracker.people.get(tid) if tid is not None else None
        dist = None
        if rec is not None and rec['prev_center'] is not None:
            dist = math.hypot(center[0] - rec['prev_center'][0], center[1] - rec['prev_center'][1])
        if rec is not None:
            rec['prev_center'] = center

        if post == 'lying':
            state = 'lying / sleeping?'
        elif dist is not None and dist < W * args.move:
            state = 'still / resting'
        else:
            state = 'active'

        visible = int((kc > 0.3).sum())
        pose_txt = ('torso {:.0f} deg, {}/17 kpts'.format(angle, visible) if angle is not None
                    else '{}/17 kpts, torso not visible'.format(visible))
        if dist is not None:
            pose_txt += ', moved {:.0f} px'.format(dist)

        mood = ''
        if emo is not None:
            face, fbox = face_crop(frame, kp, kc, box)
            if face is not None:
                label, p = emotion(emo, face)
                mood = '{} {:.0%}'.format(label, p)
                cv2.rectangle(annotated, fbox[:2], fbox[2:], (255, 200, 0), 1)

        if rec is not None:
            who = 'ID {} | visible {} (since {})'.format(tid, fmt_dur(rec['total']), fmt_clock(rec['first']))
            tag = 'ID {}  {}'.format(tid, fmt_dur(rec['total']))
        else:
            who = 'untracked'; tag = '?'
        people.append('{} | conf {:.0%} | {} | pose: {}{}'.format(
            who, pconf, state, pose_txt, ' | ' + mood if mood else ''))

        cv2.rectangle(annotated, box[:2], box[2:], (0, 255, 0), 2)
        cv2.putText(annotated, tag, (box[0], max(15, box[1] - 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, '{}{}'.format(state, ' / ' + mood if mood else ''),
                    (box[0], max(15, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return annotated, people, objects

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', default='0', help='camera index or /dev path')
    ap.add_argument('--det', default='yolo11n.pt')
    ap.add_argument('--pose', default='yolo11n-pose.pt')
    ap.add_argument('--emo', default='emotion-ferplus-8.onnx')
    ap.add_argument('--interval', type=float, default=5.0, help='seconds between full analyses')
    ap.add_argument('--track-interval', type=float, default=0.5, help='seconds between tracking updates')
    ap.add_argument('--gap', type=float, default=10.0, help='seconds unseen before a person counts as left')
    ap.add_argument('--conf', type=float, default=0.4)
    ap.add_argument('--move', type=float, default=0.06)
    ap.add_argument('--save', action='store_true')
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else 'cpu'
    log('Torch {}  device: {}'.format(torch.__version__,
        'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'))

    log('Loading models ...')
    det = YOLO(args.det)
    pose = YOLO(args.pose)
    emo = load_emotion_model(args.emo)
    log('Models loaded')

    log('Opening camera {} ...'.format(args.cam))
    cam = int(args.cam) if str(args.cam).isdigit() else args.cam
    cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    if not cap.isOpened() or not ok:
        raise SystemExit('Cannot read camera {} - check ls /dev/video*'.format(args.cam))
    W, H = frame.shape[1], frame.shape[0]
    log('Camera OK, frame {}x{}'.format(W, H))

    log('Warming up GPU (first inference can take 30-90 s) ...')
    t0 = time.time()
    det.predict(frame, device=device, verbose=False, imgsz=640)
    pose.track(frame, persist=True, device=device, verbose=False, imgsz=640, tracker='bytetrack.yaml')
    log('Warm-up done in {:.1f} s'.format(time.time() - t0))
    try:
        log('Detector running on: {}'.format(det.predictor.model.device))
        log('Pose running on:     {}'.format(pose.predictor.model.device))
    except Exception:
        log('Weights device: n/a (TensorRT engine)')

    tracker = PersonTracker(gap_limit=args.gap)
    pres = None
    now = time.time()
    next_track = now
    next_full = now
    gpu_peak = 0.0
    log('Tracking every {} s, full analysis every {} s. CTRL+C to stop.'.format(args.track_interval, args.interval))
    try:
        with jtop() as jetson:
            while jetson.ok():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.5); continue
                now = time.time()
                gpu_peak = max(gpu_peak, gpu_load(jetson))

                if now >= next_track:
                    next_track = now + args.track_interval
                    pres = track_people(frame, pose, device, args, tracker, now)

                if now >= next_full and pres is not None:
                    next_full = now + args.interval
                    t0 = time.time()
                    annotated, people, objects = analyse(frame, pres, det, emo, device, args, tracker, W, now)
                    ms = (time.time() - t0) * 1000

                    st = jetson.stats
                    gpu_peak = max(gpu_peak, gpu_load(jetson))
                    gpu_mem = torch.cuda.memory_allocated() / 1e6 if device == 0 else 0
                    ram = st['RAM'] * 100 if st['RAM'] <= 1 else st['RAM']
                    obj_txt = ', '.join('{} x{}'.format(k, v) if v > 1 else k
                                        for k, v in sorted(objects.items())) or 'none'
                    log('{:.0f} ms | GPU peak {:.0f}% @ {:.0f}C | CUDA mem {:.0f} MB | RAM {:.0f}% | {} person(s) present | objects: {}'.format(
                        ms, gpu_peak, st['Temp GPU'], gpu_mem, ram, len(tracker.present_ids()), obj_txt))
                    for p in people:
                        log('   ' + p)

                    cv2.putText(annotated, '{} present, {} objects | GPU peak {:.0f}%'.format(
                        len(tracker.present_ids()), sum(objects.values()), gpu_peak),
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    gpu_peak = 0.0
                    if args.save:
                        cv2.imwrite('latest.jpg', annotated)
                    else:
                        cv2.imshow('Scene', annotated)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
    except KeyboardInterrupt:
        log('Stopped by user')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log('---- Session summary ----')
        for tid, p in sorted(tracker.people.items()):
            log('ID {}: visible {} total, first seen {}, last seen {}{}'.format(
                tid, fmt_dur(p['total']), fmt_clock(p['first']), fmt_clock(p['last']),
                '' if p['present'] else ' (left)'))

if __name__ == '__main__':
    main()
EOF2
```

## Power mode (do this before running)

The Xavier NX ships in a low-power mode with only 2 CPU cores online, which halves inference speed and crashes some libraries. Switch to an all-core mode and max clocks:

```bash
sudo nvpmodel -q --verbose | grep -A1 "POWER_MODEL ID"   # list available modes
sudo nvpmodel -m 2            # 15 W, 6 cores  (or -m 8 for 20 W, 6 cores if listed)
sudo jetson_clocks
nproc; cat /sys/devices/system/cpu/online                # want 6 and 0-5
```

Do **not** use `nvpmodel -m 0` — on the Xavier NX that is the 15 W **2-core** mode.

If `nproc` still shows 2, bring the cores up directly:

```bash
for c in 2 3 4 5; do echo 1 | sudo tee /sys/devices/system/cpu/cpu$c/online; done
```

Make the mode persist across reboots (`jetson_clocks` itself is not persistent):

```bash
sudo nvpmodel -d 2
```

## Run

Find the camera, test which node streams, then run:

```bash
cd ~/projects/led
ls /dev/video*
python3 -c "
import cv2
for i in (0, 1):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    ok, f = cap.read()
    print('/dev/video{}: {}'.format(i, 'frames OK {}x{}'.format(f.shape[1], f.shape[0]) if ok else 'no frames'))
    cap.release()"
python3 -u yolo_scene.py --cam 0 --save
```

Most cameras expose two nodes; only the first streams. For a path that survives renumbering, use `ls -l /dev/v4l/by-id/` and pass it: `--cam /dev/v4l/by-id/usb-XXXX-video-index0`.

Expected output:

```
[08:10:00] Torch 1.12.0a0+2c916ef.nv22.3  device: GPU Xavier
[08:10:00] Loading models ...
[08:10:01] Emotion model loaded via OpenCV DNN (emotion-ferplus-8.onnx)
[08:10:01] Models loaded
[08:10:01] Opening camera 0 ...
[08:10:03] Camera OK, frame 640x480
[08:10:03] Warming up GPU (first inference can take 30-90 s) ...
[08:10:15] Warm-up done in 12.7 s
[08:10:15] Detector running on: cuda:0
[08:10:15] Pose running on:     cuda:0
[08:10:15] Tracking every 0.5 s, full analysis every 5.0 s. CTRL+C to stop.
[08:10:16] ID 1 entered the frame
[08:10:20] 271 ms | GPU peak 63% @ 41C | CUDA mem 22 MB | RAM 47% | 1 person(s) present | objects: cup, keyboard
[08:10:20]    ID 1 | visible 4s (since 08:10:16) | conf 86% | active | pose: torso 85 deg, 14/17 kpts | neutral 90%
[08:10:25] 268 ms | GPU peak 61% @ 41C | CUDA mem 22 MB | RAM 47% | 1 person(s) present | objects: chair, cup, keyboard
[08:10:25]    ID 1 | visible 9s (since 08:10:16) | conf 84% | still / resting | pose: torso 86 deg, 14/17 kpts, moved 6 px | neutral 88%
...
[08:12:51] ID 1 left the frame - visible 2m 35s (first 08:10:16, last 08:12:41)
^C
[08:13:02] Stopped by user
[08:13:02] ---- Session summary ----
[08:13:02] ID 1: visible 2m 35s total, first seen 08:10:16, last seen 08:12:41 (left)
```

`Detector running on: cuda:0` is the definitive proof the models run on the GPU. It reads `det.predictor.model.device` — the copy Ultralytics actually runs inference with — not `det.model`, which stays on CPU and misleadingly reports `cpu`. `GPU peak` is the highest jtop load seen during the whole interval (read from `jetson.gpu[...]['status']['load']`, which is a true percentage), so it reflects the inference burst rather than the idle gap between checks. For a live view run `sudo jtop` (or `tegrastats --interval 1000` and watch `GR3D_FREQ`) in a second SSH window.

Reading the pose field: `torso 86 deg` is the shoulder-to-hip angle from horizontal (≈90° upright, < 35° lying); `14/17 kpts` is how many of the 17 COCO keypoints were seen with confidence > 0.3 — at a desk it is usually the upper body only; `moved 7 px` is the person's centre displacement since the previous check and drives `active` vs `still`. The skeleton itself is drawn on each person in `latest.jpg`, with a yellow `ID 1  2m 35s` tag above the green box showing the tracker ID and accumulated visible time.

In `latest.jpg`: other objects have YOLO's colored boxes and labels, each person has a green box labeled with state and expression, and a thin yellow square marks the face crop used for the expression estimate.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--cam N` or path | 0 | Camera index (`/dev/videoN`) or a `/dev/v4l/by-id/...` path |
| `--det FILE` | `yolo11n.pt` | Object model (`.pt` or `.engine`) |
| `--pose FILE` | `yolo11n-pose.pt` | Pose model (`.pt` or `.engine`) |
| `--emo FILE` | `emotion-ferplus-8.onnx` | Expression model; skipped if missing |
| `--interval S` | 5.0 | Seconds between full analyses (objects, expression, log line, image) |
| `--track-interval S` | 0.5 | Seconds between tracking updates (person IDs and timers) |
| `--gap S` | 10.0 | Seconds a person can be unseen before counting as having left |
| `--conf C` | 0.4 | Minimum detection confidence |
| `--move F` | 0.06 | Movement threshold (fraction of frame width) for `active` |
| `--save` | off | Write `latest.jpg` instead of a window |

## Quick GPU vs CPU speed check

Confirms CUDA is actually used, without the camera:

```bash
python3 -c "
import torch, time
from ultralytics import YOLO
m = YOLO('yolo11n.pt'); img = torch.zeros(1,3,640,640)
m.predict(img, device=0, verbose=False)
t=time.time(); [m.predict(img, device=0, verbose=False) for _ in range(20)]; g=(time.time()-t)/20
t=time.time(); [m.predict(img, device='cpu', verbose=False) for _ in range(3)]; c=(time.time()-t)/3
print('GPU {:.0f} ms/frame   CPU {:.0f} ms/frame   speed-up {:.1f}x'.format(g*1000, c*1000, c/g))"
```

On this Xavier NX it printed `GPU 66 ms/frame   CPU 1795 ms/frame   speed-up 27.4x`. Near-equal numbers mean torch is not using CUDA — see the PyTorch install in Part 1.

---

## Optional: faster inference with TensorRT (both scripts)

Export once (5–10 minutes, produces `yolo11n.engine` tuned for this GPU):

```bash
cd ~/projects/led
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='engine', half=True, imgsz=640)"
python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').export(format='engine', half=True, imgsz=640)"
python3 -u yolo_person.py --model yolo11n.engine --save
python3 -u yolo_scene.py --det yolo11n.engine --pose yolo11n-pose.engine --save
```

Typical speed-up is 2–3× over the `.pt` models. `yolo_scene.py` runs two YOLO models per check, so this matters more there.

## Running in the background

```bash
cd ~/projects/led
nohup python3 -u yolo_person.py --save > yolo.log 2>&1 &      # or yolo_scene.py
tail -f yolo.log          # Ctrl+C stops tail only
pkill -f yolo_             # stop whichever detector is running
```

## Troubleshooting

**`FileNotFoundError: 'yolo11n.pt'` when loading**
The installed `ultralytics` is too old to know YOLO11. Upgrade with `pip3 install "ultralytics>=8.3,<8.4"` or download the weights with the `wget` line above.

**Stuck on "Warming up GPU"**
The first CUDA inference can take over a minute. If it exceeds two minutes, open a second SSH window and run `sudo jtop` — 0 % GPU means the camera read is hanging; try a different `--cam` index or unplug/replug the camera.

**`Cannot open camera 1`**
Run `ls /dev/video*`. Many cameras expose two nodes (e.g. `video0` and `video1`); only one delivers frames. Try the other index. Make sure no other program (another copy of the script, Cheese, etc.) has the camera open: `pkill -f yolo_person.py`.

**`torch.cuda.is_available()` prints False**
You have the CPU-only PyPI torch. Uninstall it (`pip3 uninstall torch`) and install NVIDIA's Jetson wheel from the setup section.

**Very slow (hundreds of ms per frame)**
Check `sudo jtop` — GPU should be active during inference. Confirm the `device:` line at startup says GPU. Set the power mode to maximum with `sudo nvpmodel -m 0 && sudo jetson_clocks`, or export to TensorRT.

**`Mismatch version jtop service` when starting `yolo_scene.py`**
The jtop client and service differ. `sudo pip3 install "jetson-stats==4.3.2" && sudo systemctl restart jtop.service`, then rerun.

**`Detector weights on: cpu` in the log**
Old version of the script checked `det.model`, which Ultralytics never moves — inference runs on `det.predictor.model`. The current script reports `Detector running on: cuda:0`. `CUDA mem 22 MB` and the speed check above are the other indicators.

**`GPU peak 0%` on every line**
Old version read `jetson.stats['GPU']`, which jtop 4.x returns as a fraction (0.0–1.0), so it rounded to 0. The current script reads the percentage from `jetson.gpu`. Use `sudo jtop` for a live view.

**No pose / skeleton visible**
Old version used the pose model only for the activity decision. The current script prints `pose: torso … kpts … moved …` per person and draws the skeleton with `pres.plot(img=annotated, ...)`.

**`can't open camera by index` / `Cannot read camera 1`**
The camera moved to another `/dev/videoN` after a reboot or re-plug. Run `ls /dev/video*` and the two-node test in the Run section, then pass `--cam 0` (or the `/dev/v4l/by-id/...` path).

**`pthread_setaffinity_np failed` / `Assertion '__n < this->size()' failed` / `Aborted (core dumped)` at "Loading models"**
That is onnxruntime aborting on a Jetson with CPU cores offline (2-core power mode). Two fixes, both applied here: bring all cores online (see Power mode), and run the ONNX model through `cv2.dnn` instead of onnxruntime. If you still have onnxruntime installed, `pip3 uninstall -y onnxruntime` — the script does not need it.

**`nproc` shows 2**
The board is in a 2-core power mode (`nvpmodel -m 0` puts it there). See Power mode above.

**Same person gets a new ID after leaving and coming back**
Expected. ByteTrack matches by position and motion, not appearance, and drops a track after ~30 tracked frames (≈15 s at the 0.5 s cadence). Re-identifying the same person across longer absences needs a face-recognition step, which this script does not include. Raise `--gap` if people are being marked as left during short occlusions.

**`ModuleNotFoundError: No module named 'lap'` or a tracker error at warm-up**
`pip3 install lap`. ByteTrack uses it for detection-to-track matching.

**IDs jump around when someone moves quickly**
Lower `--track-interval` (e.g. 0.25) so the tracker sees smaller displacements between updates.

**Expression always "neutral" or missing**
The face crop is too small or not frontal. Move closer to the camera (face ≥ 40 px wide) and face it; a thin yellow square in `latest.jpg` shows what the model saw.

**Everyone shows "active" on the first check**
Expected — there is no previous position to compare against. States settle from the second check onward.

**Out of memory during torchvision build**
Prefix the build with `MAX_JOBS=2` and close jtop and other heavy processes.
