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
