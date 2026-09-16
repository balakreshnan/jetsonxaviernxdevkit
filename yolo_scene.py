"""
Scene understanding from a USB camera on Jetson Xavier NX, every INTERVAL seconds:
  - all objects (YOLO11n)
  - per person: posture/activity from keypoints (YOLO11n-pose)
  - per person: facial expression (emotion-ferplus ONNX via OpenCV DNN) if the face is visible
  - pose details (torso angle, visible keypoints, movement) per person, skeleton drawn on the image
  - peak GPU utilisation, temperature and CUDA memory on every summary line (jtop)
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
# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12

def log(msg):
    print('[{}] {}'.format(time.strftime('%H:%M:%S'), msg), flush=True)

def gpu_load(jetson):
    """GPU load in percent; jtop 4.x reports fractions in some places."""
    try:
        g = next(iter(jetson.gpu.values()))
        return float(g['status']['load'])
    except Exception:
        v = jetson.stats.get('GPU', 0) or 0
        return v * 100 if v <= 1 else v

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
    """Estimate a face square from head keypoints; fall back to top of bbox."""
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
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    blob = cv2.dnn.blobFromImage(gray, scalefactor=1.0, size=(64, 64))   # 1x1x64x64 float32
    net.setInput(blob)
    logits = net.forward().flatten()
    p = np.exp(logits - logits.max()); p /= p.sum()
    i = int(p.argmax())
    return EMOTIONS[i], float(p[i])

def posture(kp, kc, box):
    """Return 'lying' or 'upright' from torso angle, falling back to box shape."""
    x1, y1, x2, y2 = box
    if min(kc[L_SHO], kc[R_SHO], kc[L_HIP], kc[R_HIP]) > 0.3:
        sho = (kp[L_SHO] + kp[R_SHO]) / 2
        hip = (kp[L_HIP] + kp[R_HIP]) / 2
        dx, dy = hip[0] - sho[0], hip[1] - sho[1]
        angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))   # 90 = vertical torso
        return ('lying' if angle < 35 else 'upright'), angle
    return ('lying' if (x2 - x1) > 1.3 * (y2 - y1) else 'upright'), None

def match_previous(center, prev, max_dist):
    best, best_d = None, max_dist
    for pc in prev:
        d = math.hypot(center[0] - pc[0], center[1] - pc[1])
        if d < best_d:
            best, best_d = pc, d
    return best, best_d

def analyse(frame, det, pose, emo, device, args, prev_centers, W):
    """Run both models on one frame. Returns (annotated, people, objects, centers)."""
    dres = det.predict(frame, device=device, conf=args.conf, imgsz=640, verbose=False)[0]
    objects = {}
    for b in dres.boxes:
        name = det.names[int(b.cls[0])]
        if name != 'person':
            objects[name] = objects.get(name, 0) + 1

    pres = pose.predict(frame, device=device, conf=args.conf, imgsz=640, verbose=False)[0]
    annotated = dres.plot()
    if len(pres.boxes):
        annotated = pres.plot(img=annotated, boxes=False, labels=False, kpt_radius=4, kpt_line=True)
    people, centers = [], []
    for i, b in enumerate(pres.boxes):
        box = tuple(map(int, b.xyxy[0]))
        pconf = float(b.conf[0])
        kp = pres.keypoints.xy[i].cpu().numpy()
        kc = pres.keypoints.conf[i].cpu().numpy() if pres.keypoints.conf is not None else np.ones(17)

        post, angle = posture(kp, kc, box)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        centers.append(center)
        _, dist = match_previous(center, prev_centers, max_dist=W * 0.4)
        moved = dist < W * 0.4 and dist > W * args.move
        if post == 'lying':
            state = 'lying / sleeping?'
        elif prev_centers and not moved:
            state = 'still / resting'
        else:
            state = 'active'

        mood = ''
        if emo is not None:
            face, fbox = face_crop(frame, kp, kc, box)
            if face is not None:
                label, p = emotion(emo, face)
                mood = '{} {:.0%}'.format(label, p)
                cv2.rectangle(annotated, fbox[:2], fbox[2:], (255, 200, 0), 1)

        visible = int((kc > 0.3).sum())
        pose_txt = ('torso {:.0f} deg, {}/17 kpts'.format(angle, visible) if angle is not None
                    else '{}/17 kpts, torso not visible'.format(visible))
        if prev_centers and dist < W * 0.4:
            pose_txt += ', moved {:.0f} px'.format(dist)
        people.append('person {:.0%} | {} | pose: {}{}'.format(pconf, state, pose_txt, ' | ' + mood if mood else ''))
        cv2.rectangle(annotated, box[:2], box[2:], (0, 255, 0), 2)
        y = max(15, box[1] - 8)
        cv2.putText(annotated, '{}{}'.format(state, ' / ' + mood if mood else ''),
                    (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return annotated, people, objects, centers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', default='0', help='camera index or /dev path (e.g. /dev/v4l/by-id/...)')
    ap.add_argument('--det', default='yolo11n.pt', help='object model (.pt or .engine)')
    ap.add_argument('--pose', default='yolo11n-pose.pt', help='pose model (.pt or .engine)')
    ap.add_argument('--emo', default='emotion-ferplus-8.onnx', help='emotion ONNX model')
    ap.add_argument('--interval', type=float, default=5.0)
    ap.add_argument('--conf', type=float, default=0.4)
    ap.add_argument('--move', type=float, default=0.06,
                    help='movement threshold as fraction of frame width to count as active')
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
    pose.predict(frame, device=device, verbose=False, imgsz=640)
    log('Warm-up done in {:.1f} s'.format(time.time() - t0))
    try:
        log('Detector running on: {}'.format(det.predictor.model.device))
        log('Pose running on:     {}'.format(pose.predictor.model.device))
    except Exception:
        log('Weights device: n/a (TensorRT engine)')

    prev_centers = []
    next_run = time.time()
    gpu_peak = 0.0
    log('Analysing every {} s. CTRL+C to stop.'.format(args.interval))
    try:
        with jtop() as jetson:
            while jetson.ok():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.5); continue
                gpu_peak = max(gpu_peak, gpu_load(jetson))
                if time.time() < next_run:
                    continue
                next_run = time.time() + args.interval

                t0 = time.time()
                annotated, people, objects, prev_centers = analyse(
                    frame, det, pose, emo, device, args, prev_centers, W)
                ms = (time.time() - t0) * 1000

                st = jetson.stats
                gpu_peak = max(gpu_peak, gpu_load(jetson))
                gpu_mem = torch.cuda.memory_allocated() / 1e6 if device == 0 else 0
                ram = st['RAM'] * 100 if st['RAM'] <= 1 else st['RAM']
                obj_txt = ', '.join('{} x{}'.format(k, v) if v > 1 else k
                                    for k, v in sorted(objects.items())) or 'none'
                log('{:.0f} ms | GPU peak {:.0f}% @ {:.0f}C | CUDA mem {:.0f} MB | RAM {:.0f}% | {} person(s) | objects: {}'.format(
                    ms, gpu_peak, st['Temp GPU'], gpu_mem, ram, len(people), obj_txt))
                for p in people:
                    log('   ' + p)

                cv2.putText(annotated, '{} persons, {} objects | GPU peak {:.0f}%'.format(
                    len(people), sum(objects.values()), gpu_peak),
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

if __name__ == '__main__':
    main()
