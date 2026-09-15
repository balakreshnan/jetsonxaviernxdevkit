# YOLO Person Detection on Jetson Xavier NX

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

## Optional: faster inference with TensorRT

Export once (5–10 minutes, produces `yolo11n.engine` tuned for this GPU):

```bash
cd ~/projects/led
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='engine', half=True, imgsz=640)"
python3 -u yolo_person.py --model yolo11n.engine --save
```

Typical speed-up is 2–3× over the `.pt` model.

## Running in the background

```bash
cd ~/projects/led
nohup python3 -u yolo_person.py --save > yolo.log 2>&1 &
tail -f yolo.log          # Ctrl+C stops tail only
pkill -f yolo_person.py   # stop the detector
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

**Out of memory during torchvision build**
Prefix the build with `MAX_JOBS=2` and close jtop and other heavy processes.
