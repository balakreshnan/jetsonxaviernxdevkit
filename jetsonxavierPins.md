# Jetson GPU Temperature LED + Fan Controller

`rlled2.py` monitors the GPU temperature on an NVIDIA Jetson (tested on Xavier NX, JetPack 5 / Python 3.8) using `jtop`. When the GPU goes above 40 °C it turns on an LED (GPIO pin 29) and runs the built-in PWM fan at 100 %. When the temperature drops back below 38 °C, both turn off. On exit the fan is handed back to the system's automatic control.

## Hardware

- NVIDIA Jetson with the stock PWM fan connected to the fan header.
- An LED (with a series resistor, ~330 Ω) between **BOARD pin 29** and GND.

## Requirements

- `jetson-stats` 4.3.x (`jtop`) — client and service must be the same version.
- `Jetson.GPIO` (provides the `RPi.GPIO` module on Jetson).
- Your user must be in the `gpio` group (or run with `sudo`).

## Setup (over SSH)

Connect to the Jetson:

```bash
ssh aioffice@aioffice-desktop
```

Install dependencies (once):

```bash
sudo pip3 install "jetson-stats==4.3.2"
sudo pip3 install Jetson.GPIO
sudo systemctl restart jtop.service
sudo usermod -aG gpio $USER     # then log out and back in
```

Check that the jtop service is running and matches the client:

```bash
systemctl status jtop.service
jtop --version
```

## Create the script

Go to the project folder and paste the following block into the terminal. It writes the file in one step:

```bash
mkdir -p ~/projects/AIPinBall1 && cd ~/projects/AIPinBall1

cat > rlled2.py << 'EOF'
import time
from jtop import jtop
import RPi.GPIO as GPIO

LED_PIN = 29          # BOARD numbering
TEMP_ON = 40.0        # turn on above this (deg C)
TEMP_OFF = 38.0       # turn off below this (small gap prevents flicker)
FAN_SPEED = 100       # percent when on
POLL_SECONDS = 5

def cooling(jetson, fan, on):
    GPIO.output(LED_PIN, GPIO.HIGH if on else GPIO.LOW)
    jetson.fan.set_speed(fan, FAN_SPEED if on else 0)

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    cooling_on = False
    fan = None
    old_profile = None

    print("Monitoring GPU temp. Press CTRL+C to exit")
    try:
        with jtop() as jetson:
            fan = list(jetson.fan.keys())[0]          # e.g. 'pwmfan'
            old_profile = jetson.fan.get_profile(fan)
            print("Fan:", fan, " profiles:", jetson.fan.all_profiles(fan), " current:", old_profile)
            jetson.fan.set_profile(fan, 'manual')     # take over from automatic control
            jetson.fan.set_speed(fan, 0)

            while jetson.ok():
                gpu_temp = jetson.stats['Temp GPU']
                print('GPU {:.1f} C  fan {}%  cooling {}'.format(
                    gpu_temp, jetson.fan.get_speed(fan), 'ON' if cooling_on else 'OFF'))

                if gpu_temp > TEMP_ON and not cooling_on:
                    cooling(jetson, fan, True)
                    cooling_on = True
                    print("Temp above {} -> LED and fan ON".format(TEMP_ON))
                elif gpu_temp < TEMP_OFF and cooling_on:
                    cooling(jetson, fan, False)
                    cooling_on = False
                    print("Temp below {} -> LED and fan OFF".format(TEMP_OFF))

                time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        if fan and old_profile:
            try:
                with jtop() as jetson:
                    jetson.fan.set_profile(fan, old_profile)   # hand control back
            except Exception:
                pass
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()

if __name__ == '__main__':
    main()
EOF
```

## Run

```bash
cd ~/projects/AIPinBall1
python3 rlled2.py
```

Expected output:

```
Monitoring GPU temp. Press CTRL+C to exit
Fan: pwmfan  profiles: ['quiet', 'cool', 'manual']  current: quiet
GPU 36.5 C  fan 0%  cooling OFF
GPU 41.0 C  fan 0%  cooling OFF
Temp above 40.0 -> LED and fan ON
GPU 39.5 C  fan 100%  cooling ON
GPU 37.5 C  fan 100%  cooling ON
Temp below 38.0 -> LED and fan OFF
```

Press **Ctrl+C** to stop. The LED turns off, GPIO is released, and the fan profile is restored.

To keep it running after you close the SSH session:

```bash
nohup python3 rlled2.py > rlled2.log 2>&1 &
tail -f rlled2.log        # watch output; Ctrl+C stops tail only
```

Stop a background run with `pkill -f rlled2.py`.

## Tuning

Edit the constants at the top of `rlled2.py`:

| Constant       | Default | Meaning                                  |
|----------------|---------|------------------------------------------|
| `LED_PIN`      | 29      | BOARD pin number for the LED             |
| `TEMP_ON`      | 40.0    | Turn cooling on above this temperature   |
| `TEMP_OFF`     | 38.0    | Turn cooling off below this temperature  |
| `FAN_SPEED`    | 100     | Fan speed (%) while cooling is on        |
| `POLL_SECONDS` | 5       | How often the temperature is checked     |

Set `TEMP_OFF` equal to `TEMP_ON` for a single hard threshold.

## Troubleshooting

**`Mismatch version jtop service: [x] and client: [y]`**
Client and service versions differ. Reinstall one version for both and restart the service:

```bash
sudo pip3 uninstall -y jetson-stats
sudo pip3 install "jetson-stats==4.3.2"
sudo systemctl restart jtop.service
```

**`all_profiles() missing 1 required positional argument`**
You are on jtop 4.x, which needs the fan name — the script above already handles this.

**Permission errors on GPIO**
Add yourself to the `gpio` group (`sudo usermod -aG gpio $USER`), log out and back in, or run with `sudo python3 rlled2.py`.

**Fan runs even when the script says OFF**
Another process (or the system's `nvfancontrol`) is driving it. Check with `sudo jtop` → CTRL tab.
