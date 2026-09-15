# Jetson GPU Temperature Fan + LED Controller

`rlled2.py` monitors the GPU temperature on an NVIDIA Jetson (tested on Xavier NX, JetPack 5 / Python 3.8) using `jtop`.

Behaviour:

| GPU temperature                | Fan  | LED (pin 29)                          |
|--------------------------------|------|----------------------------------------|
| Below 40 °C (not yet triggered)| off  | off                                    |
| Rises above 40 °C              | 100 %| solid                                  |
| Above 38 °C and falling        | 100 %| blinking (0.5 s on / 0.5 s off)        |
| Above 38 °C and rising/steady  | 100 %| solid                                  |
| Drops below 38 °C              | off  | off                                    |

The 40 °C on / 38 °C off gap stops the fan chattering when the temperature hovers at the threshold. On exit (Ctrl+C) the LED is turned off and fan control is handed back to the system's automatic profile.

## Hardware

- NVIDIA Jetson with the stock PWM fan on the fan header (controlled through `jtop`).
- One LED with a 330 Ω series resistor:
  - Long leg (anode) → 330 Ω → **BOARD pin 29**
  - Short leg (cathode) → **BOARD pin 39 (GND)**

Give the LED its own breadboard row. If other components share a row with the LED or its resistor, unused GPIO pins driven LOW can pull the LED's supply down and make it glow dimly.

Note: Jetson GPIO pins are 3.3 V and can only source ~1–2 mA. A single LED with 330 Ω is fine; anything bigger (more LEDs, relays, motors) should be switched through a transistor — see "Adding more loads" below.

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

cat > rlled2.py << 'EOF2'
import time
from jtop import jtop
import RPi.GPIO as GPIO

LED_PIN = 29
TEMP_ON = 40.0        # fan + LED on above this
TEMP_OFF = 38.0       # fan + LED off below this
FAN_SPEED = 100
POLL_SECONDS = 5
BLINK_SECONDS = 0.5

def hold(on, falling):
    """LED for POLL_SECONDS: off, solid, or blinking."""
    if not on:
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(POLL_SECONDS)
        return
    if not falling:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(POLL_SECONDS)
        return
    end = time.time() + POLL_SECONDS
    state = GPIO.HIGH
    while time.time() < end:
        GPIO.output(LED_PIN, state)
        state = GPIO.LOW if state == GPIO.HIGH else GPIO.HIGH
        time.sleep(BLINK_SECONDS)

def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    cooling_on = False
    last_temp = None
    fan = None
    old_profile = None

    print("Monitoring GPU temp. Press CTRL+C to exit")
    try:
        with jtop() as jetson:
            fan = list(jetson.fan.keys())[0]
            old_profile = jetson.fan.get_profile(fan)
            print("Fan:", fan, " profiles:", jetson.fan.all_profiles(fan), " current:", old_profile)
            jetson.fan.set_profile(fan, 'manual')
            jetson.fan.set_speed(fan, 0)

            while jetson.ok():
                gpu_temp = jetson.stats['Temp GPU']

                if gpu_temp > TEMP_ON and not cooling_on:
                    cooling_on = True
                    jetson.fan.set_speed(fan, FAN_SPEED)
                    print("Above {} -> fan and LED ON".format(TEMP_ON))
                elif gpu_temp < TEMP_OFF and cooling_on:
                    cooling_on = False
                    jetson.fan.set_speed(fan, 0)
                    print("Below {} -> fan and LED OFF".format(TEMP_OFF))

                falling = last_temp is not None and gpu_temp < last_temp
                last_temp = gpu_temp

                if cooling_on:
                    led = 'blinking (temp falling)' if falling else 'solid (temp rising)'
                else:
                    led = 'off'
                print('GPU {:.1f} C  fan {}  LED {}'.format(
                    gpu_temp, 'ON' if cooling_on else 'off', led))

                hold(cooling_on, falling)
    except KeyboardInterrupt:
        pass
    finally:
        if fan and old_profile:
            try:
                with jtop() as jetson:
                    jetson.fan.set_profile(fan, old_profile)
            except Exception:
                pass
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()

if __name__ == '__main__':
    main()
EOF2
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
GPU 39.5 C  fan off  LED off
GPU 40.0 C  fan off  LED off
Above 40.0 -> fan and LED ON
GPU 40.5 C  fan ON  LED solid (temp rising)
GPU 39.5 C  fan ON  LED blinking (temp falling)
GPU 38.5 C  fan ON  LED blinking (temp falling)
Below 38.0 -> fan and LED OFF
GPU 37.5 C  fan off  LED off
```

Press **Ctrl+C** to stop. The LED turns off, GPIO is released, and the fan profile is restored.

To keep it running after you close the SSH session:

```bash
nohup python3 rlled2.py > rlled2.log 2>&1 &
tail -f rlled2.log        # watch output; Ctrl+C stops tail only
```

Stop a background run with `pkill -f rlled2.py`.

## Testing the LED on its own

Solid for 3 s:

```bash
python3 -c "import RPi.GPIO as G,time;G.setmode(G.BOARD);G.setup(29,G.OUT);G.output(29,1);time.sleep(3);G.cleanup()"
```

Solid for 3 s, then blink 5 times:

```bash
python3 -c "
import RPi.GPIO as G, time
G.setmode(G.BOARD); G.setup(29, G.OUT)
G.output(29, 1); time.sleep(3)
for _ in range(5):
    G.output(29, 0); time.sleep(0.5); G.output(29, 1); time.sleep(0.5)
G.cleanup()"
```

## Tuning

Edit the constants at the top of `rlled2.py`:

| Constant        | Default | Meaning                                              |
|-----------------|---------|------------------------------------------------------|
| `LED_PIN`       | 29      | BOARD pin number for the LED                         |
| `TEMP_ON`       | 40.0    | Turn fan + LED on above this temperature             |
| `TEMP_OFF`      | 38.0    | Turn fan + LED off below this temperature            |
| `FAN_SPEED`     | 100     | Fan speed (%) while cooling is on                    |
| `POLL_SECONDS`  | 5       | How often the temperature is checked                 |
| `BLINK_SECONDS` | 0.5     | On/off time while blinking                           |

Set `TEMP_OFF = 40.0` for a single hard threshold. If sensor noise causes unwanted blinking, add a dead band: `falling = last_temp is not None and gpu_temp < last_temp - 0.5`.

## Adding more loads (extra LEDs, motor, relay)

GPIO pins are weak (~1–2 mA), so anything beyond one small LED should be switched with an NPN transistor as a low-side switch:

- GPIO pin → 1 kΩ → transistor base
- Transistor emitter → GND (Jetson GND and the load's supply GND tied together)
- Load supply + → load → transistor collector
- For a motor: add a flyback diode (1N4007) across the motor, stripe toward supply +, and use a separate motor supply (not the Jetson 5 V pin)

Small LEDs: 2N2222 / BC547 with the LED powered from the 5 V pin (pin 2 or 4) through 220 Ω. Motors up to a few amps: TIP120. Or use a ready-made L298N / DRV8833 motor driver module (GPIO to IN1, motor and supply to the screw terminals).

Spare GPIO pins next to 29: 31 and 33. Ground pins: 6, 9, 14, 20, 25, 30, 34, 39.

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

**LED glows dimly in the script but brightly in the one-liner test**
Another GPIO pin driven LOW is sharing a breadboard row with the LED or its resistor. Give the LED its own row, and don't set up GPIO pins the script isn't using.

**LED never lights**
Check the ground jumper (Jetson pin 39 → LED short leg) and that the LED isn't in backwards. Run the one-liner test above.

**Fan runs even when the script says off**
Another process (or the system's `nvfancontrol`) is driving it. Check with `sudo jtop` → CTRL tab, and `pkill -f rlled2.py` to make sure no old copy of the script is running.
