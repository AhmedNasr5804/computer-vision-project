# Raspberry Pi bridge scripts

`eye_to_pic.py` is the Pi-side companion to the Android app's Firebase publisher (`android_app/app/src/main/java/com/cv552/eyedemo/FirebaseClient.kt`). It polls the `/eye_monitor` node at the lab1-f7c43 Realtime Database and writes single-byte commands to the PIC microcontroller over UART so the car only moves when the driver is awake.

| Phone publishes | Pi sends to PIC | PIC effect |
|---|---|---|
| `state = "OPEN"` (driver awake) | `'E'` (one byte) | `eyes_open = 1` → state machine clears `STATE_EYES_CLOSED` → AUTO-mode drives |
| `state = "CLOSED"` | `'C'` (one byte) | `eyes_open = 0` → safety state-machine enters `STATE_EYES_CLOSED` → `Motor_Stop()` |
| `state = "UNKNOWN"` (no face) | `'C'` | as above — fail-safe to STOP |
| network down / `timestamp > 2s` old | `'C'` | as above — fail-safe to STOP |

The script is **edge-triggered**: it sends a byte only when the decision changes, plus a 5-second **heartbeat** re-send so the PIC re-syncs if a UART byte is ever lost (no PIC firmware change needed).

## Wiring (PIC ⇄ Pi UART)

| Pi pin (BCM) | PIC pin | Notes |
|---|---|---|
| GPIO 14 (TXD0) | RC7 (RX) of the PIC16F877A | Pi 3.3 V is fine for the PIC's 5 V TTL input |
| GPIO 15 (RXD0) | RC6 (TX) of the PIC16F877A | **Add a 3.3 V level shifter** (or 2-resistor divider) — the PIC's TX is 5 V and will damage the Pi GPIO over time |
| GND | GND | common ground required |

Enable the primary UART on the Pi:

```bash
sudo raspi-config nonint do_serial 2     # disable shell on /dev/serial0, enable HW UART
sudo reboot
```

After reboot `/dev/serial0` is the PL011 UART (the same baud rate the PIC firmware uses: 9600 8N1).

If you instead use a USB-to-TTL adapter, pass `--serial /dev/ttyUSB0`.

## Install + run

```bash
# Pi OS Bookworm 64-bit
sudo apt update
sudo apt install -y python3-pip
pip install --break-system-packages requests pyserial

# Drop the script onto the Pi (scp / git pull / whatever)
cd ~/eye_to_pic
chmod +x eye_to_pic.py

# Smoke test without a PIC attached -- prints every decision, never writes UART
python3 eye_to_pic.py --dry-run -v

# Real run
sudo python3 eye_to_pic.py
```

Expected output during a real run (eyes opening / closing a few times in front of the phone camera):

```
[02:39:01] bridge online | URL=https://lab1-f7c43-default-rtdb.firebaseio.com/eye_monitor.json | serial=/dev/serial0 | dry_run=False | poll=10Hz | heartbeat=5s
[02:39:01] TRANSITION state=OPEN    p_open=0.92 age=  84ms fps=30.1 -> E (awake)
[02:39:06] heartbeat -> E (awake)
[02:39:11] heartbeat -> E (awake)
[02:39:14] TRANSITION state=CLOSED  p_open=0.18 age=  77ms fps=30.4 -> C (asleep)
[02:39:19] heartbeat -> C (asleep)
[02:39:22] TRANSITION state=OPEN    p_open=0.88 age=  92ms fps=30.2 -> E (awake)
```

## Auto-start on boot (systemd)

Create `/etc/systemd/system/eye-to-pic.service`:

```ini
[Unit]
Description=Eye-state -> PIC UART bridge
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/pi/eye_to_pic
ExecStart=/usr/bin/python3 /home/pi/eye_to_pic/eye_to_pic.py
Restart=on-failure
RestartSec=2
User=root            # required for /dev/serial0 access without group fiddling
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now eye-to-pic
sudo journalctl -fu eye-to-pic     # follow logs
```

The unit's `Restart=on-failure` plus the script's own exponential-backoff retry loop together cover most transient failures (Wi-Fi drop, UART glitch). On `systemctl stop`, the script's SIGTERM handler writes one final `'C'` to the PIC before exiting, so the car halts cleanly.

## Failure modes worth being aware of

1. **Pi loses Wi-Fi mid-drive.** The first failed `requests.get(...)` doesn't immediately flip the car to STOP — the script keeps the last successful command for up to `STALE_MS` (2 s). Past that, the timestamp from the last cached fetch is stale and the script switches to `'C'`. So you get a ~2-second grace window before the safety stop fires.
2. **Phone goes to background / locks.** `FirebaseClient` stops publishing because the CameraX analyzer stops getting frames; the `timestamp` field freezes; after 2 s the Pi sends `'C'`. Same path as Wi-Fi loss.
3. **PIC reboots.** It powers up with `eyes_open = 1` (see firmware `main.c`) so the first PIC-side state is SAFE. Within at most 5 s the Pi heartbeat re-sends the current command, re-syncing the two sides.
4. **Pi reboots.** systemd restarts the bridge in ≈10 s and the publisher schedule on the phone is unchanged, so the bridge picks up where it left off.
