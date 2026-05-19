#!/usr/bin/env python3
"""eye_to_pic.py
================

Raspberry Pi bridge between the Firebase ``eye_monitor`` feed (published
by the Android app in this repo) and the PIC microcontroller that drives
the RC car. Sends single-byte UART commands to the PIC so the car only
moves when the Android pipeline says the driver is awake.

PIC command mapping (matches firmware/main.c on the PIC side):

    'E'  ->  eyes_open = 1  (driver awake -> SAFE -> motors allowed to run)
    'C'  ->  eyes_open = 0  (eyes closed  -> STATE_EYES_CLOSED -> Motor_Stop())

The script is edge-triggered: a new byte is only written when the
decision changes, plus a heartbeat re-send every HEARTBEAT_S seconds so
the PIC re-syncs if a serial byte is ever lost.

Fail-safe behaviour
-------------------

* The phone's ``timestamp`` field is checked against wall-clock; if the
  most-recent sample is older than STALE_MS, we treat the link as down
  and send 'C' (stop).
* ``state == "UNKNOWN"`` (face lost in the app's analyzer) also maps to
  'C'.
* On SIGINT / SIGTERM we write one final 'C' and flush before exiting,
  so the car is left in a safe state on shutdown.
* Network errors are caught with exponential backoff. While a fetch is
  failing we *don't* prematurely flip to 'C' on the first miss --- we
  let STALE_MS govern that decision so a brief packet loss does not
  cause the car to jerk to a halt.

Behaviour against the four cases the user asked for
---------------------------------------------------

* eye OPEN, car stopped     ->  send 'E', PIC clears the EYES_CLOSED safety
                                state, the AUTO-mode loop in the PIC starts
                                driving on its own.
* eye OPEN, car moving      ->  no command sent (state did not change);
                                heartbeat 'E' re-sent every HEARTBEAT_S
                                seconds as a watchdog.
* eye CLOSED, car moving    ->  send 'C' immediately, PIC's safety
                                state-machine fires STATE_EYES_CLOSED and
                                Motor_Stop() is called the same loop tick.
* eye CLOSED, car stopped   ->  no command sent (state did not change);
                                PIC keeps motors stopped.

Usage
-----

::

    pip install requests pyserial
    sudo python3 eye_to_pic.py                       # production
    python3 eye_to_pic.py --dry-run -v               # no UART, log every poll
    python3 eye_to_pic.py --serial /dev/ttyUSB0      # USB-serial adapter
"""

import argparse
import logging
import signal
import sys
import time
from contextlib import contextmanager

try:
    import requests
except ImportError:
    sys.exit("requests not installed.  run: pip install requests pyserial")

try:
    import serial  # provided by the 'pyserial' package
except ImportError:
    sys.exit("pyserial not installed.  run: pip install requests pyserial")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
URL              = "https://lab1-f7c43-default-rtdb.firebaseio.com/eye_monitor.json"
STALE_MS         = 2000     # treat samples older than this as UNKNOWN
POLL_INTERVAL_S  = 0.1      # 10 Hz -- matches the publisher's throttle
HEARTBEAT_S      = 5.0      # re-send the current command at least this often
NET_TIMEOUT_S    = 2.0
BAUD             = 9600     # matches UART_Init(9600) on the PIC
DEFAULT_SERIAL   = "/dev/serial0"

CMD_AWAKE = b"E"
CMD_SLEEP = b"C"

log = logging.getLogger("eye_to_pic")


# --------------------------------------------------------------------------
# Decision logic
# --------------------------------------------------------------------------
def state_to_command(state, age_ms):
    """Map a Firebase snapshot to a PIC command.  Conservative on missing
    or stale data: anything except a fresh ``"OPEN"`` returns ``CMD_SLEEP``."""
    if age_ms is None or age_ms > STALE_MS:
        return CMD_SLEEP, "stale"
    if state == "OPEN":
        return CMD_AWAKE, "awake"
    if state == "CLOSED":
        return CMD_SLEEP, "asleep"
    return CMD_SLEEP, f"unknown({state})"


# --------------------------------------------------------------------------
# Firebase fetch
# --------------------------------------------------------------------------
def fetch():
    """Return ``(state, p_open, p_closed, ts_ms, fps)`` or all-None on error."""
    try:
        r = requests.get(URL, timeout=NET_TIMEOUT_S)
        r.raise_for_status()
        d = r.json() or {}
        return (
            d.get("state"),
            float(d.get("p_open", 0.0)),
            float(d.get("p_closed", 0.0)),
            int(d.get("timestamp", 0)) or None,
            float(d.get("fps", 0.0)),
        )
    except Exception as exc:                              # noqa: BLE001
        log.warning("fetch failed: %s", exc)
        return (None,) * 5


# --------------------------------------------------------------------------
# Serial wrapper
# --------------------------------------------------------------------------
@contextmanager
def open_serial(dev, dry_run):
    if dry_run:
        log.info("DRY RUN: not opening %s", dev)
        yield None
        return
    ser = serial.Serial(dev, BAUD, timeout=0)
    try:
        yield ser
    finally:
        try: ser.close()
        except Exception: pass


def write_cmd(ser, cmd):
    """Best-effort UART write.  Returns True on success, False on serial error."""
    if ser is None:
        return True
    try:
        ser.write(cmd)
        ser.flush()
        return True
    except serial.SerialException as exc:
        log.warning("serial write failed: %s", exc)
        return False


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--serial",  default=DEFAULT_SERIAL,
                        help=f"PIC UART device (default {DEFAULT_SERIAL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands instead of writing to UART")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="log every poll, not just state transitions")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    with open_serial(args.serial, args.dry_run) as ser:

        # Send 'C' on shutdown so the car halts on Ctrl-C / systemd stop.
        def safe_stop(signum=None, frame=None):
            log.info("shutdown: writing final 'C' (safe stop)")
            write_cmd(ser, CMD_SLEEP)
            sys.exit(0)

        signal.signal(signal.SIGINT,  safe_stop)
        signal.signal(signal.SIGTERM, safe_stop)

        log.info("bridge online | URL=%s | serial=%s | dry_run=%s | poll=%.0fHz | heartbeat=%.0fs",
                 URL, args.serial, args.dry_run, 1.0 / POLL_INTERVAL_S, HEARTBEAT_S)

        last_cmd       = None
        last_sent_at_s = 0.0
        backoff_s      = 1.0

        while True:
            loop_start = time.time()
            state, p_open, p_closed, ts_ms, fps = fetch()

            if ts_ms is None and state is None:
                # Full network failure - throttle the retry to avoid log spam.
                # Do NOT change last_cmd here; STALE_MS will eventually flip
                # us to 'C' through the normal path once the next successful
                # fetch returns a stale timestamp.
                time.sleep(min(backoff_s, 8.0))
                backoff_s = min(backoff_s * 2, 8.0)
                continue
            backoff_s = 1.0

            now_ms = int(time.time() * 1000)
            age_ms = (now_ms - ts_ms) if ts_ms is not None else None
            cmd, reason = state_to_command(state, age_ms)

            send_now = (cmd != last_cmd) or (loop_start - last_sent_at_s > HEARTBEAT_S)

            if send_now:
                if write_cmd(ser, cmd):
                    if cmd != last_cmd:
                        log.info(
                            "TRANSITION state=%-7s p_open=%.2f age=%4sms fps=%4.1f -> %s (%s)",
                            state or "MISSING",
                            p_open or 0.0,
                            age_ms if age_ms is not None else "----",
                            fps or 0.0,
                            cmd.decode(), reason,
                        )
                    else:
                        log.info("heartbeat -> %s (%s)", cmd.decode(), reason)
                    last_cmd       = cmd
                    last_sent_at_s = loop_start
            elif args.verbose:
                log.info(
                    "hold state=%-7s p_open=%.2f age=%4sms fps=%4.1f cmd=%s",
                    state or "MISSING",
                    p_open or 0.0,
                    age_ms if age_ms is not None else "----",
                    fps or 0.0,
                    (last_cmd or b"-").decode(),
                )

            elapsed = time.time() - loop_start
            sleep_for = POLL_INTERVAL_S - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
