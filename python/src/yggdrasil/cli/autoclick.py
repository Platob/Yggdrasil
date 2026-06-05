"""
Simple auto-clicker with configurable interval, click count, and hotkey to stop.
"""
import time
import threading
import pyautogui
from pynput import keyboard

# Configuration
CLICK_INTERVAL = 60   # seconds between clicks
CLICK_COUNT = 0        # 0 = infinite
BUTTON = "left"        # "left", "right", or "middle"
START_DELAY = 3.0      # seconds before clicking starts
STOP_KEY = keyboard.Key.f8

# Failsafe: move mouse to a screen corner to abort
pyautogui.FAILSAFE = True

stop_event = threading.Event()


def on_press(key):
    if key == STOP_KEY:
        stop_event.set()
        return False  # stop listener


def click_loop():
    print(f"Starting in {START_DELAY}s... press F8 to stop, or slam mouse to a corner.")
    time.sleep(START_DELAY)

    count = 0
    while not stop_event.is_set():
        if CLICK_COUNT and count >= CLICK_COUNT:
            break
        try:
            pyautogui.click(button=BUTTON)
        except pyautogui.FailSafeException:
            print("Failsafe triggered.")
            break
        count += 1
        # Sleep in small slices so stop is responsive
        end = time.monotonic() + CLICK_INTERVAL
        while time.monotonic() < end and not stop_event.is_set():
            time.sleep(min(0.02, end - time.monotonic()))

    print(f"Stopped after {count} click(s).")


def main():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        click_loop()
    finally:
        stop_event.set()
        listener.stop()


if __name__ == "__main__":
    main()