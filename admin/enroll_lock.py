"""
Tiny helpers that let the enrollment window and the kiosk share one camera,
and let the dashboard tell whether a background worker is still alive.

* While enrollment needs the camera it creates ".enrollment_active". The
  kiosk sees that file, releases the camera, and re-opens it once the file is
  gone. The file is refreshed every few seconds; if the enrollment process
  dies, the file goes stale after STALE_SECONDS and the kiosk resumes on its own.
* Heartbeat writes a timestamp file at a steady rate so another process can
  tell "still working" from "crashed" without OS-specific process checks.
"""

import json
import os
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(BASE_DIR, ".enrollment_active")
STALE_SECONDS = float(os.environ.get("ENROLL_LOCK_STALE_SEC", "30"))


def acquire(owner="enroll"):
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump({"owner": owner, "pid": os.getpid(), "time": time.time()}, f)


def touch():
    try:
        os.utime(LOCK_PATH, None)
    except OSError:
        pass


def release():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def is_active():
    """True while a live enrollment holds the camera."""
    try:
        age = time.time() - os.path.getmtime(LOCK_PATH)
    except OSError:
        return False
    return age < STALE_SECONDS


class Heartbeat(threading.Thread):
    """Background thread: writes `path` (and refreshes the camera lock)."""

    def __init__(self, path=None, interval=3.0):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self._stop_evt = threading.Event()

    def beat(self):
        if self.path:
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    f.write(str(time.time()))
            except OSError:
                pass
        touch()

    def run(self):
        while not self._stop_evt.is_set():
            self.beat()
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
