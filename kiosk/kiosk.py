"""
BCREC Faculty Attendance Kiosk (Raspberry Pi).

Camera -> face detection -> passive anti-spoofing (MiniFAS) -> SFace embedding ->
cosine matching against the enrolled faculty -> attendance record.

Built to keep running: a camera that drops out is re-opened, an error while
processing one frame never stops the loop, and a scan is saved locally together
with its upload-queue entry in one database transaction.
"""
import os as _os
import sys as _sys

# Make this file's own folder and the shared ../common folder importable, however
# it is started (python app.py, python admin/app.py, python -m admin.app, ...).
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.join(_here, "..", "common"), _here):
    _p = _os.path.normpath(_p)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import os
import sqlite3
import time
import traceback
import uuid
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import kiosk_sync
from anti_spoof import AntiSpoofEngine, REAL_THRESHOLD, TEMPORAL_FRAMES
from lite_core import (CLASSIFIER_PATH, FaceEngine, ensure_db, get_faculty, identify, largest_face,
                       load_classifier, mark_attendance, open_camera)

DETECT_EVERY = int(os.environ.get("DETECT_EVERY", "2"))
MIN_FACE_PX = int(os.environ.get("MIN_FACE_PX", "90"))
EMB_FRAMES = int(os.environ.get("EMB_FRAMES", "2"))
RESULT_HOLD = float(os.environ.get("RESULT_HOLD", "3"))          # seconds a successful result stays up
REJECT_HOLD = float(os.environ.get("REJECT_HOLD", "1.5"))        # a rejected scan retries sooner
SAME_FACE_MIN = float(os.environ.get("SAME_FACE_MIN", "0.5"))    # frames of one scan must show the same face
CLASSIFIER_CHECK_SEC = float(os.environ.get("CLASSIFIER_CHECK_SEC", "3"))
BAD_READ_LIMIT = int(os.environ.get("BAD_READ_LIMIT", "40"))     # failed reads before the camera is re-opened
CAMERA_RETRY_SEC = float(os.environ.get("CAMERA_RETRY_SEC", "2"))
MAX_CONSECUTIVE_ERRORS = int(os.environ.get("MAX_CONSECUTIVE_ERRORS", "5"))
PERF_LOG_SEC = float(os.environ.get("PERF_LOG_SEC", "30"))       # 0 = no timing log
if os.name != "nt":
    cv2.setNumThreads(int(os.environ.get("OPENCV_THREADS", "2")))
FULLSCREEN = os.environ.get("KIOSK_FULLSCREEN", "1") == "1"
WIN = "BCREC Faculty Attendance"
GREEN = (60, 170, 60)
AMBER = (40, 170, 240)
RED = (60, 60, 200)
WHITE = (255, 255, 255)
NAVY = (32, 22, 11)


def put(img, text, org, scale=0.8, color=WHITE, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def log(msg):
    print(f"[kiosk] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Classifier file handling
# ---------------------------------------------------------------------------
def classifier_stamp():
    """Changes whenever face_classifier_lite.pkl is replaced (new faculty trained)."""
    try:
        st = os.stat(CLASSIFIER_PATH)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def safe_load_classifier():
    try:
        return load_classifier()
    except Exception as exc:  # noqa: BLE001
        log(f"could not load classifier: {exc}")
        return None


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------
def message_frame(line1, line2=""):
    img = np.full((480, 640, 3), NAVY, np.uint8)
    put(img, WIN, (20, 60), 0.9)
    put(img, line1, (20, 230), 0.9, AMBER)
    if line2:
        put(img, line2, (20, 280), 0.65)
    return img


def sync_line(sync):
    """Short status for the top bar: (text, colour)."""
    if sync is None:
        return "", WHITE
    try:
        s = sync.snapshot()
    except Exception:  # noqa: BLE001 - a status line must never stop the kiosk
        return "", WHITE
    pending = f" - {s['pending']} queued" if s["pending"] else ""
    if s.get("warning"):
        return "! " + s["warning"][:48], AMBER
    if s["online"]:
        return "Synced" + pending, GREEN
    if s["ever_synced"]:
        return "Offline" + pending, AMBER
    return "Connecting to admin PC...", AMBER


# ---------------------------------------------------------------------------
# Recording a scan
# ---------------------------------------------------------------------------
def record_scan(fac, sync):
    """Save attendance locally; when linked, queue the upload in the SAME transaction."""
    when = datetime.now()
    extra = None
    if sync is not None:
        event_id = uuid.uuid4().hex
        extra = lambda conn: kiosk_sync.enqueue_in(conn, event_id, fac["faculty_id"], when.timestamp())  # noqa: E731
    try:
        out = mark_attendance(fac["id"], now=when, extra=extra)
    except sqlite3.OperationalError as exc:
        if extra is None or "no such table" not in str(exc).lower():
            raise
        kiosk_sync.ensure_outbox()
        out = mark_attendance(fac["id"], now=when, extra=extra)
    if sync is not None and out[0] != "cooldown":
        sync.kick()
    return out


def build_result(res, sync=None):
    if not res["accepted"]:
        if res.get("reason") == "ambiguous":
            return {"title": "Not sure who this is", "line": "Face the camera and try again", "color": RED,
                    "hold": REJECT_HOLD}
        return {"title": "Not recognised", "line": "Please try again", "color": RED, "hold": REJECT_HOLD}
    try:
        fac = get_faculty(res["faculty_id"])
        if fac is None:
            return {"title": "Unknown ID", "line": res["faculty_id"], "color": RED, "hold": REJECT_HOLD}
        status, t, late = record_scan(fac, sync)
    except Exception as exc:  # noqa: BLE001 - never let a database hiccup stop the kiosk
        log(f"could not record the scan: {type(exc).__name__}: {exc}")
        return {"title": "System busy", "line": "Please scan again", "color": RED, "hold": REJECT_HOLD}
    labels = {
        "checkin_first": "CHECK-IN" + ("  (LATE)" if late else ""),
        "checkin": "CHECK-IN (re-entry)",
        "checkout": "CHECK-OUT",
        "cooldown": "Already recorded",
    }
    return {"title": fac["name"], "line": f"{labels[status]}   {t}",
            "color": AMBER if late or status == "cooldown" else GREEN, "hold": RESULT_HOLD}


# ---------------------------------------------------------------------------
# Timing log (so slow stages on the Pi are visible)
# ---------------------------------------------------------------------------
class Perf:
    def __init__(self):
        self.t0 = time.time()
        self.sum = {}
        self.n = {}
        self.frames = 0

    def add(self, name, seconds):
        self.sum[name] = self.sum.get(name, 0.0) + seconds
        self.n[name] = self.n.get(name, 0) + 1

    def tick(self):
        self.frames += 1
        if not PERF_LOG_SEC or time.time() - self.t0 < PERF_LOG_SEC:
            return
        el = time.time() - self.t0
        parts = " ".join(f"{k}={self.sum[k] / self.n[k] * 1000:.0f}ms" for k in sorted(self.sum))
        log(f"[perf] {self.frames / el:.1f} fps  {parts}")
        self.__init__()


# ---------------------------------------------------------------------------
# The kiosk
# ---------------------------------------------------------------------------
class Kiosk:
    def __init__(self, sync):
        self.sync = sync
        self.loaded_stamp = classifier_stamp()
        self.bundle = safe_load_classifier()
        if self.bundle is None:
            log("no trained faces yet - waiting for the first faculty registration")
        self.engine = FaceEngine()
        self.anti = AntiSpoofEngine()
        self.faces = np.empty((0, 15), np.float32)
        self.embeddings = []
        self.spoof_history = deque(maxlen=TEMPORAL_FRAMES)
        self.passed = False
        self.result = None
        self.result_until = 0.0
        self.spoof_text = "Anti-spoof: waiting"
        self.frame_i = 0
        self.last_check = 0.0
        self.errors = 0
        self.last_error_log = 0.0
        self.perf = Perf()

    # -- state ---------------------------------------------------------------
    def reset_scan(self):
        self.embeddings.clear()
        self.spoof_history.clear()
        self.passed = False

    def reset_all(self):
        self.reset_scan()
        self.faces = np.empty((0, 15), np.float32)
        self.result = None
        self.result_until = 0.0

    def reload_if_changed(self, now):
        """Pick up newly trained faces without a restart."""
        if now - self.last_check < CLASSIFIER_CHECK_SEC:
            return
        self.last_check = now
        stamp = classifier_stamp()
        if stamp == self.loaded_stamp:
            return
        self.loaded_stamp = stamp
        if stamp is None:  # file removed (e.g. the last registration was undone)
            self.bundle = None
            log("classifier removed")
            return
        new_bundle = safe_load_classifier()
        if new_bundle is not None:
            self.bundle = new_bundle
            log("classifier reloaded (changed on disk)")
        else:
            log("keeping the previous classifier")

    def force_reload(self):
        self.loaded_stamp = classifier_stamp()
        self.bundle = safe_load_classifier() or self.bundle
        log("classifier reloaded")

    # -- one frame -----------------------------------------------------------
    def process(self, frame, now):
        if self.bundle is None:
            self.faces = np.empty((0, 15), np.float32)
            self.result = None
            return
        if now < self.result_until:
            return
        self.result = None
        self.frame_i += 1
        if self.frame_i % DETECT_EVERY:
            return

        t = time.perf_counter()
        self.faces = self.engine.detect_fast(frame)
        self.perf.add("detect", time.perf_counter() - t)
        if len(self.faces) != 1:
            self.reset_scan()
            self.spoof_text = "Anti-spoof: waiting for face" if len(self.faces) == 0 else "ONE PERSON ONLY"
            return
        face = largest_face(self.faces)
        if face is None or float(face[2]) < MIN_FACE_PX:
            self.reset_scan()
            self.spoof_text = "Move closer"
            return

        t = time.perf_counter()
        anti_result = self.anti.score(frame, face)
        self.perf.add("spoof", time.perf_counter() - t)
        if anti_result is None:
            self.reset_scan()
            self.spoof_text = "Anti-spoof unavailable"
            return
        self.spoof_history.append(anti_result["real"])
        median_real = float(np.median(np.asarray(self.spoof_history)))
        real = len(self.spoof_history) >= TEMPORAL_FRAMES and median_real >= REAL_THRESHOLD
        self.spoof_text = f"Anti-spoof REAL {median_real:.2f}" if real else f"Anti-spoof {median_real:.2f}"
        if not real:
            self.embeddings.clear()
            self.passed = False
            return

        self.passed = True
        t = time.perf_counter()
        self.embeddings.append(self.engine.embed(frame, face))
        self.perf.add("embed", time.perf_counter() - t)
        if len(self.embeddings) < EMB_FRAMES:
            return

        embs, self.embeddings = self.embeddings, []
        # all frames of one scan must show the same face (not two people passing by)
        if len(embs) > 1 and min(float(e @ embs[0]) for e in embs[1:]) < SAME_FACE_MIN:
            log("scan discarded: frames did not show the same face")
            self.reset_scan()
            return
        mean = np.mean(embs, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 0:
            self.reset_scan()
            return
        res = identify(self.bundle, mean / norm)
        self.result = build_result(res, self.sync)
        self.result_until = time.time() + self.result["hold"]
        log(f"[scan] {res.get('faculty_id') or '-'} cos={res.get('cosine', 0):.2f} "
            f"next={res.get('second', 0):.2f} margin={res.get('margin', 0):.2f} "
            f"anti={median_real:.2f} {res.get('reason')}")
        self.reset_scan()

    def process_safely(self, frame, now):
        """An error while handling one frame must never stop the kiosk."""
        try:
            self.process(frame, now)
            self.errors = 0
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            self.reset_all()
            if time.time() - self.last_error_log > 5:
                self.last_error_log = time.time()
                log(f"frame skipped after an error ({self.errors} in a row): {type(exc).__name__}: {exc}")
                traceback.print_exc()
            if self.errors >= MAX_CONSECUTIVE_ERRORS:
                log("too many errors in a row - reloading the face models")
                self.errors = 0
                try:
                    self.engine = FaceEngine()
                    self.anti = AntiSpoofEngine()
                except BaseException as exc2:  # noqa: BLE001 - SystemExit from a missing model included
                    log(f"could not reload the models: {exc2}")

    # -- drawing ------------------------------------------------------------
    def draw(self, frame):
        for f in self.faces:
            x, y, w, h = map(int, f[:4])
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN if self.passed else AMBER, 2)
        H, W = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (W, 55), NAVY, -1)
        put(frame, WIN, (10, 34), 0.75)
        put(frame, datetime.now().strftime("%d %b %Y  %H:%M:%S"), (W - 300, 28), 0.60)
        line, color = sync_line(self.sync)
        if line:
            put(frame, line, (W - 300, 48), 0.42, color, 1)
        cv2.rectangle(frame, (0, H - 130), (W, H), NAVY, -1)
        if self.bundle is None:
            put(frame, "No faculty enrolled yet", (12, H - 72), 0.72, AMBER, 2)
            put(frame, "Waiting for the admin PC" if self.sync else "Register faculty from the dashboard first",
                (12, H - 32), 0.58, WHITE, 2)
        elif self.result:
            put(frame, self.result["title"], (12, H - 78), 0.95, self.result["color"], 2)
            put(frame, self.result["line"], (12, H - 40), 0.75, self.result["color"], 2)
        elif len(self.faces) == 0:
            put(frame, "Look at the camera", (12, H - 72), 0.72, WHITE, 2)
            put(frame, self.spoof_text, (12, H - 32), 0.60, AMBER, 2)
        elif not self.passed:
            put(frame, "Analyzing liveness...", (12, H - 72), 0.70, AMBER, 2)
            put(frame, self.spoof_text, (12, H - 32), 0.58, WHITE, 2)
        else:
            put(frame, "Recognizing face...", (12, H - 72), 0.68, GREEN, 2)
            put(frame, self.spoof_text, (12, H - 32), 0.58, WHITE, 2)


def main():
    ensure_db()
    try:
        sync = kiosk_sync.start_if_configured()
    except kiosk_sync.ConfigError as exc:
        raise SystemExit(f"{exc}\nFix or delete kiosk_config.json and start the kiosk again.")
    log("linked to the admin PC" if sync else "stand-alone mode (no kiosk_config.json)")

    k = Kiosk(sync)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = None
    bad_reads = 0
    try:
        while True:
            if cap is None:
                try:
                    cap = open_camera()
                    bad_reads = 0
                    k.reset_all()
                    log("camera opened")
                except SystemExit as exc:      # open_camera reports failure by exiting
                    log(f"camera not available yet: {exc}")
                    cv2.imshow(WIN, message_frame("Camera not available", "Check the USB camera. Retrying..."))
                    if cv2.waitKey(int(CAMERA_RETRY_SEC * 1000)) & 0xFF in (27, ord("q")):
                        break
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                bad_reads += 1
                if bad_reads >= BAD_READ_LIMIT:
                    log("camera stopped delivering images - re-opening it")
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    cap = None
                    continue
                time.sleep(0.05)
                continue
            bad_reads = 0

            t_frame = time.perf_counter()
            now = time.time()
            k.reload_if_changed(now)
            k.process_safely(frame, now)
            k.draw(frame)
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            k.perf.add("frame", time.perf_counter() - t_frame)
            k.perf.tick()
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                k.force_reload()
    finally:
        if sync is not None:
            sync.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
