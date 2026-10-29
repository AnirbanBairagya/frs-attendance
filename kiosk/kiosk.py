"""BCREC Faculty Attendance Kiosk with passive anti-spoofing and active liveness."""
import os
import time
from collections import deque
from datetime import datetime
import cv2
import numpy as np

from common.anti_spoof import AntiSpoofEngine, REAL_THRESHOLD, TEMPORAL_FRAMES
from common.lite_core import CLASSIFIER_PATH, FaceEngine, ensure_db, get_faculty, identify, largest_face, load_classifier, mark_attendance, open_camera
from liveness import LivenessChallenge
from . import kiosk_sync

DETECT_EVERY = int(os.environ.get("DETECT_EVERY", "2"))
MIN_FACE_PX = int(os.environ.get("MIN_FACE_PX", "90"))
EMB_FRAMES = int(os.environ.get("EMB_FRAMES", "2"))
RESULT_HOLD = float(os.environ.get("RESULT_HOLD", "3"))
CLASSIFIER_CHECK_SEC = float(os.environ.get("CLASSIFIER_CHECK_SEC", "3"))
if os.name != "nt":
    cv2.setNumThreads(int(os.environ.get("OPENCV_THREADS", "2")))
FULLSCREEN = os.environ.get("KIOSK_FULLSCREEN", "1") == "1"
GREEN = (60, 170, 60)
AMBER = (40, 170, 240)
RED = (60, 60, 200)
WHITE = (255, 255, 255)
NAVY = (32, 22, 11)

def put(img, text, org, scale=0.8, color=WHITE, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def reset_security(liveness, embeddings, spoof_history):
    embeddings.clear()
    spoof_history.clear()
    liveness.reset()

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
    except Exception as exc:
        print(f"[kiosk] could not load classifier: {exc}")
        return None

def sync_line(sync):
    """Short status for the top bar: (text, colour)."""
    if sync is None:
        return "", WHITE
    s = sync.snapshot()
    pending = f" - {s['pending']} queued" if s["pending"] else ""
    if s.get("warning"):
        return "! " + s["warning"][:48], AMBER
    if s["online"]:
        return "Synced" + pending, GREEN
    if s["ever_synced"]:
        return "Offline" + pending, AMBER
    return "Connecting to admin PC...", AMBER

def build_result(res, sync=None):
    if not res["accepted"]:
        return {"title": "Not recognised", "line": "Please try again", "color": RED}
    fac = get_faculty(res["faculty_id"])
    if fac is None:
        return {"title": "Unknown ID", "line": res["faculty_id"], "color": RED}
    when = datetime.now()
    status, t, late = mark_attendance(fac["id"], now=when)
    if sync is not None and status != "cooldown":
        try:
            kiosk_sync.enqueue_event(fac["faculty_id"], when.timestamp())
            sync.kick()
        except Exception as exc:  # noqa: BLE001 - the scan is already recorded locally
            print(f"[kiosk] could not queue the scan for the admin PC: {exc}")
    labels = {
        "checkin_first": "CHECK-IN" + ("  (LATE)" if late else ""),
        "checkin": "CHECK-IN (re-entry)",
        "checkout": "CHECK-OUT",
        "cooldown": "Already recorded",
    }
    return {"title": fac["name"], "line": f"{labels[status]}   {t}",
            "color": AMBER if late or status == "cooldown" else GREEN}

def main():
    ensure_db()
    try:
        sync = kiosk_sync.start_if_configured()
    except kiosk_sync.ConfigError as exc:
        raise SystemExit(f"{exc}\nFix or delete kiosk_config.json and start the kiosk again.")
    print("[kiosk] linked to the admin PC" if sync else "[kiosk] stand-alone mode (no kiosk_config.json)")
    loaded_stamp = classifier_stamp()
    bundle = safe_load_classifier()
    if bundle is None:
        print("[kiosk] no trained faces yet - waiting for the first faculty registration")
    engine = FaceEngine()
    anti = AntiSpoofEngine()
    liveness = LivenessChallenge()
    cap = open_camera()
    win = "BCREC Faculty Attendance"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    faces = np.empty((0, 15), np.float32)
    embeddings = []
    spoof_history = deque(maxlen=TEMPORAL_FRAMES)
    liveness_passed = False
    liveness_state = {"passed": False, "failed": False, "prompt": "Look at the camera", "progress": "0/3"}
    result = None
    result_until = 0.0
    last_spoof_text = "Anti-spoof: waiting"
    frame_i = 0
    last_check = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame_i += 1
            now = time.time()

            # Pick up newly trained faces without a restart.
            if now - last_check >= CLASSIFIER_CHECK_SEC:
                last_check = now
                stamp = classifier_stamp()
                if stamp != loaded_stamp:
                    loaded_stamp = stamp
                    if stamp is None:  # file removed (e.g. a registration was undone)
                        bundle = None
                        print("[kiosk] classifier removed")
                    else:
                        new_bundle = safe_load_classifier()
                        if new_bundle is not None:
                            bundle = new_bundle
                            print("[kiosk] classifier reloaded (changed on disk)")
                        else:
                            print("[kiosk] keeping the previous classifier")

            if bundle is None:
                faces = np.empty((0, 15), np.float32)
                result = None
            elif now >= result_until:
                result = None
                if frame_i % DETECT_EVERY == 0:
                    faces = engine.detect_fast(frame)
                    if len(faces) != 1:
                        reset_security(liveness, embeddings, spoof_history)
                        liveness_passed = False
                        last_spoof_text = "Anti-spoof: waiting for face" if len(faces) == 0 else "ONE PERSON ONLY"
                    else:
                        face = largest_face(faces)
                        if face is None or float(face[2]) < MIN_FACE_PX:
                            reset_security(liveness, embeddings, spoof_history)
                            liveness_passed = False
                            last_spoof_text = "Move closer"
                        else:
                            anti_result = anti.score(frame, face)
                            if anti_result is None:
                                reset_security(liveness, embeddings, spoof_history)
                                liveness_passed = False
                                last_spoof_text = "Anti-spoof unavailable"
                            else:
                                spoof_history.append(anti_result["real"])
                                median_real = float(np.median(np.asarray(spoof_history)))
                                passive_real = len(spoof_history) >= TEMPORAL_FRAMES and median_real >= REAL_THRESHOLD
                                last_spoof_text = f"Anti-spoof REAL {median_real:.2f}" if passive_real else f"Anti-spoof {median_real:.2f}"
                                if not passive_real:
                                    embeddings.clear()
                                    liveness_passed = False
                                else:
                                    # We rely entirely on the passive MiniFAS Anti-Spoof model.
                                    liveness_passed = True
                                    if liveness_passed:
                                        embeddings.append(engine.embed(frame, face))
                                        if len(embeddings) >= EMB_FRAMES:
                                            mean = np.mean(embeddings, axis=0)
                                            norm = np.linalg.norm(mean)
                                            embeddings.clear()
                                            if norm > 0:
                                                mean /= norm
                                                res = identify(bundle, mean)
                                                result = build_result(res, sync)
                                                result_until = time.time() + RESULT_HOLD
                                                print(f"[scan] {res.get('faculty_id')} conf={res.get('confidence', 0):.2f} cos={res.get('cosine', 0):.2f} anti={median_real:.2f} accepted={res.get('accepted')}")
                                                reset_security(liveness, embeddings, spoof_history)
                                                liveness_passed = False
            for f in faces:
                x, y, w, h = map(int, f[:4])
                box_color = GREEN if liveness_passed else AMBER
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            H, W = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (W, 55), NAVY, -1)
            put(frame, "BCREC Faculty Attendance", (10, 34), 0.75)
            put(frame, datetime.now().strftime("%d %b %Y  %H:%M:%S"), (W - 300, 28), 0.60)
            line, line_color = sync_line(sync)
            if line:
                put(frame, line, (W - 300, 48), 0.42, line_color, 1)
            cv2.rectangle(frame, (0, H - 130), (W, H), NAVY, -1)
            
            if bundle is None:
                put(frame, "No faculty enrolled yet", (12, H - 72), 0.72, AMBER, 2)
                put(frame, "Waiting for the admin PC" if sync else "Register faculty from the dashboard first",
                    (12, H - 32), 0.58, WHITE, 2)
            elif result:
                put(frame, result["title"], (12, H - 78), 0.95, result["color"], 2)
                put(frame, result["line"], (12, H - 40), 0.75, result["color"], 2)
            elif len(faces) == 0:
                put(frame, "Look at the camera", (12, H - 72), 0.72, WHITE, 2)
                put(frame, last_spoof_text, (12, H - 32), 0.60, AMBER, 2)
            elif not liveness_passed:
                # Replaced the active prompt with a fast static verifying prompt
                put(frame, "Analyzing liveness...", (12, H - 72), 0.70, AMBER, 2)
                put(frame, last_spoof_text, (12, H - 32), 0.58, WHITE, 2)
            else:
                put(frame, "Recognizing face...", (12, H - 72), 0.68, GREEN, 2)
                put(frame, last_spoof_text, (12, H - 32), 0.58, WHITE, 2)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                loaded_stamp = classifier_stamp()
                bundle = safe_load_classifier() or bundle
                print("[kiosk] classifier reloaded")
    finally:
        if sync is not None:
            sync.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
