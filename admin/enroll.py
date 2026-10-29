"""
Native faculty enrollment (camera capture).

Three guided camera captures are taken:
  1. Straight
  2. Slightly left
  3. Slightly right

This is intentionally native-camera based rather than browser Base64 capture,
so it is suitable for a Raspberry Pi kiosk.

Two ways to run it:

* From the dashboard (normal): started with --job-dir. Photos are only STAGED
  in that folder and the outcome is written to result.json. Nothing is added
  to the database or known_faces/ here; registration.py commits and trains
  afterwards, and can undo everything if a step fails.
* By hand (legacy): without --job-dir it saves photos and the database record
  directly, like the original script.

While the camera window is open, a ".enrollment_active" file tells the kiosk to
release the camera, so both can run on the same machine.
"""

import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
import uuid

import cv2
import numpy as np

from . import enroll_lock
from common.lite_core import (
    COSINE_THRESHOLD,
    DB_PATH,
    KNOWN_FACES_DIR,
    FaceEngine,
    build_reference_embeddings,
    ensure_db,
    get_faculty,
    largest_face,
    open_camera,
)

PROMPTS = [
    "Look STRAIGHT at the camera",
    "Turn slightly LEFT",
    "Turn slightly RIGHT",
]
WINDOW = "BCREC Faculty Enrollment"
MIN_FACE_PX = int(os.environ.get("MIN_FACE_PX", "90"))  # same default as kiosk.py
CAMERA_OPEN_ATTEMPTS = int(os.environ.get("ENROLL_CAMERA_ATTEMPTS", "12"))
HANDOFF_WAIT = float(os.environ.get("ENROLL_HANDOFF_WAIT", "1.5"))
MAX_BAD_READS = 150


class EnrollmentError(Exception):
    """Something went wrong; the message is shown to the admin."""


class EnrollmentCancelled(EnrollmentError):
    """The admin cancelled (ESC key or the Cancel button on the dashboard)."""


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def open_camera_with_retry(source, cancel_check, attempts=None, delay=1.0):
    """
    open_camera() exits the process when the camera is busy. While the kiosk is
    still letting go of the camera that is expected, so retry for a while.
    """
    attempts = CAMERA_OPEN_ATTEMPTS if attempts is None else attempts
    last = ""
    for _ in range(attempts):
        if cancel_check():
            raise EnrollmentCancelled("Enrollment was cancelled.")
        try:
            return open_camera(source)
        except SystemExit as exc:
            last = str(exc.code) if exc.code not in (None, 0) else ""
            time.sleep(delay)
    raise EnrollmentError(
        "Could not open the camera. Check that it is plugged in and that no other "
        "program is using it, then try again."
        + (f" ({last.splitlines()[0]})" if last else "")
    )


def capture_shots(engine, cap, cancel_check):
    """Show the live window and return 3 frames, one per prompt."""
    shots = []
    bad_reads = 0

    for i, prompt in enumerate(PROMPTS):
        while True:
            if cancel_check():
                raise EnrollmentCancelled("Enrollment was cancelled.")

            ok, frame = cap.read()
            if not ok or frame is None:
                bad_reads += 1
                if bad_reads > MAX_BAD_READS:
                    raise EnrollmentError(
                        "The camera stopped sending images. Check the connection and try again."
                    )
                time.sleep(0.03)
                continue
            bad_reads = 0

            faces = engine.detect_fast(frame)
            view = frame.copy()

            if len(faces) == 0:
                msg, ready = "No face found", False
            elif len(faces) > 1:
                msg, ready = "Only one person in view", False
            elif float(faces[0][2]) < MIN_FACE_PX:
                msg, ready = "Move closer", False
            else:
                msg, ready = "OK - press SPACE", True

            for f in faces:
                x, y, w, h = map(int, f[:4])
                color = (0, 200, 0) if ready else (0, 170, 255)
                cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)

            cv2.putText(
                view, f"[{i + 1}/3] {prompt}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )
            cv2.putText(
                view, msg, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            cv2.putText(
                view, "SPACE = capture   ESC = cancel", (10, view.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                raise EnrollmentCancelled("Enrollment was cancelled at the camera window (ESC).")

            if key == 32 and ready:
                shots.append(frame.copy())
                cv2.waitKey(700)  # short pause so one press can't take two shots
                break

    return shots


# ---------------------------------------------------------------------------
# Checks on the captured photos
# ---------------------------------------------------------------------------
def verify_shots(engine, shots):
    """
    Every photo must contain a face, all three must be the same person, and the
    person must not already be enrolled. Raises EnrollmentError otherwise.
    """
    embeddings = []
    for idx, shot in enumerate(shots):
        face = largest_face(engine.detect(shot))
        if face is None:
            raise EnrollmentError(
                f"Photo {idx + 1} no longer contains a detectable face. Please retake the photos."
            )
        embeddings.append(engine.embed(shot, face))

    for a in range(len(embeddings)):
        for b in range(a + 1, len(embeddings)):
            if float(embeddings[a] @ embeddings[b]) < COSINE_THRESHOLD:
                raise EnrollmentError(
                    "The three photos do not look like the same person. Make sure only "
                    "the person being registered is in front of the camera, then retake."
                )

    existing = build_reference_embeddings(engine)
    for emb in embeddings:
        for fid, refs in existing.items():
            if len(refs) and float(np.max(refs @ emb)) >= COSINE_THRESHOLD:
                row = get_faculty(fid)
                who = f"{fid} ({row['name']})" if row is not None else fid
                raise EnrollmentError(
                    f"This face matches enrolled faculty {who}. Nothing was saved."
                )
    return embeddings


# ---------------------------------------------------------------------------
# Dashboard mode: stage photos + write result.json
# ---------------------------------------------------------------------------
def _write_result(job_dir, data):
    path = os.path.join(job_dir, "result.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def run_staged(args):
    job_dir = os.path.abspath(args.job_dir)
    cancel_path = os.path.join(job_dir, "cancel")

    def cancel_check():
        return os.path.exists(cancel_path)

    beat = enroll_lock.Heartbeat(os.path.join(job_dir, "heartbeat"))
    beat.start()
    cap = None
    locked = False
    try:
        try:
            ensure_db()
            if get_faculty(args.id):
                raise EnrollmentError(f"Faculty ID {args.id} already exists. Nothing was changed.")

            engine = FaceEngine()

            enroll_lock.acquire("enroll.py")
            locked = True
            time.sleep(HANDOFF_WAIT)  # let the kiosk release the camera
            cap = open_camera_with_retry(args.camera, cancel_check)
            try:
                shots = capture_shots(engine, cap, cancel_check)
            finally:
                cap.release()
                cap = None
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
                enroll_lock.release()  # camera is free again
                locked = False

            verify_shots(engine, shots)

            photos_dir = os.path.join(job_dir, "photos")
            os.makedirs(photos_dir, exist_ok=True)
            names = []
            for i, shot in enumerate(shots):
                name = f"shot_{i}.jpg"
                path = os.path.join(photos_dir, name)
                if not cv2.imwrite(path, shot, [cv2.IMWRITE_JPEG_QUALITY, 95]) or not os.path.getsize(path):
                    raise EnrollmentError("Could not save a photo. The disk may be full or read-only.")
                names.append(name)

            _write_result(job_dir, {"ok": True, "photos": names})
            return 0

        except EnrollmentCancelled as exc:
            _write_result(job_dir, {"ok": False, "cancelled": True, "error": str(exc)})
        except EnrollmentError as exc:
            _write_result(job_dir, {"ok": False, "error": str(exc)})
        except SystemExit as exc:
            _write_result(job_dir, {"ok": False, "error": (str(exc.code) if exc.code not in (None, 0) else "Enrollment stopped.")})
        except cv2.error as exc:
            traceback.print_exc()
            _write_result(job_dir, {
                "ok": False,
                "error": "The camera window could not be shown or an image could not be processed. "
                         "Start the dashboard from the attendance machine's desktop session "
                         "(not as a background service).",
                "details": str(exc)[:300],
            })
        except Exception as exc:  # noqa: BLE001 - last resort; report instead of dying silently
            traceback.print_exc()
            _write_result(job_dir, {"ok": False, "error": f"Unexpected error: {type(exc).__name__}: {exc}"})
        return 1
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if locked:
            enroll_lock.release()
        beat.stop()


# ---------------------------------------------------------------------------
# Legacy mode: save straight to known_faces/ and the database
# ---------------------------------------------------------------------------
def run_legacy(args):
    ensure_db()
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    if get_faculty(args.id):
        raise SystemExit(f"Faculty ID {args.id} already exists. Nothing was changed.")

    engine = FaceEngine()
    beat = enroll_lock.Heartbeat(None)
    enroll_lock.acquire("enroll.py")
    beat.start()
    cap = None
    try:
        time.sleep(HANDOFF_WAIT)
        cap = open_camera_with_retry(args.camera, lambda: False)
        shots = capture_shots(engine, cap, lambda: False)
    except EnrollmentError as exc:
        raise SystemExit(f"{exc} No database record was created.")
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        beat.stop()
        enroll_lock.release()

    print("Checking for duplicate faces...")
    try:
        verify_shots(engine, shots)
    except EnrollmentError as exc:
        raise SystemExit(str(exc))

    uid = uuid.uuid4().hex[:8]
    saved_paths = []
    try:
        for i, shot in enumerate(shots):
            path = os.path.join(KNOWN_FACES_DIR, f"{args.id}_{uid}_{i}.jpg")
            if not cv2.imwrite(path, shot):
                raise RuntimeError(f"Could not write {path}")
            saved_paths.append(path)

        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                """
                INSERT INTO faculty
                (faculty_id, name, department, designation, image_path, email, phone, active, created_at)
                VALUES (?,?,?,?,?,?,?,?,datetime('now'))
                """,
                (args.id, args.name, args.dept, args.designation,
                 saved_paths[0], args.email, args.phone, 1),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        for path in saved_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        raise

    print(
        f"Enrolled {args.name} ({args.id}) with 3 photos.\n"
        "Run train_lite.py before expecting the new face to be recognized."
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--dept", default="")
    ap.add_argument("--designation", default="")
    ap.add_argument("--email", default="")
    ap.add_argument("--phone", default="")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--job-dir", default=None,
                    help="used by the dashboard: stage photos here instead of saving directly")
    args = ap.parse_args()

    if args.job_dir:
        return run_staged(args)
    if not args.name:
        ap.error("--name is required unless --job-dir is given")
    return run_legacy(args)


if __name__ == "__main__":
    sys.exit(main())
