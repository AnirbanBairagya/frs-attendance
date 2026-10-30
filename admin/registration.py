"""
Faculty registration as one guided, undoable flow.

    details -> camera capture -> train -> admin clicks "Register Faculty"

Each registration is a "job" kept in .registration/<job id>/ so it survives a
dashboard restart. The state machine:

    capturing --> captured --> training --> trained --> done
        |            |            |           |
        v            v            v           v
     failed      cancelled   (back to      cancelled (undo)
                              captured
                              with an error)

Nothing touches the real system until "Train" is pressed. Then ONE worker
process (this file run as a script) does, in this order:

    1. back up the current classifier
    2. copy the 3 photos into known_faces/
    3. insert the faculty row
    4. train a new classifier into a temporary file
    5. check the new classifier contains the new person
    6. swap the new classifier into place (atomic)

and every step is written to an undo log first. If ANY step fails, or the
worker crashes or times out, rollback() puts the photos, database and
classifier back exactly as they were. Only one registration can be active at a
time.
"""

import os as _os
import sys as _sys

# Shared code (lite_core.py, kiosk_proto.py, ...) lives in ../common
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))

import json
import os
import pickle
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback

import enroll_lock
from lite_core import (
    BASE_DIR,
    CLASSIFIER_PATH,
    DB_PATH,
    KNOWN_FACES_DIR,
    SFACE_PATH,
    YUNET_PATH,
    ensure_db,
    list_reference_photos,
)

JOBS_DIR = os.path.join(BASE_DIR, ".registration")
ENROLL_SCRIPT = os.path.join(BASE_DIR, "enroll.py")
WORKER_SCRIPT = os.path.abspath(__file__)

ENROLL_TIMEOUT = int(os.environ.get("REGISTRATION_ENROLL_TIMEOUT", "600"))     # seconds
TRAIN_TIMEOUT = int(os.environ.get("REGISTRATION_TRAIN_TIMEOUT", "1800"))      # seconds
IDLE_SECONDS = int(os.environ.get("REGISTRATION_IDLE_MINUTES", "30")) * 60
MIN_FREE_MB = int(os.environ.get("REGISTRATION_MIN_FREE_MB", "100"))
HEARTBEAT_STALE = 60  # seconds without a heartbeat => the process is gone
KEEP_FINISHED_DAYS = 7
EXPECTED_PHOTOS = 3

ACTIVE_STATES = {"capturing", "captured", "training", "trained"}
TERMINAL_STATES = {"done", "failed", "cancelled"}

JOB_ID_RE = re.compile(r"^[0-9a-f]{16}$")
FACULTY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^[0-9+()\-\s]{5,20}$")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

SUCCESS_MESSAGE = "Face has been captured, trained and saved."


class RegistrationError(Exception):
    """Message is safe to show to the admin."""


class JobBusy(RegistrationError):
    def __init__(self, job):
        super().__init__("Another registration is already in progress.")
        self.job = job


class InvalidState(RegistrationError):
    pass


class Abandoned(Exception):
    """The dashboard gave up on this worker (timeout / lost contact); stop quietly."""


_lock = threading.RLock()
_procs = {}  # job id -> Popen (only for processes started by this dashboard run)


# ---------------------------------------------------------------------------
# Small file helpers
# ---------------------------------------------------------------------------
def _replace(src, dst, attempts=8):
    """os.replace, retried: on Windows it fails while another process has the file open."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.25)


def _write_json(path, data):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)


def _read_json(path, attempts=5):
    for _ in range(attempts):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (ValueError, OSError):
            time.sleep(0.1)
    return None


def job_path(job_id, *parts):
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise ValueError("invalid job id")
    return os.path.join(JOBS_DIR, job_id, *parts)


def load_job(job_id):
    try:
        return _read_json(job_path(job_id, "state.json"))
    except ValueError:
        return None


def save_job(job):
    job["updated_at"] = time.time()
    _write_json(job_path(job["id"], "state.json"), job)


def set_state(job, state, **fields):
    job["state"] = state
    job["state_since"] = time.time()
    job.update(fields)
    save_job(job)
    return job


def _remove(path):
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _tail(path, chars=400):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()[-chars:].strip()
    except OSError:
        return ""


def _heartbeat_age(job):
    try:
        hb = os.path.getmtime(job_path(job["id"], "heartbeat"))
    except OSError:
        hb = 0.0
    return time.time() - max(hb, job.get("state_since", 0.0))


def elapsed(job):
    return int(max(0, time.time() - job.get("state_since", time.time())))


# ---------------------------------------------------------------------------
# Validation and pre-flight checks
# ---------------------------------------------------------------------------
def validate_faculty_fields(form):
    """Return (clean_fields, error_message_or_None)."""
    keys = ("faculty_id", "name", "department", "designation", "email", "phone")
    clean = {k: (form.get(k, "") or "").strip() for k in keys}

    if not clean["faculty_id"] or not clean["name"]:
        return clean, "Faculty ID and Name are required."
    if not FACULTY_ID_RE.match(clean["faculty_id"]):
        return clean, (
            "Faculty ID may contain only letters, digits, dot, dash and underscore "
            "(up to 40 characters) and must start with a letter or digit."
        )
    for key, label in (("name", "Name"), ("department", "Department"), ("designation", "Designation")):
        if len(clean[key]) > 100:
            return clean, f"{label} is too long (100 characters at most)."
        if CONTROL_RE.search(clean[key]):
            return clean, f"{label} contains invalid characters."
    if clean["email"] and (len(clean["email"]) > 120 or not EMAIL_RE.match(clean["email"])):
        return clean, "Please enter a valid email address."
    if clean["phone"] and not PHONE_RE.match(clean["phone"]):
        return clean, "Phone may contain only digits, spaces, + - and parentheses (5 to 20 characters)."
    return clean, None


def _id_conflict(faculty_id):
    """Why this ID can't be used, or None. Case-insensitive; also looks for leftover photos."""
    low = faculty_id.lower()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        row = conn.execute(
            "SELECT faculty_id FROM faculty WHERE lower(faculty_id)=?", (low,)
        ).fetchone()
    finally:
        conn.close()
    if row:
        return f"Faculty ID {row[0]} already exists."
    for fid in list_reference_photos():
        if fid.lower() == low:
            return (
                f"Face photos for ID {fid} already exist in known_faces/ without a faculty "
                "record. Use a different ID, or remove those photos first."
            )
    return None


def _check_disk():
    free = shutil.disk_usage(BASE_DIR).free
    if free < MIN_FREE_MB * 1024 * 1024:
        raise RegistrationError(
            f"Not enough free disk space ({free // (1024 * 1024)} MB left; at least "
            f"{MIN_FREE_MB} MB needed). Free some space and try again."
        )


def _check_models():
    for path in (YUNET_PATH, SFACE_PATH):
        if not os.path.isfile(path):
            raise RegistrationError(
                f"A face model file is missing ({os.path.basename(path)}). "
                "Copy the models folder into the project and try again."
            )
    if not os.path.isfile(ENROLL_SCRIPT):
        raise RegistrationError("enroll.py is missing from the project folder.")


def _launch(args, log_path):
    log = open(log_path, "ab")
    try:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        return subprocess.Popen(
            args,
            cwd=BASE_DIR,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Starting, polling and refreshing jobs
# ---------------------------------------------------------------------------
def start_job(fields):
    """Create a job and open the camera window. Raises JobBusy / RegistrationError."""
    with _lock:
        active = get_active_job()
        if active:
            raise JobBusy(active)

        conflict = _id_conflict(fields["faculty_id"])
        if conflict:
            raise RegistrationError(conflict)
        _check_disk()
        _check_models()

        job_id = secrets.token_hex(8)
        os.makedirs(job_path(job_id), exist_ok=True)
        now = time.time()
        job = {
            "id": job_id,
            "state": "capturing",
            "created": now,
            "state_since": now,
            "updated_at": now,
            "faculty": dict(fields),
            "photos": [],
            "error": None,
            "message": "",
            "progress": "",
            "details": "",
            "summary": {},
            "undo": {},
        }
        save_job(job)

        cmd = [sys.executable, ENROLL_SCRIPT, "--id", fields["faculty_id"],
               "--job-dir", job_path(job_id)]
        try:
            _procs[job_id] = _launch(cmd, job_path(job_id, "enroll.log"))
        except OSError as exc:
            shutil.rmtree(job_path(job_id), ignore_errors=True)  # back to the initial state
            raise RegistrationError(f"Could not open the camera window: {exc}")
        return job


def get_job(job_id):
    job = load_job(job_id)
    if job is None:
        return None
    return refresh_job(job) if job.get("state") in ACTIVE_STATES else job


def get_active_job():
    """The unfinished registration, if any (stale ones are cleaned up on the way)."""
    with _lock:
        if not os.path.isdir(JOBS_DIR):
            return None
        jobs = []
        for name in os.listdir(JOBS_DIR):
            if JOB_ID_RE.match(name):
                job = load_job(name)
                if job:
                    jobs.append(job)
        jobs.sort(key=lambda j: j.get("created", 0), reverse=True)
        for job in jobs:
            if job.get("state") in ACTIVE_STATES:
                job = refresh_job(job)
                if job["state"] in ACTIVE_STATES:
                    return job
        return None


def _discard_staging(job):
    shutil.rmtree(job_path(job["id"], "photos"), ignore_errors=True)


def _stop_enroll(job):
    """Ask the camera window to close (cancel flag), then make sure it is gone."""
    jid = job["id"]
    try:
        with open(job_path(jid, "cancel"), "w", encoding="utf-8") as f:
            f.write("1")
    except OSError:
        pass
    proc = _procs.pop(jid, None)
    if proc is not None:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
    enroll_lock.release()


def _stop_worker(job):
    proc = _procs.pop(job["id"], None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def refresh_job(job):
    """Look at the real world (processes, files, clocks) and move the job forward."""
    with _lock:
        state = job.get("state")
        jid = job["id"]
        now = time.time()
        age = now - job.get("state_since", now)

        if state == "capturing":
            proc = _procs.get(jid)
            exited = proc is not None and proc.poll() is not None  # check BEFORE reading result
            if exited:
                _procs.pop(jid, None)
            res = _read_json(job_path(jid, "result.json"))

            if res is not None:
                enroll_lock.release()
                if res.get("ok"):
                    photos = [p for p in res.get("photos", [])
                              if os.path.isfile(job_path(jid, "photos", p))]
                    if len(photos) == EXPECTED_PHOTOS:
                        return set_state(job, "captured", photos=photos, error=None, details="")
                    _discard_staging(job)
                    return set_state(job, "failed",
                                     error="The captured photos could not be found on disk. Please try again.")
                _discard_staging(job)
                if res.get("cancelled"):
                    return set_state(job, "cancelled", error=None,
                                     message="Enrollment was cancelled at the camera window. Nothing was saved.")
                return set_state(job, "failed", error=res.get("error") or "Enrollment failed.",
                                 details=res.get("details", ""))

            if age > ENROLL_TIMEOUT:
                _stop_enroll(job)
                _discard_staging(job)
                return set_state(job, "failed",
                                 error=f"No photos were taken within {ENROLL_TIMEOUT // 60} minutes, "
                                       "so the enrollment window was closed. Nothing was saved.")
            if exited:
                enroll_lock.release()
                _discard_staging(job)
                return set_state(job, "failed",
                                 error="The camera window closed before the photos were saved. Nothing was saved.",
                                 details=_tail(job_path(jid, "enroll.log")))
            if proc is None and _heartbeat_age(job) > HEARTBEAT_STALE:
                enroll_lock.release()
                _discard_staging(job)
                return set_state(job, "failed",
                                 error="Lost contact with the camera window. Nothing was saved.")
            return job

        if state == "training":
            proc = _procs.get(jid)
            exited = proc is not None and proc.poll() is not None
            if exited:
                _procs.pop(jid, None)
            fresh = load_job(jid)
            if fresh is not None and fresh.get("state") != "training":
                return fresh  # the worker finished (or failed back) on its own
            if fresh is not None:
                job = fresh

            if age > TRAIN_TIMEOUT:
                _stop_worker(job)
                return _recover_training(
                    job, f"Training took longer than {TRAIN_TIMEOUT // 60} minutes and was stopped.")
            if exited:
                return _recover_training(job, "The training process stopped unexpectedly.")
            if proc is None and _heartbeat_age(job) > HEARTBEAT_STALE:
                return _recover_training(job, "Lost contact with the training process.")
            return job

        if state == "captured" and age > IDLE_SECONDS:
            _discard_staging(job)
            return set_state(job, "cancelled", error=None,
                             message="This registration expired because it was left idle. Nothing was saved.")

        if state == "trained" and age > IDLE_SECONDS:
            return _finalize(job, "Registration was completed automatically after being left idle.")

        return job


# ---------------------------------------------------------------------------
# Admin actions
# ---------------------------------------------------------------------------
def request_train(job_id):
    """Start the commit-and-train worker. Safe against double clicks."""
    with _lock:
        job = get_job(job_id)
        if job is None:
            raise InvalidState("This registration no longer exists.")
        if job["state"] != "captured":
            raise InvalidState("Training can only start right after the photos are captured.")

        photos_ok = all(os.path.isfile(job_path(job_id, "photos", p)) for p in job["photos"])
        if len(job["photos"]) != EXPECTED_PHOTOS or not photos_ok:
            _discard_staging(job)
            set_state(job, "failed", error="The captured photos are missing. Please capture them again.")
            raise InvalidState("The captured photos are missing.")

        _remove(job_path(job_id, "heartbeat"))
        # "run" identifies this worker. If the dashboard gives up on it, the token
        # is cleared and the worker stops writing (see _still_owner).
        set_state(job, "training", error=None, progress="Starting...", undo={},
                  run=secrets.token_hex(4))
        try:
            _procs[job_id] = _launch([sys.executable, WORKER_SCRIPT, "train", job_id],
                                     job_path(job_id, "worker.log"))
        except OSError as exc:
            set_state(job, "captured", progress="",
                      error=f"Could not start training: {exc}. Nothing was saved; you can try again.")
        return job


def cancel_job(job_id):
    with _lock:
        job = get_job(job_id)
        if job is None:
            raise InvalidState("This registration no longer exists.")
        state = job["state"]
        if state in TERMINAL_STATES:
            return job
        if state == "training":
            raise InvalidState(
                "Training is in progress and can't be cancelled safely. Please wait; "
                "everything is undone automatically if it fails.")
        if state == "capturing":
            _stop_enroll(job)
            _discard_staging(job)
            return set_state(job, "cancelled", error=None, message="Registration cancelled. Nothing was saved.")
        if state == "captured":
            _discard_staging(job)
            return set_state(job, "cancelled", error=None, message="Registration cancelled. Nothing was saved.")
        # trained: everything is live, so undo it
        problems = rollback(job)
        _discard_staging(job)
        if problems:
            return set_state(job, "cancelled", error=None, message=(
                "Registration was undone, but some cleanup could not finish: "
                + "; ".join(problems) + ". Please remove those items manually."))
        return set_state(job, "cancelled", error=None,
                         message="Registration undone. The faculty record and face data were removed.")


def submit_job(job_id):
    with _lock:
        job = get_job(job_id)
        if job is None:
            raise InvalidState("This registration no longer exists.")
        if job["state"] == "done":
            return job
        if job["state"] != "trained":
            raise InvalidState("The face data has not been trained yet.")
        return _finalize(job, "Registration complete.")


def _finalize(job, message):
    _discard_staging(job)
    _remove(job_path(job["id"], "classifier_backup.pkl"))
    return set_state(job, "done", error=None, progress="", message=message)


def staged_photo_path(job_id, n):
    job = load_job(job_id)
    if not job or job.get("state") not in ACTIVE_STATES:
        return None
    photos = job.get("photos") or []
    if not (0 <= n < len(photos)):
        return None
    path = job_path(job_id, "photos", photos[n])
    return path if os.path.isfile(path) else None


def public_status(job):
    return {
        "state": job["state"],
        "progress": job.get("progress", ""),
        "error": job.get("error"),
        "elapsed": elapsed(job),
    }


def recover_on_startup():
    """Run once when the dashboard starts: settle half-finished jobs, tidy old ones."""
    with _lock:
        active = get_active_job()
        if not (active and active["state"] == "capturing"):
            enroll_lock.release()
        if not os.path.isdir(JOBS_DIR):
            return
        cutoff = time.time() - KEEP_FINISHED_DAYS * 86400
        for name in os.listdir(JOBS_DIR):
            if not JOB_ID_RE.match(name):
                continue
            job = load_job(name)
            if job is None:
                path = job_path(name)
                if os.path.getmtime(path) < time.time() - 86400:
                    shutil.rmtree(path, ignore_errors=True)
            elif job.get("state") in TERMINAL_STATES and job.get("updated_at", 0) < cutoff:
                shutil.rmtree(job_path(name), ignore_errors=True)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------
def rollback(job):
    """
    Put the database, photos and classifier back as they were before training.
    Safe to call more than once. Returns a list of things that could not be undone.
    """
    problems = []
    undo = job.get("undo") or {}
    jid = job["id"]

    # classifier
    if undo.get("replace_started"):
        backup = undo.get("backup")
        try:
            if backup and os.path.isfile(backup):
                restore_tmp = f"{CLASSIFIER_PATH}.{jid}.restore"
                shutil.copyfile(backup, restore_tmp)
                _replace(restore_tmp, CLASSIFIER_PATH)
            elif not undo.get("classifier_existed"):
                if not _remove(CLASSIFIER_PATH):
                    problems.append("could not remove the new classifier file")
            else:
                problems.append("the previous classifier backup is missing")
        except OSError as exc:
            problems.append(f"could not restore the previous classifier ({exc})")

    # database
    if undo.get("insert_started"):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=15)
            try:
                row = conn.execute(
                    "SELECT id FROM faculty WHERE faculty_id=? AND image_path=?",
                    (job["faculty"]["faculty_id"], undo.get("image_path")),
                ).fetchone()
                if row:
                    conn.execute("DELETE FROM attendance WHERE faculty_id_fk=?", (row[0],))
                    conn.execute("DELETE FROM faculty WHERE id=?", (row[0],))
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            problems.append(f"could not remove the faculty record ({exc})")

    # photos and temporary files
    for path in undo.get("copied", []):
        if not _remove(path):
            problems.append(f"could not delete {os.path.basename(path)}")
    candidate = undo.get("candidate")
    if candidate:
        _remove(candidate)
        _remove(candidate + ".part")
    _remove(f"{CLASSIFIER_PATH}.{jid}.restore")
    return problems


def _recover_training(job, reason):
    """Training did not finish: undo everything and let the admin try again."""
    job["run"] = None  # a worker that is somehow still alive must stop writing
    save_job(job)
    problems = rollback(job)
    return _back_to_captured(job, reason, problems)


def _back_to_captured(job, reason, problems):
    photos_ok = (len(job.get("photos", [])) == EXPECTED_PHOTOS and
                 all(os.path.isfile(job_path(job["id"], "photos", p)) for p in job["photos"]))
    if problems:
        msg = (f"{reason} Automatic cleanup could not finish ({'; '.join(problems)}). "
               "Please remove those items manually.")
    else:
        msg = f"{reason} Nothing was saved; the system is unchanged."
    if photos_ok:
        return set_state(job, "captured", error=msg, progress="", undo={},
                         details=_tail(job_path(job["id"], "worker.log")))
    return set_state(job, "failed", error=msg, progress="", undo={},
                     details=_tail(job_path(job["id"], "worker.log")))


# ---------------------------------------------------------------------------
# The worker: commit + train (runs as its own process)
# ---------------------------------------------------------------------------
def _friendly(exc):
    if isinstance(exc, SystemExit):
        return str(exc.code) if exc.code not in (None, 0) else "Training stopped."
    if isinstance(exc, RegistrationError) or type(exc).__name__ == "TrainingError":
        return str(exc)
    if isinstance(exc, MemoryError):
        return "The device ran out of memory while training."
    if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
        return "The database is busy right now."
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        return "The disk is full."
    if isinstance(exc, KeyboardInterrupt):
        return "Training was interrupted."
    return f"Unexpected error: {type(exc).__name__}: {exc}"


def _still_owner(job):
    """Raise Abandoned unless the saved job is still this worker's training run."""
    current = load_job(job["id"])
    if (current is None or current.get("state") != "training"
            or not job.get("run") or current.get("run") != job.get("run")):
        raise Abandoned()


def _save_owned(job):
    _still_owner(job)
    save_job(job)


def _commit_and_train(job):
    jid = job["id"]
    fields = job["faculty"]
    fid = fields["faculty_id"]
    save_job = _save_owned  # noqa: F841 - every write below first checks we still own the job

    undo = {
        "copied": [],
        "insert_started": False,
        "image_path": None,
        "classifier_existed": os.path.isfile(CLASSIFIER_PATH),
        "backup": None,
        "candidate": None,
        "replace_started": False,
    }
    job["undo"] = undo
    save_job(job)

    ensure_db()
    _check_disk()
    conflict = _id_conflict(fid)
    if conflict:
        raise RegistrationError(conflict)

    sources = [job_path(jid, "photos", p) for p in job["photos"]]
    if len(sources) != EXPECTED_PHOTOS or not all(os.path.isfile(s) for s in sources):
        raise RegistrationError("The captured photos are missing.")

    # 1. back up the current classifier
    if undo["classifier_existed"]:
        backup = job_path(jid, "classifier_backup.pkl")
        shutil.copyfile(CLASSIFIER_PATH, backup)
        undo["backup"] = backup
        save_job(job)

    # 2. copy photos (the undo log lists them BEFORE they exist)
    uid = secrets.token_hex(4)
    dests = [os.path.join(KNOWN_FACES_DIR, f"{fid}_{uid}_{i}.jpg") for i in range(len(sources))]
    undo["copied"] = dests
    save_job(job)
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    for src, dst in zip(sources, dests):
        shutil.copyfile(src, dst)

    # 3. faculty row
    undo["insert_started"] = True
    undo["image_path"] = dests[0]
    save_job(job)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    try:
        conn.execute(
            """
            INSERT INTO faculty
            (faculty_id, name, department, designation, image_path, email, phone, active, created_at)
            VALUES (?,?,?,?,?,?,?,1,datetime('now'))
            """,
            (fid, fields["name"], fields["department"], fields["designation"],
             dests[0], fields["email"], fields["phone"]),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise RegistrationError(f"Faculty ID {fid} already exists.")
    finally:
        conn.close()

    # 4. train into a temporary file
    from train_lite import train  # heavy import: only in the worker

    candidate = f"{CLASSIFIER_PATH}.{jid}.new"
    undo["candidate"] = candidate
    save_job(job)

    last_write = [0.0]

    def progress(done, total, label):
        now = time.time()
        if now - last_write[0] < 1.0 and done < total:
            return
        last_write[0] = now
        job["progress"] = f"Learning faces: photo {done} of {total}"
        save_job(job)

    summary = train(output_path=candidate, progress=progress,
                    log=lambda msg: print(msg, flush=True))

    # 5. the new classifier must really know the new person
    job["progress"] = "Checking the trained data..."
    save_job(job)
    with open(candidate, "rb") as fh:
        bundle = pickle.load(fh)
    if fid not in (bundle.get("refs") or {}):
        raise RegistrationError(
            "No usable face was found in the captured photos, so the new person can't be recognised.")

    # 6. swap it in
    undo["replace_started"] = True
    save_job(job)
    _replace(candidate, CLASSIFIER_PATH)
    undo["candidate"] = None

    _still_owner(job)
    set_state(job, "trained", error=None, progress="", message=SUCCESS_MESSAGE,
              summary={"people": summary.get("people"),
                       "holdout_accuracy": summary.get("holdout_accuracy")})


def run_training(job_id):
    job = load_job(job_id)
    if not job or job.get("state") != "training":
        print("Nothing to do: job is not in the training state.")
        return 1

    beat = enroll_lock.Heartbeat(job_path(job_id, "heartbeat"))
    beat.start()
    try:
        _commit_and_train(job)
        return 0
    except Abandoned:
        _drop_worker_temp_files(job)  # the dashboard already rolled everything back
        print("Training stopped: the dashboard gave up on this run.")
        return 1
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        traceback.print_exc()
        try:
            _still_owner(job)
        except Abandoned:
            _drop_worker_temp_files(job)
            return 1
        problems = rollback(job)
        _back_to_captured(job, _friendly(exc), problems)
        return 1
    finally:
        beat.stop()


def _drop_worker_temp_files(job):
    """Only the files this worker itself creates; safe even after the dashboard's rollback."""
    candidate = (job.get("undo") or {}).get("candidate")
    if candidate:
        _remove(candidate)
        _remove(candidate + ".part")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "train":
        sys.exit(run_training(sys.argv[2]))
    print("This file is used by the dashboard. Usage: python registration.py train <job id>")
    sys.exit(2)
