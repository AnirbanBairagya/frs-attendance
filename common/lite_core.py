"""
Lightweight, cross-platform core for the BCREC face-attendance system.

Designed to run on both:
  - Windows development PC
  - Raspberry Pi 3B+ / Raspberry Pi OS (64-bit)

Models:
  - YuNet  : face detector (ONNX, OpenCV DNN)
  - SFace  : 128-D face recognizer (ONNX, OpenCV DNN)

The same code chooses the appropriate camera backend automatically.
"""

import glob
import os
import pickle
import sqlite3
import sys
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths.
#   CODE_DIR  - where this shared file lives (e.g. common/)
#   BASE_DIR  - where THIS machine/app keeps its own data (database, classifier,
#               known_faces, secrets). It is the folder of the script that was
#               started (admin/app.py -> admin/, kiosk/kiosk.py -> kiosk/), so the
#               admin app and the kiosk never share data even if they share code.
#               Override with the FRS_DATA_DIR environment variable.
#   MODELS_DIR - the .onnx files: FRS_MODELS_DIR, else <BASE_DIR>/models if it
#               exists, else <CODE_DIR>/models (the shared common/models).
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def _data_dir():
    env = os.environ.get("FRS_DATA_DIR")
    if env:
        return os.path.abspath(env)
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    if main and os.path.isfile(main):
        return os.path.dirname(os.path.abspath(main))
    return CODE_DIR


BASE_DIR = _data_dir()
_own_models = os.path.join(BASE_DIR, "models")
MODELS_DIR = os.path.abspath(
    os.environ.get("FRS_MODELS_DIR")
    or (_own_models if os.path.isdir(_own_models) else os.path.join(CODE_DIR, "models"))
)
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "face_classifier_lite.pkl")

YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# ---------------------------------------------------------------------------
# Settings. They can be overridden with environment variables.
# ---------------------------------------------------------------------------
COLLEGE_START_TIME = os.environ.get("COLLEGE_START_TIME", "10:00:00")
COSINE_THRESHOLD = float(os.environ.get("COSINE_THRESHOLD", "0.363"))
SVM_CONFIDENCE = float(os.environ.get("SVM_CONFIDENCE", "0.60"))
COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", "60"))


def _die(message):
    raise SystemExit(message)


# ---------------------------------------------------------------------------
# OpenCV face engine
# ---------------------------------------------------------------------------
class FaceEngine:
    def __init__(self, score_threshold=0.85):
        for p in (YUNET_PATH, SFACE_PATH):
            if not os.path.isfile(p):
                _die(
                    f"Missing model file:\n{p}\n"
                    "Copy the models folder before starting the application."
                )

        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            _die(
                f"OpenCV {cv2.__version__} does not provide FaceDetectorYN / FaceRecognizerSF.\n"
                "Install OpenCV >= 4.8 with the required DNN face APIs."
            )

        self.detector = cv2.FaceDetectorYN.create(
            YUNET_PATH,
            "",
            (320, 320),
            score_threshold,
            0.3,
            5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

    def detect(self, img):
        """Detect faces at the image's native resolution."""
        if img is None or img.size == 0:
            return np.empty((0, 15), np.float32)

        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None:
            return np.empty((0, 15), np.float32)
        return faces

    def detect_fast(self, img, target_w=320):
        """Detect on a smaller frame and map boxes/landmarks to full-frame coordinates."""
        h, w = img.shape[:2]
        if w <= target_w:
            return self.detect(img)

        scale = w / float(target_w)
        target_h = max(1, int(h / scale))
        small = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        faces = self.detect(small).copy()
        if len(faces):
            faces[:, :14] *= scale
        return faces

    def embed(self, img, face):
        """Return an L2-normalised 128-D SFace embedding."""
        aligned = self.recognizer.alignCrop(img, face)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat


def largest_face(faces):
    if faces is None or len(faces) == 0:
        return None
    return faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]


def spoof_check(frame, face):
    """
    Liveness hook.

    This project version does not implement active liveness detection yet.
    Keep the hook so a real liveness module can be added later without changing
    the kiosk interface.
    """
    return True


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def _open_capture(index, backend):
    if backend is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def open_camera(source=None):
    """
    Open a camera in a Windows/Pi friendly way.

    Numeric sources use:
      - Windows: DirectShow first, then Media Foundation, then default
      - Linux: V4L2 first, then default

    URL sources use OpenCV's default backend.
    """
    source = os.environ.get("CAMERA_SOURCE", "0") if source is None else str(source)

    if source.isdigit():
        index = int(source)

        if os.name == "nt":
            backends = [
                getattr(cv2, "CAP_DSHOW", None),
                getattr(cv2, "CAP_MSMF", None),
                None,
            ]
        else:
            backends = [
                getattr(cv2, "CAP_V4L2", None),
                None,
            ]

        cap = None
        for backend in backends:
            if backend is None:
                candidate = _open_capture(index, None)
            else:
                candidate = _open_capture(index, backend)

            if candidate.isOpened():
                cap = candidate
                break
            candidate.release()

        if cap is None:
            _die(
                f"Cannot open camera index {index}.\n"
                "Check the camera connection and CAMERA_SOURCE."
            )
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            _die(
                "Cannot open camera URL / source:\n"
                f"{source}\n"
                "Check CAMERA_SOURCE."
            )

    # 640x480 is deliberate for Raspberry Pi 3B+: it reduces CPU load while
    # retaining enough pixels for a normal attendance distance.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap


# ---------------------------------------------------------------------------
# Face reference photos
# ---------------------------------------------------------------------------
def parse_faculty_id(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return base.rsplit("_", 2)[0]


def list_reference_photos():
    """Return {faculty_id: [photo paths]} sorted by filename."""
    out = {}
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(KNOWN_FACES_DIR, pattern)))

    for p in sorted(paths):
        if os.path.basename(p).startswith("_"):
            continue
        out.setdefault(parse_faculty_id(p), []).append(p)
    return out


def build_reference_embeddings(engine):
    refs = {}
    for fid, paths in list_reference_photos().items():
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                continue
            try:
                face = largest_face(engine.detect(img))
                if face is not None:
                    refs.setdefault(fid, []).append(engine.embed(img, face))
            except cv2.error:
                continue

    return {k: np.array(v, np.float32) for k, v in refs.items() if v}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def load_classifier():
    if not os.path.isfile(CLASSIFIER_PATH):
        return None
    with open(CLASSIFIER_PATH, "rb") as f:
        return pickle.load(f)


def identify(bundle, emb):
    """Return accepted/faculty_id/confidence/cosine from a trained bundle."""
    if not bundle or not bundle.get("refs"):
        return {
            "accepted": False,
            "faculty_id": "",
            "confidence": 0.0,
            "cosine": -1.0,
        }

    refs = bundle["refs"]
    clf = bundle.get("classifier")

    if clf is not None:
        probs = clf.predict_proba(emb.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        fid = str(clf.classes_[idx])
        conf = float(probs[idx])
        cos = float(np.max(refs[fid] @ emb)) if fid in refs else -1.0
        accepted = conf >= SVM_CONFIDENCE and cos >= COSINE_THRESHOLD
    else:
        candidates = []
        for k, v in refs.items():
            if len(v):
                candidates.append((k, float(np.max(v @ emb))))
        if not candidates:
            return {
                "accepted": False,
                "faculty_id": "",
                "confidence": 0.0,
                "cosine": -1.0,
            }
        fid, cos = max(candidates, key=lambda t: t[1])
        conf = cos
        accepted = cos >= COSINE_THRESHOLD

    return {
        "accepted": accepted,
        "faculty_id": fid,
        "confidence": conf,
        "cosine": cos,
    }


# ---------------------------------------------------------------------------
# SQLite schema + safe migration
# ---------------------------------------------------------------------------
def _table_columns(conn, table_name):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _add_column_if_missing(conn, table_name, column_name, definition):
    cols = _table_columns(conn, table_name)
    if column_name not in cols:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def _repair_attendance_rows(conn):
    """
    Back-fill status/late/working_minutes in older databases where those
    columns existed but were NULL.
    """
    rows = conn.execute(
        """
        SELECT id, date, check_in_time, check_out_time, status, late, working_minutes
        FROM attendance
        """
    ).fetchall()

    for row in rows:
        if not row[2]:
            continue

        late = 1 if row[2] > COLLEGE_START_TIME else 0
        status = "Late" if late else "Present"

        working_minutes = row[6]
        if row[3]:
            try:
                start = datetime.strptime(
                    f"{row[1]} {row[2]}", "%Y-%m-%d %H:%M:%S"
                )
                end = datetime.strptime(
                    f"{row[1]} {row[3]}", "%Y-%m-%d %H:%M:%S"
                )
                minutes = max(0, int((end - start).total_seconds() // 60))
                working_minutes = minutes
            except (ValueError, TypeError):
                pass

        if row[4] != status or row[5] != late or row[6] != working_minutes:
            conn.execute(
                """
                UPDATE attendance
                SET status=?, late=?, working_minutes=?
                WHERE id=?
                """,
                (status, late, working_minutes, row[0]),
            )


def ensure_db():
    """Create the complete schema, then safely add any missing columns."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS faculty (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faculty_id TEXT UNIQUE,
            name TEXT,
            department TEXT,
            designation TEXT,
            image_path TEXT,
            email TEXT,
            phone TEXT,
            active INTEGER DEFAULT 1,
            created_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faculty_id_fk INTEGER,
            date TEXT,
            check_in_time TEXT,
            check_out_time TEXT,
            status TEXT,
            late INTEGER DEFAULT 0,
            working_minutes INTEGER,
            FOREIGN KEY(faculty_id_fk) REFERENCES faculty(id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
        """
    )

    # Safe upgrades for older versions of the project.
    faculty_columns = {
        "faculty_id": "TEXT",
        "name": "TEXT",
        "department": "TEXT",
        "designation": "TEXT",
        "image_path": "TEXT",
        "email": "TEXT",
        "phone": "TEXT",
        "active": "INTEGER DEFAULT 1",
        "created_at": "TEXT",
    }
    for col, definition in faculty_columns.items():
        _add_column_if_missing(conn, "faculty", col, definition)

    attendance_columns = {
        "faculty_id_fk": "INTEGER",
        "date": "TEXT",
        "check_in_time": "TEXT",
        "check_out_time": "TEXT",
        "status": "TEXT",
        "late": "INTEGER DEFAULT 0",
        "working_minutes": "INTEGER",
    }
    for col, definition in attendance_columns.items():
        _add_column_if_missing(conn, "attendance", col, definition)

    admin_columns = {
        "username": "TEXT",
        "password_hash": "TEXT",
        "created_at": "TEXT",
    }
    for col, definition in admin_columns.items():
        _add_column_if_missing(conn, "admin_users", col, definition)

    # Populate missing timestamps/active values where possible.
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE faculty SET active=1 WHERE active IS NULL")
    conn.execute("UPDATE faculty SET created_at=? WHERE created_at IS NULL", (now,))

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_attendance_faculty_date ON attendance(faculty_id_fk, date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)"
    )

    _repair_attendance_rows(conn)
    conn.commit()
    conn.close()


def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def get_faculty(faculty_id):
    conn = _conn()
    row = conn.execute(
        "SELECT * FROM faculty WHERE faculty_id=?",
        (faculty_id,),
    ).fetchone()
    conn.close()
    return row


def faculty_id_exists(faculty_id):
    return get_faculty(faculty_id) is not None


def _is_late(time_str):
    return time_str > COLLEGE_START_TIME


def mark_attendance(faculty_pk, now=None):
    """
    Attendance state machine:
      - first scan of the day -> CHECK-IN
      - next scan while an IN row is open -> CHECK-OUT
      - later scan -> new CHECK-IN re-entry
      - repeated scan inside cooldown -> Already recorded

    The first daily check-in is the one that determines Late/Present status.

    `now` lets the admin PC replay a scan that a kiosk recorded earlier (for
    example while the network was down) at the time it really happened.
    """
    now = now or datetime.now()
    today = now.strftime("%Y-%m-%d")
    t = now.strftime("%H:%M:%S")

    conn = _conn()
    try:
        rows = conn.execute(
            """
            SELECT id, check_in_time, check_out_time, late, status
            FROM attendance
            WHERE faculty_id_fk=? AND date=?
            ORDER BY id DESC
            """,
            (faculty_pk, today),
        ).fetchall()

        if rows:
            last_t = rows[0]["check_out_time"] or rows[0]["check_in_time"]
            if last_t:
                try:
                    last_dt = datetime.strptime(
                        f"{today} {last_t}", "%Y-%m-%d %H:%M:%S"
                    )
                    if (now - last_dt).total_seconds() < COOLDOWN_SEC:
                        return "cooldown", last_t, bool(rows[0]["late"])
                except (TypeError, ValueError):
                    pass

            open_row = next(
                (row for row in rows if row["check_out_time"] is None),
                None,
            )

            if open_row is not None:
                working_minutes = None
                try:
                    start = datetime.strptime(
                        f"{today} {open_row['check_in_time']}",
                        "%Y-%m-%d %H:%M:%S",
                    )
                    working_minutes = max(
                        0,
                        int((now - start).total_seconds() // 60),
                    )
                except (TypeError, ValueError):
                    pass

                conn.execute(
                    """
                    UPDATE attendance
                    SET check_out_time=?, working_minutes=?
                    WHERE id=?
                    """,
                    (t, working_minutes, open_row["id"]),
                )
                conn.commit()
                return "checkout", t, bool(open_row["late"])

            # Re-entry: no second late mark for the same calendar day.
            conn.execute(
                """
                INSERT INTO attendance
                (faculty_id_fk, date, check_in_time, status, late)
                VALUES (?,?,?,?,?)
                """,
                (faculty_pk, today, t, "Present", 0),
            )
            conn.commit()
            return "checkin", t, False

        late = 1 if _is_late(t) else 0
        status = "Late" if late else "Present"
        conn.execute(
            """
            INSERT INTO attendance
            (faculty_id_fk, date, check_in_time, status, late)
            VALUES (?,?,?,?,?)
            """,
            (faculty_pk, today, t, status, late),
        )
        conn.commit()
        return "checkin_first", t, bool(late)
    finally:
        conn.close()
