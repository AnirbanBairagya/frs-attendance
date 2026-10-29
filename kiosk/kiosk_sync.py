"""
Kiosk (Raspberry Pi) side of the admin-PC link.

The kiosk keeps working on its own copy of the data:
  * faculty list and trained classifier  <- pulled from the admin PC
  * attendance scans                     -> queued in a local outbox and pushed

If the admin PC is off or the network is down, scans simply stay in the outbox
and are delivered later, in order, with their real time.

Configuration: kiosk_config.json next to kiosk.py (created from the dashboard's
Kiosks page). Without that file the kiosk runs stand-alone, as before.
Uses only the standard library for networking.
"""
import json
import os
import pickle
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import cv2

from . import kiosk_proto as proto
from common.lite_core import COMMON_DIR, CLASSIFIER_PATH, DB_PATH, SFACE_PATH, YUNET_PATH

CONFIG_PATH = os.environ.get("KIOSK_CONFIG", os.path.join(COMMON_DIR, "kiosk_config.json"))
STATE_PATH = os.path.join(COMMON_DIR, ".kiosk_state.json")
HTTP_TIMEOUT = float(os.environ.get("KIOSK_HTTP_TIMEOUT", "8"))
MAX_BACKOFF = 60.0
KEEP_ATTENDANCE_DAYS = 8
SKEW_WARN_SECONDS = 120


class ConfigError(Exception):
    pass


class SyncError(Exception):
    """Something went wrong talking to the admin PC; retried automatically."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path=None):
    """Return the config dict, or None when this kiosk is not linked to an admin PC."""
    path = path or CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ConfigError(f"kiosk_config.json is unreadable: {exc}")
    if not isinstance(raw, dict):
        raise ConfigError("kiosk_config.json must contain a JSON object.")
    url = str(raw.get("admin_url", "")).strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigError("kiosk_config.json: admin_url must look like http://192.168.1.20:5000")
    kiosk_id = str(raw.get("kiosk_id", "")).strip()
    if not proto.KIOSK_ID_RE.match(kiosk_id):
        raise ConfigError("kiosk_config.json: kiosk_id is missing or malformed.")
    secret = str(raw.get("secret", "")).strip()
    if len(secret) < 16:
        raise ConfigError("kiosk_config.json: secret is missing or too short.")
    try:
        poll = float(raw.get("poll_seconds", 5))
    except (TypeError, ValueError):
        raise ConfigError("kiosk_config.json: poll_seconds must be a number.")
    return {"admin_url": url, "kiosk_id": kiosk_id, "secret": secret,
            "poll_seconds": min(max(poll, 1.0), 300.0), "ca_file": raw.get("ca_file") or None}


# ---------------------------------------------------------------------------
# Local database: outbox + faculty mirror
# ---------------------------------------------------------------------------
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_outbox():
    conn = _db()
    conn.execute("""CREATE TABLE IF NOT EXISTS outbox (
        event_id TEXT PRIMARY KEY,
        faculty_id TEXT NOT NULL,
        ts REAL NOT NULL,
        queued_at REAL)""")
    conn.commit()
    conn.close()


def enqueue_event(faculty_id, ts):
    event_id = uuid.uuid4().hex
    conn = _db()
    try:
        conn.execute("INSERT INTO outbox (event_id,faculty_id,ts,queued_at) VALUES (?,?,?,?)",
                     (event_id, faculty_id, float(ts), time.time()))
        conn.commit()
    finally:
        conn.close()
    return event_id


def pending_events(limit=proto.MAX_EVENTS_PER_SYNC):
    conn = _db()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT event_id,faculty_id,ts FROM outbox ORDER BY ts, queued_at LIMIT ?", (limit,))]
    finally:
        conn.close()


def pending_count():
    conn = _db()
    try:
        return conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    finally:
        conn.close()


def _delete_events(ids):
    if not ids:
        return
    conn = _db()
    try:
        conn.executemany("DELETE FROM outbox WHERE event_id=?", [(i,) for i in ids])
        conn.commit()
    finally:
        conn.close()


def local_faculty_rows():
    conn = _db()
    try:
        return conn.execute("SELECT faculty_id,name,department,designation,email,phone,active FROM faculty "
                            "WHERE faculty_id IS NOT NULL AND faculty_id<>''").fetchall()
    finally:
        conn.close()


def apply_faculty(rows):
    """Make the local faculty table equal to the admin PC's list (one transaction)."""
    conn = _db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = {r["faculty_id"]: r["id"] for r in conn.execute("SELECT id,faculty_id FROM faculty")}
        wanted = set()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in rows:
            fid = r["faculty_id"]
            wanted.add(fid)
            vals = (r["name"], r["department"], r["designation"], r["email"], r["phone"], int(r["active"]))
            if fid in existing:
                conn.execute("UPDATE faculty SET name=?,department=?,designation=?,email=?,phone=?,active=? "
                             "WHERE id=?", vals + (existing[fid],))
            else:
                conn.execute("INSERT INTO faculty (faculty_id,name,department,designation,email,phone,active,"
                             "image_path,created_at) VALUES (?,?,?,?,?,?,?,?,?)", (fid,) + vals + ("", stamp))
        for fid, pk in existing.items():
            if fid not in wanted:
                conn.execute("DELETE FROM faculty WHERE id=?", (pk,))
        conn.execute("DELETE FROM faculty WHERE faculty_id IS NULL OR faculty_id=''")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def prune_attendance():
    """The admin PC keeps the history; the kiosk only needs recent days."""
    cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - KEEP_ATTENDANCE_DAYS * 86400))
    conn = _db()
    try:
        conn.execute("DELETE FROM attendance WHERE date < ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Small local state (has this kiosk ever mirrored the admin PC?)
# ---------------------------------------------------------------------------
def _load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state):
    tmp = STATE_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_PATH)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Local facts reported to the admin PC
# ---------------------------------------------------------------------------
_sha_cache = {}


def file_sha(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    hit = _sha_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _sha_cache[path] = (key, h.hexdigest())
    return h.hexdigest()


def models_fingerprint():
    parts = [file_sha(p) or "missing" for p in (YUNET_PATH, SFACE_PATH)]
    return proto.sha256_hex("|".join(parts).encode())


def sklearn_version():
    try:
        import sklearn
        return sklearn.__version__
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class SyncClient:
    def __init__(self, cfg, opener=None):
        self.cfg = cfg
        self._opener = opener or self._default_open
        self.last_error = None
        self.warning = None
        self.skew = None

    # -- transport ---------------------------------------------------------
    def _default_open(self, req, timeout):
        import ssl
        ctx = None
        if req.full_url.startswith("https") and self.cfg.get("ca_file"):
            ctx = ssl.create_default_context(cafile=self.cfg["ca_file"])
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)

    def _call(self, method, path, payload=None):
        """Signed request. Returns (status, headers, body) of a verified reply."""
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload is not None else b""
        ts = int(time.time())
        nonce = proto.new_nonce()
        sig = proto.sign_request(self.cfg["secret"], method, path, ts, nonce, body)
        req = urllib.request.Request(self.cfg["admin_url"] + path, data=body if method == "POST" else None,
                                     method=method)
        req.add_header("X-Kiosk-Id", self.cfg["kiosk_id"])
        req.add_header("X-Timestamp", str(ts))
        req.add_header("X-Nonce", nonce)
        req.add_header("X-Signature", sig)
        req.add_header("Content-Type", "application/json")
        try:
            resp = self._opener(req, HTTP_TIMEOUT)
            status, headers, data = resp.status, resp.headers, resp.read()
            resp.close()
        except urllib.error.HTTPError as exc:
            status, headers, data = exc.code, exc.headers, exc.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise SyncError(f"Cannot reach the admin PC ({reason})")

        if status == 401 or status == 400 or (status == 403 and not headers.get("X-Signature")):
            # Rejected before signing a reply: explain it in plain words.
            try:
                info = json.loads(data)
            except ValueError:
                info = {}
            if info.get("error") == "clock_skew" and isinstance(info.get("server_time"), (int, float)):
                self.skew = info["server_time"] - time.time()
            raise SyncError(info.get("message") or f"Admin PC refused the request (HTTP {status})")
        expected = proto.sign_response(self.cfg["secret"], nonce, status, data)
        if not proto.same(expected, headers.get("X-Signature", "")):
            raise SyncError("Reply from the admin PC failed verification (wrong key, or it was tampered with)")
        return status, headers, data

    # -- one cycle ---------------------------------------------------------
    def _status(self, pending):
        return {
            "classifier_sha": file_sha(CLASSIFIER_PATH) or "",
            "faculty_hash": proto.faculty_digest(local_faculty_rows()),
            "models_fp": models_fingerprint(),
            "opencv": cv2.__version__,
            "sklearn": sklearn_version(),
            "pending": pending,
            "last_error": (self.last_error or "")[:180],
            "warning": (self.warning or "")[:180],
            "skew": round(self.skew, 1) if self.skew is not None else 0,
            "version": 1,
        }

    def sync_once(self):
        """One round trip. Returns {"sent": n, "changed": bool}. Raises SyncError."""
        events = pending_events()
        t0 = time.time()
        status, _h, data = self._call("POST", "/api/kiosk/sync",
                                      {"events": events, "status": self._status(pending_count())})
        t1 = time.time()
        try:
            reply = json.loads(data)
        except ValueError:
            raise SyncError("Unreadable reply from the admin PC")
        if status != 200 or not reply.get("ok"):
            raise SyncError(reply.get("message") or f"Admin PC error (HTTP {status})")

        # The admin PC has recorded these; drop them from the queue.
        sent_ids = {e["event_id"] for e in events}
        _delete_events([i for i in reply.get("results", {}) if i in sent_ids])

        if isinstance(reply.get("server_time"), (int, float)):
            self.skew = reply["server_time"] - (t0 + t1) / 2
        self.warning = None
        if self.skew is not None and abs(self.skew) > SKEW_WARN_SECONDS:
            self.warning = f"Kiosk clock is {abs(self.skew) / 60:.0f} min off - fix the time"

        changed = self._apply_manifest(reply)
        prune_attendance()
        return {"sent": len(sent_ids & set(reply.get("results", {}))), "changed": changed,
                "more": pending_count() > 0}

    # -- applying what the admin PC holds ----------------------------------
    def _apply_manifest(self, m):
        faculty = m.get("faculty")
        clf = m.get("classifier") or {}
        if not isinstance(faculty, list):
            raise SyncError("Reply from the admin PC has no faculty list")
        state = _load_state()
        local = local_faculty_rows()
        changed = False

        # Safety net for the first link-up: a kiosk that used to be stand-alone
        # still holds the real faculty data, while a brand-new admin PC is empty.
        if not faculty and local and not state.get("mirrored"):
            self.warning = (f"Admin PC has no faculty yet but this kiosk has {len(local)}. Copy attendance.db, "
                            "known_faces and the classifier to the admin PC first. Nothing was changed.")
            return False

        # 1) Download + verify the classifier (the slow part) before touching anything.
        candidate = None
        install_error = None
        want_sha = clf.get("sha256") if clf.get("present") else None
        have_sha = file_sha(CLASSIFIER_PATH)
        if want_sha and want_sha != have_sha:
            try:
                candidate = self._fetch_classifier(clf)
            except SyncError as exc:
                install_error = str(exc)
        # 2) Faculty list (fast, one transaction) ...
        new_hash = proto.faculty_digest(faculty)
        if new_hash != proto.faculty_digest(local):
            apply_faculty(faculty)
            changed = True
        # 3) ... then swap the classifier in right after it.
        if candidate:
            os.replace(candidate, CLASSIFIER_PATH)
            changed = True
        elif not want_sha and not clf.get("present") and have_sha and not faculty:
            try:
                os.remove(CLASSIFIER_PATH)   # the admin PC has nobody trained any more
                changed = True
            except OSError:
                pass
        if faculty and not state.get("mirrored"):
            state["mirrored"] = True
            _save_state(state)
        if install_error:
            # Reachable and in contact, but the new face data could not be installed.
            self.warning = install_error
        return changed

    def _fetch_classifier(self, clf):
        mine_fp = models_fingerprint()
        if clf.get("models_fp") and clf["models_fp"] != mine_fp:
            raise SyncError("Model files on this kiosk differ from the admin PC's - copy the 'models' folder over")
        mine_sk = sklearn_version()
        if clf.get("sklearn") and mine_sk and proto.major_minor(clf["sklearn"]) != proto.major_minor(mine_sk):
            raise SyncError(f"scikit-learn {mine_sk} here but {clf['sklearn']} on the admin PC - "
                            f"run: pip install scikit-learn=={clf['sklearn']}")
        status, headers, data = self._call("GET", "/api/kiosk/classifier")
        if status != 200:
            raise SyncError("Admin PC has no classifier to send")
        sha = proto.sha256_hex(data)
        if not proto.same(sha, headers.get("X-Content-SHA256", "")):
            raise SyncError("Downloaded classifier is corrupt (checksum mismatch)")
        # Signature + checksum verified, so it is now safe to unpickle for a trial load.
        try:
            bundle = pickle.loads(data)
            if not isinstance(bundle, dict) or "refs" not in bundle:
                raise ValueError("unexpected content")
        except Exception as exc:  # noqa: BLE001
            raise SyncError(f"Downloaded classifier cannot be loaded here: {exc}")
        tmp = CLASSIFIER_PATH + ".incoming"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return tmp


# ---------------------------------------------------------------------------
# Background thread used by kiosk.py
# ---------------------------------------------------------------------------
class SyncThread(threading.Thread):
    def __init__(self, cfg, client=None):
        super().__init__(daemon=True, name="kiosk-sync")
        self.cfg = cfg
        self.client = client or SyncClient(cfg)
        self._wake = threading.Event()
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._state = {"configured": True, "online": False, "ever_synced": False, "last_ok": None,
                       "error": None, "warning": None, "pending": 0}
        self.fails = 0

    def snapshot(self):
        with self._lock:
            snap = dict(self._state)
        snap["pending"] = pending_count()
        return snap

    def kick(self):
        self._wake.set()

    def stop(self):
        self._halt.set()
        self._wake.set()

    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    def run_cycle(self):
        """One attempt; returns seconds to wait before the next."""
        try:
            res = self.client.sync_once()
        except SyncError as exc:
            self.fails += 1
            self.client.last_error = str(exc)
            self._set(online=False, error=str(exc), warning=self.client.warning)
            base = self.cfg["poll_seconds"] * (2 ** min(self.fails, 6))
            return min(MAX_BACKOFF, base) * random.uniform(0.8, 1.2)
        except Exception as exc:  # noqa: BLE001 - the thread must never die
            self.fails += 1
            msg = f"Unexpected sync problem: {type(exc).__name__}: {exc}"
            self.client.last_error = msg
            self._set(online=False, error=msg)
            return min(MAX_BACKOFF, self.cfg["poll_seconds"] * (2 ** min(self.fails, 6)))
        self.fails = 0
        self.client.last_error = None
        self._set(online=True, ever_synced=True, last_ok=time.time(), error=None, warning=self.client.warning)
        if res["more"]:
            return 0.2          # more queued scans: keep draining
        return 1.0 if res["changed"] else self.cfg["poll_seconds"]

    def run(self):
        while not self._halt.is_set():
            wait = self.run_cycle()
            self._wake.wait(wait)
            self._wake.clear()


def start_if_configured():
    """Return a running SyncThread, or None for a stand-alone kiosk."""
    cfg = load_config()
    if cfg is None:
        return None
    ensure_outbox()
    t = SyncThread(cfg)
    t.start()
    return t
