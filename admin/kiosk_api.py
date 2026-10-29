"""
Admin PC side of the kiosk link.

The kiosk (Raspberry Pi) calls these endpoints every few seconds:

  POST /api/kiosk/sync        send queued scans + status; get the faculty list
                              and the fingerprint of the trained classifier
  GET  /api/kiosk/classifier  download the trained classifier

Authentication is a per-kiosk secret created on the dashboard's Kiosks page.
Requests and replies are HMAC-signed (see kiosk_proto.py); there are no
cookies and no admin login involved.
"""
import json
import math
import os
import sqlite3
import threading
import time
from datetime import datetime

import cv2
from flask import Blueprint, Response, g, request

import common.kiosk_proto as proto
from common.lite_core import CLASSIFIER_PATH, DB_PATH, MODELS_DIR, SFACE_PATH, YUNET_PATH, mark_attendance

bp = Blueprint("kiosk_api", __name__, url_prefix="/api/kiosk")

ONLINE_WITHIN_SECONDS = float(os.environ.get("KIOSK_ONLINE_SECONDS", "45"))
FUTURE_TOLERANCE = 120      # a scan "from the future" is clamped to now
STATUS_KEYS = ("classifier_sha", "faculty_hash", "models_fp", "opencv", "sklearn",
               "pending", "last_error", "warning", "skew", "version")

_nonces = {}
_nonce_lock = threading.Lock()
_info_cache = {}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables():
    conn = _db()
    conn.execute("""CREATE TABLE IF NOT EXISTS kiosks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kiosk_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        secret TEXT NOT NULL,
        created_at TEXT,
        last_seen REAL,
        last_ip TEXT,
        status_json TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS kiosk_events (
        event_id TEXT PRIMARY KEY,
        kiosk_id TEXT,
        faculty_id TEXT,
        scanned_at REAL,
        result TEXT,
        received_at REAL)""")
    conn.commit()
    conn.close()


def create_kiosk(name):
    name = " ".join(str(name or "").split())[:60]
    if not name:
        raise ValueError("Give the kiosk a name, for example 'Main gate'.")
    ensure_tables()
    conn = _db()
    try:
        for _ in range(5):
            kiosk_id = proto.new_kiosk_id()
            if conn.execute("SELECT 1 FROM kiosks WHERE kiosk_id=?", (kiosk_id,)).fetchone() is None:
                break
        secret = proto.new_secret()
        conn.execute("INSERT INTO kiosks (kiosk_id,name,secret,created_at) VALUES (?,?,?,?)",
                     (kiosk_id, name, secret, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    finally:
        conn.close()
    return {"kiosk_id": kiosk_id, "name": name, "secret": secret}


def regenerate_secret(kiosk_id):
    conn = _db()
    try:
        secret = proto.new_secret()
        cur = conn.execute("UPDATE kiosks SET secret=?, status_json=NULL WHERE kiosk_id=?", (secret, kiosk_id))
        conn.commit()
        if cur.rowcount == 0:
            return None
        name = conn.execute("SELECT name FROM kiosks WHERE kiosk_id=?", (kiosk_id,)).fetchone()["name"]
    finally:
        conn.close()
    return {"kiosk_id": kiosk_id, "name": name, "secret": secret}


def remove_kiosk(kiosk_id):
    conn = _db()
    try:
        cur = conn.execute("DELETE FROM kiosks WHERE kiosk_id=?", (kiosk_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def _get_kiosk(kiosk_id):
    conn = _db()
    try:
        return conn.execute("SELECT * FROM kiosks WHERE kiosk_id=?", (kiosk_id,)).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# What the admin PC currently holds (faculty list + classifier + models)
# ---------------------------------------------------------------------------
def _file_sha(path):
    """sha256 of a file, cached until its size/mtime changes. (None, 0) if missing."""
    try:
        st = os.stat(path)
    except OSError:
        return None, 0
    key = (st.st_mtime_ns, st.st_size)
    cached = _info_cache.get(path)
    if cached and cached[0] == key:
        return cached[1], st.st_size
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _info_cache[path] = (key, h.hexdigest())
    return h.hexdigest(), st.st_size


def models_fingerprint():
    parts = []
    for p in (YUNET_PATH, SFACE_PATH):
        sha, _ = _file_sha(p)
        parts.append(sha or "missing")
    return proto.sha256_hex("|".join(parts).encode())


def sklearn_version():
    try:
        import sklearn
        return sklearn.__version__
    except Exception:  # noqa: BLE001
        return ""


def faculty_rows():
    conn = _db()
    try:
        return conn.execute(
            "SELECT faculty_id,name,department,designation,email,phone,active FROM faculty "
            "WHERE faculty_id IS NOT NULL AND faculty_id<>'' ORDER BY faculty_id").fetchall()
    finally:
        conn.close()


def current_state():
    sha, size = _file_sha(CLASSIFIER_PATH)
    rows = faculty_rows()
    return {
        "faculty": [dict(zip(proto.FACULTY_FIELDS, proto.faculty_row(r))) for r in rows],
        "faculty_hash": proto.faculty_digest(rows),
        "classifier": {
            "present": sha is not None, "sha256": sha, "size": size,
            "models_fp": models_fingerprint(), "opencv": cv2.__version__, "sklearn": sklearn_version(),
        },
    }


def overview():
    """Every kiosk with online / up-to-date flags, for the dashboard."""
    state = current_state()
    out = []
    conn = _db()
    try:
        rows = conn.execute("SELECT * FROM kiosks ORDER BY name COLLATE NOCASE").fetchall()
        pending_events = {}
    finally:
        conn.close()
    now = time.time()
    for r in rows:
        try:
            st = json.loads(r["status_json"] or "{}")
        except ValueError:
            st = {}
        seen = r["last_seen"]
        online = bool(seen) and now - seen <= ONLINE_WITHIN_SECONDS
        has_status = bool(st)
        faculty_ok = has_status and st.get("faculty_hash") == state["faculty_hash"]
        want = state["classifier"]["sha256"]
        classifier_ok = has_status and (st.get("classifier_sha") or None) == want
        out.append({
            "kiosk_id": r["kiosk_id"], "name": r["name"], "created_at": r["created_at"],
            "last_seen": seen, "last_seen_ago": None if not seen else int(now - seen),
            "last_ip": r["last_ip"], "online": online,
            "synced": online and faculty_ok and classifier_ok,
            "faculty_ok": faculty_ok, "classifier_ok": classifier_ok,
            "pending": st.get("pending"), "last_error": st.get("last_error"),
            "warning": st.get("warning"), "skew": st.get("skew"),
            "version_note": _compat_note(st, state),
        })
    return out


def _compat_note(st, state):
    c = state["classifier"]
    if not st:
        return None
    if st.get("models_fp") and st["models_fp"] != c["models_fp"]:
        return "Model files differ from the admin PC"
    if st.get("sklearn") and c["sklearn"] and proto.major_minor(st["sklearn"]) != proto.major_minor(c["sklearn"]):
        return f"scikit-learn {st['sklearn']} on the kiosk, {c['sklearn']} here"
    return None


# ---------------------------------------------------------------------------
# Request authentication
# ---------------------------------------------------------------------------
def _err(code, message, status, extra=None):
    body = {"ok": False, "error": code, "message": message, "server_time": time.time()}
    body.update(extra or {})
    return Response(json.dumps(body), status=status, mimetype="application/json")


def _remember_nonce(kiosk_id, nonce):
    now = time.time()
    with _nonce_lock:
        if len(_nonces) > 5000:
            for k in [k for k, exp in _nonces.items() if exp < now]:
                _nonces.pop(k, None)
        key = (kiosk_id, nonce)
        if key in _nonces and _nonces[key] > now:
            return False
        _nonces[key] = now + 2 * proto.MAX_SKEW_SECONDS
        return True


@bp.before_request
def authenticate():
    h = request.headers
    kiosk_id = h.get("X-Kiosk-Id", "")
    nonce = h.get("X-Nonce", "")
    sig = h.get("X-Signature", "")
    try:
        ts = int(h.get("X-Timestamp", ""))
    except ValueError:
        return _err("bad_request", "Missing or invalid X-Timestamp.", 400)
    if not (proto.KIOSK_ID_RE.match(kiosk_id) and 8 <= len(nonce) <= 64 and sig):
        return _err("bad_request", "Missing authentication headers.", 400)
    row = _get_kiosk(kiosk_id)
    if row is None:
        return _err("unknown_kiosk", "This kiosk is not registered (or was removed) on the admin PC.", 401)
    body = request.get_data()
    expected = proto.sign_request(row["secret"], request.method, request.path, ts, nonce, body)
    if not proto.same(expected, sig):
        return _err("bad_signature", "Wrong kiosk key. Re-create kiosk_config.json from the Kiosks page.", 401)
    if abs(time.time() - ts) > proto.MAX_SKEW_SECONDS:
        return _err("clock_skew", "The kiosk clock differs from the admin PC by more than 5 minutes.", 401)
    if not _remember_nonce(kiosk_id, nonce):
        return _err("replay", "Request already seen.", 401)
    g.kiosk = row
    g.nonce = nonce
    return None


def reply(data, status=200, raw=None, mimetype="application/json", headers=None):
    body = raw if raw is not None else json.dumps(data, separators=(",", ":")).encode("utf-8")
    resp = Response(body, status=status, mimetype=mimetype)
    resp.headers["X-Signature"] = proto.sign_response(g.kiosk["secret"], g.nonce, status, body)
    for k, v in (headers or {}).items():
        resp.headers[k] = v
    return resp


# ---------------------------------------------------------------------------
# Scans coming from kiosks
# ---------------------------------------------------------------------------
def _apply_event(conn, kiosk_id, ev, received):
    """Record one scan. Returns a short result string; never raises for bad data."""
    if not isinstance(ev, dict):
        return None, "invalid"
    eid = ev.get("event_id")
    if not isinstance(eid, str) or not proto.EVENT_ID_RE.match(eid):
        return None, "invalid"
    fid = ev.get("faculty_id")
    ts = ev.get("ts")
    if (not isinstance(fid, str) or not fid or len(fid) > 64 or isinstance(ts, bool)
            or not isinstance(ts, (int, float)) or not math.isfinite(ts) or ts < 946684800):
        return eid, "invalid"

    cur = conn.execute("INSERT OR IGNORE INTO kiosk_events (event_id,kiosk_id,faculty_id,scanned_at,result,received_at)"
                       " VALUES (?,?,?,?,?,?)", (eid, kiosk_id, fid, ts, "pending", received))
    conn.commit()
    if cur.rowcount == 0:
        prev = conn.execute("SELECT result FROM kiosk_events WHERE event_id=?", (eid,)).fetchone()
        if prev is not None and prev["result"] != "pending":
            return eid, "duplicate"
        # 'pending' = the admin PC stopped half-way last time. Replaying the same
        # time is safe: the attendance cooldown absorbs a second copy.

    fac = conn.execute("SELECT id FROM faculty WHERE faculty_id=?", (fid,)).fetchone()
    if fac is None:
        result = "unknown_faculty"
    else:
        when = min(ts, received + FUTURE_TOLERANCE)
        status, _t, _late = mark_attendance(fac["id"], now=datetime.fromtimestamp(when))
        result = status if when == ts else status + "+clock_ahead"
    conn.execute("UPDATE kiosk_events SET result=? WHERE event_id=?", (result, eid))
    conn.commit()
    return eid, result


@bp.route("/sync", methods=["POST"])
def sync():
    try:
        data = json.loads(request.get_data() or b"{}")
        if not isinstance(data, dict):
            raise ValueError
    except ValueError:
        return reply({"ok": False, "error": "bad_request", "message": "Body must be a JSON object."}, 400)
    events = data.get("events") or []
    if not isinstance(events, list) or len(events) > proto.MAX_EVENTS_PER_SYNC:
        return reply({"ok": False, "error": "bad_request", "message": "Too many events in one sync."}, 400)

    received = time.time()
    results = {}
    conn = _db()
    try:
        # Oldest scan first, so check-in / check-out pair up correctly.
        valid = [e for e in events if isinstance(e, dict) and isinstance(e.get("ts"), (int, float))
                 and not isinstance(e.get("ts"), bool) and math.isfinite(e["ts"])]
        junk = [e for e in events if e not in valid]
        for ev in sorted(valid, key=lambda e: e["ts"]) + junk:
            eid, result = _apply_event(conn, g.kiosk["kiosk_id"], ev, received)
            if eid:
                results[eid] = result
    finally:
        conn.close()

    status = data.get("status") if isinstance(data.get("status"), dict) else {}
    clean = {}
    for k in STATUS_KEYS:
        v = status.get(k)
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            clean[k] = v[:200] if isinstance(v, str) else v
    conn = _db()
    try:
        conn.execute("UPDATE kiosks SET last_seen=?, last_ip=?, status_json=? WHERE kiosk_id=?",
                     (received, request.remote_addr, json.dumps(clean), g.kiosk["kiosk_id"]))
        conn.commit()
    finally:
        conn.close()

    out = {"ok": True, "server_time": time.time(), "results": results}
    out.update(current_state())
    return reply(out)


@bp.route("/classifier", methods=["GET"])
def classifier():
    try:
        with open(CLASSIFIER_PATH, "rb") as f:
            data = f.read()
    except OSError:
        return reply({"ok": False, "error": "no_classifier", "message": "No trained classifier yet."}, 404)
    return reply(None, raw=data, mimetype="application/octet-stream",
                 headers={"X-Content-SHA256": proto.sha256_hex(data)})
