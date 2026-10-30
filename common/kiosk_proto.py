"""
Shared by the admin PC (kiosk_api.py) and the kiosk (kiosk_sync.py).

Every request from a kiosk is signed with a per-kiosk secret (HMAC-SHA256) and
every reply from the admin PC is signed too. That matters because the trained
classifier is a pickle file: a pickle must never be loaded unless we are sure
it came from the admin PC and was not altered on the way.

Standard library only, so the Raspberry Pi needs nothing extra for this.
"""
import hashlib
import hmac
import json
import re
import secrets
import time

MAX_SKEW_SECONDS = 300          # allowed clock difference for a signed request
MAX_EVENTS_PER_SYNC = 100
KIOSK_ID_RE = re.compile(r"^k-[0-9a-f]{8}$")
EVENT_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
FACULTY_FIELDS = ("faculty_id", "name", "department", "designation", "email", "phone", "active")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_nonce() -> str:
    return secrets.token_hex(12)


def new_secret() -> str:
    return secrets.token_urlsafe(32)


def new_kiosk_id() -> str:
    return "k-" + secrets.token_hex(4)


def _mac(secret: str, text: str) -> str:
    return hmac.new(secret.encode("utf-8"), text.encode("utf-8"), hashlib.sha256).hexdigest()


def sign_request(secret, method, path, ts, nonce, body: bytes) -> str:
    return _mac(secret, "\n".join(["req", method.upper(), path, str(int(ts)), nonce, sha256_hex(body)]))


def sign_response(secret, nonce, status, body: bytes) -> str:
    # Bound to the request's nonce, so an old reply cannot be replayed later.
    return _mac(secret, "\n".join(["resp", nonce, str(int(status)), sha256_hex(body)]))


def same(a: str, b: str) -> bool:
    return hmac.compare_digest(str(a or ""), str(b or ""))


def faculty_row(row) -> list:
    """Canonical form of one faculty record (identical on both machines)."""
    def text(v):
        return "" if v is None else str(v)
    get = row.__getitem__
    return [text(get("faculty_id")), text(get("name")), text(get("department")),
            text(get("designation")), text(get("email")), text(get("phone")),
            0 if get("active") in (0, "0", False) else 1]


def faculty_digest(rows) -> str:
    """Fingerprint of the whole faculty list; equal lists give equal digests."""
    canon = sorted((faculty_row(r) for r in rows), key=lambda r: r[0])
    return sha256_hex(json.dumps(canon, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def major_minor(version: str) -> str:
    parts = re.findall(r"\d+", str(version))
    return ".".join(parts[:2])


def now() -> float:
    return time.time()
