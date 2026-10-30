"""
Lightweight Flask dashboard for the BCREC faculty attendance system.

Recognition is handled by kiosk.py. Flask provides management pages, daily
attendance, monthly reports, CSV export, and a native-camera enrollment
launcher.

Access is protected by an admin login. Accounts live in the existing
admin_users table and are managed with create_admin.py.
"""

import os as _os
import sys as _sys

# Shared code (lite_core.py, kiosk_proto.py, ...) lives in ../common
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))

import calendar
import csv
import hashlib
import hmac
import io
import os
import json
import secrets
import socket
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from urllib.parse import urlparse

from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

import kiosk_api
import kiosk_proto
import registration
from lite_core import BASE_DIR, DB_PATH, ensure_db

# 0.0.0.0 so kiosks on the LAN can sync. The dashboard itself is login-protected.
APP_HOST = os.environ.get("ATTENDANCE_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("ATTENDANCE_PORT", "5000"))
APP_DEBUG = os.environ.get("ATTENDANCE_DEBUG", "0") == "1"

# --- Login settings (all overridable with environment variables) -----------
SESSION_HOURS = float(os.environ.get("ATTENDANCE_SESSION_HOURS", "8"))
COOKIE_SECURE = os.environ.get("ATTENDANCE_COOKIE_SECURE", "0") == "1"
MAX_FAILED_LOGINS = int(os.environ.get("ATTENDANCE_MAX_FAILED_LOGINS", "5"))
LOCKOUT_SECONDS = int(os.environ.get("ATTENDANCE_LOCKOUT_SECONDS", "900"))


def load_secret_key():
    """
    Key used to sign session cookies.

    Order: ATTENDANCE_SECRET_KEY env var, then a .secret_key file next to this
    script (created on first run, owner-only permissions). If neither can be
    used, fall back to a temporary key, which logs everyone out on restart.
    """
    env_key = os.environ.get("ATTENDANCE_SECRET_KEY")
    if env_key:
        return env_key

    path = os.path.join(BASE_DIR, ".secret_key")
    try:
        with open(path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    except FileNotFoundError:
        pass
    except OSError:
        return secrets.token_hex(32)

    key = secrets.token_hex(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(key)
    except OSError:
        print("[auth] Could not save .secret_key; sessions reset on restart.")
    return key


app = Flask(__name__)
app.config.update(
    SECRET_KEY=load_secret_key(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=COOKIE_SECURE,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=SESSION_HOURS),
)
ensure_db()
kiosk_api.ensure_tables()
app.register_blueprint(kiosk_api.bp)
try:
    registration.recover_on_startup()  # settle any half-finished registration
except Exception as exc:  # noqa: BLE001 - never stop the dashboard from starting
    print(f"[registration] startup recovery skipped: {exc}")


# Database helper
# Connect and get database object
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Admin authentication
# ---------------------------------------------------------------------------
# Compared against when the username does not exist, so a wrong username and a
# wrong password take about the same time.
DUMMY_HASH = generate_password_hash("not-a-real-password")

PUBLIC_ENDPOINTS = {"login", "static"}

_failed_logins = {}
_failed_lock = threading.Lock()


def lockout_remaining(ip):
    """Seconds until this IP may try again (0 if it is not locked out)."""
    now = time.time()
    with _failed_lock:
        stamps = [t for t in _failed_logins.get(ip, []) if now - t < LOCKOUT_SECONDS]
        if stamps:
            _failed_logins[ip] = stamps
        else:
            _failed_logins.pop(ip, None)
        if len(stamps) >= MAX_FAILED_LOGINS:
            return int(LOCKOUT_SECONDS - (now - stamps[0])) + 1
    return 0


def record_failed_login(ip):
    with _failed_lock:
        _failed_logins.setdefault(ip, []).append(time.time())


def clear_failed_logins(ip):
    with _failed_lock:
        _failed_logins.pop(ip, None)


def admin_exists():
    conn = get_db()
    try:
        return conn.execute("SELECT 1 FROM admin_users LIMIT 1").fetchone() is not None
    finally:
        conn.close()


def _password_fingerprint(password_hash):
    # Stored in the session. If the password is reset, the fingerprint no
    # longer matches and old sessions stop working.
    return hashlib.sha256((password_hash or "").encode("utf-8")).hexdigest()[:16]


def current_admin():
    """Return the logged-in admin row, or None. Cached per request."""
    if "admin" in g:
        return g.admin

    g.admin = None
    admin_id = session.get("admin_id")
    if admin_id is not None:
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT id, username, password_hash FROM admin_users WHERE id=?",
                (admin_id,),
            ).fetchone()
        finally:
            conn.close()

        if row and hmac.compare_digest(
            _password_fingerprint(row["password_hash"]),
            str(session.get("pwv", "")),
        ):
            g.admin = row
        else:
            session.clear()
    return g.admin


def safe_next(target):
    """Only allow redirects to a path on this site (no open redirects)."""
    if not target:
        return url_for("dashboard")
    parsed = urlparse(target)
    if (
        parsed.scheme
        or parsed.netloc
        or not target.startswith("/")
        or target.startswith("//")
        or "\\" in target
    ):
        return url_for("dashboard")
    return target


def same_origin():
    """Block cross-site form posts. SameSite=Lax cookies are the first layer."""
    origin = request.headers.get("Origin") or request.headers.get("Referer")
    if not origin:
        return True
    return urlparse(origin).netloc == request.host


@app.before_request
def require_login():
    if request.method == "POST" and not same_origin():
        return Response("Cross-site request blocked.", status=403)

    # Kiosk sync calls are authenticated by their own signature (kiosk_api.py).
    if request.endpoint in PUBLIC_ENDPOINTS or request.blueprint == "kiosk_api":
        return None

    if current_admin() is None:
        if request.method == "GET":
            return redirect(url_for("login", next=request.full_path.rstrip("?")))
        return redirect(url_for("login"))
    return None


@app.after_request
def harden_response(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if request.endpoint != "static":
        resp.headers["Cache-Control"] = "no-store"

    return resp


@app.context_processor
def inject_admin():
    """Makes {{ admin_username }} available to every template (used by the nav)."""
    admin = g.get("admin")
    return {"admin_username": admin["username"] if admin else ""}


@app.route("/login", methods=["GET", "POST"])
def login():
    next_url = safe_next(request.values.get("next", ""))

    if current_admin() is not None:
        return redirect(next_url)

    def page(error=None, status=200):
        return (
            render_template(
                "login.html",
                error=error,
                next_url=next_url,
                no_admin=not admin_exists(),
            ),
            status,
        )

    if request.method == "GET":
        return page()

    ip = request.remote_addr or "unknown"
    wait = lockout_remaining(ip)
    if wait:
        minutes = max(1, (wait + 59) // 60)
        return page(
            f"Too many failed attempts. Try again in {minutes} minute(s).", 429
        )

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    conn = get_db()
    try:
        admin = conn.execute(
            "SELECT id, username, password_hash FROM admin_users WHERE username=?",
            (username,),
        ).fetchone()
    finally:
        conn.close()

    stored_hash = admin["password_hash"] if admin and admin["password_hash"] else DUMMY_HASH
    password_ok = check_password_hash(stored_hash, password)

    if admin is None or not password_ok:
        record_failed_login(ip)
        return page("Invalid username or password.", 401)

    clear_failed_logins(ip)
    session.clear()  # new session on login
    session.permanent = True
    session["admin_id"] = admin["id"]
    session["pwv"] = _password_fingerprint(admin["password_hash"])
    return redirect(next_url)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------
def format_working_time(minutes):
    if minutes is None:
        return "-"
    minutes = max(0, int(minutes))
    return f"{minutes // 60}h {minutes % 60}m"


def calculate_session_minutes(attendance_date, check_in, check_out):
    if not check_in or not check_out:
        return None
    try:
        start = datetime.strptime(
            f"{attendance_date} {check_in}", "%Y-%m-%d %H:%M:%S"
        )
        end = datetime.strptime(
            f"{attendance_date} {check_out}", "%Y-%m-%d %H:%M:%S"
        )
        return max(0, int((end - start).total_seconds() // 60))
    except (ValueError, TypeError):
        return None


# Working days: Monday-Saturday; Sunday excluded. Future dates are excluded
# from the current month.
def get_working_days(year, month):
    today = date.today()
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])

    if year == today.year and month == today.month:
        last_day = today

    if first_day > today:
        return 0

    working_days = 0
    current = first_day
    while current <= last_day:
        if current.weekday() != 6:
            working_days += 1
        current += timedelta(days=1)
    return working_days


# Dashboard
@app.route("/")
def dashboard():
    today = date.today().isoformat()
    conn = get_db()
    try:
        total_faculty = conn.execute(
            "SELECT COUNT(*) FROM faculty WHERE active = 1"
        ).fetchone()[0]

        present_today = conn.execute(
            """
            SELECT COUNT(DISTINCT faculty_id_fk)
            FROM attendance
            WHERE date=? AND check_in_time IS NOT NULL
            """,
            (today,),
        ).fetchone()[0]

        late_today = conn.execute(
            """
            SELECT COUNT(DISTINCT faculty_id_fk)
            FROM attendance
            WHERE date=? AND late=1
            """,
            (today,),
        ).fetchone()[0]

        absent_today = max(total_faculty - present_today, 0)
    finally:
        conn.close()

    return render_template(
        "dashboard.html",
        total_faculty=total_faculty,
        present_today=present_today,
        late_today=late_today,
        absent_today=absent_today,
    )


# Faculty Management
@app.route("/faculty")
def faculty():
    conn = get_db()
    try:
        faculty_list = conn.execute(
            "SELECT * FROM faculty ORDER BY name"
        ).fetchall()
    finally:
        conn.close()

    return render_template("faculty.html", faculty_list=faculty_list)


# ---------------------------------------------------------------------------
# Faculty registration: details -> camera capture -> train -> submit.
# The long-running steps live in registration.py; these routes only start,
# poll and close them, so the browser never waits on the camera or training.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Kiosks (Raspberry Pi devices linked to this admin PC)
# ---------------------------------------------------------------------------
def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the LAN interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def public_url():
    return (os.environ.get("ATTENDANCE_PUBLIC_URL") or f"http://{lan_ip()}:{APP_PORT}").rstrip("/")


def ago_text(seconds):
    if seconds is None:
        return "never"
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def kiosk_note():
    """One line for the registration page: have the kiosks picked up the new face data?"""
    try:
        items = kiosk_api.overview()
    except Exception:  # noqa: BLE001 - purely informational
        return ""
    if not items:
        return "No kiosk is linked to this admin PC yet - add one on the Kiosks page."
    ready = [k for k in items if k["synced"]]
    note = f"Kiosks up to date: {len(ready)} of {len(items)}."
    waiting = [f"{k['name']} ({'updating' if k['online'] else 'offline'})" for k in items if not k["synced"]]
    if waiting:
        note += " Waiting for: " + ", ".join(waiting) + ". Offline kiosks catch up when they reconnect."
    return note


def _kiosk_page(new=None, error=None, status=200):
    items = kiosk_api.overview()
    for k in items:
        k["seen_text"] = ago_text(k["last_seen_ago"])
    state = kiosk_api.current_state()["classifier"]
    return render_template(
        "kiosks.html", kiosks=items, new=new, error=error, admin_url=public_url(), port=APP_PORT,
        versions={"sklearn": state["sklearn"] or "?", "opencv": state["opencv"]},
    ), status


def _new_key_view(info):
    config = {"admin_url": public_url(), "kiosk_id": info["kiosk_id"], "secret": info["secret"], "poll_seconds": 5}
    return {"name": info["name"], "config": json.dumps(config, indent=2)}


@app.route("/kiosks")
def kiosks():
    return _kiosk_page()


@app.route("/kiosks/status")
def kiosks_status():
    return jsonify(kiosk_api.overview())


@app.route("/kiosks/add", methods=["POST"])
def kiosk_add():
    try:
        info = kiosk_api.create_kiosk(request.form.get("name", ""))
    except ValueError as exc:
        return _kiosk_page(error=str(exc), status=400)
    return _kiosk_page(new=_new_key_view(info))


@app.route("/kiosks/<kiosk_id>/regenerate", methods=["POST"])
def kiosk_regenerate(kiosk_id):
    info = kiosk_api.regenerate_secret(kiosk_id) if kiosk_proto.KIOSK_ID_RE.match(kiosk_id) else None
    if info is None:
        flash("That kiosk no longer exists.")
        return redirect(url_for("kiosks"))
    return _kiosk_page(new=_new_key_view(info))


@app.route("/kiosks/<kiosk_id>/remove", methods=["POST"])
def kiosk_remove(kiosk_id):
    if kiosk_proto.KIOSK_ID_RE.match(kiosk_id) and kiosk_api.remove_kiosk(kiosk_id):
        flash("Kiosk removed. It will stop syncing.")
    else:
        flash("That kiosk no longer exists.")
    return redirect(url_for("kiosks"))


def _job_redirect(job_id):
    return redirect(url_for("registration_job", job_id=job_id))


@app.route("/faculty/register", methods=["GET", "POST"])
def register_faculty():
    active = registration.get_active_job()
    if active:  # one registration at a time; resume it instead of starting another
        return _job_redirect(active["id"])

    if request.method == "GET":
        retry = registration.get_job(request.args.get("retry", ""))
        form = retry.get("faculty", {}) if retry else {}
        return render_template("register_faculty.html", message=None, error=None, form=form)

    clean, error = registration.validate_faculty_fields(request.form)
    if error:
        return render_template("register_faculty.html", message=None, error=error, form=clean)

    try:
        job = registration.start_job(clean)
    except registration.JobBusy as exc:
        return _job_redirect(exc.job["id"])
    except registration.RegistrationError as exc:
        return render_template("register_faculty.html", message=None, error=str(exc), form=clean)
    return _job_redirect(job["id"])


@app.route("/faculty/register/<job_id>")
def registration_job(job_id):
    job = registration.get_job(job_id)
    if job is None:
        return redirect(url_for("register_faculty"))
    return render_template(
        "register_job.html",
        job=job,
        kiosk_note=kiosk_note(),
        elapsed=registration.elapsed(job),
        expected_photos=registration.EXPECTED_PHOTOS,
    )


@app.route("/faculty/register/<job_id>/status")
def registration_status(job_id):
    job = registration.get_job(job_id)
    if job is None:
        return jsonify({"state": "missing"}), 404
    data = registration.public_status(job)
    data["kiosk_note"] = kiosk_note()
    return jsonify(data)


@app.route("/faculty/register/<job_id>/photo/<int:n>")
def registration_photo(job_id, n):
    path = registration.staged_photo_path(job_id, n)
    if not path:
        abort(404)
    return send_file(path, mimetype="image/jpeg", max_age=0)


def _run_job_action(job_id, action):
    try:
        action(job_id)
    except registration.InvalidState as exc:
        flash(str(exc))
    return _job_redirect(job_id)


@app.route("/faculty/register/<job_id>/train", methods=["POST"])
def registration_train(job_id):
    return _run_job_action(job_id, registration.request_train)


@app.route("/faculty/register/<job_id>/submit", methods=["POST"])
def registration_submit(job_id):
    return _run_job_action(job_id, registration.submit_job)


@app.route("/faculty/register/<job_id>/cancel", methods=["POST"])
def registration_cancel(job_id):
    try:
        job = registration.cancel_job(job_id)
    except registration.InvalidState as exc:
        flash(str(exc))
        return _job_redirect(job_id)
    if request.form.get("retake") == "1" and job["state"] == "cancelled":
        return redirect(url_for("register_faculty", retry=job_id))
    return _job_redirect(job_id)


# Daily Attendance
@app.route("/attendance")
def attendance():
    selected_date = request.args.get("date", date.today().isoformat())
    conn = get_db()
    try:
        records = conn.execute(
            """
            SELECT
                a.id,
                f.faculty_id,
                f.name,
                f.department,
                f.designation,
                a.date,
                a.check_in_time,
                a.check_out_time,
                a.status,
                a.late,
                a.working_minutes
            FROM attendance a
            JOIN faculty f ON f.id = a.faculty_id_fk
            WHERE a.date=?
            ORDER BY a.check_in_time, f.name
            """,
            (selected_date,),
        ).fetchall()
    finally:
        conn.close()

    rows = []
    for row in records:
        working_minutes = row["working_minutes"]
        if working_minutes is None:
            working_minutes = calculate_session_minutes(
                row["date"], row["check_in_time"], row["check_out_time"]
            )
        rows.append({**dict(row), "working_time": format_working_time(working_minutes)})

    return render_template(
        "attendance.html",
        records=rows,
        selected_date=selected_date,
    )


# Monthly report builder
def build_monthly_report(month, department="ALL"):
    try:
        selected_year, selected_month = map(int, month.split("-"))
        if not (2000 <= selected_year <= 2100 and 1 <= selected_month <= 12):
            raise ValueError
    except (ValueError, AttributeError):
        month = date.today().strftime("%Y-%m")
        selected_year = date.today().year
        selected_month = date.today().month

    working_days = get_working_days(selected_year, selected_month)
    conn = get_db()

    try:
        departments = conn.execute(
            """
            SELECT DISTINCT department
            FROM faculty
            WHERE active=1
              AND department IS NOT NULL
              AND department!=''
            ORDER BY department
            """
        ).fetchall()

        query = """
            SELECT id, faculty_id, name, department
            FROM faculty
            WHERE active=1
        """
        params = []
        if department != "ALL":
            query += " AND department=?"
            params.append(department)
        query += " ORDER BY name"

        faculty_rows = conn.execute(query, params).fetchall()
        report_rows = []

        for member in faculty_rows:
            attendance_rows = conn.execute(
                """
                SELECT date, check_in_time, check_out_time, late, working_minutes
                FROM attendance
                WHERE faculty_id_fk=? AND substr(date,1,7)=?
                ORDER BY date, check_in_time
                """,
                (member["id"], month),
            ).fetchall()

            present_dates = set()
            late_dates = set()
            total_minutes = 0

            for row in attendance_rows:
                if row["check_in_time"]:
                    present_dates.add(row["date"])
                if row["late"]:
                    late_dates.add(row["date"])

                minutes = row["working_minutes"]
                if minutes is None:
                    minutes = calculate_session_minutes(
                        row["date"], row["check_in_time"], row["check_out_time"]
                    )
                if minutes is not None:
                    total_minutes += minutes

            days_present = len(present_dates)
            days_late = len(late_dates)
            days_absent = max(working_days - days_present, 0)
            attendance_percentage = (
                (days_present / working_days) * 100 if working_days else 0.0
            )

            report_rows.append(
                {
                    "faculty_id": member["faculty_id"],
                    "name": member["name"],
                    "department": member["department"],
                    "present_days": days_present,
                    "absent_days": days_absent,
                    "late_days": days_late,
                    "total_minutes": total_minutes,
                    "working_time": format_working_time(total_minutes),
                    "total_hours": total_minutes / 60.0,
                    "attendance_percentage": attendance_percentage,
                }
            )

        return {
            "month": month,
            "selected_year": selected_year,
            "selected_month": selected_month,
            "month_name": calendar.month_name[selected_month],
            "department": department,
            "departments": departments,
            "working_days": working_days,
            "report_rows": report_rows,
        }
    finally:
        conn.close()


@app.route("/reports")
def reports():
    month = request.args.get("month", date.today().strftime("%Y-%m"))
    department = request.args.get("department", "ALL")
    data = build_monthly_report(month, department)
    return render_template("reports.html", **data)


# CSV export
@app.route("/export-csv")
def export_csv():
    month = request.args.get("month", date.today().strftime("%Y-%m"))
    department = request.args.get("department", "ALL")
    data = build_monthly_report(month, department)

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "Faculty ID",
            "Name",
            "Department",
            "Days Present",
            "Days Absent",
            "Days Late",
            "Total Working Time",
            "Attendance %",
        ]
    )

    for row in data["report_rows"]:
        writer.writerow(
            [
                row["faculty_id"],
                row["name"],
                row["department"],
                row["present_days"],
                row["absent_days"],
                row["late_days"],
                row["working_time"],
                f"{row['attendance_percentage']:.2f}%",
            ]
        )

    response = Response(output.getvalue(), mimetype="text/csv; charset=utf-8")
    response.headers["Content-Disposition"] = (
        f"attachment; filename=monthly_attendance_{data['month']}.csv"
    )
    return response


if __name__ == "__main__":
    print(f"BCREC Attendance Dashboard: http://{APP_HOST}:{APP_PORT}")
    print(f"Kiosks sync to: {public_url()}   (link a kiosk on the dashboard's Kiosks page)")
    if not admin_exists():
        print("WARNING: no admin account exists yet. Run: python create_admin.py")
    if APP_HOST not in ("127.0.0.1", "localhost", "::1") and not COOKIE_SECURE:
        print(
            "NOTE: the dashboard is reachable over the network on plain HTTP. "
            "Passwords are not encrypted in transit; put it behind HTTPS "
            "(then set ATTENDANCE_COOKIE_SECURE=1) or keep it on a trusted network."
        )
    app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=APP_DEBUG,
        threaded=False,
    )
