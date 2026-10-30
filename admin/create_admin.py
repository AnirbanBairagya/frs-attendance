"""
Create an admin account for the dashboard, or reset an admin's password.

Usage (run on the machine that hosts attendance.db):

    python create_admin.py                      # create an admin (prompts)
    python create_admin.py --username principal # create with a given username
    python create_admin.py --reset principal    # set a new password
    python create_admin.py --list               # show existing admins

Passwords are typed at a hidden prompt and stored only as a salted hash.
Resetting a password signs that admin out of all active sessions.
"""

import os as _os
import sys as _sys

# Shared code (lite_core.py, kiosk_proto.py, ...) lives in ../common
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))

import argparse
import getpass
import re
import sqlite3
from datetime import datetime

from werkzeug.security import generate_password_hash

from lite_core import DB_PATH, ensure_db

MIN_PASSWORD_LENGTH = 8
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{3,50}$")


def validate_username(username):
    if not USERNAME_RE.match(username):
        return "Username must be 3-50 characters: letters, digits, _ . @ -"
    return None


def validate_password(username, password):
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if password.lower() == username.lower():
        return "Password must not be the same as the username."
    return None


def create_admin(username, password):
    """Insert a new admin. Raises ValueError with a readable message on problems."""
    error = validate_username(username) or validate_password(username, password)
    if error:
        raise ValueError(error)

    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        if conn.execute(
            "SELECT 1 FROM admin_users WHERE username=?", (username,)
        ).fetchone():
            raise ValueError(f"Admin '{username}' already exists. Use --reset to change the password.")
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, created_at) VALUES (?,?,?)",
            (
                username,
                generate_password_hash(password),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def reset_password(username, password):
    """Change an admin's password. Raises ValueError with a readable message."""
    error = validate_password(username, password)
    if error:
        raise ValueError(error)

    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "UPDATE admin_users SET password_hash=? WHERE username=?",
            (generate_password_hash(password), username),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No admin named '{username}'.")
        conn.commit()
    finally:
        conn.close()


def list_admins():
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute(
            "SELECT username, created_at FROM admin_users ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def prompt_password(username):
    while True:
        first = getpass.getpass("Password: ")
        error = validate_password(username, first)
        if error:
            print(error)
            continue
        if first != getpass.getpass("Confirm password: "):
            print("Passwords do not match. Try again.")
            continue
        return first


def main():
    ap = argparse.ArgumentParser(description="Manage dashboard admin accounts.")
    ap.add_argument("--username", help="username for the new admin")
    ap.add_argument("--reset", metavar="USERNAME", help="reset this admin's password")
    ap.add_argument("--list", action="store_true", help="list admin usernames")
    args = ap.parse_args()

    try:
        if args.list:
            admins = list_admins()
            if not admins:
                print("No admin accounts yet.")
            for username, created in admins:
                print(f"{username}   (created {created})")
            return 0

        if args.reset:
            reset_password(args.reset, prompt_password(args.reset))
            print(f"Password updated for '{args.reset}'. Existing sessions are signed out.")
            return 0

        username = (args.username or input("New admin username: ")).strip()
        error = validate_username(username)
        if error:
            print(error)
            return 1
        create_admin(username, prompt_password(username))
        print(f"Admin '{username}' created. You can now sign in at /login.")
        return 0
    except ValueError as exc:
        print(exc)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
