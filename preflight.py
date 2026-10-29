"""
Read-only deployment preflight check.

This script never changes the database, photos, models, or classifier.
Run it on Windows before staging and on Raspberry Pi after copying.
"""

import os
import platform
import sqlite3
import sys

import cv2
import numpy as np

from common.lite_core import (
    COMMON_DIR,
    ADMIN_DIR,
    KIOSK_DIR,
    CLASSIFIER_PATH,
    DB_PATH,
    KNOWN_FACES_DIR,
    SFACE_PATH,
    YUNET_PATH,
)

REQUIRED_FACULTY = {
    "id",
    "faculty_id",
    "name",
    "department",
    "designation",
    "image_path",
    "email",
    "phone",
    "active",
    "created_at",
}
REQUIRED_ATTENDANCE = {
    "id",
    "faculty_id_fk",
    "date",
    "check_in_time",
    "check_out_time",
    "status",
    "late",
    "working_minutes",
}
REQUIRED_ADMIN = {"id", "username", "password_hash", "created_at"}


def check_file(path, label):
    ok = os.path.isfile(path) or os.path.isdir(path)
    print(f"{'OK' if ok else 'MISSING':8s} {label}: {path}")
    return ok


def columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def main():
    print("BCREC face-attendance deployment preflight")
    print("=" * 48)
    print(f"OS          : {platform.system()} {platform.release()}")
    print(f"Machine     : {platform.machine()}")
    print(f"Python      : {platform.python_version()}")
    print(f"OpenCV      : {cv2.__version__}")
    print(f"NumPy       : {np.__version__}")
    print(f"Common      : {COMMON_DIR}")
    print(f"Admin       : {ADMIN_DIR}")
    print(f"Kiosk       : {KIOSK_DIR}")
    print()

    all_ok = True
    all_ok &= check_file(YUNET_PATH, "YuNet model")
    all_ok &= check_file(SFACE_PATH, "SFace model")
    all_ok &= check_file(KNOWN_FACES_DIR, "known_faces folder")
    all_ok &= check_file(DB_PATH, "attendance.db")
    all_ok &= check_file(CLASSIFIER_PATH, "face_classifier_lite.pkl")

    print()
    print("OpenCV APIs:")
    detector_api = hasattr(cv2, "FaceDetectorYN")
    recognizer_api = hasattr(cv2, "FaceRecognizerSF")
    print(f"{'OK' if detector_api else 'MISSING':8s} FaceDetectorYN")
    print(f"{'OK' if recognizer_api else 'MISSING':8s} FaceRecognizerSF")
    all_ok &= detector_api and recognizer_api

    photos = []
    if os.path.isdir(KNOWN_FACES_DIR):
        for name in os.listdir(KNOWN_FACES_DIR):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                photos.append(name)
    print(f"\nKnown face images: {len(photos)}")
    if not photos:
        print("WARNING   No face photos are present.")

    if os.path.isfile(DB_PATH):
        print("\nDatabase schema:")
        conn = sqlite3.connect(DB_PATH)
        try:
            existing_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            print(f"{'OK' if 'faculty' in existing_tables else 'MISSING':8s} faculty table")
            print(f"{'OK' if 'attendance' in existing_tables else 'MISSING':8s} attendance table")
            print(f"{'OK' if 'admin_users' in existing_tables else 'MISSING':8s} admin_users table")

            if "faculty" in existing_tables:
                missing = REQUIRED_FACULTY - columns(conn, "faculty")
                if missing:
                    all_ok = False
                    print(f"ERROR     faculty missing columns: {sorted(missing)}")
            if "attendance" in existing_tables:
                missing = REQUIRED_ATTENDANCE - columns(conn, "attendance")
                if missing:
                    all_ok = False
                    print(f"ERROR     attendance missing columns: {sorted(missing)}")
            if "admin_users" in existing_tables:
                missing = REQUIRED_ADMIN - columns(conn, "admin_users")
                if missing:
                    all_ok = False
                    print(f"ERROR     admin_users missing columns: {sorted(missing)}")
        finally:
            conn.close()

    print()
    print("Result:")
    if all_ok:
        print("READY      Required runtime files and OpenCV face APIs are present.")
        return 0

    print("CHECK      One or more required items are missing or invalid.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
