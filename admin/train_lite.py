"""
Train the lightweight SFace classifier from photos in known_faces/.

The dashboard runs this for you right after a new faculty member is enrolled,
so you normally never start it by hand. It still works as a script:

    python train_lite.py                  # writes face_classifier_lite.pkl
    python train_lite.py --output x.pkl   # write somewhere else

For Raspberry Pi deployment, training may be done on the development PC and
face_classifier_lite.pkl can be copied to the Pi, provided the same models are
used there.
"""

import os as _os
import sys as _sys

# Shared code (lite_core.py, kiosk_proto.py, ...) lives in ../common
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))

import argparse
import os
import pickle

import cv2
import numpy as np
from sklearn.svm import SVC

from lite_core import CLASSIFIER_PATH, FaceEngine, largest_face, list_reference_photos

AUG_PER_PHOTO = int(os.environ.get("AUG_PER_PHOTO", "8"))
rng = np.random.default_rng(42)


class TrainingError(RuntimeError):
    """Training could not finish. The message is safe to show to an admin."""


def augment(img):
    out = cv2.convertScaleAbs(
        img,
        alpha=rng.uniform(0.7, 1.3),
        beta=rng.uniform(-40, 40),
    )
    h, w = out.shape[:2]
    m = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        rng.uniform(-12, 12),
        1.0,
    )
    return cv2.warpAffine(
        out,
        m,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )


def embed_image(engine, img):
    face = largest_face(engine.detect(img))
    return None if face is None else engine.embed(img, face)


def make_clf():
    return SVC(kernel="linear", C=1.0, probability=True)


def train(output_path=CLASSIFIER_PATH, progress=None, log=print):
    """
    Train on every photo in known_faces/ and write the bundle to output_path.

    The file is written to a temporary name first and moved into place only
    when complete, so a crash can never leave a half-written classifier.
    progress(done, total, faculty_id) is called after each photo.
    Raises TrainingError if training cannot finish.
    Returns {"people": n, "labels": [...], "holdout_accuracy": float | None}.
    """
    global rng
    rng = np.random.default_rng(42)  # same results every run

    photos = list_reference_photos()
    if not photos:
        raise TrainingError("No face photos found in known_faces/. Enrol faculty first.")

    engine = FaceEngine()
    total = sum(len(paths) for paths in photos.values())
    done = 0
    Xtr, ytr, Xte, yte, refs = [], [], [], [], {}

    for fid, paths in photos.items():
        held_out = paths[-1] if len(paths) >= 2 else None
        log(f"{fid}: {len(paths)} photo(s)")

        for path in paths:
            img = cv2.imread(path)
            emb = None if img is None else embed_image(engine, img)
            done += 1
            if progress:
                progress(done, total, fid)
            if emb is None:
                log(f"  ! no face found in {path}, skipped")
                continue

            refs.setdefault(fid, []).append(emb)
            samples = [emb]

            for _ in range(AUG_PER_PHOTO):
                augmented_emb = embed_image(engine, augment(img))
                if augmented_emb is not None:
                    samples.append(augmented_emb)

            X, y = (Xte, yte) if path == held_out else (Xtr, ytr)
            X.extend(samples)
            y.extend([fid] * len(samples))

    refs = {k: np.array(v, np.float32) for k, v in refs.items()}
    labels = sorted(refs)
    if not labels:
        raise TrainingError("No usable faces found in the photos.")

    holdout_accuracy = None
    if len(labels) < 2:
        log("Only one person enrolled: cosine matching only, no SVM.")
        clf = None
    else:
        if Xte and set(ytr) == set(labels):
            probe = make_clf().fit(np.array(Xtr), ytr)
            holdout_accuracy = float(probe.score(np.array(Xte), yte))
            log(f"\nHeld-out accuracy (one unseen photo per person): {holdout_accuracy * 100:.1f}%")
            log("  (small sample; validate with live scans as well)")

        clf = make_clf().fit(np.array(Xtr + Xte), ytr + yte)

    tmp_path = output_path + ".part"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(
                {
                    "classifier": clf,
                    "refs": refs,
                    "model": "SFace",
                    "opencv_version": cv2.__version__,
                },
                f,
            )
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    log(f"\nSaved {output_path} ({len(labels)} people)")
    return {"people": len(labels), "labels": labels, "holdout_accuracy": holdout_accuracy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=CLASSIFIER_PATH)
    args = ap.parse_args()
    try:
        train(args.output)
    except TrainingError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
