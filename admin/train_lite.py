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

# Make this file's own folder and the shared ../common folder importable, however
# it is started (python app.py, python admin/app.py, python -m admin.app, ...).
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.join(_here, "..", "common"), _here):
    _p = _os.path.normpath(_p)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import os
import pickle

import cv2
import numpy as np
from lite_core import CLASSIFIER_PATH, FaceEngine, largest_face, list_reference_photos, verify_person

AUG_PER_PHOTO = int(os.environ.get("AUG_PER_PHOTO", "4"))
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


def train(output_path=CLASSIFIER_PATH, progress=None, log=print):
    """
    Train on every photo in known_faces/ and write the bundle to output_path.

    The file is written to a temporary name first and moved into place only
    when complete, so a crash can never leave a half-written classifier.
    progress(done, total, faculty_id) is called after each photo.
    Raises TrainingError if training cannot finish.
    Returns {"people": n, "labels": [...], "holdout_accuracy": fraction of people passing the self-check}.
    """
    global rng
    rng = np.random.default_rng(42)  # same results every run

    photos = list_reference_photos()
    if not photos:
        raise TrainingError("No face photos found in known_faces/. Enrol faculty first.")

    engine = FaceEngine()
    total = sum(len(paths) for paths in photos.values())
    done = 0
    orig, aug = {}, {}

    for fid, paths in photos.items():
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
            orig.setdefault(fid, []).append(emb)
            for _ in range(AUG_PER_PHOTO):          # lighting / small rotation variants
                augmented_emb = embed_image(engine, augment(img))
                if augmented_emb is not None:
                    aug.setdefault(fid, []).append(augmented_emb)

    labels = sorted(orig)
    if not labels:
        raise TrainingError("No usable faces found in the photos.")
    orig = {k: np.array(v, np.float32) for k, v in orig.items()}
    refs = {k: np.vstack([orig[k]] + ([np.array(aug[k], np.float32)] if k in aug else [])) for k in labels}

    bundle = {
        "classifier": None,          # matching is cosine-nearest-reference (see lite_core.identify)
        "refs": refs,                # original + augmented embeddings, used by the kiosk
        "orig": orig,                # original photos only, used for the self-check
        "model": "SFace",
        "opencv_version": cv2.__version__,
        "format": 2,
    }

    # Self-check of the trained data, exactly as the kiosk would match it.
    passed = 0
    for fid in labels:
        ok, detail = verify_person(bundle, fid)
        passed += ok
        if not ok:
            log(f"  ! self-check failed for {fid}: {detail}")
    holdout_accuracy = passed / len(labels)
    log(f"\nSelf-check (each photo vs the others): {passed}/{len(labels)} people recognised")

    tmp_path = output_path + ".part"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(bundle, f)
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
