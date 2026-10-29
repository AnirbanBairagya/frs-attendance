"""
Train the lightweight SFace classifier from photos in known_faces/.

Run this after adding or changing faculty photos.
For Raspberry Pi deployment, training may be done on the development PC and
face_classifier_lite.pkl can be copied to the Pi, provided the same models are
used there.
"""

import os
import pickle

import cv2
import numpy as np
from sklearn.svm import SVC

from common.lite_core import CLASSIFIER_PATH, FaceEngine, largest_face, list_reference_photos

AUG_PER_PHOTO = int(os.environ.get("AUG_PER_PHOTO", "8"))
rng = np.random.default_rng(42)


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


def main():
    photos = list_reference_photos()
    if not photos:
        raise SystemExit("No photos in known_faces/. Enrol faculty first.")

    engine = FaceEngine()
    Xtr, ytr, Xte, yte, refs = [], [], [], [], {}

    for fid, paths in photos.items():
        held_out = paths[-1] if len(paths) >= 2 else None
        print(f"{fid}: {len(paths)} photo(s)")

        for path in paths:
            img = cv2.imread(path)
            emb = None if img is None else embed_image(engine, img)
            if emb is None:
                print(f"  ! no face found in {path}, skipped")
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
        raise SystemExit("No usable faces found.")

    if len(labels) < 2:
        print("Only one person enrolled: cosine matching only, no SVM.")
        clf = None
    else:
        if Xte and set(ytr) == set(labels):
            probe = make_clf().fit(np.array(Xtr), ytr)
            acc = probe.score(np.array(Xte), yte)
            print(f"\nHeld-out accuracy (one unseen photo per person): {acc * 100:.1f}%")
            print("  (small sample; validate with live scans as well)")

        Xall = np.array(Xtr + Xte)
        yall = ytr + yte
        clf = make_clf().fit(Xall, yall)

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "refs": refs,
                "model": "SFace",
                "opencv_version": cv2.__version__,
            },
            f,
        )

    print(f"\nSaved {CLASSIFIER_PATH} ({len(labels)} people)")


if __name__ == "__main__":
    main()
