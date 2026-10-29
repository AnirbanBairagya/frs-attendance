"""Lightweight MiniFAS ONNX anti-spoof engine."""
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.onnx")
REAL_THRESHOLD = float(os.environ.get("ANTI_SPOOF_THRESHOLD", "0.45"))
TEMPORAL_FRAMES = int(os.environ.get("ANTI_SPOOF_FRAMES", "2"))
CROP_SCALE = 2.7
INPUT_SIZE = 128

class AntiSpoofEngine:
    def __init__(self):
        if not os.path.isfile(MODEL_PATH):
            raise SystemExit(f"Missing anti-spoof model: {MODEL_PATH}")
        try:
            self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        except Exception as exc:
            raise SystemExit(f"Could not load anti-spoof model:\n{exc}")

    @staticmethod
    def crop_face(frame, face, scale=CROP_SCALE):
        h, w = frame.shape[:2]
        x, y, fw, fh = [float(v) for v in face[:4]]
        if fw <= 1 or fh <= 1:
            return None
        cx, cy = x + fw / 2.0, y + fh / 2.0
        nw, nh = fw * scale, fh * scale
        x1 = max(0, int(cx - nw / 2.0))
        y1 = max(0, int(cy - nh / 2.0))
        x2 = min(w, int(cx + nw / 2.0))
        y2 = min(h, int(cy + nh / 2.0))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None

    @staticmethod
    def make_input(crop):
        img = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    @staticmethod
    def softmax(logits):
        logits = logits.astype(np.float32)
        logits -= np.max(logits, axis=1, keepdims=True)
        expv = np.exp(logits)
        return expv / np.sum(expv, axis=1, keepdims=True)

    def score(self, frame, face):
        crop = self.crop_face(frame, face)
        if crop is None:
            return None
        self.net.setInput(self.make_input(crop))
        output = self.net.forward()
        logits = np.asarray(output).reshape(1, -1)
        probs = self.softmax(logits)[0]
        return {"real": float(probs[0]), "spoof": float(probs[1])}
