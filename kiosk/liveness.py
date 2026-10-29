"""Active random liveness challenge used after passive anti-spoofing."""
import math
import secrets
import time
import numpy as np

ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN")
ACTION_TEXT = {
    "LEFT": "Turn your head LEFT",
    "RIGHT": "Turn your head RIGHT",
    "UP": "Tilt your head UP",
    "DOWN": "Tilt your head DOWN",
}

class LivenessChallenge:
    def __init__(self, sequence_length=3, calibration_frames=8, action_timeout=3.0,
                 session_timeout=20.0, hold_frames=2, yaw_threshold=0.09,
                 pitch_threshold=0.05, center_yaw_threshold=0.045,
                 center_pitch_threshold=0.035):
        self.sequence_length = sequence_length
        self.calibration_frames = calibration_frames
        self.action_timeout = action_timeout
        self.session_timeout = session_timeout
        self.hold_frames = hold_frames
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.center_yaw_threshold = center_yaw_threshold
        self.center_pitch_threshold = center_pitch_threshold
        self.rng = secrets.SystemRandom()
        self.reset()

    def reset(self):
        self.active = False
        self.phase = "waiting"
        self.started_at = 0.0
        self.action_started_at = 0.0
        self.calibration = []
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.sequence = []
        self.step = 0
        self.matched_frames = 0
        self.last_prompt = "Look at the camera"
        self.last_failed = False

    @staticmethod
    def _pose(face):
        if face is None or len(face) < 15:
            return None
        x, y, w, h = [float(v) for v in face[:4]]
        if w <= 1 or h <= 1:
            return None
        p1 = np.asarray(face[5:7], dtype=np.float32)
        p2 = np.asarray(face[7:9], dtype=np.float32)
        nose = np.asarray(face[9:11], dtype=np.float32)
        if not np.all(np.isfinite(np.concatenate([p1, p2, nose]))):
            return None
        eye1, eye2 = sorted((p1, p2), key=lambda p: float(p[0]))
        d1 = float(np.linalg.norm(nose - eye1))
        d2 = float(np.linalg.norm(nose - eye2))
        if d1 < 1.0 or d2 < 1.0:
            return None
        yaw = math.log((d1 + 1e-6) / (d2 + 1e-6))
        pitch = (float(nose[1]) - y) / h
        return yaw, pitch

    def _fail(self, message):
        self.active = False
        self.phase = "failed"
        self.last_prompt = message
        self.last_failed = True
        return {"passed": False, "failed": True, "prompt": message,
                "progress": f"{self.step}/{self.sequence_length}"}

    def _start(self, now):
        self.active = True
        self.phase = "calibrating"
        self.started_at = now
        self.calibration = []
        self.sequence = self.rng.sample(list(ACTIONS), self.sequence_length)
        self.step = 0
        self.matched_frames = 0
        self.action_started_at = now
        self.last_prompt = "Look straight at the camera"
        self.last_failed = False

    def _target_reached(self, action, dyaw, dpitch):
        if action == "LEFT":
            return dyaw <= -self.yaw_threshold
        if action == "RIGHT":
            return dyaw >= self.yaw_threshold
        if action == "UP":
            return dpitch <= -self.pitch_threshold
        if action == "DOWN":
            return dpitch >= self.pitch_threshold
        return False

    @staticmethod
    def _centered(dyaw, dpitch, yaw_limit, pitch_limit):
        return abs(dyaw) <= yaw_limit and abs(dpitch) <= pitch_limit

    def update(self, face):
        now = time.monotonic()
        pose = self._pose(face)
        if pose is None:
            return {"passed": False, "failed": False,
                    "prompt": "Keep one clear face in view",
                    "progress": f"{self.step}/{self.sequence_length}"}
        yaw, pitch = pose
        if not self.active:
            self._start(now)
        if now - self.started_at > self.session_timeout:
            return self._fail("Liveness timeout - try again")
        if self.phase == "calibrating":
            self.calibration.append((yaw, pitch))
            if len(self.calibration) < self.calibration_frames:
                return {"passed": False, "failed": False,
                        "prompt": "Look straight at the camera",
                        "progress": f"0/{self.sequence_length}"}
            values = np.asarray(self.calibration, dtype=np.float32)
            self.baseline_yaw = float(np.median(values[:, 0]))
            self.baseline_pitch = float(np.median(values[:, 1]))
            self.phase = "action"
            self.step = 0
            self.matched_frames = 0
            self.action_started_at = now
        action = self.sequence[self.step]
        dyaw = yaw - self.baseline_yaw
        dpitch = pitch - self.baseline_pitch
        if self.phase == "action":
            if self._target_reached(action, dyaw, dpitch):
                self.matched_frames += 1
            else:
                self.matched_frames = 0
            if self.matched_frames >= self.hold_frames:
                self.phase = "return"
                self.matched_frames = 0
                self.action_started_at = now
                return {"passed": False, "failed": False,
                        "prompt": "Return to center",
                        "progress": f"{self.step + 1}/{self.sequence_length}"}
            if now - self.action_started_at > self.action_timeout:
                return self._fail("Movement not detected - try again")
            prompt = (f"Liveness {self.step + 1}/{self.sequence_length}: "
                      f"{ACTION_TEXT[action]}")
            self.last_prompt = prompt
            return {"passed": False, "failed": False, "prompt": prompt,
                    "progress": f"{self.step}/{self.sequence_length}"}
        if self.phase == "return":
            if self._centered(dyaw, dpitch, self.center_yaw_threshold, self.center_pitch_threshold):
                self.matched_frames += 1
            else:
                self.matched_frames = 0
            if self.matched_frames >= self.hold_frames:
                self.step += 1
                self.matched_frames = 0
                self.action_started_at = now
                if self.step >= self.sequence_length:
                    self.active = False
                    self.phase = "passed"
                    self.last_prompt = "Liveness verified"
                    return {"passed": True, "failed": False,
                            "prompt": "Liveness verified",
                            "progress": f"{self.sequence_length}/{self.sequence_length}"}
                return {"passed": False, "failed": False, "prompt": "Next challenge",
                        "progress": f"{self.step}/{self.sequence_length}"}
            if now - self.action_started_at > self.action_timeout:
                return self._fail("Return to center - try again")
            self.last_prompt = "Return to center"
            return {"passed": False, "failed": False, "prompt": "Return to center",
                    "progress": f"{self.step + 1}/{self.sequence_length}"}
        return {"passed": self.phase == "passed", "failed": self.phase == "failed",
                "prompt": self.last_prompt,
                "progress": f"{min(self.step, self.sequence_length)}/{self.sequence_length}"}
