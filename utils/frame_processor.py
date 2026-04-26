"""
Converts raw BGR screen captures into model-ready grayscale observations.
Returns observation, frame hash, and animation flag in a single call.
"""
import hashlib
import cv2
import numpy as np

from config import OBS_HEIGHT, OBS_WIDTH, ANIMATION_DIFF_THRESH


class FrameProcessor:
    """
    Pipeline per call: BGR (H,W,3) → grayscale → resize(80×72) → (72,80,1) uint8
    Also computes MD5 hash and detects minor animations vs genuine new states.
    """

    def __init__(self):
        self._last_gray: np.ndarray | None = None

    def process(self, bgr_frame: np.ndarray) -> tuple[np.ndarray, str, bool]:
        """
        Args:
            bgr_frame: Raw screen capture from mss, shape (H, W, 3), dtype uint8.

        Returns:
            obs:          (OBS_HEIGHT, OBS_WIDTH, 1) uint8 — ready for gym observation.
            frame_hash:   MD5 hex string uniquely identifying this frame.
            is_animation: True if pixel-sum diff vs previous frame is below threshold,
                          meaning this is a minor animation (sparkle, HP bar tick)
                          rather than a meaningfully new game state.
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        # cv2.resize takes (width, height)
        resized = cv2.resize(gray, (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA)
        obs = resized[:, :, np.newaxis]  # (72, 80, 1)

        frame_hash = hashlib.md5(resized.tobytes()).hexdigest()

        is_animation = False
        if self._last_gray is not None:
            diff = int(np.sum(np.abs(resized.astype(np.int16) - self._last_gray.astype(np.int16))))
            is_animation = diff < ANIMATION_DIFF_THRESH
        self._last_gray = resized.copy()

        return obs, frame_hash, is_animation

    def reset(self):
        self._last_gray = None
