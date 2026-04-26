"""
FireRedEnv: gymnasium.Env wrapping mGBA running Pokemon FireRed (GBA).

Observation space: Box(0, 255, (72, 80, 1), uint8) — single grayscale frame.
                   train.py wraps this in VecFrameStack(n=3) → (72, 80, 3).

Action space:      Discrete(7) — UP DOWN LEFT RIGHT A B START

Episode lifecycle:
  First reset()  → launches mGBA subprocess with ROM + initial save state.
  Later resets() → sends Shift+F1 hotkey to reload the same save state.
  close()        → terminates the mGBA process.
"""
import logging
import subprocess
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    ROM_PATH, STATE_PATH, MGBA_EXE, MGBA_SCALE,
    MGBA_LAUNCH_WAIT, MGBA_RESET_WAIT, MGBA_FAST_FORWARD,
    MGBA_WINDOW_TITLE,
    OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS, N_ACTIONS,
    MAX_STEPS_PER_EPISODE,
)
from environment.screen_capture import ScreenCapture
from environment.input_handler import InputHandler
from environment.reward_calculator import RewardCalculator
from utils.frame_processor import FrameProcessor
from utils.hash_tracker import HashTracker

logger = logging.getLogger(__name__)


class FireRedEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.render_mode = render_mode

        self._capture    = ScreenCapture()
        self._inputs     = InputHandler()
        self._processor  = FrameProcessor()
        self._tracker    = HashTracker()
        self._reward_calc = RewardCalculator()

        self._step_count:    int = 0
        self._episode_count: int = 0
        self._grace_steps:   int = 0   # death detection disabled for first N steps
        self._mgba_proc: subprocess.Popen | None = None

        # Cached per-step outputs (set in reset + step to avoid double processing)
        self._last_bgr: np.ndarray | None = None
        self._last_obs: np.ndarray | None = None

    # ── reset ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Always restart mGBA — the -t flag loads the save state cleanly at boot,
        # avoiding any in-game menu interaction. At ~40 min per episode the ~3s
        # restart cost is negligible.
        self._restart_mgba()

        self._step_count  = 0
        self._grace_steps = 60   # ignore death detection for first 60 steps (loading screen)
        self._episode_count += 1
        self._processor.reset()
        self._tracker.reset_episode()
        self._reward_calc.reset()

        obs = self._capture_obs()
        return obs, {}

    def _restart_mgba(self):
        """
        Kill any existing mGBA process, then launch a fresh one with the save
        state loaded via the -t flag. No keyboard shortcuts or menus required.
        """
        if self._mgba_proc and self._mgba_proc.poll() is None:
            self._mgba_proc.terminate()
            try:
                self._mgba_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._mgba_proc.kill()
            time.sleep(0.5)

        cmd = [
            str(MGBA_EXE),
            f"-{MGBA_SCALE}",
            "-t", str(STATE_PATH),
            "-C", f"fastForwardRatio={MGBA_FAST_FORWARD}",
            "-C", "fastForward=1",   # start with fast-forward enabled
            str(ROM_PATH),
        ]
        logger.info("Starting mGBA (episode %d): %s", self._episode_count + 1, " ".join(cmd))
        self._mgba_proc = subprocess.Popen(cmd)

        found = self._capture.find_window(timeout=MGBA_LAUNCH_WAIT + 5.0)
        if not found:
            raise RuntimeError(
                f"mGBA window '{MGBA_WINDOW_TITLE}' did not appear. "
                "Check MGBA_EXE path in config.py."
            )

        time.sleep(MGBA_LAUNCH_WAIT)
        self._capture.activate_window()

    # ── step ───────────────────────────────────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._inputs.send_action(action)

        bgr = self._capture.grab()
        self._last_bgr = bgr
        obs, frame_hash, is_anim = self._processor.process(bgr)
        self._last_obs = obs

        is_new, is_stuck, is_stall = self._tracker.update(frame_hash)
        reward, info = self._reward_calc.compute(bgr, is_new, is_stuck, is_stall, is_anim)

        self._step_count += 1
        if self._grace_steps > 0:
            self._grace_steps -= 1
            info.pop("died", None)   # suppress death signal during loading screen
        truncated  = self._step_count >= MAX_STEPS_PER_EPISODE
        terminated = False  # blacking out in Pokemon just warps you to the Pokemon Center
                            # — penalise it but keep the episode running

        info["step"]           = self._step_count
        info["episode"]        = self._episode_count
        info["total_explored"] = len(self._tracker.global_seen)

        return obs, reward, terminated, truncated, info

    # ── helpers ────────────────────────────────────────────────────────────

    def _capture_obs(self) -> np.ndarray:
        bgr = self._capture.grab()
        self._last_bgr = bgr
        obs, _, _ = self._processor.process(bgr)
        self._last_obs = obs
        return obs

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array" and self._last_bgr is not None:
            import cv2
            return cv2.cvtColor(self._last_bgr, cv2.COLOR_BGR2RGB)
        return None

    def close(self):
        if self._mgba_proc and self._mgba_proc.poll() is None:
            self._mgba_proc.terminate()
            try:
                self._mgba_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._mgba_proc.kill()
        self._capture.close()
        super().close()
