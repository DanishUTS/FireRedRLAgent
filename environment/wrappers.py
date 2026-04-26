"""Gymnasium wrappers for the stable-retro FireRed env."""

from __future__ import annotations

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Names of the GBA buttons in stable-retro's mGBA core action vector.
# Order from cores/mgba.json:
#   ["B", null, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", null, "L", "R"]
GBA_BUTTONS = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT",
               "A", None, "L", "R"]

# Compact discrete action set for FireRed:
#   0: NOOP, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT, 5: A, 6: B, 7: START
DISCRETE_ACTIONS: list[list[str]] = [
    [],          # 0 NOOP
    ["UP"],      # 1
    ["DOWN"],    # 2
    ["LEFT"],    # 3
    ["RIGHT"],   # 4
    ["A"],       # 5
    ["B"],       # 6
    ["START"],   # 7
]


class DiscreteActions(gym.ActionWrapper):
    """Maps a flat Discrete(N) into the multi-hot button array stable-retro expects."""

    def __init__(self, env: gym.Env, combos: list[list[str]] = DISCRETE_ACTIONS):
        super().__init__(env)
        self._combos = combos
        self._lookup = np.zeros((len(combos), len(GBA_BUTTONS)), dtype=np.int8)
        for i, buttons in enumerate(combos):
            for b in buttons:
                self._lookup[i, GBA_BUTTONS.index(b)] = 1
        self.action_space = spaces.Discrete(len(combos))

    def action(self, act: int) -> np.ndarray:
        return self._lookup[int(act)].copy()


class FrameSkip(gym.Wrapper):
    """Repeats the chosen action `skip` frames; returns the last observation
    and the summed reward. One grid step in FireRed = ~24 frames."""

    def __init__(self, env: gym.Env, skip: int = 24):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        obs = None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class Grayscale84(gym.ObservationWrapper):
    """RGB → grayscale → resized to (84, 84, 1) uint8 — matches NatureCNN input."""

    def __init__(self, env: gym.Env, size: tuple[int, int] = (84, 84)):
        super().__init__(env)
        self._size = size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size[0], size[1], 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self._size, interpolation=cv2.INTER_AREA)
        return resized[:, :, None]


class StatusBarOverlay(gym.ObservationWrapper):
    """Bakes interpretable scalar game state into the bottom rows of the
    observation as bar graphs. The CNN can learn to read these directly,
    so the policy gets game state without needing recurrence.

    Encodes 4 bars in the bottom 8 rows:
      row 0-1: HP fraction
      row 2-3: party-level progress (0..100 → 0..1)
      row 4-5: badge progress (count / 8)
      row 6-7: exploration fraction (capped at 1000 unique tiles)
    """

    BAR_ROWS = 8
    EXPLORATION_CAP = 1000

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # observation_space is (H, W, 1) from Grayscale84 — unchanged
        self.observation_space = env.observation_space
        self._state = {
            "hp_frac": 0.0,
            "level_frac": 0.0,
            "badge_frac": 0.0,
            "explore_frac": 0.0,
        }

    def update_state(self, hp_frac: float, sum_levels: int, badge_count: int,
                     explore_count: int) -> None:
        self._state["hp_frac"] = float(np.clip(hp_frac, 0.0, 1.0))
        self._state["level_frac"] = float(np.clip(sum_levels / 100.0, 0.0, 1.0))
        self._state["badge_frac"] = float(np.clip(badge_count / 8.0, 0.0, 1.0))
        self._state["explore_frac"] = float(
            np.clip(explore_count / self.EXPLORATION_CAP, 0.0, 1.0)
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        h, w, _ = out.shape
        bars = [
            self._state["hp_frac"],
            self._state["level_frac"],
            self._state["badge_frac"],
            self._state["explore_frac"],
        ]
        for i, frac in enumerate(bars):
            row_start = h - self.BAR_ROWS + i * 2
            fill = int(round(frac * w))
            out[row_start:row_start + 2, :fill, 0] = 255
            out[row_start:row_start + 2, fill:, 0] = 0
        return out
