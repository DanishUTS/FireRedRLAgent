"""Pokemon FireRed gymnasium env, headless via stable-retro + libretro mGBA core."""

from __future__ import annotations

import hashlib

import gymnasium as gym
import numpy as np
import stable_retro as retro

import config
from environment.memory_reader import MemoryReader
from environment.reward_shaper import RewardShaper, RewardWeights
from environment.wrappers import (
    DISCRETE_ACTIONS,
    DiscreteActions,
    FrameSkip,
    Grayscale84,
    StatusBarOverlay,
)
from utils.hash_tracker import HashTracker
from utils.integrations import GAME_NAME, has_start_state, register_custom_integration


def _make_base_env(render_mode: str | None = None) -> retro.RetroEnv:
    """Build the raw stable-retro env (no wrappers)."""
    register_custom_integration()
    state = retro.State.DEFAULT if has_start_state() else retro.State.NONE
    return retro.make(
        game=GAME_NAME,
        state=state,
        use_restricted_actions=retro.Actions.ALL,
        inttype=retro.data.Integrations.ALL,
        render_mode=render_mode,
    )


class FireRedEnv(gym.Env):
    """Wraps the libretro env, owns memory reader + reward shaper.

    The wrapper stack on top is added by `make_env()` so SubprocVecEnv workers
    and the eval/play scripts get an identical pipeline.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, render_mode: str | None = None,
                 weights: RewardWeights | None = None):
        super().__init__()
        self._retro = _make_base_env(render_mode=render_mode)
        self.action_space = self._retro.action_space
        self.observation_space = self._retro.observation_space
        self.render_mode = render_mode

        self._reader = MemoryReader()
        self._shaper = RewardShaper(weights)
        self._hash_tracker = HashTracker()
        self._step_count = 0
        self._last_state: dict = {}

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = self._retro.reset(seed=seed, options=options)
        self._shaper.reset()
        self._hash_tracker.reset_episode()
        self._step_count = 0
        self._last_state = self._reader.read(self._retro.get_ram())
        info = self._merge_info(info)
        return obs, info

    def step(self, action):
        obs, _retro_reward, terminated, truncated, info = self._retro.step(action)
        self._step_count += 1

        gs = self._reader.read(self._retro.get_ram())
        self._last_state = gs

        is_new, _is_stuck, _is_stall = self._hash_tracker.update(_md5(obs))

        # Enemy HP fraction lives in the battle struct, which moves around in
        # memory. Pass None for now; we'll wire that in once a battle parser is
        # in place. The shaper handles None gracefully.
        reward = self._shaper.step(gs, is_new_hash=is_new, enemy_hp_frac=None)

        if self._step_count >= config.MAX_STEPS_PER_EPISODE:
            truncated = True

        info = self._merge_info(info)
        return obs, reward, terminated, truncated, info

    def _merge_info(self, info: dict) -> dict:
        info = dict(info)
        gs = self._last_state
        info.update({
            "x": gs.get("x", 0),
            "y": gs.get("y", 0),
            "map_id": gs.get("map_id", 0),
            "sum_levels": gs.get("sum_levels", 0),
            "badge_count": gs.get("badge_count", 0),
            "hp_frac": gs.get("hp_frac", 0.0),
            "coords_seen": self._shaper.coords_seen_count,
            "reward_components": dict(self._shaper.components),
        })
        return info

    def render(self):
        return self._retro.render()

    def close(self):
        self._retro.close()

    @property
    def status_overlay(self) -> dict:
        gs = self._last_state
        return {
            "hp_frac": gs.get("hp_frac", 0.0),
            "sum_levels": gs.get("sum_levels", 0),
            "badge_count": gs.get("badge_count", 0),
            "explore_count": self._shaper.coords_seen_count,
        }


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


class _OverlayUpdater(gym.Wrapper):
    """Pulls live status from FireRedEnv into StatusBarOverlay each step."""

    def __init__(self, env: gym.Env, overlay: StatusBarOverlay, base: FireRedEnv):
        super().__init__(env)
        self._overlay = overlay
        self._base = base
        self.observation_space = overlay.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sync()
        return self._overlay.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._sync()
        return self._overlay.observation(obs), reward, terminated, truncated, info

    def _sync(self) -> None:
        s = self._base.status_overlay
        self._overlay.update_state(
            hp_frac=s["hp_frac"],
            sum_levels=s["sum_levels"],
            badge_count=s["badge_count"],
            explore_count=s["explore_count"],
        )


def make_env(render_mode: str | None = None,
             weights: RewardWeights | None = None) -> gym.Env:
    """Build the full wrapper stack used for both training and eval."""
    base = FireRedEnv(render_mode=render_mode, weights=weights)
    env: gym.Env = base
    env = FrameSkip(env, skip=config.FRAME_SKIP)
    env = DiscreteActions(env, combos=DISCRETE_ACTIONS)
    env = Grayscale84(env, size=(config.OBS_HEIGHT, config.OBS_WIDTH))
    overlay = StatusBarOverlay(env)
    env = _OverlayUpdater(env, overlay, base)
    return env
