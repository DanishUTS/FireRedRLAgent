"""Phase 2 training: parallel PPO on stable-retro / mGBA, GPU.

Usage:
  python train.py                  # fresh start, 12 envs, full schedule
  python train.py --resume <zip>   # resume a checkpoint
  python train.py --smoke          # 10k step smoke test on 4 envs
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecVideoRecorder,
)

import config
from environment import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# ── Callbacks ────────────────────────────────────────────────────────────────


class RewardLoggingCallback(BaseCallback):
    """Logs per-component reward + game-state stats to TensorBoard each rollout."""

    def __init__(self):
        super().__init__()
        self._comp_sum: dict[str, float] = defaultdict(float)
        self._comp_n: int = 0
        self._sum_levels: list[int] = []
        self._badges: list[int] = []
        self._coords: list[int] = []
        self._map_ids: set[int] = set()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            comps = info.get("reward_components", {})
            for k, v in comps.items():
                self._comp_sum[k] += float(v)
            if comps:
                self._comp_n += 1
            if "sum_levels" in info:
                self._sum_levels.append(info["sum_levels"])
                self._badges.append(info["badge_count"])
                self._coords.append(info["coords_seen"])
                self._map_ids.add(info["map_id"])
        return True

    def _on_rollout_end(self) -> None:
        if self._comp_n:
            for k, total in self._comp_sum.items():
                self.logger.record(f"firered_reward/{k}", total / self._comp_n)
        if self._sum_levels:
            self.logger.record("firered/mean_sum_levels",
                               float(np.mean(self._sum_levels)))
            self.logger.record("firered/max_sum_levels",
                               float(np.max(self._sum_levels)))
            self.logger.record("firered/mean_badges",
                               float(np.mean(self._badges)))
            self.logger.record("firered/coords_seen",
                               float(np.max(self._coords)))
            self.logger.record("firered/maps_visited", float(len(self._map_ids)))
        self._comp_sum.clear()
        self._comp_n = 0
        self._sum_levels.clear()
        self._badges.clear()
        self._coords.clear()


# ── Env factories ────────────────────────────────────────────────────────────


def _env_fn(rank: int, seed: int):
    def _init():
        env = make_env()
        env.reset(seed=seed + rank)
        return env
    return _init


def build_train_env(n_envs: int, seed: int) -> VecMonitor:
    if n_envs == 1:
        vec = DummyVecEnv([_env_fn(0, seed)])
    else:
        vec = SubprocVecEnv([_env_fn(i, seed) for i in range(n_envs)],
                            start_method="spawn")
    vec = VecMonitor(vec, filename=str(config.LOG_DIR / "monitor.csv"))
    vec = VecFrameStack(vec, n_stack=config.N_STACK, channels_order="last")
    return vec


def build_eval_env(seed: int) -> VecMonitor:
    vec = DummyVecEnv([_env_fn(0, seed + 999)])
    vec = VecMonitor(vec)
    vec = VecFrameStack(vec, n_stack=config.N_STACK, channels_order="last")
    return vec


# ── Model ────────────────────────────────────────────────────────────────────


def build_model(env, resume: Path | None) -> PPO:
    policy_kwargs = {
        "features_extractor_kwargs": {"features_dim": config.FEATURES_DIM},
        "net_arch": dict(pi=[512, 512], vf=[512, 512]),
    }
    if resume:
        log.info("Resuming from %s", resume)
        return PPO.load(
            resume, env=env, device=config.DEVICE,
            custom_objects={"learning_rate": config.LEARNING_RATE,
                            "clip_range": config.CLIP_RANGE},
        )
    return PPO(
        "CnnPolicy",
        env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        tensorboard_log=str(config.LOG_DIR),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=config.SEED,
        device=config.DEVICE,
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="10k-step smoke test with 4 envs")
    p.add_argument("--n-envs", type=int, default=None)
    args = p.parse_args()

    config.MODEL_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)

    n_envs = args.n_envs or (4 if args.smoke else config.N_ENVS)
    total_steps = 10_000 if args.smoke else config.TOTAL_TIMESTEPS

    run_name = f"{config.CHECKPOINT_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}"
    log.info("Run %s | envs=%d | total_steps=%s | device=%s",
             run_name, n_envs, f"{total_steps:,}", config.DEVICE)

    train_env = build_train_env(n_envs=n_envs, seed=config.SEED)
    eval_env = build_eval_env(seed=config.SEED)

    model = build_model(train_env, args.resume)

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(1, config.CHECKPOINT_FREQ // n_envs),
            save_path=str(config.MODEL_DIR),
            name_prefix=config.CHECKPOINT_PREFIX,
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(config.MODEL_DIR / "best"),
            log_path=str(config.LOG_DIR / "eval"),
            eval_freq=max(1, config.EVAL_FREQ // n_envs),
            n_eval_episodes=config.EVAL_EPISODES,
            deterministic=False,
            verbose=1,
        ),
        RewardLoggingCallback(),
    ])

    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=(args.resume is None),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        log.info("Interrupted — saving final checkpoint.")
    finally:
        model.save(str(config.MODEL_DIR / f"{config.CHECKPOINT_PREFIX}_final"))
        train_env.close()
        eval_env.close()
        log.info("Saved final model. tensorboard --logdir %s", config.LOG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
