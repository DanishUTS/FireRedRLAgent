"""
PPO training script for the FireRed RL Agent.

Usage:
  python train.py                                          # fresh start
  python train.py --resume models/ppo_firered_10000_steps.zip  # resume
"""
import argparse
import logging
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from config import (
    BATCH_SIZE, CHECKPOINT_FREQ, CHECKPOINT_PREFIX,
    CLIP_RANGE, DEVICE, ENT_COEF, FEATURES_DIM,
    GAE_LAMBDA, GAMMA, LEARNING_RATE, LOG_DIR,
    MAX_GRAD_NORM, MODEL_DIR, N_EPOCHS, N_STACK, N_STEPS,
    RESUME_CHECKPOINT, SEED, TOTAL_TIMESTEPS, VF_COEF,
)
from environment.fire_red_env import FireRedEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Callback ───────────────────────────────────────────────────────────────

class RewardLoggingCallback(BaseCallback):
    """
    Logs per-component reward breakdowns to TensorBoard each rollout.
    SB3 only logs episode mean reward by default; this gives visibility into
    which reward signals are actually firing (explore, stuck, hp, stall).
    Also logs total_explored (unique screen hashes seen) as a progress metric.
    """

    _KEYS = ("reward_explore", "reward_stuck", "reward_hp", "reward_stall",
             "total_reward", "enemy_hp", "player_hp", "total_explored")

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._accum: dict[str, list[float]] = {k: [] for k in self._KEYS}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in self._KEYS:
                if key in info:
                    self._accum[key].append(float(info[key]))
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self._accum.items():
            if values:
                self.logger.record(f"firered/{key}", sum(values) / len(values))
            self._accum[key].clear()


# ── Environment factory ────────────────────────────────────────────────────

def make_env():
    def _init():
        return FireRedEnv(render_mode=None)
    return _init


def build_env() -> VecFrameStack:
    """
    FireRedEnv → DummyVecEnv (n=1) → VecMonitor → VecFrameStack(3)

    VecMonitor records episode stats (length, reward) for SB3's ep_rew_mean log.
    VecFrameStack stacks the last 3 grayscale frames → shape (72, 80, 3).
    PPO('CnnPolicy') automatically applies VecTransposeImage → (3, 72, 80)
    which is the format NatureCNN expects.
    """
    env = DummyVecEnv([make_env()])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


# ── Model construction ─────────────────────────────────────────────────────

def build_model(env: VecFrameStack, resume_path: str | None) -> PPO:
    policy_kwargs = {
        "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
        "net_arch": dict(pi=[512, 512], vf=[512, 512]),
    }

    if resume_path:
        logger.info("Resuming from: %s", resume_path)
        model = PPO.load(
            resume_path,
            env=env,
            device=DEVICE,
            custom_objects={"learning_rate": LEARNING_RATE, "clip_range": CLIP_RANGE},
        )
    else:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            tensorboard_log=str(LOG_DIR),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=SEED,
            device=DEVICE,
        )
    return model


# ── Training loop ──────────────────────────────────────────────────────────

def train(resume_path: str | None = None):
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    run_name = f"{CHECKPOINT_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Run: %s", run_name)

    env = build_env()
    model = build_model(env, resume_path)

    callbacks = [
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=str(MODEL_DIR),
            name_prefix=CHECKPOINT_PREFIX,
            verbose=1,
        ),
        RewardLoggingCallback(),
    ]

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=(resume_path is None),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted — saving final checkpoint.")
    finally:
        model.save(str(MODEL_DIR / f"{CHECKPOINT_PREFIX}_final"))
        env.close()
        logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=RESUME_CHECKPOINT,
                        help="Path to a checkpoint .zip to resume training from")
    args = parser.parse_args()
    train(resume_path=args.resume)
