"""Loads a trained PPO checkpoint and watches the agent play in a window.

Usage:
    python scripts/play.py models/ppo_firered_500000_steps.zip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from environment import make_env  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    vec = DummyVecEnv([lambda: make_env(render_mode="rgb_array")])
    vec = VecFrameStack(vec, n_stack=config.N_STACK, channels_order="last")
    vec = VecTransposeImage(vec)

    model = PPO.load(args.checkpoint, env=vec, device=config.DEVICE)
    obs = vec.reset()

    win = "FireRed — agent play"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 480, 320)

    delay = 1.0 / args.fps
    try:
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec.step(action)
            frame = vec.envs[0].unwrapped.render() if hasattr(vec.envs[0], "unwrapped") \
                    else vec.envs[0].render()
            if frame is not None:
                cv2.imshow(win, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(delay)
    finally:
        cv2.destroyAllWindows()
        vec.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
