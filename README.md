# FireRedRLAgent

A reinforcement-learning agent that learns to play Pokemon FireRed (GBA) end-to-end from pixels. Immediate goal: beat the first gym (Brock). Long-term goal: progress as far through the game as the reward signal allows.

The agent runs headless via [stable-retro](https://github.com/Farama-Foundation/stable-retro) (libretro mGBA core), trains with PPO from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) on a CUDA GPU, and uses 12 parallel emulator processes to collect experience.

## Architecture

```
SubprocVecEnv (12 workers)
  └─ Monitor → FrameStack(3) → StatusBarOverlay → Grayscale84 → DiscreteActions(8)
       → FrameSkip(24) → FireRedEnv → stable-retro (libretro mGBA core)
                              ├─ MemoryReader  (parses get_ram() each step)
                              └─ RewardShaper  (memory-based, monotonic)
                              ↓
                        PPO CnnPolicy on CUDA
```

Reward is composed from values read directly out of mGBA RAM, so there's no pixel-sampling or HP-bar calibration step. Components include:

- Coordinate-tile exploration `(map_id, x, y)`
- Hash-based exploration (backup signal for visually unique screens)
- Monotonically-increasing party level (sqrt-scaled)
- Badge bonus
- HP-delta in battles
- Per-step time cost (discourages stalling)

## Setup

Linux or WSL with an NVIDIA GPU.

```bash
# 1. Clone + create venv
git clone <repo>
cd FireRedRLAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Provide your own legal ROM (US 1.0, BPRE, SHA1 41cb23d8...)
python scripts/import_rom.py /path/to/PokemonFireRed.gba

# 3. Capture an initial save state (one-time, interactive)
#    Play to your desired starting point (e.g. just after picking your starter
#    in Oak's lab) and press F2.
python scripts/capture_state.py
```

Verify GPU passthrough:
```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Train

```bash
python train.py --smoke          # 10k-step smoke test on 4 envs
python train.py                  # full run, 12 envs, 50M steps
python train.py --resume models/ppo_firered_500000_steps.zip
tensorboard --logdir runs/       # watch metrics (auto-refreshes)
```

Per-component reward breakdown, level/badge progress, number of unique tiles visited, and number of distinct maps reached are logged to TensorBoard alongside SB3's standard `ep_rew_mean` and learning-rate curves.

## Watch a trained agent

```bash
python scripts/play.py models/best/best_model.zip
```

## Repo layout

```
config.py                      # all hyperparameters + paths
train.py                       # PPO + SubprocVecEnv + callbacks

environment/
  fire_red_env.py              # gymnasium env wrapping stable-retro
  memory_reader.py             # FireRed RAM → structured game state
  reward_shaper.py             # memory-based multi-component reward
  wrappers.py                  # FrameSkip, DiscreteActions, Grayscale84, StatusBarOverlay

utils/
  integrations.py              # registers retro_integration/ with stable-retro
  hash_tracker.py              # backup hash-based exploration signal

retro_integration/
  PokemonFireRed-GbAdvance/    # rom.sha, metadata, data, scenario, start.state

scripts/
  import_rom.py                # one-shot ROM import
  capture_state.py             # interactive save-state grabber
  play.py                      # render a checkpoint playing the game

models/                        # checkpoints (gitignored)
runs/                          # TensorBoard logs (gitignored)
```

## ML topics covered

1. **Reinforcement Learning** — PPO from `stable-baselines3`
2. **Convolutional Neural Networks** — NatureCNN feature extractor
3. **Reward shaping** — multi-component, memory-derived, monotonic
4. **Parallel data collection** — `SubprocVecEnv`
5. **Experience stacking** — `VecFrameStack` for short-horizon memory

## Roadmap

- ⏭ Battle-state memory parser (enemy HP, in-battle flag) for tighter battle reward
- ⏭ Per-tile exploration heatmap + episode video grid
- ⏭ Recurrent policy (`sb3-contrib RecurrentPPO`) once short-horizon memory becomes the bottleneck
