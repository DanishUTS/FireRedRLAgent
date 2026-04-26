"""Single source of truth for Phase 2 (stable-retro + parallel PPO)."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent.resolve()
ROM_PATH  = ROOT_DIR / "PokemonFireRed.gba"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR   = ROOT_DIR / "runs"          # TensorBoard root
INTEGRATION_DIR = ROOT_DIR / "retro_integration"
GAME_NAME = "PokemonFireRed-GbAdvance"

# ── Parallelism ──────────────────────────────────────────────────────────────
# 16 cores total → 12 envs leaves headroom for OS, PPO learner, TB
N_ENVS  = 12
DEVICE  = "cuda"           # CUDA 13.2 RTX 3060 Ti in WSL
SEED    = 42

# ── Episode ──────────────────────────────────────────────────────────────────
FRAME_SKIP            = 24       # one grid space per discrete action
MAX_STEPS_PER_EPISODE = 20_480   # ~20 min of in-game time

# ── Observation ──────────────────────────────────────────────────────────────
OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS = 84, 84, 1
N_STACK = 3                       # VecFrameStack → (3, 84, 84) for CnnPolicy

# ── PPO ──────────────────────────────────────────────────────────────────────
LEARNING_RATE = 2.5e-4
N_STEPS       = 2_048             # per env per update → batch = 12 × 2048 = 24,576
BATCH_SIZE    = 2_048             # 12 minibatches per epoch
N_EPOCHS      = 3
GAMMA         = 0.998
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
ENT_COEF      = 0.01
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
FEATURES_DIM  = 512               # NatureCNN output

TOTAL_TIMESTEPS  = 50_000_000
CHECKPOINT_FREQ  = 100_000        # global step count (across all envs)
CHECKPOINT_PREFIX = "ppo_firered"

# ── Eval / video ─────────────────────────────────────────────────────────────
EVAL_FREQ        = 50_000         # global steps between evals
EVAL_EPISODES    = 3
VIDEO_FREQ       = 200_000        # global steps between recorded videos
VIDEO_LENGTH     = 1_000          # frames per recorded video
