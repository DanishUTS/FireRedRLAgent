"""
Single source of truth for all configuration.
Change values here — nothing else hardcodes numbers or paths.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.resolve()
ROM_PATH   = ROOT_DIR / "PokemonFireRed.gba"
STATE_PATH = ROOT_DIR / "states" / "start.ss0"
MODEL_DIR  = ROOT_DIR / "models"
LOG_DIR    = ROOT_DIR / "logs"

# Adjust if mGBA is installed elsewhere — run calibrate.py to verify
MGBA_EXE = Path("C:/Program Files/mGBA/mGBA.exe")

# ── Emulator ───────────────────────────────────────────────────────────────
MGBA_SCALE        = 3          # 3x scale = 720×480 game canvas
MGBA_WINDOW_TITLE = "mGBA"     # substring matched by pygetwindow
MGBA_LAUNCH_WAIT  = 4.0        # seconds to wait after Popen before capturing
MGBA_RESET_WAIT   = 1.5        # seconds after Shift+F1 state reload
MGBA_FAST_FORWARD = 16         # fast-forward multiplier (16x real-time speed)

# ── Action space ───────────────────────────────────────────────────────────
# mGBA default GBA bindings: A=X, B=Z, Start=Enter, D-pad=Arrow keys
# Verify in mGBA → Settings → Input before training
ACTION_KEYS = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "x",        # GBA A button
    5: "z",        # GBA B button
    6: "return",   # GBA Start button
}
N_ACTIONS = len(ACTION_KEYS)  # 7

# At 16x fast-forward, 24 GBA frames ≈ 0.025s real time. Use 0.04s for safety margin.
ACTION_HOLD_SECS  = 0.04
ACTION_PAUSE_SECS = 0.02

# ── Observation ────────────────────────────────────────────────────────────
OBS_HEIGHT   = 72
OBS_WIDTH    = 80
OBS_CHANNELS = 1      # single grayscale frame; VecFrameStack adds N_STACK
N_STACK      = 3      # frames stacked → (72, 80, 3) fed to CnnPolicy

# ── Episode ────────────────────────────────────────────────────────────────
MAX_STEPS_PER_EPISODE = 40_960  # ~40 min real time at 16× speed

# ── Reward ─────────────────────────────────────────────────────────────────
REWARD_NEW_FRAME      = 1.0    # +1 per genuinely new screen hash
REWARD_STUCK_PENALTY  = -0.1   # per step when stuck in one area
REWARD_STALL_PENALTY  = -0.5   # per step when stalling in battle
REWARD_DEATH_PENALTY  = -5.0   # on blackout / whiteout detection

# Pixel-sum absolute diff between consecutive frames below this → animation
# (not a genuinely new game state — suppress exploration reward)
ANIMATION_DIFF_THRESH = 500

# Stuck detection: if same hash appears this many times in the window → stuck
STUCK_WINDOW_SIZE  = 100
STUCK_REPEAT_THRESH = 20

# Battle stall: same hash this many consecutive steps → stalling
BATTLE_STALL_STEPS = 30

# ── PPO Hyperparameters ────────────────────────────────────────────────────
LEARNING_RATE    = 2.5e-4
N_STEPS          = 2_048      # rollout length (single env)
BATCH_SIZE       = 512
N_EPOCHS         = 4
GAMMA            = 0.998      # high discount: Pallet→Brock takes hundreds of steps
GAE_LAMBDA       = 0.95
CLIP_RANGE       = 0.2
ENT_COEF         = 0.01       # entropy bonus to maintain exploration
VF_COEF          = 0.5
MAX_GRAD_NORM    = 0.5
FEATURES_DIM     = 512        # NatureCNN output dimension

TOTAL_TIMESTEPS  = 10_000_000
CHECKPOINT_FREQ  = 10_000
CHECKPOINT_PREFIX = "ppo_firered"
SEED   = 42
DEVICE = "cuda"

# Set to a .zip path to resume training from a checkpoint, or None to start fresh
RESUME_CHECKPOINT: str | None = None
