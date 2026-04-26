"""
All reward components in one place, cleanly separated from the env loop.

Phase 1: screen-only signals (exploration hash, HP bar pixels, death detection).
Phase 2 stubs are present but raise NotImplementedError until memory reading lands.
"""
import numpy as np

from config import (
    REWARD_NEW_FRAME,
    REWARD_STUCK_PENALTY,
    REWARD_STALL_PENALTY,
    REWARD_DEATH_PENALTY,
    MGBA_SCALE,
)

# ── HP bar pixel coordinates (at MGBA_SCALE=3, 720×480 game canvas) ────────
# GBA native resolution: 240×160.  At 3× scale each native pixel = 3×3 pixels.
# Native HP bar positions (approximate):
#   Enemy  HP bar: y≈43, x=100..198  →  scaled: y≈129, x=300..594
#   Player HP bar: y≈53, x=218..318  →  scaled: y≈159, x=654..954
# mGBA adds a title bar (~30px) and menu bar (~22px) above the game canvas.
# The exact offsets shift with the OS theme; run calibrate.py to verify.
_SCALE = MGBA_SCALE
_TITLE_BAR_PX = 52   # approximate; calibrate.py will print the real value

ENEMY_HP_Y  = 180
ENEMY_HP_X1 = 246
ENEMY_HP_X2 = 615

PLAYER_HP_Y  = 371
PLAYER_HP_X1 = 246
PLAYER_HP_X2 = 615


class RewardCalculator:

    def __init__(self):
        self._prev_enemy_hp: float  = 1.0
        self._prev_player_hp: float = 1.0

    def compute(
        self,
        bgr_frame: np.ndarray,
        is_new_global: bool,
        is_stuck: bool,
        is_battle_stall: bool,
        is_animation: bool,
    ) -> tuple[float, dict]:
        """
        Compute the total reward for this step.

        Returns:
            reward: scalar float
            info:   dict with per-component breakdowns for TensorBoard logging
        """
        reward = 0.0
        info: dict = {}

        # 1. Exploration: new screen state discovered
        if is_new_global and not is_animation:
            reward += REWARD_NEW_FRAME
            info["reward_explore"] = REWARD_NEW_FRAME

        # 2. Stuck penalty
        if is_stuck:
            reward += REWARD_STUCK_PENALTY
            info["reward_stuck"] = REWARD_STUCK_PENALTY

        # 3. Battle stall penalty
        if is_battle_stall:
            reward += REWARD_STALL_PENALTY
            info["reward_stall"] = REWARD_STALL_PENALTY

        # 4. HP delta reward (pixel-based approximation)
        enemy_hp, player_hp = self._sample_hp_bars(bgr_frame)
        enemy_delta  = self._prev_enemy_hp  - enemy_hp    # positive = enemy lost HP (good)
        player_delta = self._prev_player_hp - player_hp   # positive = player lost HP (bad)
        hp_reward = enemy_delta * 2.0 - player_delta * 1.0
        if abs(hp_reward) > 0.001:  # skip noise
            reward += hp_reward
            info["reward_hp"] = round(hp_reward, 4)
        info["enemy_hp"]  = round(enemy_hp, 3)
        info["player_hp"] = round(player_hp, 3)
        self._prev_enemy_hp  = enemy_hp
        self._prev_player_hp = player_hp

        # 5. Death detection (full blackout or whiteout)
        if self._detect_death(bgr_frame):
            reward += REWARD_DEATH_PENALTY
            info["died"] = True

        info["total_reward"] = round(reward, 4)
        return reward, info

    def _sample_hp_bars(self, bgr: np.ndarray) -> tuple[float, float]:
        """
        Sample a horizontal stripe across each HP bar region.
        Count pixels whose color matches green / yellow / red (health colours).
        Returns (enemy_ratio, player_ratio) in [0.0, 1.0].

        BGR thresholds:
          Green:  B<100, G>120, R<100
          Yellow: B<80,  G>120, R>120
          Red:    B<80,  G<80,  R>100
        """
        h, w = bgr.shape[:2]

        def _ratio(row: np.ndarray) -> float:
            if row.size == 0:
                return 0.0
            b, g, r = row[:, 0], row[:, 1], row[:, 2]
            health = ((b < 100) & (g > 120) & (r < 100)) | \
                     ((b < 80)  & (g > 120) & (r > 120)) | \
                     ((b < 80)  & (g < 80)  & (r > 100))
            return float(np.sum(health)) / row.shape[0]

        ey = min(ENEMY_HP_Y,  h - 1)
        py = min(PLAYER_HP_Y, h - 1)
        enemy_row  = bgr[ey, max(0, ENEMY_HP_X1):min(ENEMY_HP_X2,  w)]
        player_row = bgr[py, max(0, PLAYER_HP_X1):min(PLAYER_HP_X2, w)]

        return _ratio(enemy_row), _ratio(player_row)

    def _detect_death(self, bgr: np.ndarray) -> bool:
        # Only detect blackout (losing all Pokemon → sent to Pokemon Center).
        # A blackout is a sustained BLACK screen, not a white flash.
        # FireRed has frequent white flashes (battle intros, moves) so > 250
        # would fire constantly and is removed entirely.
        mean_brightness = float(np.mean(bgr))
        return mean_brightness < 5

    def reset(self):
        self._prev_enemy_hp  = 1.0
        self._prev_player_hp = 1.0

    # ── Phase 2 stubs ──────────────────────────────────────────────────────

    def compute_level_reward(self, level_sum: int) -> float:
        """
        Phase 2: monotonically-increasing level reward.
        Tracks max-ever-seen to prevent PC deposit exploit (depositing a
        Pokemon causes level_sum to drop, which would otherwise create a
        traumatic negative spike — the exact issue documented in the
        PokemonRedExperiments video).
        """
        raise NotImplementedError("Phase 2: requires memory reading")

    def compute_coordinate_reward(self, map_id: int, x: int, y: int) -> float:
        """Phase 2: grid-based exploration reward replacing hash-based."""
        raise NotImplementedError("Phase 2: requires memory reading")
