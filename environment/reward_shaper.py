"""Memory-based multi-component reward for Pokemon FireRed.

All reward signals are derived from `MemoryReader` output (no pixel sampling).
Designed to fix the failure modes from the PokemonRedExperiments video:
  * monotonic level reward → no PC-deposit trauma, no Magikarp-buying exploit
  * coordinate-tile exploration → progress in visually-uniform caves
  * per-step time cost → can't escape losing battles by stalling
  * badge bonus dominates → keeps the agent goal-directed
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import config


@dataclass
class RewardWeights:
    coord_explore: float = 1.0
    hash_explore: float = 0.5
    level_delta: float = 1.0      # multiplied by Δ(sqrt(sum_levels))
    badge: float = 50.0
    enemy_hp_loss: float = 0.5    # per % of enemy HP lost
    player_hp_loss: float = 0.25  # per % of own HP lost (penalty)
    faint_all: float = 1.0        # one-shot penalty when whole party faints
    time: float = 0.001           # per-step cost
    stuck_map: float = 0.5        # penalty when too many steps with no new tile


@dataclass
class _State:
    coords_seen: set = field(default_factory=set)
    max_sqrt_levels: float = 0.0
    last_badge_count: int = 0
    last_hp_frac: float = 1.0
    last_enemy_hp_frac: float = 1.0
    fainted_announced: bool = False
    steps_on_map: int = 0
    last_map_id: int | None = None
    new_tile_in_window: bool = False


class RewardShaper:
    """Stateful per-env reward composer."""

    STUCK_MAP_WINDOW = 2_000  # steps without a new tile before stuck penalty

    def __init__(self, weights: RewardWeights | None = None):
        self.w = weights or RewardWeights()
        self._s = _State()
        self.components: dict[str, float] = {}

    def reset(self) -> None:
        # Note: coords_seen persists across resets to drive long-horizon exploration.
        self._s.max_sqrt_levels = 0.0
        self._s.last_badge_count = 0
        self._s.last_hp_frac = 1.0
        self._s.last_enemy_hp_frac = 1.0
        self._s.fainted_announced = False
        self._s.steps_on_map = 0
        self._s.last_map_id = None
        self._s.new_tile_in_window = False

    def step(self, gs: dict, is_new_hash: bool, enemy_hp_frac: float | None) -> float:
        """Returns total shaped reward. `gs` is MemoryReader.read()."""
        c: dict[str, float] = {}

        # 1. Coordinate-tile exploration
        tile = (gs["map_id"], gs["x"], gs["y"])
        if tile not in self._s.coords_seen:
            self._s.coords_seen.add(tile)
            c["coord"] = self.w.coord_explore
            self._s.new_tile_in_window = True
        else:
            c["coord"] = 0.0

        # 2. Hash-based exploration backup
        c["hash"] = self.w.hash_explore if is_new_hash else 0.0

        # 3. Monotonic level reward (sqrt scaling damps Magikarp grinding)
        sqrt_levels = math.sqrt(max(0, gs["sum_levels"]))
        if sqrt_levels > self._s.max_sqrt_levels:
            delta = sqrt_levels - self._s.max_sqrt_levels
            self._s.max_sqrt_levels = sqrt_levels
            c["level"] = self.w.level_delta * delta
        else:
            c["level"] = 0.0

        # 4. Badge bonus
        new_badges = gs["badge_count"] - self._s.last_badge_count
        c["badge"] = self.w.badge * max(0, new_badges)
        self._s.last_badge_count = gs["badge_count"]

        # 5. HP delta in battles (only counts negative deltas; positive is healing)
        c["enemy_hp"] = 0.0
        c["player_hp"] = 0.0
        if enemy_hp_frac is not None:
            d_enemy = self._s.last_enemy_hp_frac - enemy_hp_frac
            if d_enemy > 0:
                c["enemy_hp"] = self.w.enemy_hp_loss * 100.0 * d_enemy
            self._s.last_enemy_hp_frac = enemy_hp_frac

        d_player = self._s.last_hp_frac - gs["hp_frac"]
        if d_player > 0:
            c["player_hp"] = -self.w.player_hp_loss * 100.0 * d_player
        self._s.last_hp_frac = gs["hp_frac"]

        # 6. Whole-party faint (one-shot)
        if gs["all_fainted"] and not self._s.fainted_announced:
            c["faint"] = -self.w.faint_all
            self._s.fainted_announced = True
        else:
            c["faint"] = 0.0
            if not gs["all_fainted"]:
                self._s.fainted_announced = False

        # 7. Time cost
        c["time"] = -self.w.time

        # 8. Stuck-on-map penalty
        c["stuck_map"] = 0.0
        if gs["map_id"] != self._s.last_map_id:
            self._s.steps_on_map = 0
            self._s.new_tile_in_window = False
            self._s.last_map_id = gs["map_id"]
        else:
            self._s.steps_on_map += 1
            if self._s.steps_on_map >= self.STUCK_MAP_WINDOW:
                if not self._s.new_tile_in_window:
                    c["stuck_map"] = -self.w.stuck_map
                self._s.steps_on_map = 0
                self._s.new_tile_in_window = False

        self.components = c
        return float(sum(c.values()))

    @property
    def coords_seen_count(self) -> int:
        return len(self._s.coords_seen)
