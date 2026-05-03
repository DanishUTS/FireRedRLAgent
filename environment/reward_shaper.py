"""Memory-based multi-component reward for Pokemon FireRed.

All reward signals are derived from `MemoryReader` output (no pixel sampling).
Designed to fix the failure modes from the PokemonRedExperiments video and
from our own observations:
  * monotonic level reward            → no PC-deposit trauma
  * sqrt-scaled level reward          → no Magikarp-buying exploit
  * coordinate-tile exploration       → progress in visually-uniform caves
  * per-map diminishing returns       → can't milk a small map for tile reward
  * big bonus on entering a new map   → strong directional pull toward Brock
  * non-trivial per-step time cost    → can't escape losing battles by stalling
  * win-battle / ran-from-battle      → focus on actually battling, not running
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    coord_explore: float = 1.0
    coord_explore_taper: float = 0.1   # reward for tiles beyond per-map cap
    coord_explore_cap: int = 10        # tiles per map at full reward
    new_map: float = 20.0              # one-shot bonus for entering a new map_id
    hash_explore: float = 0.5
    level_delta: float = 1.0           # multiplied by Δ(sqrt(sum_levels))
    badge: float = 50.0
    enemy_hp_loss: float = 0.5         # per % of enemy HP lost
    player_hp_loss: float = 0.25       # per % of own HP lost (penalty)
    faint_all: float = 1.0             # whole-party faint (one-shot)
    time: float = 0.02                 # per-step cost (2× the old value)
    stuck_map: float = 0.5             # penalty for too many steps with no new tile
    win_battle: float = 2.0            # bonus on enemy fainting
    ran_from_battle: float = 5.0       # penalty when battle ends without a winner
    revisit: float = 1.0               # penalty per step onto an already-known tile


@dataclass
class _State:
    coords_seen: set = field(default_factory=set)
    coords_per_map: dict = field(default_factory=lambda: defaultdict(int))
    maps_seen: set = field(default_factory=set)
    max_sqrt_levels: float = 0.0
    last_badge_count: int = 0
    last_hp_frac: float = 1.0
    last_enemy_hp_frac: float = 1.0
    last_in_battle: bool = False
    fainted_announced: bool = False
    steps_on_map: int = 0
    last_map_id: int | None = None
    last_tile: tuple | None = None
    new_tile_in_window: bool = False


class RewardShaper:
    """Stateful per-env reward composer."""

    STUCK_MAP_WINDOW = 2_000  # steps without a new tile before stuck penalty

    def __init__(self, weights: RewardWeights | None = None):
        self.w = weights or RewardWeights()
        self._s = _State()
        self.components: dict[str, float] = {}

    def reset(self) -> None:
        # coords_seen / maps_seen / coords_per_map persist across resets so
        # exploration drives long-horizon learning across episodes.
        self._s.max_sqrt_levels = 0.0
        self._s.last_badge_count = 0
        self._s.last_hp_frac = 1.0
        self._s.last_enemy_hp_frac = 1.0
        self._s.last_in_battle = False
        self._s.fainted_announced = False
        self._s.steps_on_map = 0
        self._s.last_map_id = None
        self._s.new_tile_in_window = False

    def step(self, gs: dict, is_new_hash: bool) -> float:
        """Returns total shaped reward. `gs` is MemoryReader.read()."""
        c: dict[str, float] = {}

        # 1. Map-first-visit bonus (BIG — pulls the agent toward new areas)
        if gs["map_id"] not in self._s.maps_seen:
            self._s.maps_seen.add(gs["map_id"])
            c["new_map"] = self.w.new_map
        else:
            c["new_map"] = 0.0

        # 2. Coordinate-tile exploration with per-map diminishing returns,
        #    plus a revisit penalty when the agent moves onto a known tile.
        #    The revisit penalty only fires on the *transition* (last_tile →
        #    new_tile), so standing still doesn't accumulate the cost.
        tile = (gs["map_id"], gs["x"], gs["y"])
        c["revisit"] = 0.0
        if tile not in self._s.coords_seen:
            self._s.coords_seen.add(tile)
            cnt = self._s.coords_per_map[gs["map_id"]]
            c["coord"] = (self.w.coord_explore if cnt < self.w.coord_explore_cap
                          else self.w.coord_explore_taper)
            self._s.coords_per_map[gs["map_id"]] += 1
            self._s.new_tile_in_window = True
        else:
            c["coord"] = 0.0
            if self._s.last_tile is not None and tile != self._s.last_tile:
                c["revisit"] = -self.w.revisit
        self._s.last_tile = tile

        # 3. Hash-based exploration backup
        c["hash"] = self.w.hash_explore if is_new_hash else 0.0

        # 4. Monotonic, sqrt-scaled level reward
        sqrt_levels = math.sqrt(max(0, gs["sum_levels"]))
        if sqrt_levels > self._s.max_sqrt_levels:
            delta = sqrt_levels - self._s.max_sqrt_levels
            self._s.max_sqrt_levels = sqrt_levels
            c["level"] = self.w.level_delta * delta
        else:
            c["level"] = 0.0

        # 5. Badge bonus
        new_badges = gs["badge_count"] - self._s.last_badge_count
        c["badge"] = self.w.badge * max(0, new_badges)
        self._s.last_badge_count = gs["badge_count"]

        # 6. Battle HP deltas
        c["enemy_hp"] = 0.0
        c["player_hp"] = 0.0
        enemy_hp_frac = gs.get("enemy_hp_frac")
        if enemy_hp_frac is not None:
            d_enemy = self._s.last_enemy_hp_frac - enemy_hp_frac
            if d_enemy > 0:
                c["enemy_hp"] = self.w.enemy_hp_loss * 100.0 * d_enemy
            self._s.last_enemy_hp_frac = enemy_hp_frac

        d_player = self._s.last_hp_frac - gs["hp_frac"]
        if d_player > 0:
            c["player_hp"] = -self.w.player_hp_loss * 100.0 * d_player
        self._s.last_hp_frac = gs["hp_frac"]

        # 7. Whole-party faint (one-shot)
        if gs["all_fainted"] and not self._s.fainted_announced:
            c["faint"] = -self.w.faint_all
            self._s.fainted_announced = True
        else:
            c["faint"] = 0.0
            if not gs["all_fainted"]:
                self._s.fainted_announced = False

        # 8. Battle-outcome reward — fires on the in_battle → not_in_battle edge
        in_battle = gs.get("in_battle", False)
        c["win_battle"] = 0.0
        c["ran"] = 0.0
        if self._s.last_in_battle and not in_battle:
            # Battle just ended.
            if self._s.last_enemy_hp_frac <= 0.001:
                c["win_battle"] = self.w.win_battle           # we KO'd them
            elif gs["all_fainted"]:
                pass                                          # already penalised by faint
            else:
                c["ran"] = -self.w.ran_from_battle            # ran or got teleported
            self._s.last_enemy_hp_frac = 1.0                  # reset for next battle
        self._s.last_in_battle = in_battle

        # 9. Time cost
        c["time"] = -self.w.time

        # 10. Stuck-on-map penalty (no new tile in N steps on the same map)
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

    @property
    def maps_seen_count(self) -> int:
        return len(self._s.maps_seen)
