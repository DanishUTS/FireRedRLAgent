"""Parses Pokemon FireRed (US 1.0, BPRE) game state out of mGBA RAM.

stable-retro's `env.get_ram()` returns the GBA EWRAM as a flat uint8 array
(256 KB starting at GBA address 0x02000000). All offsets below are relative
to that base.

References (pret/pokefirered decompilation):
  - gPlayerParty       = 0x02024284
  - gPlayerPartyCount  = 0x02024029
  - gEnemyParty        = 0x0202402C  (in battle only)
  - SaveBlock1 base    = 0x02025734  (typical allocation in vanilla FireRed)

If the start.state was captured on a different revision, the SaveBlock1
base may differ — adjust SAVEBLOCK1_OFFSET below.
"""

import numpy as np

EWRAM_BASE = 0x02000000

# Direct EWRAM offsets
PARTY_COUNT_OFFSET = 0x024029
PARTY_DATA_OFFSET = 0x024284  # gPlayerParty
ENEMY_PARTY_OFFSET = 0x024744  # gEnemyParty (= gPlayerParty + 6 × 100 bytes)
SAVEBLOCK1_OFFSET = 0x025594  # discovered empirically for this start.state

PARTY_MON_SIZE = 100
MAX_PARTY = 6

# Within a party Pokemon struct (first 32 bytes are encrypted, but we only
# need the unencrypted tail at offset 80+):
MON_LEVEL = 84
MON_HP = 86            # u16, current HP
MON_MAX_HP = 88        # u16, max HP
MON_SPECIES_OK = 0     # personality value (just used to detect empty slots)

# Within SaveBlock1:
SB1_POS_X = 0x00       # s16
SB1_POS_Y = 0x02       # s16
SB1_MAP_GROUP = 0x04
SB1_MAP_NUM = 0x05
SB1_MONEY = 0x0290     # u32, XOR-encrypted with security key (we ignore enc.)
SB1_FLAGS = 0x0EE0     # bitfield (288 bytes)

# Badge flags (byte offset within SB1_FLAGS, bit number)
# FLAG_BADGE01_GET = 0x820 ... FLAG_BADGE08_GET = 0x827
BADGE_FLAG_START = 0x820
N_BADGES = 8


def _u8(ram: np.ndarray, off: int) -> int:
    return int(ram[off])


def _u16(ram: np.ndarray, off: int) -> int:
    return int(ram[off]) | (int(ram[off + 1]) << 8)


def _s16(ram: np.ndarray, off: int) -> int:
    v = _u16(ram, off)
    return v - 0x10000 if v >= 0x8000 else v


class MemoryReader:
    """Parses one RAM snapshot into a structured game-state dict."""

    def __init__(self, saveblock1_offset: int = SAVEBLOCK1_OFFSET):
        self.sb1 = saveblock1_offset

    def read(self, ram: np.ndarray) -> dict:
        if ram is None or len(ram) < 0x40000:
            return self._empty()

        party_count = _u8(ram, PARTY_COUNT_OFFSET)
        party_count = max(0, min(MAX_PARTY, party_count))

        party_levels = []
        party_hp = []
        party_max_hp = []
        for i in range(party_count):
            base = PARTY_DATA_OFFSET + i * PARTY_MON_SIZE
            party_levels.append(_u8(ram, base + MON_LEVEL))
            party_hp.append(_u16(ram, base + MON_HP))
            party_max_hp.append(_u16(ram, base + MON_MAX_HP))

        sum_levels = sum(party_levels)
        sum_hp = sum(party_hp)
        sum_max_hp = sum(party_max_hp)
        hp_frac = (sum_hp / sum_max_hp) if sum_max_hp > 0 else 0.0
        all_fainted = (sum_max_hp > 0 and sum_hp == 0)

        x = _s16(ram, self.sb1 + SB1_POS_X)
        y = _s16(ram, self.sb1 + SB1_POS_Y)
        map_group = _u8(ram, self.sb1 + SB1_MAP_GROUP)
        map_num = _u8(ram, self.sb1 + SB1_MAP_NUM)
        map_id = (map_group << 8) | map_num

        badge_count = self._count_badges(ram)

        # Battle detection via enemy party slot 0. Outside of battle the slot
        # holds stale/zero data; in battle it's populated with the opponent's
        # active mon. Sanity-check ranges to filter false positives.
        enemy_level = _u8(ram, ENEMY_PARTY_OFFSET + MON_LEVEL)
        enemy_hp = _u16(ram, ENEMY_PARTY_OFFSET + MON_HP)
        enemy_max_hp = _u16(ram, ENEMY_PARTY_OFFSET + MON_MAX_HP)
        in_battle = (1 <= enemy_level <= 100
                     and 0 < enemy_max_hp < 1000
                     and enemy_hp <= enemy_max_hp)
        enemy_hp_frac = (enemy_hp / enemy_max_hp) if in_battle else None

        return {
            "party_count": party_count,
            "party_levels": party_levels,
            "sum_levels": sum_levels,
            "sum_hp": sum_hp,
            "sum_max_hp": sum_max_hp,
            "hp_frac": hp_frac,
            "all_fainted": all_fainted,
            "x": x,
            "y": y,
            "map_id": map_id,
            "badge_count": badge_count,
            "in_battle": in_battle,
            "enemy_level": enemy_level if in_battle else 0,
            "enemy_hp_frac": enemy_hp_frac,
        }

    def _count_badges(self, ram: np.ndarray) -> int:
        count = 0
        for i in range(N_BADGES):
            flag_id = BADGE_FLAG_START + i
            byte_off = self.sb1 + SB1_FLAGS + (flag_id >> 3)
            bit = flag_id & 7
            if 0 <= byte_off < len(ram) and (ram[byte_off] >> bit) & 1:
                count += 1
        return count

    @staticmethod
    def _empty() -> dict:
        return {
            "party_count": 0,
            "party_levels": [],
            "sum_levels": 0,
            "sum_hp": 0,
            "sum_max_hp": 0,
            "hp_frac": 0.0,
            "all_fainted": False,
            "x": 0,
            "y": 0,
            "map_id": 0,
            "badge_count": 0,
            "in_battle": False,
            "enemy_level": 0,
            "enemy_hp_frac": None,
        }
