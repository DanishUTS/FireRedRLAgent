"""
Tracks exploration progress and detects stuck / battle-stall conditions.

global_seen: persists across episodes — drives the exploration reward signal.
recent_window: per-episode sliding window — drives stuck and stall detection.
"""
from collections import Counter, deque

from config import STUCK_WINDOW_SIZE, STUCK_REPEAT_THRESH, BATTLE_STALL_STEPS


class HashTracker:

    def __init__(self):
        self.global_seen: set[str] = set()
        self.recent_window: deque[str] = deque(maxlen=STUCK_WINDOW_SIZE)
        self._consecutive_same = 0
        self._last_hash: str | None = None

    def update(self, frame_hash: str) -> tuple[bool, bool, bool]:
        """
        Args:
            frame_hash: MD5 hex of the current processed frame.

        Returns:
            is_new_global:   True if this hash has never been seen before (reward trigger).
            is_stuck:        True if the agent is looping in the same area.
            is_battle_stall: True if the agent is holding a button to stall a battle.
        """
        is_new_global = frame_hash not in self.global_seen
        self.global_seen.add(frame_hash)
        self.recent_window.append(frame_hash)

        # Stuck: one hash dominates the recent window
        is_stuck = False
        if len(self.recent_window) == STUCK_WINDOW_SIZE:
            top_count = Counter(self.recent_window).most_common(1)[0][1]
            is_stuck = top_count >= STUCK_REPEAT_THRESH

        # Battle stall: same hash repeated consecutively
        if frame_hash == self._last_hash:
            self._consecutive_same += 1
        else:
            self._consecutive_same = 0
        self._last_hash = frame_hash
        is_battle_stall = self._consecutive_same >= BATTLE_STALL_STEPS

        return is_new_global, is_stuck, is_battle_stall

    def reset_episode(self):
        """Clear per-episode state. global_seen is intentionally preserved."""
        self.recent_window.clear()
        self._consecutive_same = 0
        self._last_hash = None
