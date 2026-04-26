"""
Maps discrete action indices to mGBA keypresses via pydirectinput.

Uses pydirectinput (DirectInput scan codes) rather than pyautogui (VK codes)
because mGBA — like most game emulators — uses DirectInput and will silently
ignore VK-based input from pyautogui.
"""
import time
import pydirectinput

from config import ACTION_KEYS, ACTION_HOLD_SECS, ACTION_PAUSE_SECS

# Disable pydirectinput's built-in inter-call pause (we control timing manually)
pydirectinput.PAUSE = 0.0


class InputHandler:

    def send_action(self, action: int) -> None:
        """Press and release the key mapped to the given action index."""
        key = ACTION_KEYS[action]
        pydirectinput.keyDown(key)
        time.sleep(ACTION_HOLD_SECS)
        pydirectinput.keyUp(key)
        time.sleep(ACTION_PAUSE_SECS)

    def press_key(self, key: str, hold: float = 0.1) -> None:
        """Utility for one-off key presses (state loading, fast-forward toggle)."""
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(0.05)

    def load_state_slot1(self) -> None:
        """Load mGBA save state slot 1 (key configured in config.py)."""
        self.press_key(MGBA_LOAD_STATE_KEY, hold=0.1)

    def enable_fast_forward(self) -> None:
        """Toggle mGBA fast-forward ON (Tab key, mGBA default binding)."""
        self.press_key("tab", hold=0.05)
