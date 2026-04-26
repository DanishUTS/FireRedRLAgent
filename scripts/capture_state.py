"""Interactive helper to capture an initial save state for FireRed.

Opens the ROM in stable-retro with a window, takes keyboard input, and
saves the emulator state to retro_integration/PokemonFireRed-GbAdvance/start.state
when you press F2. Use this to skip the title screen + intro / new game
flow once, so RL training can begin from a known checkpoint.

Controls:
    Arrow keys → D-pad
    Z          → A
    X          → B
    Enter      → START
    Right Shift→ SELECT
    F2         → Save state to start.state
    ESC        → Quit (without saving)

Usage:
    python scripts/capture_state.py
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import cv2
import numpy as np
import stable_retro as retro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.integrations import (  # noqa: E402
    GAME_NAME,
    integration_path,
    register_custom_integration,
)
from environment.wrappers import GBA_BUTTONS  # noqa: E402

# OpenCV waitKey codes → GBA button names
KEY_TO_BUTTON = {
    ord('z'): "A",
    ord('x'): "B",
    13: "START",       # Enter
    32: "SELECT",      # Space
    81: "LEFT",        # Linux arrow-left
    82: "UP",
    83: "RIGHT",
    84: "DOWN",
    ord('a'): "LEFT",  # WASD fallback (Windows OpenCV doesn't always send arrow codes)
    ord('w'): "UP",
    ord('d'): "RIGHT",
    ord('s'): "DOWN",
}


def buttons_to_action(active: set[str]) -> np.ndarray:
    arr = np.zeros(len(GBA_BUTTONS), dtype=np.int8)
    for i, name in enumerate(GBA_BUTTONS):
        if name in active:
            arr[i] = 1
    return arr


def main() -> int:
    register_custom_integration()
    env = retro.make(
        game=GAME_NAME,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        inttype=retro.data.Integrations.ALL,
    )
    env.reset()
    print("FireRed state-capture window opened.")
    print("Play to your desired starting point, then press F2 to save.")
    print("ESC to quit without saving.")

    out_path = integration_path() / "start.state"
    win = "FireRed — capture start state"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 480, 320)

    try:
        while True:
            obs, _, term, trunc, _ = env.step(np.zeros(len(GBA_BUTTONS), dtype=np.int8))
            cv2.imshow(win, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(16) & 0xFFFF
            if key == 27:  # ESC
                print("Quit without saving.")
                return 0
            if key in (0xBE, 191, ord('c')):  # F2 / 'c' fallback
                state_bytes = env.unwrapped.em.get_state() if hasattr(env.unwrapped, "em") \
                              else env.get_wrapper_attr("em").get_state()
                # stable-retro reads .state files through gzip — must gzip on save
                out_path.write_bytes(gzip.compress(state_bytes))
                print(f"Saved → {out_path}  ({len(state_bytes):,} bytes raw, gzipped)")
                return 0
            active = {KEY_TO_BUTTON[key]} if key in KEY_TO_BUTTON else set()
            for _ in range(2):  # hold the key briefly so the game registers it
                env.step(buttons_to_action(active))
            if term or trunc:
                env.reset()
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
