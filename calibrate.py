"""
Pre-training calibration tool.

Step 1: Open mGBA with the ROM and get into a battle so both HP bars are visible.
Step 2: python calibrate.py
Step 3: A window will show the mGBA capture. Click on:
          - The ENEMY  HP bar (top bar)    → left click
          - The PLAYER HP bar (bottom bar) → left click
        Two clicks total. The script saves the Y coordinates and X range.
Step 4: Verify the printed coordinates look right, then run python train.py.
"""
import ctypes
import sys
import time
import cv2
import numpy as np

# Must be set before any window coordinate queries so Win32 returns physical pixels
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# ── Shared state for mouse callback ───────────────────────────────────────
_clicks: list[tuple[int, int]] = []
_frame: np.ndarray | None = None


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicks) < 2:
        _clicks.append((x, y))
        print(f"  Click {len(_clicks)}: x={x}, y={y}")


def capture_mgba_frame() -> np.ndarray:
    """Grab a frame from the running mGBA window using physical pixel coordinates."""
    import mss
    import pygetwindow as gw
    from config import MGBA_WINDOW_TITLE
    from environment.screen_capture import _get_physical_rect

    monitor = _get_physical_rect(MGBA_WINDOW_TITLE)
    if not monitor:
        print(f"ERROR: No window found with title containing '{MGBA_WINDOW_TITLE}'.")
        print("Open windows:")
        for w in gw.getAllWindows():
            if w.title.strip():
                print(f"  '{w.title}'")
        sys.exit(1)

    print(f"  Window physical rect: {monitor}")
    print(f"  Expected at 3× scale: ~720×480 (plus window chrome ~52px title bar)")
    with mss.mss() as sct:
        shot = sct.grab(monitor)
    frame = np.array(shot, dtype=np.uint8)[:, :, :3]
    return frame


def write_coordinates(enemy_y: int, player_y: int, x1: int, x2: int, frame_w: int):
    """Patch reward_calculator.py with the calibrated coordinates."""
    import re
    path = "environment/reward_calculator.py"
    with open(path, "r") as f:
        src = f.read()

    # Replace the four coordinate constants
    replacements = {
        r"ENEMY_HP_Y\s*=.*":   f"ENEMY_HP_Y  = {enemy_y}",
        r"ENEMY_HP_X1\s*=.*":  f"ENEMY_HP_X1 = {x1}",
        r"ENEMY_HP_X2\s*=.*":  f"ENEMY_HP_X2 = {x2}",
        r"PLAYER_HP_Y\s*=.*":  f"PLAYER_HP_Y  = {player_y}",
        r"PLAYER_HP_X1\s*=.*": f"PLAYER_HP_X1 = {x1}",
        r"PLAYER_HP_X2\s*=.*": f"PLAYER_HP_X2 = {x2}",
    }
    for pattern, replacement in replacements.items():
        src = re.sub(pattern, replacement, src)

    with open(path, "w") as f:
        f.write(src)
    print(f"  Saved to {path}")


def run():
    global _frame, _clicks

    print("=" * 60)
    print("FireRed RL — Interactive HP Bar Calibration")
    print("=" * 60)
    print("\nMake sure mGBA is open and a BATTLE is on screen.")
    print("Click on the mGBA window NOW, then come back here.")
    print("Capturing in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print()

    print("Capturing mGBA screen...")
    _frame = capture_mgba_frame()
    h, w = _frame.shape[:2]
    print(f"Frame size: {w}×{h}")

    # ── Check mGBA exe while we're at it ─────────────────────────────────
    from config import MGBA_EXE, STATE_PATH, ROM_PATH
    print(f"\nmGBA exe:  {'OK' if MGBA_EXE.exists() else 'NOT FOUND — fix MGBA_EXE in config.py'}")
    print(f"ROM:       {'OK' if ROM_PATH.exists() else 'NOT FOUND'}")
    print(f"State:     {'OK' if STATE_PATH.exists() else 'NOT FOUND'}")

    # ── Interactive click ─────────────────────────────────────────────────
    print("\nA window will open. Click on:")
    print("  Click 1 → anywhere on the ENEMY  HP bar  (top bar in battle UI)")
    print("  Click 2 → anywhere on the PLAYER HP bar  (bottom bar in battle UI)")
    print("Press Q after two clicks to confirm.\n")

    win_name = "Calibration — click HP bars (Q to confirm)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, _on_mouse)

    display = _frame.copy()
    while True:
        show = display.copy()

        # Draw crosshairs for each click
        for i, (cx, cy) in enumerate(_clicks):
            colour = (0, 0, 255) if i == 0 else (255, 0, 0)
            label  = "Enemy HP"  if i == 0 else "Player HP"
            cv2.line(show, (0, cy), (w, cy), colour, 2)
            cv2.putText(show, label, (10, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        cv2.imshow(win_name, show)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") and len(_clicks) == 2:
            break
        if key == ord("r"):   # allow redo
            _clicks.clear()
            print("Clicks cleared — click again.")

    cv2.destroyAllWindows()

    if len(_clicks) < 2:
        print("Not enough clicks. Exiting.")
        sys.exit(1)

    enemy_y  = _clicks[0][1]
    player_y = _clicks[1][1]

    # X range: assume HP bars span the middle 40–80% of the frame width
    x1 = int(w * 0.30)
    x2 = int(w * 0.75)

    print(f"\nCalibrated coordinates:")
    print(f"  Enemy  HP bar: y={enemy_y},  x={x1}..{x2}")
    print(f"  Player HP bar: y={player_y}, x={x1}..{x2}")

    # Annotate and save screenshot for reference
    annotated = _frame.copy()
    cv2.line(annotated, (x1, enemy_y),  (x2, enemy_y),  (0, 0, 255), 3)
    cv2.line(annotated, (x1, player_y), (x2, player_y), (255, 0, 0), 3)
    cv2.putText(annotated, "Enemy HP",  (x1, enemy_y  - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(annotated, "Player HP", (x1, player_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite("calibration_screenshot.png", annotated)
    print("  Screenshot saved: calibration_screenshot.png")

    # Write coordinates into reward_calculator.py
    print("\nUpdating environment/reward_calculator.py ...")
    write_coordinates(enemy_y, player_y, x1, x2, w)

    print("\n" + "=" * 60)
    print("Checklist before python train.py:")
    print("  [ ] HP bars calibrated (done above)")
    print("  [ ] Battle animations DISABLED  (mGBA Settings)")
    print("  [ ] Fast-forward set to 4× or 6×  (mGBA Settings → Emulation)")
    print("  [ ] Key bindings: A=X  B=Z  Start=Enter  D-pad=Arrows  (mGBA Settings → Input)")
    print("  [ ] Save state in Slot 1:")
    print("        mGBA → File → Load State → Load File → states/start.ss0")
    print("        mGBA → File → Save State → Slot 1  (hotkey: Shift+F1)")
    print("        Test: press F10 — game should jump back to start")
    print("=" * 60)


if __name__ == "__main__":
    run()
