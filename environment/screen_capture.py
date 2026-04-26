"""
Finds the mGBA window and captures its frame via mss.

Uses Win32 GetWindowRect (via ctypes) for the bounding box.
pygetwindow returns logical pixels which are scaled down on high-DPI displays
(125%, 150% etc.), causing mss — which works in physical pixels — to capture
only a partial window. Setting DPI awareness first makes GetWindowRect return
physical pixel coordinates that match what mss expects.
"""
import ctypes
import ctypes.wintypes
import time

import mss
import numpy as np
import pygetwindow as gw

from config import MGBA_WINDOW_TITLE

# Make this process DPI-aware so all Win32 coordinate calls use physical pixels.
# Must happen before any window queries. Tries the modern API first, falls back
# to the legacy one (Windows 7 era).
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Per-monitor DPI aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Win32 EnumWindows callback type
_EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_size_t, ctypes.c_size_t)


def _get_physical_rect(title_substring: str) -> dict | None:
    """
    Find a visible window whose title contains title_substring and return its
    bounding box in PHYSICAL pixels using Win32 GetWindowRect.

    This is DPI-safe: logical coordinates (pygetwindow) differ from physical
    coordinates (mss) when Windows display scaling is not 100%.
    """
    found: list[dict] = []

    def _callback(hwnd, _lparam):
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        if title_substring.lower() in buf.value.lower():
            rect = ctypes.wintypes.RECT()
            ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            found.append({
                "left":   rect.left,
                "top":    rect.top,
                "width":  rect.right  - rect.left,
                "height": rect.bottom - rect.top,
            })
        return True

    ctypes.windll.user32.EnumWindows(_EnumWindowsProc(_callback), 0)
    return found[0] if found else None


class ScreenCapture:

    def __init__(self):
        self._sct = mss.mss()
        self._monitor: dict | None = None

    def find_window(self, timeout: float = 10.0) -> bool:
        """Poll until the mGBA window appears. Returns True if found in time."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            rect = _get_physical_rect(MGBA_WINDOW_TITLE)
            if rect:
                self._monitor = rect
                return True
            time.sleep(0.2)
        return False

    def refresh_window_bounds(self):
        """Re-read window position (e.g. if the window was moved)."""
        rect = _get_physical_rect(MGBA_WINDOW_TITLE)
        if rect:
            self._monitor = rect

    def grab(self) -> np.ndarray:
        """
        Capture the mGBA window and return a BGR uint8 array (H, W, 3).
        mss returns BGRA; alpha is dropped.
        """
        if self._monitor is None:
            raise RuntimeError("Call find_window() before grab()")
        shot = self._sct.grab(self._monitor)
        frame = np.array(shot, dtype=np.uint8)
        return frame[:, :, :3]

    def activate_window(self):
        """Bring mGBA to foreground so pydirectinput keypresses land correctly."""
        windows = gw.getWindowsWithTitle(MGBA_WINDOW_TITLE)
        if windows:
            try:
                windows[0].activate()
                time.sleep(0.1)
            except Exception:
                pass

    def close(self):
        self._sct.close()
