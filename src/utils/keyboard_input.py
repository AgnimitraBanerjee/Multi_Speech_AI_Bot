"""
keyboard_input.py
-----------------
Cross-platform single-keypress capture (no Enter required).

Windows : msvcrt.getch()
Linux/macOS : termios + tty raw mode
"""

import sys
import platform
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_keypress() -> str:
    """
    Block until the user presses a single key and return it as a
    lowercase string. Does NOT require Enter.
    """
    system = platform.system()
    try:
        if system == "Windows":
            return _windows_getch()
        else:
            return _unix_getch()
    except Exception as exc:
        logger.warning(f"get_keypress fell back to input(): {exc}")
        return _fallback_input()


# ------------------------------------------------------------------
# Platform implementations
# ------------------------------------------------------------------

def _windows_getch() -> str:
    import msvcrt
    key = msvcrt.getch()
    # msvcrt returns bytes
    try:
        return key.decode("utf-8").lower()
    except UnicodeDecodeError:
        return key.decode("latin-1").lower()


def _unix_getch() -> str:
    import tty
    import termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
        return key.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _fallback_input() -> str:
    """Plain input() fallback for environments without a real TTY."""
    raw = input("Press S/1 to Start or Q/2 to Quit: ").strip().lower()
    return raw[0] if raw else ""