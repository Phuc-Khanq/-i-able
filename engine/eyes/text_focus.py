"""Is the keyboard focus somewhere that takes text? Asked before the dictation card is typed (Khang, 2026-11-22):
if the user hasn't picked a text box, the card waits and asks them to click one, so the text is never typed into
nothing and lost.

focused_text_box() -> True (a text box, document or editable field has the focus), False (something else has it,
e.g. a button, the desktop or a web page with nothing selected), None (can't tell: not Windows, or UI Automation
failed). Callers type as before on None. Two checks, either is enough for True:
  * UI Automation (the uiautomation package, already used for dwell): the focused control is an Edit or Document
    control, or a combo box whose text can be typed into;
  * the Windows caret: classic Win32 text boxes show a blinking caret (GetGUIThreadInfo's hwndCaret).
"""
from __future__ import annotations

import sys

TEXT_TYPES = {"EditControl", "DocumentControl"}


def _caret() -> bool | None:
    """True if the foreground thread shows a text caret, None if it can't be asked."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class GUITHREADINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hwndActive", wintypes.HWND),
                        ("hwndFocus", wintypes.HWND), ("hwndCapture", wintypes.HWND),
                        ("hwndMenuOwner", wintypes.HWND), ("hwndMoveSize", wintypes.HWND),
                        ("hwndCaret", wintypes.HWND), ("rcCaret", wintypes.RECT)]

        info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
        if not ctypes.windll.user32.GetGUIThreadInfo(0, ctypes.byref(info)):
            return None
        return bool(info.hwndCaret)
    except Exception:
        return None


def _uia() -> bool | None:
    """True / False from UI Automation's focused control, None if it can't be asked."""
    if sys.platform != "win32":
        return None
    try:
        import uiautomation as auto
    except Exception:
        return None
    try:
        with auto.UIAutomationInitializerInThread():
            c = auto.GetFocusedControl()
            if c is None:
                return None
            kind = c.ControlTypeName
            if kind in TEXT_TYPES:
                return True
            if kind == "ComboBoxControl":           # an address bar or a search box with a drop-down
                try:
                    return not c.GetValuePattern().IsReadOnly
                except Exception:
                    return False
            return False
    except Exception:
        return None


def focused_text_box() -> bool | None:
    """True, False or None (see the module notes). Never raises."""
    uia = _uia()
    if uia:
        return True
    if _caret():
        return True
    return uia            # False when UI Automation saw something that isn't text, None when nobody could tell
