"""Moves the real cursor, clicks, scrolls and types. Only the mode manager calls click and type.

Route: Khang and Hung (Hung leads)
Publishes: nothing
Subscribes: cursor moves from smoothing; click and type requests from mode_manager
APIs: pynput mouse and keyboard; ctypes SetProcessDpiAwarenessContext so pixels match at 125% and 150%

All coordinates are physical screen pixels, the same pixels mss screenshots use, once
make_dpi_aware() has run. pynput is imported on first use so the engine still starts on
machines without a desktop session.
"""
import sys
import time

_mouse = None
_keyboard = None
_dpi_aware = False
_screen: tuple[int, int] | None = None

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 is the handle value -4.
PER_MONITOR_AWARE_V2 = -4
CLICK_SETTLE_S = 0.03   # let the app see the cursor arrive before the press


def make_dpi_aware() -> None:
    """Call once at startup, before any screenshot or cursor move."""
    global _dpi_aware
    if _dpi_aware or sys.platform != "win32":
        _dpi_aware = True
        return
    import ctypes

    user32 = ctypes.windll.user32
    try:
        ok = user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(PER_MONITOR_AWARE_V2))
    except AttributeError:  # older than Windows 10 1703
        ok = False
    if not ok:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()
    _dpi_aware = True


def screen_size() -> tuple[int, int]:
    """Primary screen size in physical pixels, read once."""
    global _screen
    if _screen is None:
        _screen = _read_screen_size()
    return _screen


def _read_screen_size() -> tuple[int, int]:
    make_dpi_aware()
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))  # SM_CXSCREEN, SM_CYSCREEN
    try:
        import mss

        with mss.mss() as sct:
            mon = sct.monitors[1]
            return int(mon["width"]), int(mon["height"])
    except Exception:
        return 1920, 1080


def _get_mouse():
    global _mouse
    if _mouse is None:
        make_dpi_aware()
        from pynput.mouse import Controller

        _mouse = Controller()
    return _mouse


def _get_keyboard():
    global _keyboard
    if _keyboard is None:
        from pynput.keyboard import Controller

        _keyboard = Controller()
    return _keyboard


def move(x: int, y: int) -> None:
    w, h = screen_size()
    _get_mouse().position = (min(max(int(round(x)), 0), w - 1), min(max(int(round(y)), 0), h - 1))


def position() -> tuple[int, int]:
    x, y = _get_mouse().position
    return int(x), int(y)


def click(x: int, y: int, button: str = "left", count: int = 1) -> None:
    """Move to (x, y) and click. button is "left" or "right"; count=2 double-clicks."""
    from pynput.mouse import Button

    move(x, y)
    time.sleep(CLICK_SETTLE_S)
    _get_mouse().click(Button.right if button == "right" else Button.left, count)


def scroll(dy: int, dx: int = 0) -> None:
    """Scroll by notches; dy > 0 scrolls up, like a mouse wheel."""
    _get_mouse().scroll(dx, dy)


_wheel_leftover = 0.0


def scroll_units(units: int) -> None:
    """Scroll by 1/120 of a notch per unit (positive = up): the smooth, trackpad-like scrolling.
    Windows takes these small wheel steps directly. Elsewhere only whole notches work, so the rest is carried."""
    global _wheel_leftover
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(units), 0)   # MOUSEEVENTF_WHEEL
        return
    _wheel_leftover += units / 120.0
    whole = int(_wheel_leftover)
    if whole:
        _wheel_leftover -= whole
        scroll(whole)


def type_text(text: str) -> None:
    _get_keyboard().type(text)


def press(key: str) -> None:
    """Press and release a named key, e.g. "enter", "backspace", "tab", "esc"."""
    from pynput.keyboard import Key

    _get_keyboard().tap(getattr(Key, key))


def combo(keys: str) -> None:
    """A key or a shortcut: "enter", "shift+enter", "ctrl+z". Modifiers are held while the last key taps.
    On macOS, "ctrl" is Control; use "cmd+z" for Command-Z."""
    from pynput.keyboard import Key

    parts = [p.strip().lower() for p in keys.split("+") if p.strip()]
    if not parts:
        return
    named = [getattr(Key, p) if hasattr(Key, p) else p for p in parts]  # "z" stays a character
    kb = _get_keyboard()
    for mod in named[:-1]:
        kb.press(mod)
    try:
        kb.tap(named[-1])
    finally:
        for mod in reversed(named[:-1]):
            kb.release(mod)


# ------------------------------------------------------------------ the app in front (for the dictation clean-up)
# The trained clean-up model was taught with plain app kinds ("email", "Slack", "document" ...), so the program in
# front is mapped to one. The window title is read only to tell apps apart inside a browser; it is never logged or
# kept, and the kind goes only to the local clean-up model.
APP_BY_PROGRAM = {"outlook": "email", "olk": "email", "thunderbird": "email", "hxoutlook": "email",
                  "slack": "Slack", "ms-teams": "Teams", "teams": "Teams", "discord": "Discord",
                  "whatsapp": "WhatsApp", "telegram": "chat", "signal": "chat", "zoom": "chat",
                  "winword": "document", "soffice": "document", "notepad": "notes", "onenote": "notes",
                  "obsidian": "notes", "excel": "spreadsheet", "powerpnt": "slides", "code": "code editor"}
BROWSERS = {"chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "arc"}
APP_BY_TITLE = [("gmail", "email"), ("outlook", "email"), ("mail", "email"), ("whatsapp", "WhatsApp"),
                ("messenger", "chat"), ("slack", "Slack"), ("discord", "Discord"), ("teams", "Teams"),
                ("google docs", "document"), ("word", "document"), ("linkedin", "LinkedIn"),
                ("facebook", "Facebook"), ("instagram", "Instagram"), ("github", "GitHub"), ("twitch", "Twitch")]


def app_kind(program: str, title: str = "") -> str:
    """"OUTLOOK.EXE" -> "email"; a browser by its title ("Inbox - Gmail" -> "email", else "browser"); "" if unknown."""
    name = program.lower().rsplit("\\", 1)[-1].removesuffix(".exe")
    if name in APP_BY_PROGRAM:
        return APP_BY_PROGRAM[name]
    if name in BROWSERS:
        low = title.lower()
        return next((kind for key, kind in APP_BY_TITLE if key in low), "browser")
    return ""


def foreground_app() -> str:
    """The kind of app in front on Windows (app_kind), or "" when it can't be told."""
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        user32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title, 256)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = k32.OpenProcess(0x1000, False, pid.value)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        try:
            path, size = ctypes.create_unicode_buffer(520), wintypes.DWORD(520)
            if not k32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                return ""
        finally:
            k32.CloseHandle(handle)
        return app_kind(path.value, title.value)
    except Exception:                                    # never let this stop dictation
        return ""
