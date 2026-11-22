"""S036: the settings file (settings.json in DATA_ROOT) as a frozen, strictly checked dataclass.

Safe defaults: no consent (consent_version 0) and the OS cursor off. A broken file is never
silently reset: load_settings raises, so the error is seen. Save is atomic (.tmp + os.replace).
"""
import json
import math
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from iable_eye.click import METHODS, ClickConfig
from iable_eye.paths import DATA_ROOT, ROOT, assert_data_path

SETTINGS_VERSION = 1; CONSENT_VERSION = 1; SETTINGS_NAME = "settings.json"  # noqa: E702
MODIFIERS = ("ctrl", "alt", "shift", "win")
_KEYS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789") | frozenset(f"f{i}" for i in range(1, 13))


class SettingsError(ValueError):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code  # "field", "value", "version", "file"


def parse_hotkey(s: str) -> tuple:
    """'Ctrl+Alt+P' -> (frozenset({'ctrl', 'alt'}), 'p'). Anything else -> SettingsError 'value'."""
    if not isinstance(s, str):
        raise SettingsError("value", f"hotkey {s!r} is not a string")
    parts = [p.strip() for p in s.lower().split("+")]
    if any(p == "" for p in parts) or len(set(parts)) != len(parts):
        raise SettingsError("value", f"hotkey {s!r} has an empty or repeated part")
    mods = frozenset(p for p in parts if p in MODIFIERS)
    keys = [p for p in parts if p not in MODIFIERS]
    if not mods or len(keys) != 1 or keys[0] not in _KEYS:
        raise SettingsError("value", f"hotkey {s!r} needs >= 1 modifier and one key a-z, 0-9, f1-f12")
    return mods, keys[0]


def _number(name, v):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        raise SettingsError("value", f"{name} = {v!r} is not a finite number")


@dataclass(frozen=True)
class Settings:
    version: int = 1
    click_method: str = "dwell"
    dwell_s: float = 0.6
    dwell_radius_px: float = 30.0
    pause_hotkey: str = "ctrl+alt+p"
    cursor_enabled: bool = False
    consent_version: int = 0

    def __post_init__(self):
        if type(self.version) is not int or self.version != SETTINGS_VERSION:
            raise SettingsError("version", f"version {self.version!r} is not {SETTINGS_VERSION}")
        if not isinstance(self.click_method, str) or self.click_method not in METHODS:
            raise SettingsError("value", f"click_method {self.click_method!r} not in {METHODS}")
        _number("dwell_s", self.dwell_s)
        if not 0.3 <= self.dwell_s <= 3.0:
            raise SettingsError("value", f"dwell_s {self.dwell_s} not in [0.3, 3.0]")
        _number("dwell_radius_px", self.dwell_radius_px)
        if not 5 <= self.dwell_radius_px <= 200:
            raise SettingsError("value", f"dwell_radius_px {self.dwell_radius_px} not in [5, 200]")
        parse_hotkey(self.pause_hotkey)
        if type(self.cursor_enabled) is not bool:
            raise SettingsError("value", f"cursor_enabled {self.cursor_enabled!r} is not a bool")
        if type(self.consent_version) is not int or self.consent_version not in (0, CONSENT_VERSION):
            raise SettingsError("value", f"consent_version {self.consent_version!r} not in (0, {CONSENT_VERSION})")


_FIELDS = frozenset(f.name for f in fields(Settings))


def settings_path(data_root=DATA_ROOT, root=ROOT) -> Path:
    return assert_data_path(Path(data_root) / SETTINGS_NAME, data_root, root)


def to_json(s: Settings) -> str:
    return json.dumps(asdict(s), sort_keys=True, indent=2) + "\n"


def from_json(text: str) -> Settings:
    try:
        d = json.loads(text)
    except (TypeError, ValueError) as e:
        raise SettingsError("file", f"not valid JSON: {e}") from None
    if not isinstance(d, dict):
        raise SettingsError("file", "the JSON is not an object")
    if set(d) != _FIELDS:
        missing, extra = sorted(_FIELDS - set(d)), sorted(set(d) - _FIELDS)
        raise SettingsError("field", f"missing {missing}, unknown {extra}")
    return Settings(**d)


def save_settings(s: Settings, path) -> None:
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(to_json(s), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a reader sees the old file or the new one, never half


def load_settings(path) -> Settings:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return Settings()
    except UnicodeDecodeError as e:
        raise SettingsError("file", f"not UTF-8 text: {e}") from None
    return from_json(text)


def to_click_config(s: Settings) -> ClickConfig:
    return replace(ClickConfig(), dwell_s=s.dwell_s, dwell_radius_px=s.dwell_radius_px)
