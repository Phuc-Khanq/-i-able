"""S036: the app state machine. Consent first, pause always wins.

Every start call checks consent first, then pause. Pausing stops recording (it never comes back
by itself) but keeps tracking and cursor; output is blocked only because paused is True.
GatedOutput and record_with_consent put the gate where it matters: at the output and the recorder.
"""
from dataclasses import replace

from iable_eye import recorder
from iable_eye.output import Output
from iable_eye.settings import CONSENT_VERSION, Settings


class StateError(ValueError):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code  # "consent", "paused", "cursor_off", "tracking", "busy"


class AppState:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._consent = settings.consent_version == CONSENT_VERSION
        self._paused = self._tracking = self._cursor = self._recording = False

    settings = property(lambda self: self._settings)
    consent = property(lambda self: self._consent)
    paused = property(lambda self: self._paused)
    tracking = property(lambda self: self._tracking)
    cursor = property(lambda self: self._cursor)
    recording = property(lambda self: self._recording)

    def _need(self, *checks):
        for code in checks:
            bad = {"consent": not self._consent, "paused": self._paused,
                   "cursor_off": not self._settings.cursor_enabled, "tracking": not self._tracking,
                   "busy": self._recording}[code]
            if bad:
                raise StateError(code, f"refused: {code}")

    def give_consent(self) -> Settings:
        self._settings = replace(self._settings, consent_version=CONSENT_VERSION)
        self._consent = True
        return self._settings

    def withdraw_consent(self) -> Settings:
        self._settings = replace(self._settings, consent_version=0, cursor_enabled=False)
        self._consent = self._tracking = self._cursor = self._recording = False
        return self._settings

    def set_cursor_enabled(self, on: bool) -> Settings:
        self._settings = replace(self._settings, cursor_enabled=on)  # Settings checks on is a bool
        if not on:
            self._cursor = False
        return self._settings

    def set_click_method(self, method: str) -> Settings:
        self._settings = replace(self._settings, click_method=method)  # Settings checks the method
        return self._settings

    def start_tracking(self) -> None:
        self._need("consent", "paused")
        self._tracking = True

    def stop_tracking(self) -> None:
        self._tracking = self._cursor = False

    def start_cursor(self) -> None:
        self._need("consent", "paused", "cursor_off", "tracking")
        self._cursor = True

    def stop_cursor(self) -> None:
        self._cursor = False

    def start_recording(self) -> None:
        self._need("consent", "paused", "busy")
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        if self._paused:
            self._recording = False  # the recorder caller must stop; resume does not restart it
        return self._paused

    def allow_output(self) -> bool:
        return self._consent and not self._paused and self._cursor

    def allow_record(self) -> bool:
        return self._consent and not self._paused and self._recording


class GatedOutput:
    """Wraps an Output. A refused call sends nothing and adds 1 to .dropped.
    .sent is read-only and equals inner.sent, so output.apply_step works through the gate."""

    sent = property(lambda self: self.inner.sent)  # S061: apply_step reads it; refused calls send nothing

    def __init__(self, state: AppState, inner: Output):
        self.state, self.inner, self.dropped = state, inner, 0

    def move(self, t, x, y) -> bool:
        if not self.state.allow_output():
            self.dropped += 1
            return False
        return self.inner.move(t, x, y)

    def click(self, t, x, y) -> None:
        if not self.state.allow_output():
            self.dropped += 1
            return
        self.inner.click(t, x, y)


def record_with_consent(state: AppState, **kwargs):
    state.start_recording()  # raises before the recorder is touched
    try:
        return recorder.record_session(**kwargs)
    finally:
        state.stop_recording()
