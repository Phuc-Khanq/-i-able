"""The tracker interface (S007): one frame in, one checked FaceObs out.

Every tracker (the simulator one here, the MediaPipe one in S008) is a BaseTracker.
`track()` checks each output again, so a subclass cannot hand out a bad record.

Standard library, numpy and the project only. No file is opened here.
"""
from __future__ import annotations

import abc

from .faceobs import FaceObs, FaceObsError, stack, unstack, validate_stacked


class TrackerError(ValueError):
    """A tracker refused an input or produced a bad output. `code` names the check."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


class BaseTracker(abc.ABC):
    """Subclasses write `_track`; callers use `track`."""

    def __init__(self):
        self._last_t = None

    @abc.abstractmethod
    def _track(self, frame, t_s: float) -> FaceObs:
        """Turn one frame (taken at time t_s, in seconds) into a FaceObs."""

    def track(self, frame, t_s) -> FaceObs:
        obs = self._track(frame, t_s)
        if not isinstance(obs, FaceObs):                                       # 1
            raise TrackerError("type", f"_track returned {type(obs).__name__}, not FaceObs")
        if obs.t != float(t_s):                                                # 2
            raise TrackerError("time", f"the record has t={obs.t!r}, the frame has t={float(t_s)!r}")
        if self._last_t is not None and obs.t <= self._last_t:                 # 3
            raise TrackerError("order", f"t={obs.t!r} is not after the last t={self._last_t!r}")
        try:                                                                   # 4
            stack([obs])
        except FaceObsError as e:
            raise TrackerError("output", f"{e.code}: the record fails the FaceObs checks ({e})") from e
        self._last_t = obs.t                                                   # 5
        return obs

    def reset(self) -> None:
        """Forget the last time (for a new session)."""
        self._last_t = None

    def close(self) -> None:
        """Free resources. Nothing to free here."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class SimTracker(BaseTracker):
    """Plays back stored arrays. The 'frame' is the frame index i."""

    def __init__(self, arrays: dict):
        super().__init__()
        validate_stacked(arrays)
        self._frames = unstack(arrays)

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def _track(self, frame, t_s: float) -> FaceObs:
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < len(self._frames):
            raise TrackerError("frame", f"frame must be an int in [0, {len(self._frames)}), got {frame!r}")
        return self._frames[frame]
