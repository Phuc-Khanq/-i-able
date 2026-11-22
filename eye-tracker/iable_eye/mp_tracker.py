"""MpTracker: the MediaPipe Face Landmarker behind BaseTracker (S008).

VIDEO running mode. The model is read once as bytes from `iable_data\\models_base`.
The conversion itself is in `mp_convert.py` (pure, tested on fakes).
No file is created here; frames stay in memory.

Installed API names (mediapipe 1.2.3): the running mode enum is `vision.RunningMode`
(older docs call it `VisionRunningMode`).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .faceobs import FaceObs
from .mp_convert import check_frame, result_to_faceobs, video_ms
from .paths import DATA_ROOT, ROOT, DataPathError, assert_data_path
from .tracker import BaseTracker, TrackerError

DEFAULT_MODEL = "models_base/face_landmarker.task"


def _mediapipe_factory(model_bytes: bytes, num_faces: int, image_mode: bool = False):
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.IMAGE if image_mode else vision.RunningMode.VIDEO,
        num_faces=num_faces,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)


class MpTracker(BaseTracker):
    """One frame (h, w, 3) uint8 RGB array in, one FaceObs out. `last_n_faces` = faces seen."""

    def __init__(self, model_path=None, *, num_faces=2, data_root=DATA_ROOT, root=ROOT,
                 _landmarker_factory=None, image_mode=False):
        super().__init__()
        if model_path is None:
            model_path = Path(data_root) / DEFAULT_MODEL
        try:
            p = assert_data_path(model_path, data_root, root)
        except DataPathError as e:
            raise TrackerError("path", str(e)) from e
        if p.suffix.lower() != ".task" or not p.is_file():
            raise TrackerError("model", f"the model must be an existing .task file: {p}")
        self.model_path = p
        self._model_bytes = p.read_bytes()
        self.model_bytes_sha256 = hashlib.sha256(self._model_bytes).hexdigest()
        self._num_faces = int(num_faces)
        # S060: image_mode=True = MediaPipe IMAGE mode (independent photos, no tracking between them).
        self.image_mode = bool(image_mode)
        if _landmarker_factory is None:
            self._factory = lambda b, n: _mediapipe_factory(b, n, self.image_mode)
        else:
            self._factory = _landmarker_factory
        self._lm = self._factory(self._model_bytes, self._num_faces)
        self._last_ms = None
        self._closed = False
        self.last_n_faces = 0

    def _track(self, frame, t_s: float) -> FaceObs:
        import mediapipe as mp

        if self._closed:
            raise TrackerError("closed", "the tracker is closed")
        h, w = check_frame(frame)
        ms = video_ms(t_s, self._last_ms)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = self._lm.detect(image) if self.image_mode else self._lm.detect_for_video(image, ms)
        self._last_ms = ms
        obs, n = result_to_faceobs(result, t_s, w, h)
        self.last_n_faces = n
        return obs

    def reset(self) -> None:
        """New session: forget the times and build a fresh landmarker."""
        super().reset()
        if self._lm is not None:
            self._lm.close()
        self._lm = self._factory(self._model_bytes, self._num_faces)
        self._last_ms = None
        self._closed = False

    def close(self) -> None:
        """Close the landmarker once; a second call does nothing."""
        if self._closed:
            return
        self._closed = True
        if self._lm is not None:
            self._lm.close()
            self._lm = None
