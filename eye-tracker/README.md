# eye-tracker: head + eye cursor for a normal webcam

A standalone Python package (`iable_eye`) and a live demo. It runs on the CPU with MediaPipe Face Landmarker.
It lives in its own folder and does not touch the rest of the repository.

## What is in here

| Part | What it does |
| --- | --- |
| `iable_eye/head_pointer.py` | The head pointer: a relative pointer driven by head yaw and pitch, with a speed gain, a rest gate and a smoothed gain speed. |
| `iable_eye/live_demo.py` | The per-frame demo loop: head mode (default) or fusion mode (gaze jumps the cursor near the target, the head finishes). Dwell and key clicks, pause, reach and click tests. |
| `iable_eye/fusion.py`, `candidates.py` | Gaze to screen: ridge and kernel ridge on eye features, plus the jump rules. |
| `tools/demo.py` | The live tool (camera, full-screen overlay, optional real OS cursor). |
| `tools/record.py`, `tools/record_day.py` | Record calibration sessions (numbers only). |
| `tools/train_demo_model.py`, `tools/check_demo_data.py`, `tools/run_real_scores.py` | Train, check and score the personal gaze model. |

## Privacy rules (kept by the code)

- No camera frame, video or audio is ever written to disk. Only numbers (landmarks, head angles, cursor path).
- Recordings and models go to `~/iable_data/` (outside the repository). People are codes (`P001`), never names.
- No network calls in the product code.

## Setup

```text
pip install numpy scipy scikit-learn opencv-contrib-python mediapipe pytest
```

Download the MediaPipe **Face Landmarker** model (`face_landmarker.task`, Apache 2.0, from the MediaPipe model page)
and put it at `~/iable_data/models_base/face_landmarker.task`. The model file is not in this repository.

## Run the demo

```text
python eye-tracker/tools/demo.py --person P001                       # head pointer
python eye-tracker/tools/demo.py --person P001 --test reach          # edge / corner reach test
python eye-tracker/tools/demo.py --person P001 --mode fusion --test click   # head + gaze (needs a trained model)
```

Keys: SPACE click, c recenter, p pause, q quit. Ctrl+Alt+P pauses from any window.
`--os-cursor` moves the real Windows cursor (also needs the environment variable `IABLE_OS_CURSOR=1`).
Useful options: `--smooth`, `--gain-slow`, `--gain-fast`, `--gaze-smooth`, `--jump-scale`.
Every live run is saved as numbers to `~/iable_data/<person>/demo/` (`--no-record` skips it).

Fusion mode needs a personal gaze model: record sessions with `tools/record.py`, then run
`python eye-tracker/tools/train_demo_model.py --person P001`.

## Tests

```text
cd eye-tracker
python -m pytest tests -q
```

## Status

The head pointer reached all 8 edge and corner targets in 16 of 16 trials (median about 1.1 s) in a live test
on one laptop. Gaze accuracy is still coarse (about 120 to 270 px on one person), so gaze is used only for large
jumps. These are single-person results, not a fairness test.
