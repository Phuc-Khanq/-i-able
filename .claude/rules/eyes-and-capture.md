---
paths:
  - "engine/eyes/**"
  - "engine/capture/**"
  - "tests/test_capture.py"
  - "tests/test_pointer.py"
  - "tests/test_head_push.py"
---

# Eyes, pointer and data capture

## Status

Head tracking work is **paused** until Khoi (CTO) finishes researching open-source trackers to build on. Don't start pointer rewrites before that, unless Khang asks.

`engine/capture/` and `tests/test_capture.py` are an **unreviewed draft** written before the design was agreed. Khang may delete or redo it. Don't extend it or depend on it.

## Goals Khang set for head tracking

1. More accurate pointing.
2. **Direction tracking:** if the person slides or leans without turning their head, the cursor must not move. Use 3D head rotation (yaw and pitch from the head matrix), not where the nose is in the image.
3. **Per-user sensitivity, like mouse DPI:** separate horizontal and vertical, presets, and a quick way to change it hands-free.

## Agreed direction for training data

- Record **numbers only**: tracker outputs over time (head angles and position, blendshapes, landmarks, timestamps). Never video or images.
- Five things to learn, all in scope: direction intent, gesture vs no gesture, voluntary vs involuntary motion (the main concerns are direction intent and this one), personal sensitivity, gaze.
- Labels come from the task (known dot targets, "slide or lean while keeping your direction" means the cursor should not move, cued gestures, long no-gesture blocks).
- Participants are codes (P001), never names. Train and test on different people, never random frames.
- The base tracker may change, so data loading should not depend on MediaPipe internals.

## Before changing pointer or gesture behaviour

Measure first and after with the existing tools (`tests/head_eval.py`, `tests/pointer_feel.py`, `tests/blink_record.py` and `tests/blink_replay.py`). Accidental clicks matter more than speed: every gesture keeps a threshold, a hold time and a way back.
