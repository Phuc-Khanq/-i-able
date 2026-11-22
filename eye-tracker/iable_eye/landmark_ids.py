"""The one table of face-mesh landmark numbers used by the project.

"Left" and "right" are the person's own left and right (not the image's).

These numbers are written from memory of the MediaPipe face mesh (478 points:
468 face points + 10 iris points). They are NOT yet checked on real output:
step S008 checks them on a real frame. Don't "fix" them from memory.

This file imports nothing.
"""

NOSE_TIP, CHIN, FOREHEAD, FACE_RIGHT, FACE_LEFT = 1, 152, 10, 234, 454
R_EYE_OUTER, R_EYE_INNER, R_LID_UPPER, R_LID_LOWER = 33, 133, 159, 145
L_EYE_OUTER, L_EYE_INNER, L_LID_UPPER, L_LID_LOWER = 263, 362, 386, 374

# Iris: the centre first, then 4 ring points. Use the ring as a set: its order is not promised.
R_IRIS = (468, 469, 470, 471, 472)
L_IRIS = (473, 474, 475, 476, 477)

# All 23 numbers above, sorted.
NAMED = tuple(sorted((
    NOSE_TIP, CHIN, FOREHEAD, FACE_RIGHT, FACE_LEFT,
    R_EYE_OUTER, R_EYE_INNER, R_LID_UPPER, R_LID_LOWER,
    L_EYE_OUTER, L_EYE_INNER, L_LID_UPPER, L_LID_LOWER,
) + R_IRIS + L_IRIS))

# Blendshape numbers (MediaPipe's 52 scores; index 0 is "_neutral").
# From memory, NOT checked on real output: S008 checks them (and whether "Left" is the person's left).
BS_EYE_BLINK_LEFT, BS_EYE_BLINK_RIGHT = 9, 10
