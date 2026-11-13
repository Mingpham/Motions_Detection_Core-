# 5. Load one hand model, chosen by the gesture options

**Status:** Accepted (2026-11-13)

## Context

M4 needs hand landmarks (for rules, and later for the custom classifier and swipes) and MediaPipe's built-in gesture labels. MediaPipe ships two models for this:

- `hand_landmarker.task`: palm detector + landmark model.
- `gesture_recognizer.task`: the same palm detector and landmark model, plus a gesture classifier. Its result includes the landmarks and world landmarks too.

Running both would detect and track every hand twice. Measured at 640×480 on the M1: ~15 ms per frame for the landmarker and ~20 ms for the recognizer with no hand in view, ~27 ms for either with hands.

## Decision

The hands module loads exactly one model:

- `gestures.builtin_recognizer: true` → the Gesture Recognizer, used for both landmarks and gesture labels.
- `false` → the plain Hand Landmarker.

Both are wrapped by `hands/landmarker.py` (`HandDetector`), which returns the same `Hand` objects either way, so nothing else in MoveCore depends on which model ran. MediaPipe is imported inside `HandDetector`, not at module import, because the import alone takes ~1 s and motion-only runs don't need it.

## Consequences

- One copy of the hand networks in memory, and one pass per frame: the whole v1 pipeline runs at ~25 FPS with ~250 MB peak memory on a 1280×720 test clip.
- The planned separate `gesture_recognizer.py` isn't needed; `landmarker.py` covers both.
- Turning the recognizer off saves little time (the landmark model is the expensive part), but is the setting to use if the 7 built-in gestures aren't wanted.
- The models aren't in git or CI. Unit tests use a fake detector; tests against the real models skip when the files are missing.
