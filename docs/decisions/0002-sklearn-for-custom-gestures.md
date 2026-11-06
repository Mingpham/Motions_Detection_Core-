# 2. Use scikit-learn for custom gestures

**Status:** Accepted (2026-11-06)

## Context

Custom static gestures need a trained classifier. MediaPipe Model Maker is the official route, but it depends on TensorFlow, which is heavy and version-fragile on an 8GB M1.

## Decision

Classify normalized landmarks with scikit-learn. Each hand becomes 63 features (21 landmarks × x, y, z), made wrist-relative and divided by hand size so they don't depend on where the hand is or how far it is from the camera.

## Consequences

- Training takes seconds on small datasets, with no TensorFlow and little memory.
- Features come from landmarks, so accuracy is capped by landmark quality.
- If scikit-learn plateaus, PyTorch with the MPS backend is the planned next step.
