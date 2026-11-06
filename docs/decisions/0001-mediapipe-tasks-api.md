# 1. Use the MediaPipe Tasks API

**Status:** Accepted (2026-11-06)

## Context

MoveCore needs hand landmarks and gesture recognition. MediaPipe offers these through the Tasks API (`mediapipe.tasks.python.vision`). Most online tutorials use the older `mp.solutions` API, which was removed in MediaPipe 0.10.31 and had bugs on Apple Silicon.

## Decision

Use the Tasks API only. Translate tutorial code to it rather than copying `mp.solutions` calls. Models are `.task` files fetched by `scripts/download_models.py` and kept out of git.

## Consequences

- Runs natively on the M1 and stays on the supported API.
- Model files are a separate setup step, and tests that need them can't run without it.
- `RunningMode.VIDEO` requires strictly increasing timestamps, so the pipeline must supply them.
