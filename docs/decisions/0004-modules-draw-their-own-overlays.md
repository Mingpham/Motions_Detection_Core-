# 4. Modules draw their own overlays

**Status:** Accepted (2026-11-09)

## Context

From M2 on, modules produce things worth showing: motion boxes and zones now, hand landmarks and gesture labels later. Someone has to draw them. If `app.py` or `output/overlay.py` knew how to draw each module's results, every new module would mean changing that shared code, which breaks the requirement that a module can be added without modifying the core.

## Decision

`Module` gets an optional `draw(frame, result)` method. After the pipeline runs, the app calls `draw` on each module with a copy of the frame and that module's own result. `output/overlay.py` keeps only generic helpers (FPS counter, labels, side-by-side view) that modules may use.

Modules also take their own config section (`modules.<name>`) as their only constructor argument and validate it themselves.

## Consequences

- A new module brings its own drawing and option checks; the only shared change is one line registering it in `app.MODULES`.
- Modules draw on a copy, so later modules and outputs (such as the recorder) still see a clean frame.
- Modules draw in pipeline order, so a later module's drawing lands on top of an earlier one's.
