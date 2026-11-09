# 3. Select the camera by name

**Status:** Accepted (2026-11-09)

## Context

OpenCV opens cameras by index. On macOS it numbers them by sorting each device's unique ID. With Continuity Camera on, a nearby iPhone appears as an extra camera, and on this Mac its ID (`33F1…`) sorts before the built-in camera's (`EAB7…`). So index 0 is the iPhone when the phone is nearby and the built-in camera when it isn't. A fixed index can't reliably mean "the Mac's camera", and turning Continuity Camera off isn't wanted.

## Decision

Configure the camera by name (`input.camera: FaceTime HD`). At startup, read the camera list with `system_profiler SPCameraDataType -json`, sort it by unique ID as OpenCV does, and use the index of the matching name. An exact name match wins over a partial one; no match or several matches is an error that lists the cameras found. A number given with `--source` still opens that index directly.

## Consequences

- The project uses the built-in camera whether or not the iPhone is nearby, with no new dependency.
- It relies on OpenCV's ordering rule, which is internal behavior. A future OpenCV version could change it; the startup line printing the camera name and frame size makes that visible.
- OpenCV also lists "muxed" (audio and video) devices, which `system_profiler` may not. A capture card of that kind could shift the indexes.
- The camera list could change between reading it and opening the camera. That window is a fraction of a second.
- Choosing by name only works on macOS; elsewhere, use an index.
