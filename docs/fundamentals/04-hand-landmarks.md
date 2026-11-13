# 04 — Hand landmarks and gestures

Notes for milestone M4. Up to M3, everything was hand-written math: formulas that react to pixels changing. M4 adds the project's first **machine learning**: neural networks trained by Google on many hand images. They know what a hand *looks like*, so they find a hand even when it holds perfectly still, which motion detection can't do.

The path every frame takes through the hands module:

```
BGR frame → RGB → palm detector (only when no hand is tracked) → landmark model → 21 points per hand
          → rules (geometry) + built-in recognizer (ML) → debounced events
```

## 1. Two networks, one after the other

A neural network is a long chain of simple calculations (multiply, add, keep only positive values) whose millions of numbers, the **weights**, were tuned during training so that the output matches labelled examples. The `.task` files in `models/` are those weights. Nothing is trained on your machine; the networks only run (**inference**).

**Palm detector (BlazePalm).** A small convolutional network (CNN) looks at the whole frame and returns a box around each palm. It looks for the palm rather than the whole hand because a palm is rigid and roughly square, while fingers can take any shape.

**Landmark model.** A second CNN looks only inside that box, rotated upright, and outputs 21 points. Each point has:

- $x, y$: position as fractions of the frame's width and height, in $[0, 1]$
- $z$: depth relative to the wrist (smaller is closer to the camera)

It also outputs the same 21 points as **world landmarks**: real 3D positions in metres around the hand's center, estimated from the hand's shape. Distances between world landmarks don't change when the hand moves closer to the camera, which makes them the right input for gesture rules.

**Tracking.** In `VIDEO` mode, the box for the next frame is predicted from the current frame's 21 points, so the palm detector is skipped while a hand stays in view. That's why MediaPipe needs **increasing timestamps**: it uses them to know the frames are in order.

Measured on the M1 at 640×480: about 15 ms per frame with no hand (the palm detector runs every frame, looking), about 27 ms with hands (the landmark model runs per hand).

## 2. The 21 points

```
        8   12  16  20        ← fingertips (TIP)
        |   |   |   |
        7   11  15  19        ← DIP joints
        |   |   |   |
   4    6   10  14  18        ← PIP joints (thumb: 3 is the IP joint)
   |    |   |   |   |
   3    5---9---13--17        ← knuckles (MCP)
    \   |           /
     2  |          /
      \ |         /
       1|        /
        0--------             ← wrist
```

## 3. Rule-based gestures: geometry, no training

All rules use world landmarks and compare **ratios of distances**, so they don't depend on how big the hand appears or where it is in the frame.

**Is a finger extended?** A straight finger reaches far from the wrist; a curled one folds its tip back in. So compare the tip's distance from the wrist with the middle (PIP) joint's:

$$\text{extended} \iff \frac{\lVert P_\text{tip} - P_\text{wrist} \rVert}{\lVert P_\text{PIP} - P_\text{wrist} \rVert} > 1.1$$

On the sample photos, extended fingers measured 1.19–1.61 and curled ones 0.60–0.99, so 1.1 sits in the gap. Unlike the common tutorial rule "is the tip above the knuckle?", this works whether the hand points up, sideways or down.

The **thumb** bends sideways across the palm rather than toward it, so its distances are measured from the pinky knuckle (point 17) instead of the wrist: a thumb folded across the palm brings its tip closer to the pinky side.

**Finger count** = how many of the five are extended.

**Pinch:** thumb tip and index tip closer than `pinch_ratio` hand sizes, where hand size is wrist to middle knuckle (0 → 9), a length that doesn't change when fingers move:

$$\text{pinch} \iff \lVert P_4 - P_8 \rVert < \text{pinch\_ratio} \cdot \lVert P_9 - P_0 \rVert$$

**Thumb up / down:** only the thumb is extended, and it points within 45° of straight up or down, meaning the vertical part of the thumb's direction (2 → 4) is larger than the horizontal part: $|\Delta y| > |\Delta x|$.

Each hand gets one label, checked in order: `pinch`, then `thumb_up` / `thumb_down`, then `fingers_<count>`.

**Limits:** a hidden finger's position is the model's guess. With the thumb tucked behind the hand, the guessed thumb tip can land close to the index tip and briefly read as a pinch.

## 4. The built-in recognizer: a third network

MediaPipe's Gesture Recognizer contains the same palm detector and landmark model, plus a small classifier network that turns the 21 points into one of 7 labels with a confidence score: `closed_fist`, `open_palm`, `pointing_up`, `thumb_down`, `thumb_up`, `victory`, `i_love_you` (or none). MoveCore ignores labels below `min_gesture_score`.

Because it already includes the landmarker, MoveCore loads only this one model when `builtin_recognizer` is on (ADR 0005). Rules and recognizer run side by side, so you can compare a transparent rule with a trained model on the same hand.

## 5. Left and right

MediaPipe's documentation says it names each hand's side assuming a **mirrored** (selfie) image. In practice, with MediaPipe 1.2.2, it's the other way round:

- On an ordinary photo (not mirrored), it names both hands correctly.
- On MoveCore's mirrored camera view (`input.flip: true`), it called the right hand "Left", found in the first live check.

So `hands.mirrored: true` (the default, matching `input.flip: true`) swaps the sides back, and the labels match your real hands. For a video that isn't mirrored, set both to `false`.

Lesson: documentation describes intent; a test with a known answer shows what the code really does.

Hands are told apart by side: MediaPipe in `VIDEO` mode doesn't give hands an ID, but the side is stable. If both hands get the same side for a moment, the one the model is surer of counts for events.

## 6. Events

Hands and gestures flicker for a frame or two, so events are debounced like motion (02, section 6):

| Event | Payload | When |
|---|---|---|
| `hand.enter` | `hand` | seen for `start_frames` (3) frames in a row |
| `hand.leave` | `hand`, `duration_ms` | missing for `stop_frames` (10) frames in a row |
| `gesture.detected` | `hand`, `gesture`, `method` (`rules` or `builtin`), `score` (builtin) | a new gesture held for `stable_frames` (5) frames |

A gesture held still is reported once. A hand that disappears for a frame or two keeps its gesture; one that leaves and comes back reports its gesture again.

## Classic CV vs machine learning, side by side

| | M1–M3 (classic) | M4 (machine learning) |
|---|---|---|
| Who wrote the rules | People: formulas | Learned from thousands of labelled hand images |
| Training data | None | Needed once, by Google |
| What it can tell | Something changed / moved, and how | *This is a hand*, and where each joint is |
| Still objects | Fade into the background (MOG2) | Found every frame |
| Cost per frame | ~1–5 ms | ~15–27 ms |
