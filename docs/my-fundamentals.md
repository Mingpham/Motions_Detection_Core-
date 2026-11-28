# My fundamentals

## 📚 Computer Vision Fundamentals

### 1. An image is a matrix of numbers

A color frame is a 3D array of shape `H × W × 3`. Each pixel holds three intensity values from 0 to 255. OpenCV stores channels in **BGR** order (not RGB). A video is simply a sequence of these frames over time:

$$I(x, y, t) \in [0, 255]$$

Many algorithms first convert to **grayscale** (one channel), which reduces computation by 3× and keeps the structure needed for motion analysis. A **Gaussian blur** is usually applied to suppress camera noise before comparing frames.

### 2. Frame differencing — "did anything change?"

The simplest motion detector compares the current frame with the previous one:

$$D(x, y, t) = \left| I(x, y, t) - I(x, y, t-1) \right|$$

A pixel is marked as moving when $D(x, y, t) > T$ for a chosen threshold $T$. The result is a binary **motion mask**. It is fast but sensitive to noise, lighting changes, and camera shake.

### 3. Background subtraction — "what is foreground?"

Instead of comparing against the last frame, the system learns a **background model**. MOG2 models each pixel as a mixture of Gaussian distributions that adapts over time. Pixels that don't fit the background model are classified as foreground (moving objects). MOG2 can also detect and label shadows.

### 4. Cleaning the mask — morphology and contours

Raw masks contain speckles and holes. **Morphological operations** fix this:

- **Erosion** removes small noise points.
- **Dilation** fills gaps inside moving regions.
- **Opening** (erode → dilate) and **closing** (dilate → erode) combine both.

Then **contours** are extracted from the mask. Each contour large enough to matter becomes a **bounding box** — a detected moving region.

### 5. Optical flow — "where and how fast is it moving?"

Optical flow estimates the motion vector $(u, v)$ of pixels between frames. It relies on the **brightness constancy assumption**: a point keeps its brightness as it moves.

$$I(x, y, t) = I(x + \Delta x,\ y + \Delta y,\ t + \Delta t)$$

A first-order Taylor expansion gives the **optical flow constraint equation**:

$$I_x u + I_y v + I_t = 0$$

where $I_x, I_y$ are spatial gradients and $I_t$ is the temporal gradient.

- **Lucas-Kanade (sparse):** solves the equation over a small window around selected corner points. Fast; good for tracking specific features.
- **Farneback (dense):** estimates a flow vector for every pixel. Heavier; produces a full motion field, often visualized with color (hue = direction, brightness = speed).

### 6. Hand landmarks — from pixels to keypoints

MediaPipe's Hand Landmarker uses a **two-stage neural network pipeline**:

1. A **palm detector** finds hands in the full frame.
2. A **landmark model** predicts **21 3D keypoints** (wrist, knuckles, finger joints, fingertips) inside each hand region.

Coordinates are normalized to `[0, 1]` relative to the image size; `z` is the relative depth with the wrist as origin. Once hands are found, MediaPipe tracks them across frames instead of re-detecting every frame, which keeps it fast on CPU.

```
        8   12  16  20        ← fingertips
        |   |   |   |
        7   11  15  19
        |   |   |   |
   4    6   10  14  18
   |    |   |   |   |
   3    5---9---13--17        ← knuckles (MCP)
    \   |           /
     2  |          /
      \ |         /
       1|        /
        0--------             ← wrist
```

### 7. Gesture recognition — from keypoints to meaning

| Approach | How it works | Strength |
|---|---|---|
| **Rule-based** | Geometry on landmarks: is the fingertip farther from the wrist than its middle joint? Distance between thumb and index tips? | Transparent, no training data |
| **Built-in recognizer** | MediaPipe's pretrained model for 7 common gestures | Strong baseline |
| **Custom classifier** | Landmarks are normalized (relative to wrist, scaled by hand size) into a 63-value feature vector, then classified with scikit-learn | Learns your own gestures |
| **Dynamic gestures** | Track fingertip positions over a sliding time window; analyze velocity and direction | Detects movement like swipes |

**Normalization matters:** subtracting the wrist position makes features independent of where the hand is in the frame, and dividing by hand size makes them independent of distance from the camera.

---

## 🧰 Tech Stack

All tools are free and open source.

| Purpose | Tool | License |
|---|---|---|
| Language | Python 3.12 (managed with [uv](https://github.com/astral-sh/uv)) | PSF / MIT-Apache |
| Vision core | [OpenCV](https://opencv.org/) (`opencv-python`) | Apache 2.0 |
| Hand models | [MediaPipe](https://ai.google.dev/edge/mediapipe) — **Tasks API** | Apache 2.0 |
| Numerical | NumPy | BSD |
| Gesture classifier | scikit-learn | BSD |
| Configuration | PyYAML | MIT |
| Lint / format | ruff | MIT |
| Testing | pytest | MIT |
| Security | pip-audit, gitleaks, Dependabot | Apache 2.0 / MIT |
| Diagrams | Mermaid (rendered by GitHub) | MIT |

> ⚠️ **MediaPipe note:** this project uses the **Tasks API** (`mediapipe.tasks.vision`). The legacy `mp.solutions` API used by many older tutorials has been removed from recent MediaPipe releases.

---

## ✅ Requirements

### Functional

- Accept webcam or video file input
- Detect and box moving regions; emit events when motion enters or leaves configured zones
- Estimate motion direction and speed with optical flow
- Track up to 2 hands with 21 landmarks each
- Recognize rule-based, built-in, custom and dynamic (swipe) gestures
- Display annotated output and log events as JSON

### Non-functional

| Requirement | Target |
|---|---|
| Throughput | ≥ 20 FPS at 640×480, both modules enabled, on Apple M1 |
| Memory | < 1.5 GB RAM |
| Connectivity | Fully offline |
| Extensibility | New module added without modifying the core pipeline |
| Reproducibility | Locked dependencies; tests run on recorded clips |

---

## 🗺️ Roadmap

| Milestone | Deliverable | Status |
|---|---|---|
| **M0** | Repo setup, environment, CI & security checks | ✅ |
| **M1** | Capture pipeline (webcam + video) with FPS counter | ✅ |
| **M2** | Frame differencing → background subtraction → zones & events | ✅ |
| **M3** | Optical flow & motion direction | ✅ |
| **M4** | Hand landmarks, rule-based gestures, built-in recognizer | ✅ |
| **M5** | Gesture recorder + custom classifier | ✅ |
| **M6** | Dynamic swipe gestures, benchmarks, documentation | ✅ |
| **v2** | Body pose module | 🔮 |
| **v3** | Face landmarks & expression module | 🔮 |
