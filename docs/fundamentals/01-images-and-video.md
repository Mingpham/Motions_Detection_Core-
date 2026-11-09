# 01 — Images and video

Notes for milestone M1: what a frame is, how a camera or file turns into frames, and the preparation every frame gets before any module looks at it.

## A frame is an array of numbers

A color frame is a NumPy array with shape `(H, W, 3)` and type `uint8`, so each value is between 0 and 255. Indexing is **row first**: `frame[y, x]` is the pixel in row `y`, column `x`. Getting this backwards is a common bug, and `cv2.resize` makes it worse because it takes the size as `(width, height)`.

OpenCV stores the three channels as **BGR**, not RGB. MediaPipe expects RGB, so from M4 on frames are converted with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before they reach it.

A video is a sequence of frames, written as a function of position and time:

$$I(x, y, t) \in [0, 255]$$

Motion detection (M2) is about how $I$ changes as $t$ moves forward.

## Frame rate and the time budget

Frames per second (FPS) sets how much time each frame may take. At the project's target of 20 FPS:

$$\text{budget} = \frac{1000\ \text{ms}}{20} = 50\ \text{ms per frame}$$

Everything — reading, preprocessing, every module, drawing — has to fit in that. In M1, reading a 1280×720 video and preparing its frames ran at about 345 FPS (≈ 2.9 ms per frame), so almost the whole budget is left for the motion and hand modules.

The on-screen FPS is smoothed with an exponential moving average, otherwise it jumps every frame:

$$\hat{f}_t = \alpha\,\hat{f}_{t-1} + (1 - \alpha)\,f_t, \qquad \alpha = 0.9$$

Here $f_t$ is the instant rate ($1 / \Delta t$) and $\hat{f}_t$ the value shown. A larger $\alpha$ gives a steadier but slower-reacting number.

## Resolution and aspect ratio

Every module works on 640×480 frames, set in `input.width` and `input.height`. Smaller frames mean less work: 1920×1080 has about 6.75 times as many pixels as 640×480.

Cameras rarely give exactly that. The FaceTime HD camera is 16:9 (1280×720) while 640×480 is 4:3. Resizing straight from one to the other squashes the picture sideways, and squashed hands give wrong geometry for gesture rules. So `fit_to_size` first **center-crops** to the target aspect ratio, then resizes. For a frame of width $W$ and height $H$, with target $w \times h$:

$$W_\text{crop} = H \cdot \frac{w}{h} \qquad \text{(when the frame is too wide)}$$

For 1280×720 that is $720 \cdot \tfrac{4}{3} = 960$, so 160 columns are dropped from each side.

The resize uses `INTER_AREA` when shrinking, which averages the pixels that merge into one, and `INTER_LINEAR` when enlarging. Shrinking with plain sampling skips pixels and gives jagged edges and flicker.

## Mirroring

With `input.flip: true`, each frame is flipped left-right ($x' = W - 1 - x$) so the webcam behaves like a mirror: you move your right hand, and the hand on the right of the screen moves. That is more natural to use, but it swaps left and right for anything that reads them, such as MediaPipe's handedness labels in M4.

## Timestamps

Every frame carries a timestamp in milliseconds, and they must be **strictly increasing**: MediaPipe's VIDEO mode rejects a repeated or earlier one.

- **Camera:** time since the camera started, from a monotonic clock (one that never jumps back, unlike the wall clock).
- **Video file:** the video's own clock, $t_k = \text{round}(k \cdot 1000 / \text{fps})$ for frame $k$. The same file then always gives the same timestamps, however fast it is processed, which keeps tests repeatable. When `--loop` restarts the file, $k$ keeps counting.

In both cases a timestamp that would not increase is bumped to one more than the previous.

## Cameras on macOS

- **Startup:** a camera often returns a few empty frames while it starts. `CameraSource` retries for up to 50 reads (about 2.5 s) before giving up.
- **Permission:** camera access belongs to the app that starts the program (Terminal, VS Code). If it is missing, the camera fails to open.
- **Numbering:** OpenCV numbers cameras by sorting their unique IDs. An iPhone with Continuity Camera has its own ID and can take number 0 when it is nearby, so cameras are chosen by name. See [ADR 0003](../decisions/0003-select-camera-by-name.md).
