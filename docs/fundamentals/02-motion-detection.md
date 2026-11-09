# 02 — Motion detection

Notes for milestone M2: how MoveCore decides which pixels moved, turns them into boxes, and turns boxes into events. Everything here is classic computer vision: no training, no models.

The path every frame takes through the motion module:

```
frame → (gray for diff) → blur → mask (diff / MOG2 / KNN) → clean → contours → boxes → events
```

Watch it live with `uv run movecore --show-mask`: the right half of the window is the mask.

## 1. Gray (for differencing) and blur first

Frame differencing only asks "did the brightness change?", so it works on grayscale: one value per pixel instead of three. MOG2 and KNN get the color frame instead, because their shadow test needs color (see section 3).

A camera's pixels flicker slightly even when nothing moves (sensor noise). A Gaussian blur replaces each pixel with a weighted average of its neighbours, which cancels most of that flicker before frames are compared. `blur: 21` is the kernel size: bigger smooths more but also blurs away small movements. It must be odd so the kernel has a center pixel.

## 2. Frame differencing (`method: diff`)

Compare each frame with the one before:

$$D(x, y, t) = \left| I(x, y, t) - I(x, y, t-1) \right|$$

A pixel is moving when the change exceeds a threshold $T$ (`threshold: 25`):

$$M(x, y, t) = \begin{cases} 255 & \text{if } D(x, y, t) > T \\ 0 & \text{otherwise} \end{cases}$$

$M$ is the **motion mask**: white where something changed.

Two things to know:

- **Absolute difference matters.** Frames are `uint8`, so plain subtraction wraps around: $10 - 20$ gives $246$, not $-10$. `cv2.absdiff` does it properly.
- **It only sees edges of motion.** A square moving right shows up as two strips, where it arrived and where it left. Its middle looks the same in both frames. A person who stops moving disappears at once.

## 3. Background subtraction (`method: mog2` or `knn`)

Instead of the previous frame, compare with a **model of the empty scene** learned over time.

**MOG2** (Mixture of Gaussians) models each pixel's brightness as a mix of a few bell curves (Gaussians), each with a mean $\mu_k$, spread $\sigma_k$ and weight $w_k$. A swaying tree or a flickering screen gets several curves, one per state it shows. For each new value $x$, a pixel is background if it fits one of the heavier curves:

$$\frac{(x - \mu_k)^2}{\sigma_k^2} < \lambda$$

`var_threshold: 16` is $\lambda$. Because it is squared, 16 means "within 4 standard deviations". Lower values flag more pixels as foreground.

The model keeps learning. `history: 500` sets roughly how many frames it remembers, so the update rate is about $\alpha \approx 1/500$ per frame. With a learning rate $\alpha$, a curve's mean moves toward each new value:

$$\mu \leftarrow (1 - \alpha)\,\mu + \alpha\,x$$

This is why something that stops and stays (a parked bag) slowly fades into the background: after enough frames, it *is* the background.

**Why still objects fade so fast, and blind spots.** Each remembered curve has a weight (how often it was seen recently). Curves count as background from the heaviest down until their weights add up to `background_ratio` (0.9). Hold an object still and the wall's weight decays by $(1 - \alpha)$ per frame, so after

$$n = \frac{\ln(\text{background\_ratio})}{\ln(1 - 1/\text{history})} \approx 53 \text{ frames} \; (\approx 1.8\ \text{s})$$

the wall alone no longer reaches 0.9 and the object's color joins the background. When the object leaves, its color stays "background" at that spot until the wall's weight recovers (about 30 s after a 20 s hold), so the same object passing over that spot is partly invisible: a **blind spot**. Measured with `history: 500`: ratio 0.7 keeps a held object visible ~6 s with ~16 s blind spots; 0.5 gives ~12 s and ~6 s. Lower values flag flickering things (screens, plants) as motion more often, so the default stays 0.9. Restarting clears the model, and the hand module (M4) never uses it.

**KNN** keeps actual past samples per pixel instead of curves, and calls a pixel background if enough of its stored samples are close to the new value. It handles some scenes better and is a little slower here.

**Shadows:** with `detect_shadows: true`, a pixel that is a darker version of the background is marked `127` instead of `255`. MoveCore keeps only `255`, otherwise a person's shadow would stretch their box along the floor.

The shadow test counts a pixel as shadow when it is between 50% and 100% of the background's brightness (`shadowThreshold = 0.5`). On a **grayscale** image that is all it can check, so a mid-dark object in front of a light wall looks exactly like a shadow and is thrown away, while anything brighter than the background is always kept. MoveCore first fed MOG2 gray frames and showed exactly that bias. Tested against a wall of brightness 200:

| Object | On gray input | On color input |
|---|---|---|
| Brighter (230) | kept | kept |
| Dimmer gray (120), a real shadow | thrown away | thrown away |
| Orange (B 40, G 110, R 190), ~63% as bright | **thrown away** | kept |

On a **color** image, a shadow must also keep the background's color: every channel drops by about the same fraction. The orange sleeve isn't a dimmer white, so it's kept. Gray-ish objects that really are a dimmer version of the background are still lost; `detect_shadows: false` keeps them, at the cost of boxing shadows too.

With color, "how unusual" becomes a distance between colors, summed over the three channels:

$$\frac{(\Delta B)^2 + (\Delta G)^2 + (\Delta R)^2}{\sigma^2} > \lambda$$

Noise in each channel adds up, so if a real camera shows more speckle with color than with gray, raise `var_threshold`.

**Startup:** KNN marks the whole frame as moving for its first few frames, until it has enough remembered samples to compare against. MOG2 can start from the first frame. `warmup_frames` hides both.

## 4. Cleaning the mask (morphology)

Raw masks have specks (noise that got through) and holes (parts of an object that matched the background). Morphology fixes both by sliding a small shape, the **kernel** (`kernel_size: 5`), over the mask:

- **Erosion:** a pixel stays white only if the whole kernel around it is white. Small specks vanish; shapes shrink.
- **Dilation:** a pixel becomes white if any pixel under the kernel is white. Holes fill; shapes grow.
- **Opening** = erode, then dilate: removes specks, keeps the size of big shapes.
- **Closing** = dilate, then erode: fills holes, keeps the size.

MoveCore opens, closes, then dilates twice more so that a moving arm and body become one shape. That is why boxes look a little larger than the object.

## 5. Contours and boxes

`cv2.findContours` traces the outline of each white shape. Shapes smaller than `min_area` (500 pixels, about 22×22) are dropped as noise, and `cv2.boundingRect` gives the box around each remaining one.

The area is measured along the outline, through pixel centers, so a filled 40×40 square measures $39 \times 39 = 1521$, not 1600.

## 6. Debouncing: from flicker to events

"Are there boxes?" flips between yes and no from frame to frame. Firing an event on every flip would flood the log. A **debouncer** adds hysteresis, meaning the state has some resistance to change:

- `start_frames: 2`: motion must last 2 frames in a row before `motion.start`.
- `stop_frames: 15`: stillness must last 15 frames (about 0.5 s) before `motion.stop`.

Starting fast and stopping slowly means a short pause, such as someone hesitating mid-step, doesn't split one movement into two.

`warmup_frames: 30` suppresses events for the first second, while the background model is still learning and the camera is adjusting its exposure.

## 7. Zones

A zone is a named rectangle in **fractions** of the frame, `[x, y, width, height]`. Fractions keep a zone correct if the resolution changes. With `flip: true`, they are measured on the mirrored image, as you see it on screen.

A zone is active when at least `min_area` moving pixels fall inside it, the same size rule as for boxes. Using a fixed pixel count rather than a fraction of the zone means a large zone is as sensitive as a small one. Each zone has its own debouncer, giving `zone.enter` and `zone.leave` events.

## 8. Events and the log

| Event | Payload |
|---|---|
| `motion.start` | `boxes`: `[x, y, width, height]` for each box |
| `motion.stop` | `duration_ms` since the matching start |
| `zone.enter` | `zone` |
| `zone.leave` | `zone`, `duration_ms` |

Events print in the terminal and are appended to `logs/events.jsonl`, one JSON object per line (JSONL). One line per event means the file can be read while it's being written (`tail -f logs/events.jsonl`) and a crash never leaves it half-written.

## Limits of classic motion detection

- **Lighting changes** (a cloud, a lamp, the camera adjusting exposure) change many pixels at once and look like motion everywhere.
- **Camera shake** moves every pixel, so the whole frame looks like it moved.
- **Stopped objects** vanish from differencing at once and from background subtraction over time.

These are why M3 adds optical flow (which measures direction and speed rather than just change) and later milestones add models that recognize *what* is moving.
