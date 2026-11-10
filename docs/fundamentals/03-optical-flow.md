# 03 — Optical flow

Notes for milestone M3. Motion detection (M2) says *where* pixels changed. Optical flow says *how they moved*: for points in one frame, where are they in the next? From that come direction and speed.

## The core assumption: brightness constancy

A point on a moving object keeps its brightness as it moves. If it moves by $(\Delta x, \Delta y)$ in time $\Delta t$:

$$I(x, y, t) = I(x + \Delta x,\ y + \Delta y,\ t + \Delta t)$$

Expanding the right side to first order (Taylor series) and dividing by $\Delta t$ gives the **optical flow constraint**:

$$I_x u + I_y v + I_t = 0$$

where $I_x, I_y$ are how brightness changes across the image (the gradient), $I_t$ is how it changes over time, and $(u, v)$ is the velocity we want.

That is **one equation with two unknowns**. Looking through a small window at a straight edge, you can tell how fast it moves *across* the edge but not *along* it. This is the **aperture problem**, and both methods below solve it by adding an assumption.

## Lucas-Kanade (sparse, the default)

**Assumption:** all pixels in a small window (21×21 here) move together. Every pixel in the window gives one constraint, so there are 441 equations for 2 unknowns, solved by least squares:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \left(A^\top A\right)^{-1} A^\top b, \qquad A = \begin{bmatrix} I_{x_1} & I_{y_1} \\ \vdots & \vdots \\ I_{x_n} & I_{y_n} \end{bmatrix},\ b = -\begin{bmatrix} I_{t_1} \\ \vdots \\ I_{t_n} \end{bmatrix}$$

$A^\top A$ can only be inverted when the window has gradients in two directions, which is a **corner**. A flat area has no gradient; an edge has gradient in one direction only (the aperture problem again). That's why the method starts with `cv2.goodFeaturesToTrack`, which picks the points where $A^\top A$ is most invertible (Shi-Tomasi corners). Plain, untextured surfaces give few corners: a white square has only four.

Other details in `motion/optical_flow.py`:

- **Pyramids:** the Taylor expansion only holds for small moves. The image is repeatedly halved (4 levels: full, ½, ¼, ⅛), flow is found on the smallest first, then refined going up. A 40 px move is 5 px on the smallest level.
- **Forward-backward check:** each point is tracked forward, then back. If it doesn't land within 1 px of where it started, it was tracked wrongly and is dropped.
- **Only where things move:** corners are looked for inside M2's motion mask, so still background doesn't fill the result with zeros.
- **Overall direction:** the **median** of the moving points' velocities. A few badly tracked points would drag a mean with them; the median ignores them.

## Farneback (dense)

**Assumption:** around each pixel, brightness looks like a smooth curved surface (a quadratic polynomial). Fitting that surface in both frames and comparing how it shifted gives the displacement of **every pixel**, not just corners.

It's heavier, so it runs on frames shrunk by `dense_scale` (0.5 = 4× fewer pixels) and the result is scaled back up. Because each pixel's flow comes from a neighbourhood (`winsize: 15`), flow spreads slightly past an object's edges.

**Picturing it:** with every pixel having a direction and a speed, the flow becomes an image in HSV color: **hue = direction**, **brightness = speed**. Run with `mode: dense` to see moving regions tinted by direction.

## Direction and speed

Flow gives a move in pixels per frame. Dividing by the time between frames gives a velocity in **pixels per second**, which doesn't depend on the frame rate:

$$\vec{v} = \frac{(\Delta x, \Delta y)}{\Delta t}, \qquad \text{speed} = \lVert \vec{v} \rVert$$

Anything below `min_speed` (40 px/s) counts as still, so camera noise doesn't produce a direction.

**Image y points down.** So "up" on screen is negative $\Delta y$, and the angle is measured as on screen, counter-clockwise from right:

$$\theta = \operatorname{atan2}(-v_y,\ v_x) \bmod 360°$$

The direction name is whichever axis dominates. With `flip: true` everything is measured on the mirrored image, as you see it.

A `flow.direction` event fires when a new direction has held for `stable_frames` (3) frames in a row, so a single jittery frame doesn't flip it.

## Sparse or dense?

| | Sparse (Lucas-Kanade) | Dense (Farneback) |
|---|---|---|
| Measures | ~200 corners | every pixel |
| Needs | texture (corners) | nothing special |
| Speed in M3 (with motion, 1280×720 clip) | ~65 FPS | ~47 FPS |
| Good for | direction and speed of objects | seeing the whole flow field |

Sparse is the default because hand tracking (M4) still needs room in the 50 ms per frame budget.

## Limits

- **Brightness must stay constant:** a light switching on breaks the core assumption.
- **Untextured objects** give few corners (sparse) or smeared flow (dense).
- **Fast moves** beyond what the pyramid can catch, or motion blur, lose points.
- **Camera movement** makes everything flow; separating camera motion from object motion is a bigger topic.
