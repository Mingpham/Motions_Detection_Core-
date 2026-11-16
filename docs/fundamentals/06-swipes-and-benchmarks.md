# 06 — Swipes and measuring performance

Notes for milestone M6. Every gesture so far was a **pose**: one frame is enough to tell a fist from a victory sign. A swipe is different. No single frame contains it; it only exists as **change over time**. The second half of this note is about measuring whether all of MoveCore runs fast enough.

## 1. From a pose to a path

Each frame, the hands module adds points to short histories, one per hand. Each history is a **sliding time window**: points older than the window are dropped. Windows are defined in time, not frames, so a gesture means the same at 15 FPS and at 30 FPS. Two points are followed, for two different jobs:

| | Palm path | Swipe |
|---|---|---|
| Point | Palm center: mean of wrist and the 4 knuckles (0, 5, 9, 13, 17) | Index fingertip (8) |
| Window | `path_window_ms` (1 s) | `swipe_window_ms` (250 ms) |
| Moves when | The whole hand moves | The hand moves **or** the finger flicks |
| Used for | Kept as `paths` for later (drawing in the air, following a hand across zones); drawn in gray | `gesture.swipe` events; drawn in magenta |

$$c_t = \frac{1}{5} \sum_{i \in \{0, 5, 9, 13, 17\}} p_i(t) \qquad f_t = p_8(t)$$

**Design history.** The first M6 version detected swipes on the palm path, so the palm had to travel 1.5 hand sizes in 0.5 s. Tried live, it felt like path tracking, not swiping: a swipe is a quick flick led by the fingertip. Swipes now follow the fingertip in a short window; the palm path stays as its own feature.

## 2. When is a fingertip path a swipe?

Let the window run from $t_0$ to $t_1$, with fingertip displacement $\Delta = f_{t_1} - f_{t_0} = (\Delta x, \Delta y)$ in pixels, and hand size $s$ (median wrist-to-middle-knuckle length in pixels over the window). All five tests must pass:

1. **The index finger is extended the whole time.** When it isn't, the history is cleared. Without this, opening a fist would read as an "up" swipe: the fingertip shoots upward as the finger straightens.

2. **Far enough, fast**, in hand sizes, so distance from the camera doesn't matter:
   $$\frac{\lVert \Delta \rVert}{s} \ge \text{swipe\_min\_distance} \quad (0.8)$$
   Because this must happen inside the 250 ms window, it is also a **speed** test: at least 3.2 hand sizes per second. A slow drift never fits in the window.

3. **Straight enough**: the straight-line distance compared with the path actually walked,
   $$\frac{\lVert \Delta \rVert}{\sum_k \lVert f_{k+1} - f_k \rVert} \ge 0.7$$
   A zigzag or a circle walks a long path but ends near where it started.

4. **One clear direction**: the larger of $|\Delta x|, |\Delta y|$ must be at least twice the other (within about 27° of an axis). Diagonals are ignored rather than guessed.

5. **Not a repeat**: after a swipe, the history is cleared and nothing is reported for `swipe_cooldown_ms` (400 ms). A second swipe in the **same** direction also waits until the fingertip slows down, below a third of the slowest swipe speed, so one long movement isn't counted twice. Changing direction is allowed right after the cooldown, so a quick back-and-forth gives "right, left".

Directions are as seen on screen. With the mirrored camera view (`flip: true`), moving your hand to your right moves it right on screen.

Event: `gesture.swipe` with `hand`, `direction`, `distance` (hand sizes), `speed` (hand sizes per second) and `duration_ms`. On screen, the fingertip's last 250 ms is drawn in magenta, the palm's last second in gray, and the label shows `swipe left` for a moment.

**Measured** on a clip of a pointing hand with four flicks (about 1.3 hand sizes in 165 ms) separated by pauses: exactly four swipes, in the right directions, none during the pauses.

**Note:** a whole hand waved fast also moves the fingertip fast, so it counts as a swipe too. If only finger flicks should count, the fingertip could be measured *relative to the palm*; that's a small change to make if live use asks for it.

### How this compares with optical flow (M3)

| | Optical flow (M3) | Swipes (M6) |
|---|---|---|
| What moves | Pixels: anything in the image | One tracked point on a recognized hand (the index fingertip) |
| Units | Pixels per second | Hand sizes per second |
| Knows it's a hand | No | Yes (from the M4 model) |
| Typical use | "Something moved left" | "Your right hand swiped left" |

## 3. Measuring performance honestly

`uv run python scripts/benchmark.py --source clip.mp4` runs each config headless and prints FPS, milliseconds per frame and peak memory against the v1 targets (≥ 20 FPS, < 1.5 GB).

Three choices make the numbers trustworthy:

- **Warm-up excluded.** The first 30 frames include model start-up and cache filling; timing starts after them. Otherwise short runs look slower than the steady state.
- **One process per config.** Peak memory (the **resident set size**, the RAM a process actually holds) can only go up during a process's life. Measuring three configs in one process would report the largest one three times.
- **Machine load reported.** Other programs share the same CPU. The script prints the load average (how many tasks were waiting for a CPU core, on average, over the last minute) and warns when it's above half the cores.

**Measured during M6** (two-hands clip, 640×480, load ~4 on 8 cores from iCloud sync and the editor):

| Config | FPS | ms/frame | Peak MB |
|---|---|---|---|
| motion only | 70.7 | 14.1 | 163 |
| hands only | 23.5 | 42.5 | 217 |
| everything | 19.0 | 52.6 | 275 |

Per module, per frame: preprocess 1.8 ms, motion 3.8 ms, optical flow 6.2 ms, **hands 37.7 ms** (two hands: the landmark model runs once per hand), swipes +0.2 ms. The same clip ran at 25.4 FPS in M4 on a quieter machine. A fanless MacBook Air also slows itself down when it stays hot. So run the benchmark with other apps closed before judging the target. If more headroom is needed, the levers in order of effect: `max_hands: 1`, then `optical_flow.enabled: false`, then `builtin_recognizer: false`.

Memory is far below the limit: about 0.3 GB of the 1.5 GB budget.
