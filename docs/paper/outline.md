# MoveCore: from pixel differences to learned hand gestures on a laptop CPU

*Paper outline (draft, M6). Each section lists what goes in it and where the material already is.*

## Abstract (≈150 words, write last)

A single offline pipeline that goes from classic motion detection to learned hand gestures in real time on an 8 GB Apple M1, and what each step costs and adds.

## 1. Introduction

- Motivation: understand computer vision by building each level yourself, from formulas to trained models.
- Contributions:
  1. A plug-in pipeline where classic CV and ML modules share one frame loop and one event format.
  2. A side-by-side comparison of rule-based, pretrained and self-trained gesture recognition on the same landmarks.
  3. Practical findings: handedness under mirroring, data leakage in gesture datasets, safe model storage, honest benchmarking.

## 2. Background

- Frame differencing, background subtraction (MOG2, KNN). → `docs/fundamentals/02`
- Optical flow: brightness constancy, aperture problem, Lucas-Kanade, Farneback. → `03`
- Hand landmark models (palm detector + landmark regression + tracking). → `04`
- Supervised classification: logistic regression, MLP, cross-validation. → `05`

## 3. System design

- Frame contract, module registry order, event bus, per-module overlays. → `docs/architecture/README.md`, ADR 0004
- Configuration and validation as a security and usability feature.
- Offline, permissive licenses only (why not YOLO/AGPL, why not TensorFlow). → ADR 0001, 0002

## 4. Methods

### 4.1 Classic motion
Color input for MOG2 (shadow test bias on gray, with the measured table), burn-in and blind spots ($n = \ln r / \ln(1 - 1/h)$), zones by pixel count. → `02`

### 4.2 Optical flow
Sparse vs dense, forward-backward check, median velocity, direction by dominant axis. → `03`

### 4.3 Hand gestures
- Rules from distance ratios on world landmarks (rotation and scale invariant).
- One MediaPipe model for landmarks and built-in gestures. → ADR 0005
- Handedness under mirroring: documented behaviour vs measured behaviour. → `04` §5
- Custom classifier: features, session-grouped cross-validation, NumPy inference from `.npz`. → `05`, ADR 0006
- Swipes: index fingertip in a 250 ms window, five tests, the re-arm rule; why the first palm-based version felt like path tracking. Palm path kept as a separate feature. → `06`

## 5. Evaluation

- **Accuracy**
  - Rules vs built-in recognizer on the MediaPipe sample photos (5/5 agreement).
  - Custom classifier: cross-validated by session on Ming's own recordings. *To do: record ≥ 4 gestures × 2 sessions; report the confusion matrix.*
- **Swipes:** 4/4 flicks detected in the right direction on a synthetic clip, none during pauses. *To do: live test, N swipes per direction, count misses and false triggers.*
- **Speed and memory:** benchmark table per config, per-module cost breakdown, effect of machine load and thermal throttling. → `06` §3. *To do: rerun on an idle machine for the final numbers.*
- **Optical flow:** measured vs true speed on synthetic motion (within 4%). → M3 notes

## 6. Lessons learned

- Documentation describes intent; tests with a known answer show behaviour (handedness).
- Two kinds of "mirror": the hand's real side vs the side its shape has in the image (features).
- Near-duplicate frames make accuracy look better than it is (data leakage).
- Loading pickles runs code; plain arrays don't.
- Benchmarks need warm-up, isolation and a note on what else was running.

## 7. Limitations and future work

- Hidden fingers are guessed (false pinches); single camera, no depth sensor.
- Hands are keyed by side, not tracked identities.
- v2 body pose, v3 face and expression; object detection with a permissive model (RF-DETR).

## References

To collect: Shi & Tomasi 1994; Lucas & Kanade 1981; Farnebäck 2003; Zivkovic 2004 (MOG2); Zivkovic & van der Heijden 2006 (KNN); Zhang et al. 2020 (MediaPipe Hands); scikit-learn (Pedregosa et al. 2011).
