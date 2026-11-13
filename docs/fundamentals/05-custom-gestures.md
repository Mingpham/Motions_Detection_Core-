# 05 — Custom gestures: training your own classifier

Notes for milestone M5. In M4 the neural networks came pretrained from Google. Here you collect the data, train a model, and measure how good it is. The model is small, but every step is the same one used for big models.

```
record_gestures.py  →  data/gestures/*.csv  →  features (63 numbers)  →  train + cross-validate
                    →  models/gesture_clf.npz  →  live: gesture.detected method=custom
```

## 1. Supervised learning in one paragraph

You show the model examples with the right answer attached (**labels**): "these 21 points are `rock`". Training adjusts the model's numbers (**weights**) until its answers on those examples are mostly right. The real test is examples it has **never seen**: a model that only memorized its training data is useless live. That's the whole game, and most of this note is about measuring it honestly.

## 2. Recording: what makes good data

`uv run python scripts/record_gestures.py rock` keeps every 3rd frame while recording (about 10 samples per second), only when exactly one hand is visible. It stores raw world landmarks, the real hand side, and whether the image was mirrored, so the feature recipe can change later without recording again.

Good data covers the variety you'll see live: both hands, nearer and farther, slightly turned, different light. A model can only learn what it has been shown. Record each gesture in **two or more sessions** (separate runs, ideally on different days): section 5 explains why.

A **"none" gesture** is worth recording too: your hand relaxed, or doing other things. Without it, the classifier must pick one of your gestures for every hand it sees. `min_classifier_score` (0.7) also helps: answers it's less sure of are dropped.

## 3. Features: removing what shouldn't matter

A classifier compares numbers. If the same gesture gives very different numbers depending on where the hand is, it needs far more examples to learn. So each hand becomes 63 numbers (21 points × $x, y, z$), with three effects removed:

1. **Position:** subtract the wrist, $p_i' = p_i - p_0$.
2. **Size / distance:** divide by the hand size $s = \lVert p_9 - p_0 \rVert$ (wrist to middle knuckle).
3. **Which hand:** a left hand is a mirror image of a right one, so left-hand shapes get $x \to -x$. What counts is the shape *in the image*: in the mirrored camera view your right hand has a left-hand shape. M5's first end-to-end test caught this. Recording mirrored and running unmirrored turned `pointing_up` into `victory` until features used the shape's side (`Hand.looks_like`) instead of the real hand's side.

$$f = \operatorname{flatten}\left( \frac{p_i - p_0}{\lVert p_9 - p_0 \rVert} \cdot (\pm 1, 1, 1) \right) \in \mathbb{R}^{63}$$

**Rotation is kept**: thumb up and thumb down are the same shape turned over.

## 4. Two models

**Logistic regression (multinomial).** For each gesture $k$ there is a weight vector $w_k$ and a bias $b_k$; the score is a weighted sum of the features, $z_k = w_k \cdot f + b_k$. **Softmax** turns scores into probabilities that add up to 1:

$$P(k \mid f) = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

Training finds the $w, b$ that give the right gestures high probability (minimizing **cross-entropy**, $-\log P(\text{true label})$, averaged over the samples). It can only draw straight boundaries between gestures in the 63-dimensional space, but with good features that's often enough.

**Small neural network (multi-layer perceptron).** First a hidden layer of 64 units, $h = \operatorname{ReLU}(W_1 f + b_1)$ where $\operatorname{ReLU}(x) = \max(0, x)$, then the same softmax on $W_2 h + b_2$. The hidden layer lets it learn combinations ("index up **and** middle down"), so it can draw curved boundaries. With more weights it also memorizes more easily (**overfitting**).

The training script tries both and keeps the better one; on a tie, the simpler one.

## 5. Measuring honestly: cross-validation and data leakage

**Cross-validation:** split the data into $k$ parts (folds), train on $k-1$, test on the one left out, rotate until every sample has been tested once. Every sample gets a prediction from a model that didn't train on it.

**The trap:** samples from one recording run are near-copies (same light, same hand position, a tenth of a second apart). If copies end up on both sides of a split, testing is like checking answers the model has already seen: accuracy looks great and means little. This is **data leakage**.

**The fix:** keep each session together. When every gesture has at least 2 sessions, the script holds out *whole sessions* (`StratifiedGroupKFold`), so it's tested on recordings it never saw. With only one session it falls back to ordinary folds and says the number is optimistic.

## 6. Reading the confusion matrix

Rows are the true gesture, columns the predicted one. The diagonal is correct answers; anything off it shows which gestures get mixed up:

```
true \ predicted    peace   rock   thumb
peace                  38      2       0
rock                    1     39       0
thumb                   0      0      40
```

Here `peace` is sometimes taken for `rock`: record more varied samples of both, or check whether they're really distinct poses.

## 7. Saving without pickle

The trained weights are copied into plain arrays and saved as `.npz`. Loading uses `allow_pickle=False`, so a model file can't run code (ADR 0006). The live app runs the model with a few lines of NumPy, about 0.04 ms per hand, and never loads scikit-learn.

## Workflow

```bash
uv run python scripts/record_gestures.py rock          # SPACE to start/pause, q to quit
uv run python scripts/record_gestures.py call_me
uv run python scripts/record_gestures.py none          # relaxed hand, other movements
# ...later or in other light, record each again (a second session)
uv run python scripts/train_gesture_classifier.py      # accuracy, confusion matrix, saves the model
# set modules.hands.gestures.custom_classifier: true, then:
uv run movecore
```
