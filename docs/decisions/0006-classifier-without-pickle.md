# 6. Save the gesture classifier as plain arrays, not pickle

**Status:** Accepted (2026-11-13)

## Context

The usual way to save a scikit-learn model is `joblib.dump`, which uses Python's pickle format. Loading a pickle can run any code stored in it: a model file shared online, swapped in `models/`, or downloaded by mistake could take over the machine. The original plan named `models/gesture_clf.joblib`.

The models M5 trains (logistic regression, a small neural network) are only matrices and vectors. Running them is a few lines of NumPy.

## Decision

- Train with scikit-learn (`hands/training.py`, imported only by `scripts/train_gesture_classifier.py`).
- Copy the trained weights into a `GestureClassifier` and save it as `models/gesture_clf.npz`: named number arrays plus a format version.
- Load it with `np.load(..., allow_pickle=False)`, which refuses any array that would need pickle, and check every layer's shape before use.
- Run it with NumPy (`hands/classifier.py`): hidden layers are matrix multiply + bias + ReLU, the output is softmax.

## Consequences

- A model file can hold only numbers, so loading one can't run code.
- The live app never imports scikit-learn (~2 s to import warm, up to ~9 s cold). Classification costs ~0.04 ms per hand.
- Only model types that `to_classifier` knows can be deployed: `LogisticRegression` and `MLPClassifier` with ReLU. Others (random forests, SVMs) would need their own export. A test checks the NumPy copy gives the same probabilities as scikit-learn.
- If PyTorch is ever needed (ADR 0002), its weights can be exported to the same format, or to safetensors, which avoids pickle for the same reason.
