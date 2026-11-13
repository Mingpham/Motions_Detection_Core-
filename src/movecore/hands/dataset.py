"""Recorded gesture samples: one CSV file per gesture, one row per hand snapshot.

Rows keep the raw world landmarks rather than finished features, so the feature recipe can
change later without recording everything again.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np

from movecore.hands.features import FEATURE_COUNT, hand_features
from movecore.hands.landmarks import OTHER_SIDE, Hand

# The label becomes a file name, so only plain characters are allowed: no "../" tricks.
LABEL_PATTERN = re.compile(r"[a-z0-9_]{1,32}")
POINT_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
# handedness is the real hand; mirrored says whether the image was, so features can use
# the side the hand looked like whichever way it was recorded.
COLUMNS = ["label", "session", "handedness", "mirrored", *POINT_COLUMNS]
SIDES = ("Left", "Right")


class DatasetError(ValueError):
    """A gesture file or label is invalid."""


def check_label(label: str) -> str:
    if not LABEL_PATTERN.fullmatch(label):
        raise DatasetError(f"Gesture label must be 1-32 characters of a-z, 0-9 or _, got {label!r}")
    return label


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray  # (samples, 63)
    labels: np.ndarray  # (samples,) gesture names
    sessions: np.ndarray  # (samples,) "label/session", one per recording run

    def __len__(self) -> int:
        return len(self.labels)

    def counts(self) -> dict[str, int]:
        names, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(names.tolist(), counts.tolist(), strict=True))

    def session_counts(self) -> dict[str, int]:
        """How many recording runs each gesture has."""
        return {
            label: len(np.unique(self.sessions[self.labels == label])) for label in self.counts()
        }


class SampleWriter:
    """Appends hands to ``<folder>/<label>.csv``, writing the header if the file is new."""

    def __init__(self, folder: str | Path, label: str, session: str) -> None:
        self.label = check_label(label)
        self.session = session
        self.path = Path(folder) / f"{label}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        if is_new:
            self._csv.writerow(COLUMNS)

    def write(self, hand: Hand) -> None:
        points = [f"{value:.6f}" for value in hand.world.reshape(FEATURE_COUNT)]
        row = [self.label, self.session, hand.handedness, int(hand.mirrored), *points]
        self._csv.writerow(row)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def load_samples(folder: str | Path) -> Dataset:
    """Read every ``*.csv`` in ``folder`` and turn each row into features."""
    features, labels, sessions = [], [], []
    for path in sorted(Path(folder).glob("*.csv")):
        for row in _read_rows(path):
            world = np.array([float(row[column]) for column in POINT_COLUMNS]).reshape(21, 3)
            side = row["handedness"]
            looks_like = OTHER_SIDE[side] if row["mirrored"] == "1" else side
            features.append(hand_features(world, looks_like))
            labels.append(row["label"])
            sessions.append(f"{row['label']}/{row['session']}")
    if not features:
        raise DatasetError(f"No samples found in {folder}; record some with record_gestures.py")
    return Dataset(np.array(features), np.array(labels), np.array(sessions))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != COLUMNS:
            raise DatasetError(f"{path}: unexpected columns; was it written by record_gestures.py?")
        rows = list(reader)
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        try:
            check_label(row["label"])
            if row["handedness"] not in SIDES:
                raise DatasetError(f"handedness must be Left or Right, got {row['handedness']!r}")
            if row["mirrored"] not in ("0", "1"):
                raise DatasetError(f"mirrored must be 0 or 1, got {row['mirrored']!r}")
            for column in POINT_COLUMNS:
                if not np.isfinite(float(row[column])):
                    raise DatasetError(f"{column} is not a finite number")
        except (DatasetError, ValueError, TypeError) as exc:
            raise DatasetError(f"{path}, line {line}: {exc}") from exc
    return rows
