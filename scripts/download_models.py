"""Download the MediaPipe models into models/. Safe to re-run."""

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "hand_landmarker.task": (
        f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "gesture_recognizer.task": (
        f"{BASE_URL}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    ),
}
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TIMEOUT_SECONDS = 60


def download(url: str, destination: Path) -> None:
    # Write to a temporary name first so an interrupted download never looks complete.
    partial = destination.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            partial.write_bytes(response.read())
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, url in MODELS.items():
        destination = MODELS_DIR / filename
        if destination.exists():
            print(f"exists      {filename}")
            continue
        print(f"downloading {filename}")
        try:
            download(url, destination)
        except OSError as exc:
            print(f"error: could not download {filename}: {exc}", file=sys.stderr)
            return 1
        size_mb = destination.stat().st_size / 1_000_000
        print(f"saved       {filename} ({size_mb:.1f} MB, sha256 {sha256(destination)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
