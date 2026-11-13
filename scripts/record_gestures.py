"""Record hand landmark samples for one custom gesture into data/gestures/<label>.csv.

    uv run python scripts/record_gestures.py rock
    uv run python scripts/record_gestures.py rock --samples 150

Show one hand making the gesture and press SPACE to start or pause. Move it around a little
while recording: closer, farther, slightly turned, left and right hand. Run it again on
another day or in another light to add a second session; training tests on whole sessions
the model hasn't seen, which gives an honest accuracy.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from movecore.capture import SourceError, open_source
from movecore.core.config import ConfigError, load_config
from movecore.hands.dataset import DatasetError, SampleWriter, check_label
from movecore.hands.gesture_rules import extended_fingers
from movecore.hands.landmarker import HandDetector
from movecore.hands.module import draw_hand
from movecore.hands.recorder import GestureRecorder
from movecore.hands.settings import DOWNLOAD_HINT, HandSettings
from movecore.output import Display, draw_label
from movecore.preprocess import preprocess

SPACE = ord(" ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record samples of one custom gesture.")
    parser.add_argument("label", help="gesture name: a-z, 0-9 and _, e.g. rock")
    parser.add_argument("--samples", type=int, default=100, help="samples to record (100)")
    parser.add_argument(
        "--config", default="configs/hands_only.yaml", help="camera and hand options"
    )
    parser.add_argument("--source", help="camera index or name, or a video file")
    parser.add_argument("--data", default="data/gestures", help="folder for the CSV files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check_label(args.label)
        if args.samples < 1:
            raise DatasetError("--samples must be at least 1")
        config = load_config(args.config)
        settings = HandSettings.from_options(config.modules.get("hands", {}))
    except (DatasetError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Recording needs landmarks only, so the plain landmarker is enough.
    model = Path(settings.landmarker_model)
    if not model.is_file():
        print(f"error: hand model not found: {model}; {DOWNLOAD_HINT}", file=sys.stderr)
        return 2
    detector = HandDetector(
        model,
        recognizer=False,
        max_hands=2,  # two hands are detected so the recorder can say "show one hand"
        min_detection_confidence=settings.min_detection_confidence,
        min_tracking_confidence=settings.min_tracking_confidence,
        mirrored=settings.mirrored,
    )

    session = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    width, height = config.input["width"], config.input["height"]
    flip = config.input.get("flip", False)
    try:
        with (
            open_source(args.source, config.input) as source,
            SampleWriter(args.data, args.label, session) as writer,
            Display(f"Record: {args.label}") as display,
        ):
            recorder = GestureRecorder(writer, args.samples)
            print(f"Recording '{args.label}' into {writer.path} (session {session})")
            for raw, timestamp_ms in source.frames():
                frame = preprocess(raw, width, height, flip)
                hands = detector.detect(frame, timestamp_ms)
                recorder.offer(hands)
                for hand in hands:
                    draw_hand(frame, hand, extended_fingers(hand.world), hand.handedness)
                draw_label(frame, recorder.status(hands))
                if not display.show(frame):
                    break
                if display.last_key == SPACE:
                    recorder.toggle()
    except (SourceError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        detector.close()
    print(f"Saved {recorder.count} samples of '{args.label}' to {writer.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
