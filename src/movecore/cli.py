"""Command-line entry point."""

import argparse
import sys
from collections.abc import Sequence

from movecore import __version__
from movecore.app import run
from movecore.capture import SourceError
from movecore.core.config import ConfigError, load_config


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="movecore", description="Movement detection core.")
    parser.add_argument("--config", default="configs/default.yaml", help="path to a YAML config")
    parser.add_argument(
        "--source",
        help='camera index (0), camera name ("FaceTime HD") or video file; '
        "defaults to input.camera in the config",
    )
    parser.add_argument("--loop", action="store_true", help="restart a video file when it ends")
    parser.add_argument("--headless", action="store_true", help="run without opening a window")
    parser.add_argument("--max-frames", type=positive_int, help="stop after this many frames")
    parser.add_argument(
        "--show-mask", action="store_true", help="show the motion mask next to the image"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Config:  {args.config}")
    print(f"Modules: {', '.join(config.enabled_modules()) or 'none'}")
    try:
        summary = run(
            config,
            args.source,
            headless=args.headless,
            loop=args.loop,
            max_frames=args.max_frames,
            show_mask=args.show_mask,
        )
    except ConfigError as exc:  # module options are checked when the modules are built
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (SourceError, OSError) as exc:  # OSError: e.g. the event log can't be written
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
        return 130

    print(
        f"Done:    {summary.frames} frames in {summary.seconds:.1f} s "
        f"({summary.average_fps:.1f} FPS average), {summary.events} events"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
