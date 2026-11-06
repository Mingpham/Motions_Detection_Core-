"""Command-line entry point."""

import argparse
import sys
from collections.abc import Sequence

from movecore import __version__
from movecore.core.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="movecore", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="path to a YAML config")
    parser.add_argument("--source", default="0", help="camera index or video file path")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Capture and display arrive in M1; until then, show what would run.
    print(f"Config:  {args.config}")
    print(f"Source:  {args.source}")
    print(f"Input:   {config.input['width']}x{config.input['height']}")
    print(f"Modules: {', '.join(config.enabled_modules()) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
