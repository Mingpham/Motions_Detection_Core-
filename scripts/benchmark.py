"""Check the v1 performance targets: >= 20 FPS and < 1.5 GB with every module on.

    uv run python scripts/benchmark.py                          # the camera from the config
    uv run python scripts/benchmark.py --source data/clips/hands.mp4
    uv run python scripts/benchmark.py --frames 600 --configs configs/default.yaml

Each config runs in its own process, headless, for --frames frames after a short warm-up.
A video file with hands in it gives the most repeatable numbers; record one with your
phone or Photo Booth, at least a few seconds long (it's looped).
"""

import argparse
import json
import subprocess  # noqa: S404 - runs this same script with fixed arguments, no shell
import sys
from pathlib import Path

from movecore.benchmark import TARGET_FPS, TARGET_MEMORY_MB, busy_warning, machine_load, measure
from movecore.capture import SourceError
from movecore.core.config import ConfigError, load_config

DEFAULT_CONFIGS = ["configs/motion_only.yaml", "configs/hands_only.yaml", "configs/default.yaml"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure FPS and peak memory per config.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="configs to compare")
    parser.add_argument("--source", help="camera index or name, or a video file")
    parser.add_argument("--frames", type=int, default=300, help="frames to time per config")
    # Internal: measure one config in this process and print the result as JSON.
    parser.add_argument("--child", help=argparse.SUPPRESS)
    return parser


def run_child(config_path: str, source: str | None, frames: int) -> int:
    try:
        result = measure(load_config(config_path), source, frames)
    except (ConfigError, SourceError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps(result.to_dict()))
    return 0


def measure_in_new_process(config_path: str, source: str | None, frames: int) -> dict:
    command = [sys.executable, __file__, "--child", config_path, "--frames", str(frames)]
    if source is not None:
        command += ["--source", source]
    # MediaPipe writes log lines to stderr; only the last stdout line is the result.
    done = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    lines = done.stdout.strip().splitlines()
    if not lines:
        return {"error": done.stderr.strip().splitlines()[-1] if done.stderr else "no output"}
    return json.loads(lines[-1])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 1:
        print("error: --frames must be at least 1", file=sys.stderr)
        return 2
    if args.child:
        return run_child(args.child, args.source, args.frames)

    missing = [path for path in args.configs if not Path(path).is_file()]
    if missing:
        print(f"error: config not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    load, cores = machine_load()
    print(f"Source: {args.source or 'camera from each config'}, {args.frames} frames per config")
    print(f"Machine load: {load:.1f} on {cores} cores")
    print(f"\n{'config':<28}{'FPS':>8}{'ms/frame':>10}{'peak MB':>10}  targets")

    failed = False
    for path in args.configs:
        result = measure_in_new_process(path, args.source, args.frames)
        if "error" in result:
            print(f"{path:<28}  error: {result['error']}")
            failed = True
            continue
        fps, memory = result["fps"], result["peak_memory_mb"]
        ok = fps >= TARGET_FPS and memory < TARGET_MEMORY_MB
        failed |= not ok
        ms = 1000 / fps if fps else float("inf")
        print(f"{path:<28}{fps:8.1f}{ms:10.1f}{memory:10.0f}  {'met' if ok else 'MISSED'}")

    print(f"\nTargets: >= {TARGET_FPS:.0f} FPS and < {TARGET_MEMORY_MB:.0f} MB per config.")
    warning = busy_warning()
    if warning:
        print(f"Note: {warning}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
