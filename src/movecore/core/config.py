"""Load and validate YAML configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """The configuration file is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    input: dict[str, Any]
    modules: dict[str, dict[str, Any]]
    output: dict[str, Any]

    def enabled_modules(self) -> list[str]:
        return [name for name, options in self.modules.items() if options.get("enabled") is True]


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")

    return AppConfig(
        input=_validate_input(raw.get("input")),
        modules=_validate_modules(raw.get("modules")),
        output=_validate_output(raw.get("output")),
    )


def _validate_input(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ConfigError("'input' must be a mapping")
    for key in ("width", "height"):
        value = section.get(key)
        # bool is a subclass of int, so `width: true` would otherwise pass.
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"'input.{key}' must be a positive integer, got {value!r}")
    if "camera" in section and not _is_camera(section["camera"]):
        raise ConfigError("'input.camera' must be a camera name or a non-negative index")
    if not isinstance(section.get("flip", False), bool):
        raise ConfigError("'input.flip' must be true or false")
    return section


def _is_camera(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_modules(section: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(section, dict):
        raise ConfigError("'modules' must map module names to option mappings")
    for name, options in section.items():
        if not isinstance(options, dict):
            raise ConfigError(f"'modules.{name}' must be a mapping")
    return section


def _validate_output(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        raise ConfigError("'output' must be a mapping")
    return section
