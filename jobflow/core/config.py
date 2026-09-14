"""Configuration loading.

Fails loudly at startup with an actionable message. A config error found
mid-run is far more expensive than one found before the browser opens.

Author: Jashan Sadioura
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from jobflow.core.models import Profile, SearchConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        # A fresh clone has only the .example files: say so, rather than
        # failing with a bare path the reader has to interpret.
        example = path.with_suffix(".example.yaml")
        if example.exists():
            raise ConfigError(
                f"Missing config file: {path.name}\n\n"
                f"  Copy the example and fill in your own details:\n"
                f"    cp {example.name} {path.name}\n"
                f"  (on Windows: copy {example.name} {path.name})\n\n"
                f"  Location: {path.parent}"
            )
        raise ConfigError(f"Missing config file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name} is not valid YAML:\n{e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a YAML mapping at the top level")
    return data


def _format_errors(name: str, e: ValidationError) -> str:
    lines = [f"{name} failed validation:"]
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def load_profile(path: Path | None = None) -> Profile:
    path = path or CONFIG_DIR / "profile.yaml"
    try:
        return Profile.model_validate(_load_yaml(path))
    except ValidationError as e:
        raise ConfigError(_format_errors(path.name, e)) from e


def load_search_config(path: Path | None = None) -> SearchConfig:
    path = path or CONFIG_DIR / "search.yaml"
    try:
        return SearchConfig.model_validate(_load_yaml(path))
    except ValidationError as e:
        raise ConfigError(_format_errors(path.name, e)) from e


def load_all(
    profile_path: Path | None = None, search_path: Path | None = None
) -> tuple[Profile, SearchConfig]:
    return load_profile(profile_path), load_search_config(search_path)


# ---------------------------------------------------------------------------
#
# Thanks for using JobFlow.
#
# Built because an application carries your name, so the tool should stop and
# ask rather than guess. If it helped your search, pass it on to someone else
# who is looking.
#
# Jashan Sadioura  ·  https://www.linkedin.com/in/sadioura-jashan/
#
# ---------------------------------------------------------------------------
