"""
utils/yaml_reader.py - YAML file loading utility.

Wraps PyYAML with type safety and helpful error messages.
All configuration loading in the platform goes through this module.

Usage:
    from synthetic_data.src.utils.yaml_reader import load_yaml

    config = load_yaml("synthetic_data/config/generator.yaml")
"""

from pathlib import Path
from typing import Any

import yaml

from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path:
        Path to the YAML file. Can be a string or pathlib.Path.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path.resolve()}")

    if not file_path.suffix.lower() in (".yaml", ".yml"):
        raise ValueError(f"File is not a YAML file: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        try:
            content: dict[str, Any] = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"Failed to parse YAML file '{file_path}': {exc}") from exc

    logger.debug("Loaded YAML config: %s", file_path.name)
    return content


def load_yaml_strict(path: str | Path, required_keys: list[str]) -> dict[str, Any]:
    """
    Load a YAML file and validate that required top-level keys are present.

    Parameters
    ----------
    path:
        Path to the YAML file.
    required_keys:
        List of keys that must exist at the top level of the YAML.

    Returns
    -------
    dict[str, Any]

    Raises
    ------
    KeyError
        If any required key is missing from the parsed YAML.
    """
    content = load_yaml(path)
    missing = [k for k in required_keys if k not in content]
    if missing:
        raise KeyError(
            f"YAML file '{Path(path).name}' is missing required keys: {missing}"
        )
    return content
