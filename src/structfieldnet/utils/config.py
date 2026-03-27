"""JSON configuration utilities for StructFieldNet."""

import json
from pathlib import Path
from typing import Dict, Any, Union


def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON configuration file.

    Args:
        config_path: Path to the JSON file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config suffix is not `.json`.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON config files are supported, got: {path.suffix}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_project_paths(config: Dict[str, Any], project_root: Union[str, Path]) -> Dict[str, Any]:
    """Resolve configured filesystem paths relative to the project root.

    Args:
        config: Parsed configuration dictionary.
        project_root: Absolute or relative project root directory.

    Returns:
        Configuration dictionary with normalized absolute paths.
    """
    resolved_config = dict(config)
    resolved_paths = dict(resolved_config.get("paths", {}))
    project_root = Path(project_root).expanduser().resolve()

    for key in ("dataset_dir", "output_dir"):
        if key not in resolved_paths:
            continue
        raw_path = Path(str(resolved_paths[key])).expanduser()
        resolved_paths[key] = str(raw_path if raw_path.is_absolute() else (project_root / raw_path).resolve())

    resolved_config["paths"] = resolved_paths
    return resolved_config
