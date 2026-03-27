# JSON configuration utilities
# Author: Shengning Wang

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
