"""
Project maintenance helpers for StructFieldNet.

This module keeps lightweight research-code utilities in one place:
1. Cleaning Python cache artifacts.
2. Printing a compact project tree for quick inspection.
"""

import pathlib
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set, Union


def setup_project_root(relative_depth: int = 2) -> pathlib.Path:
    """Insert the project root into ``sys.path`` and return it."""
    current_file = pathlib.Path(__file__).resolve()
    project_root = current_file.parents[relative_depth - 1]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def clean_python_artifacts(
    target_dir: Union[str, pathlib.Path] = ".",
    verbose: bool = True,
) -> List[pathlib.Path]:
    """Remove ``__pycache__`` folders and compiled Python artifacts."""
    base_path = pathlib.Path(target_dir).resolve()
    removed_items: List[pathlib.Path] = []

    for pattern in ("*.pyc", "*.pyo"):
        for file_path in base_path.rglob(pattern):
            try:
                file_path.unlink()
                removed_items.append(file_path)
                if verbose:
                    print(f"removed file: {file_path.relative_to(base_path)}")
            except OSError as error:
                if verbose:
                    print(f"failed to remove {file_path}: {error}")

    for cache_dir in base_path.rglob("__pycache__"):
        if cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir)
                removed_items.append(cache_dir)
                if verbose:
                    print(f"removed dir:  {cache_dir.relative_to(base_path)}/")
            except OSError as error:
                if verbose:
                    print(f"failed to remove {cache_dir}: {error}")

    if verbose:
        print(f"cleanup complete: {len(removed_items)} item(s) removed from {base_path}")

    return removed_items


def generate_tree(
    directory: Union[str, pathlib.Path],
    prefix: str = "",
    ignore_dirs: Optional[Set[str]] = None,
    ignore_patterns: Optional[Set[str]] = None,
    max_depth: Optional[int] = None,
    current_depth: int = 0,
) -> str:
    """Generate a formatted directory tree string."""
    path = Path(directory)

    if not path.exists():
        raise ValueError(f"directory does not exist: {directory}")
    if not path.is_dir():
        raise ValueError(f"path is not a directory: {directory}")

    if ignore_dirs is None:
        ignore_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".idea",
            ".vscode",
            ".venv",
            "venv",
            "build",
            "dist",
        }
    if ignore_patterns is None:
        ignore_patterns = {".DS_Store", "*.log", "*.pyc", "*.pyo"}

    if max_depth is not None and current_depth > max_depth:
        return ""

    try:
        items = list(path.iterdir())
    except PermissionError:
        return f"{prefix}[Permission Denied]\n"

    filtered_items = []
    for item in items:
        name = item.name
        if name.startswith(".") and name != ".env":
            continue
        if item.is_dir() and name in ignore_dirs:
            continue

        skip = False
        for pattern in ignore_patterns:
            suffix = pattern.replace("*", "")
            if name == suffix or name.endswith(suffix):
                skip = True
                break
        if not skip:
            filtered_items.append(item)

    filtered_items.sort(key=lambda item: (not item.is_dir(), item.name.lower()))

    lines: List[str] = []
    for index, item in enumerate(filtered_items):
        is_last = index == len(filtered_items) - 1
        connector = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        lines.append(f"{prefix}{connector}{item.name}")

        if item.is_dir():
            subtree = generate_tree(
                item,
                prefix=prefix + next_prefix,
                ignore_dirs=ignore_dirs,
                ignore_patterns=ignore_patterns,
                max_depth=max_depth,
                current_depth=current_depth + 1,
            )
            if subtree:
                lines.append(subtree.rstrip())

    return "\n".join(lines)


def print_tree(
    directory: Optional[Union[str, pathlib.Path]] = None,
    root_name: Optional[str] = None,
    max_depth: Optional[int] = None,
) -> str:
    """Print and return a project tree."""
    if directory is None:
        directory = setup_project_root(relative_depth=2)

    path = Path(directory).resolve()
    display_name = root_name or path.name
    tree_body = generate_tree(path, max_depth=max_depth)
    full_tree = display_name if not tree_body else f"{display_name}\n{tree_body}"
    print(full_tree)
    return full_tree
