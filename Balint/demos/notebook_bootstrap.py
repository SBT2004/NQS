"""Notebook bootstrap helpers for the retained user-facing demos."""

from __future__ import annotations

import sys
from pathlib import Path

def _find_project_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "src" / "nqs").is_dir() and (candidate / "demos").is_dir():
            return candidate

    raise RuntimeError(f"Could not locate the project root from {start!r}.")


def ensure_repo_root_on_path(start: str | Path | None = None) -> Path:
    """Insert the project root and source root on ``sys.path`` and return the project root."""

    anchor = Path.cwd() if start is None else Path(start)
    project_root = _find_project_root(anchor)
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    source_root = project_root / "src"
    source_root_str = str(source_root)
    if source_root.is_dir() and source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)
    return project_root


def enable_notebook_x64() -> None:
    """Enable JAX x64 for notebook-only exact/reference workflows."""

    import jax

    jax.config.update("jax_enable_x64", True)


def bootstrap_notebook(start: str | Path | None = None, *, enable_x64: bool = True) -> Path:
    """Prepare imports for a notebook executed from any ``demos/`` directory."""

    project_root = ensure_repo_root_on_path(start)
    if enable_x64:
        enable_notebook_x64()
    return project_root


__all__ = ["bootstrap_notebook", "enable_notebook_x64", "ensure_repo_root_on_path"]
