"""Reproducibility: environment recording."""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def log_environment() -> dict[str, str]:
    """Collect environment info for reproducibility logging."""
    info: dict[str, str] = {"python_version": platform.python_version()}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = str(torch.version.cuda)
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        )
    except ImportError:
        logger.debug("torch not available; skipping torch env info")
    return info


def dump_pip_freeze(out_path: Path) -> None:
    """Write installed-package list to ``out_path`` for run reproducibility.

    Tries ``uv pip freeze`` first (works in uv-managed venvs that have no pip
    binary), then falls back to ``python -m pip freeze``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for cmd in (["uv", "pip", "freeze"], [sys.executable, "-m", "pip", "freeze"]):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            out_path.write_text(result.stdout)
            return
    logger.warning("pip freeze failed; requirements.txt will not be written")
