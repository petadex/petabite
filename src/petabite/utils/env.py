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
    """Write ``pip freeze`` output to ``out_path`` for run reproducibility."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    out_path.write_text(result.stdout)
