"""Reproducibility utilities: environment snapshot and package list.

Each training run saves its resolved config and the installed-package list
alongside the checkpoint, so a checkpoint found later can be matched to the code
and the environment that produced it.
"""

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
    """Write the installed-package list to ``out_path``.

    This used to try ``uv pip freeze`` first and fall back to pip. The fallback
    was unreachable: ``subprocess.run`` raises ``FileNotFoundError`` when the
    executable is missing rather than returning a non-zero code, so on a machine
    without ``uv`` the whole run died here, after the model had already loaded.
    This environment is module-loaded plus a venv and has no ``uv`` at all.

    Failing to record the package list is never worth losing a run over, so
    everything here is best effort.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        logger.warning("could not run pip freeze (%s); %s not written", exc, out_path)
        return

    if result.returncode != 0:
        logger.warning(
            "pip freeze exited %d; %s not written", result.returncode, out_path
        )
        return
    out_path.write_text(result.stdout)
