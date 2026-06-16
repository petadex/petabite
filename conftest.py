"""Pytest configuration: anchor the working directory to the repo root.

Several tests reference repo-relative paths (``data/sample_petase.csv``,
``run/conf``, ``run/pipeline``). Changing into the repo root at collection time
keeps them robust regardless of the directory pytest is invoked from (e.g. CI).
"""

from __future__ import annotations

import os
from pathlib import Path

os.chdir(Path(__file__).parent)
