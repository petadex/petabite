# Petabite ESM-C PETase Activity Prediction + Active Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold a PyTorch repo that fine-tunes ESM-C (HuggingFace) with LoRA for PETase activity prediction (regression + classification, config-driven) and runs an uncertainty-based pool active-learning loop.

**Architecture:** Factory/Registry pattern throughout. `src/petabite/` holds data/model/trainer/active_learning/utils modules; `run/conf` holds Hydra configs; `run/pipeline` holds entrypoints. First-commit scope is **interfaces + stubs**: registries, configs, and class skeletons are real and importable; heavy logic (ESM-C load, training, MC-dropout) raises `NotImplementedError` with a TODO. Tests assert contracts and import wiring; stubbed behavior is marked `xfail`/`skip`.

**Tech Stack:** Python ≥3.10, uv, PyTorch, HuggingFace `transformers`, `peft` (LoRA), Hydra + OmegaConf, pytest, ruff, mypy. ESM-C via `transformers` (EvolutionaryScale checkpoints).

**Conventions:** Conventional Commits. Files 200–400 lines. Every package `__init__.py` defines `__all__`. Type hints + docstrings on all public functions. `logging.getLogger(__name__)` (no `print`). Specific exceptions only.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project metadata, deps, ruff/mypy/pytest config |
| `README.md` | purpose, install, usage, AL loop overview |
| `src/petabite/__init__.py` | package marker, version |
| `src/petabite/utils/registry.py` | generic `Registry` class |
| `src/petabite/utils/seed.py` | `set_seed()` |
| `src/petabite/utils/env.py` | `log_environment()`, `dump_pip_freeze()` |
| `src/petabite/utils/logging.py` | `setup_logging()` |
| `src/petabite/utils/__init__.py` | re-exports, `__all__` |
| `src/petabite/data_module/utils.py` | CSV/FASTA IO, `dataset_hash()` |
| `src/petabite/data_module/tokenization.py` | `ESMCTokenizerWrapper` |
| `src/petabite/data_module/dataset/petase_dataset.py` | `PETaseDataset(cfg)` |
| `src/petabite/data_module/dataset/__init__.py` | dataset registry |
| `src/petabite/data_module/splits.py` | `make_splits()`, `make_al_split()` |
| `src/petabite/data_module/__init__.py` | factory exports, `__all__` |
| `src/petabite/model_module/backbone/esmc_backbone.py` | `ESMCBackbone(cfg)` HF load + LoRA wrap |
| `src/petabite/model_module/backbone/__init__.py` | backbone registry |
| `src/petabite/model_module/heads/regression_head.py` | `RegressionHead(cfg)` |
| `src/petabite/model_module/heads/classification_head.py` | `ClassificationHead(cfg)` |
| `src/petabite/model_module/heads/__init__.py` | head registry |
| `src/petabite/model_module/model.py` | `ESMCActivityModel(cfg)` |
| `src/petabite/model_module/uncertainty.py` | `mc_dropout_predict()` |
| `src/petabite/model_module/__init__.py` | factories, `__all__` |
| `src/petabite/trainer_module/metrics.py` | `regression_metrics()`, `classification_metrics()` |
| `src/petabite/trainer_module/trainer.py` | `ActivityTrainer` (HF Trainer wrapper) |
| `src/petabite/trainer_module/__init__.py` | exports, `__all__` |
| `src/petabite/active_learning/acquisition/base.py` | `AcquisitionFunction` interface |
| `src/petabite/active_learning/acquisition/uncertainty.py` | `UncertaintyAcquisition` |
| `src/petabite/active_learning/acquisition/random_baseline.py` | `RandomAcquisition` |
| `src/petabite/active_learning/acquisition/__init__.py` | acquisition registry |
| `src/petabite/active_learning/selection.py` | `select_top_k()`, `export_query()` |
| `src/petabite/active_learning/loop.py` | `ActiveLearningLoop` |
| `src/petabite/active_learning/__init__.py` | factory, `__all__` |
| `run/conf/config.yaml` + `data/`, `model/`, `trainer/`, `active_learning/` | Hydra configs |
| `run/pipeline/train.py`, `predict.py`, `active_learning.py` | entrypoints |
| `data/sample_petase.csv` | tiny synthetic example |
| `tests/test_*.py` | contract tests |

---

## Task 1: Project scaffold (uv + pyproject + package skeleton)

**Files:**
- Create: `pyproject.toml`
- Create: `src/petabite/__init__.py`
- Create: `README.md`
- Modify: `.gitignore` (append `outputs/`, `.venv/`, caches)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "petabite"
version = "0.1.0"
description = "ESM-C fine-tuning for PETase activity prediction with active learning"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "transformers>=4.44",
    "peft>=0.11",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "numpy>=1.26",
    "pandas>=2.0",
    "scikit-learn>=1.4",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5", "mypy>=1.10"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/petabite"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
packages = ["petabite"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/petabite/__init__.py`**

```python
"""Petabite: ESM-C fine-tuning for PETase activity prediction with active learning."""

__version__ = "0.1.0"

__all__ = ["__version__"]
```

- [ ] **Step 3: Create `README.md`**

```markdown
# Petabite

ESM-C fine-tuning for **PETase activity prediction** with an uncertainty-based
**active learning** loop.

## Install

```bash
uv venv
uv sync --extra dev
```

## Layout

- `src/petabite/` — data, model, trainer, active_learning, utils modules
- `run/conf/` — Hydra configs
- `run/pipeline/` — train / predict / active_learning entrypoints
- `data/sample_petase.csv` — tiny synthetic example (`sequence,label,id`)

## Usage

```bash
# Train (LoRA fine-tune ESM-C)
uv run python run/pipeline/train.py

# Predict on a CSV
uv run python run/pipeline/predict.py data.csv=path/to/seqs.csv

# Run one active-learning round
uv run python run/pipeline/active_learning.py
```

## Status

First-commit scope is **interfaces + stubs**. Registries, configs, and class
skeletons are wired and importable; ESM-C loading, training, and MC-dropout
raise `NotImplementedError` with TODO markers to be filled in.
```

- [ ] **Step 4: Append to `.gitignore`**

Append these lines (only if not already present):

```
# Petabite
outputs/
.venv/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

- [ ] **Step 5: Create the venv and verify the package imports**

Run:
```bash
uv venv && uv sync --extra dev
uv run python -c "import petabite; print(petabite.__version__)"
```
Expected: prints `0.1.0`

- [ ] **Step 6: Commit**

```bash
git checkout -b feat/petabite-scaffold
git add pyproject.toml src/petabite/__init__.py README.md .gitignore
git commit -m "chore: scaffold petabite uv project and package skeleton"
```

---

## Task 2: Utilities (registry, seed, env, logging)

**Files:**
- Create: `src/petabite/utils/registry.py`
- Create: `src/petabite/utils/seed.py`
- Create: `src/petabite/utils/env.py`
- Create: `src/petabite/utils/logging.py`
- Create: `src/petabite/utils/__init__.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utils.py
from petabite.utils import Registry, set_seed


def test_registry_register_and_get():
    reg: Registry = Registry("demo")

    @reg.register("a")
    class A:
        pass

    assert reg.get("a") is A
    assert "a" in reg.keys()


def test_registry_unknown_key_raises():
    reg: Registry = Registry("demo")
    try:
        reg.get("missing")
    except KeyError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_set_seed_is_deterministic():
    import random
    set_seed(123)
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_utils.py -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` (utils not created).

- [ ] **Step 3: Implement `registry.py`**

```python
# src/petabite/utils/registry.py
"""Generic name->class registry for factory patterns."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry:
    """Maps string keys to classes for config-driven construction."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._table: Dict[str, Type] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator registering a class under ``key``."""

        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._table:
                logger.warning("Overwriting %s registry key '%s'", self._name, key)
            self._table[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> Type:
        """Return the class registered under ``key``."""
        if key not in self._table:
            raise KeyError(
                f"Unknown {self._name} key '{key}'. Valid keys: {sorted(self._table)}"
            )
        return self._table[key]

    def keys(self) -> list[str]:
        """Return registered keys."""
        return sorted(self._table)
```

- [ ] **Step 4: Implement `seed.py`**

```python
# src/petabite/utils/seed.py
"""Reproducibility: global seed control."""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.debug("numpy not available; skipping numpy seed")
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not available; skipping torch seed")
```

- [ ] **Step 5: Implement `env.py`**

```python
# src/petabite/utils/env.py
"""Reproducibility: environment recording."""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def log_environment() -> Dict[str, str]:
    """Collect environment info for reproducibility logging."""
    info: Dict[str, str] = {"python_version": platform.python_version()}
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
        ["python", "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    out_path.write_text(result.stdout)
```

- [ ] **Step 6: Implement `logging.py`**

```python
# src/petabite/utils/logging.py
"""Logging setup helper."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging format once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
```

- [ ] **Step 7: Implement `utils/__init__.py`**

```python
# src/petabite/utils/__init__.py
from .env import dump_pip_freeze, log_environment
from .logging import setup_logging
from .registry import Registry
from .seed import set_seed

__all__ = [
    "Registry",
    "set_seed",
    "log_environment",
    "dump_pip_freeze",
    "setup_logging",
]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: PASS (3 tests).

- [ ] **Step 9: Commit**

```bash
git add src/petabite/utils tests/test_utils.py
git commit -m "feat: add registry, seed, env, and logging utilities"
```

---

## Task 3: Data module IO + tokenization

**Files:**
- Create: `src/petabite/data_module/utils.py`
- Create: `src/petabite/data_module/tokenization.py`
- Create: `data/sample_petase.csv`
- Test: `tests/test_data_io.py`

- [ ] **Step 1: Create the synthetic sample CSV**

```csv
id,sequence,label
pet01,MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGS,0.82
pet02,MKLLVLGLPGAGKGTQAQFIMEKYGIPQIST,0.13
pet03,MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEY,0.45
pet04,MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDK,0.67
pet05,MASNFTQFVLVDNGGTGDVTVAPSNFANGVAEW,0.21
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_data_io.py
from pathlib import Path

from petabite.data_module.utils import dataset_hash, read_activity_csv


def test_read_activity_csv():
    rows = read_activity_csv(Path("data/sample_petase.csv"))
    assert len(rows) == 5
    assert set(rows[0].keys()) >= {"id", "sequence", "label"}
    assert isinstance(rows[0]["sequence"], str)
    assert isinstance(rows[0]["label"], float)


def test_read_activity_csv_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("seq,value\nABC,1.0\n")
    try:
        read_activity_csv(bad)
    except ValueError as e:
        assert "sequence" in str(e)
    else:
        raise AssertionError("expected ValueError for missing columns")


def test_dataset_hash_stable():
    h1 = dataset_hash(Path("data/sample_petase.csv"))
    h2 = dataset_hash(Path("data/sample_petase.csv"))
    assert h1 == h2 and len(h1) == 12
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_data_io.py -v`
Expected: FAIL with `ModuleNotFoundError` (data_module.utils missing).

- [ ] **Step 4: Implement `data_module/utils.py`**

```python
# src/petabite/data_module/utils.py
"""Data IO and hashing utilities."""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"sequence", "label"}


def read_activity_csv(path: Path) -> List[Dict[str, object]]:
    """Read a PETase activity CSV with columns sequence,label[,id].

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Activity CSV not found: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"CSV {path} missing required columns: {sorted(missing)}")
        rows: List[Dict[str, object]] = []
        for i, raw in enumerate(reader):
            rows.append(
                {
                    "id": raw.get("id", str(i)),
                    "sequence": raw["sequence"].strip(),
                    "label": float(raw["label"]),
                }
            )
    return rows


def dataset_hash(path: Path) -> str:
    """Return a 12-char SHA256 prefix of the dataset file for version tracking."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()[:12]
```

- [ ] **Step 5: Implement `data_module/tokenization.py`**

```python
# src/petabite/data_module/tokenization.py
"""ESM-C tokenizer wrapper (HuggingFace)."""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ESMCTokenizerWrapper:
    """Wraps the HuggingFace ESM-C tokenizer for batch encoding.

    Args:
        model_name: HF checkpoint id for the ESM-C tokenizer.
        max_length: max sequence length (truncation).
    """

    def __init__(self, model_name: str, max_length: int = 1024) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None  # lazy-loaded

    def _load(self) -> None:
        # TODO: load HF tokenizer:
        #   from transformers import AutoTokenizer
        #   self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        raise NotImplementedError(
            "ESM-C tokenizer loading not implemented; wire transformers.AutoTokenizer"
        )

    def encode(self, sequences: List[str]) -> Dict[str, "object"]:
        """Tokenize a list of sequences into model input tensors."""
        if self._tokenizer is None:
            self._load()
        # TODO: return self._tokenizer(sequences, padding=True, truncation=True,
        #   max_length=self.max_length, return_tensors="pt")
        raise NotImplementedError("ESM-C tokenization not implemented")
```

- [ ] **Step 6: Run tests to verify IO tests pass**

Run: `uv run pytest tests/test_data_io.py -v`
Expected: PASS (3 tests). (Tokenization is a stub; not tested for behavior here.)

- [ ] **Step 7: Commit**

```bash
git add src/petabite/data_module/utils.py src/petabite/data_module/tokenization.py data/sample_petase.csv tests/test_data_io.py
git commit -m "feat: add data IO, hashing, and ESM-C tokenizer wrapper stub"
```

---

## Task 4: PETaseDataset, splits, and data factory

**Files:**
- Create: `src/petabite/data_module/dataset/petase_dataset.py`
- Create: `src/petabite/data_module/dataset/__init__.py`
- Create: `src/petabite/data_module/splits.py`
- Create: `src/petabite/data_module/__init__.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
from pathlib import Path

from petabite.data_module import DatasetFactory, make_al_split, make_splits


def _records():
    return [
        {"id": str(i), "sequence": "ACDEFGHIKL", "label": float(i) / 10}
        for i in range(10)
    ]


def test_dataset_factory_builds_petase_dataset():
    cls = DatasetFactory("petase")
    ds = cls(records=_records())
    assert len(ds) == 10
    sample = ds[0]
    assert "sequence" in sample and "label" in sample


def test_make_splits_partitions_without_overlap():
    train, val, test = make_splits(_records(), val_frac=0.2, test_frac=0.2, seed=1)
    ids = [r["id"] for r in train + val + test]
    assert len(ids) == len(set(ids)) == 10


def test_make_al_split_labeled_and_pool():
    labeled, pool = make_al_split(_records(), init_labeled=3, seed=1)
    assert len(labeled) == 3 and len(pool) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `dataset/petase_dataset.py`**

```python
# src/petabite/data_module/dataset/petase_dataset.py
"""PETase activity dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from ..utils import read_activity_csv

logger = logging.getLogger(__name__)


class PETaseDataset(Dataset):
    """Sequence-label dataset for PETase activity.

    Provide either ``csv_path`` or pre-loaded ``records``.

    Args:
        csv_path: optional path to a CSV with sequence,label[,id].
        records: optional list of dicts with keys id,sequence,label.
    """

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        records: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        if records is not None:
            self.records = records
        elif csv_path is not None:
            self.records = read_activity_csv(Path(csv_path))
        else:
            raise ValueError("PETaseDataset needs either csv_path or records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.records[idx]
```

- [ ] **Step 4: Implement `dataset/__init__.py` (registry)**

```python
# src/petabite/data_module/dataset/__init__.py
from petabite.utils import Registry

from .petase_dataset import PETaseDataset

DATASET_REGISTRY = Registry("dataset")
DATASET_REGISTRY.register("petase")(PETaseDataset)

__all__ = ["DATASET_REGISTRY", "PETaseDataset"]
```

- [ ] **Step 5: Implement `splits.py`**

```python
# src/petabite/data_module/splits.py
"""Train/val/test and active-learning splits."""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

Record = Dict[str, object]


def make_splits(
    records: List[Record],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Record], List[Record], List[Record]]:
    """Shuffle and partition records into train/val/test."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def make_al_split(
    records: List[Record], init_labeled: int, seed: int = 42
) -> Tuple[List[Record], List[Record]]:
    """Split into an initial labeled set and an unlabeled pool."""
    if init_labeled > len(records):
        raise ValueError("init_labeled exceeds dataset size")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:init_labeled], shuffled[init_labeled:]
```

- [ ] **Step 6: Implement `data_module/__init__.py` (factory)**

```python
# src/petabite/data_module/__init__.py
from .dataset import DATASET_REGISTRY, PETaseDataset
from .splits import make_al_split, make_splits


def DatasetFactory(name: str):
    """Return the dataset class registered under ``name``."""
    return DATASET_REGISTRY.get(name)


__all__ = [
    "DatasetFactory",
    "DATASET_REGISTRY",
    "PETaseDataset",
    "make_splits",
    "make_al_split",
]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS (3 tests).

- [ ] **Step 8: Commit**

```bash
git add src/petabite/data_module tests/test_dataset.py
git commit -m "feat: add PETaseDataset, splits, and dataset factory"
```

---

## Task 5: Model heads + head registry

**Files:**
- Create: `src/petabite/model_module/heads/regression_head.py`
- Create: `src/petabite/model_module/heads/classification_head.py`
- Create: `src/petabite/model_module/heads/__init__.py`
- Test: `tests/test_heads.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_heads.py
import torch

from petabite.model_module.heads import HEAD_REGISTRY


def test_regression_head_output_shape():
    cls = HEAD_REGISTRY.get("regression")
    head = cls(input_dim=16, output_dim=1)
    out = head(torch.randn(4, 16))
    assert out.shape == (4, 1)


def test_classification_head_output_shape():
    cls = HEAD_REGISTRY.get("classification")
    head = cls(input_dim=16, output_dim=3)
    out = head(torch.randn(4, 16))
    assert out.shape == (4, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_heads.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `heads/regression_head.py`**

```python
# src/petabite/model_module/heads/regression_head.py
"""Regression head for continuous activity prediction."""

from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    """MLP head mapping pooled embeddings to a continuous value.

    Args:
        input_dim: embedding dimension from the backbone.
        output_dim: number of regression targets (default 1).
        dropout: dropout probability.
    """

    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return predicted values of shape (batch, output_dim)."""
        return self.net(embeddings)
```

- [ ] **Step 4: Implement `heads/classification_head.py`**

```python
# src/petabite/model_module/heads/classification_head.py
"""Classification head for binned/active-inactive activity prediction."""

from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """MLP head mapping pooled embeddings to class logits.

    Args:
        input_dim: embedding dimension from the backbone.
        output_dim: number of classes.
        dropout: dropout probability.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (batch, output_dim)."""
        return self.net(embeddings)
```

- [ ] **Step 5: Implement `heads/__init__.py` (registry)**

```python
# src/petabite/model_module/heads/__init__.py
from petabite.utils import Registry

from .classification_head import ClassificationHead
from .regression_head import RegressionHead

HEAD_REGISTRY = Registry("head")
HEAD_REGISTRY.register("regression")(RegressionHead)
HEAD_REGISTRY.register("classification")(ClassificationHead)

__all__ = ["HEAD_REGISTRY", "RegressionHead", "ClassificationHead"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_heads.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add src/petabite/model_module/heads tests/test_heads.py
git commit -m "feat: add regression and classification heads with registry"
```

---

## Task 6: ESM-C backbone (LoRA) + model composition

**Files:**
- Create: `src/petabite/model_module/backbone/esmc_backbone.py`
- Create: `src/petabite/model_module/backbone/__init__.py`
- Create: `src/petabite/model_module/model.py`
- Create: `src/petabite/model_module/uncertainty.py`
- Create: `src/petabite/model_module/__init__.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import pytest

from petabite.model_module import BackboneFactory, ModelFactory
from petabite.model_module.backbone import BACKBONE_REGISTRY


def test_backbone_registered():
    assert "esmc" in BACKBONE_REGISTRY.keys()


def test_model_factory_returns_class():
    assert ModelFactory("esmc_activity") is not None
    assert BackboneFactory("esmc") is not None


@pytest.mark.xfail(reason="ESM-C backbone load is a stub", strict=True)
def test_backbone_forward_stub_raises():
    cls = BackboneFactory("esmc")
    backbone = cls(model_name="dummy", lora_r=8)
    backbone.forward(input_ids=None, attention_mask=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backbone/esmc_backbone.py`**

```python
# src/petabite/model_module/backbone/esmc_backbone.py
"""ESM-C backbone with LoRA adapters (HuggingFace + peft)."""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ESMCBackbone(nn.Module):
    """Loads a HuggingFace ESM-C model and wraps it with LoRA adapters.

    The base weights are frozen; only LoRA adapters train.

    Args:
        model_name: HF checkpoint id (e.g. an EvolutionaryScale ESM-C model).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
    """

    def __init__(
        self,
        model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.hidden_size: int = 0  # set after loading
        self._model = None  # lazy-loaded
        # TODO: load and wrap:
        #   from transformers import AutoModel
        #   from peft import LoraConfig, get_peft_model
        #   base = AutoModel.from_pretrained(model_name)
        #   for p in base.parameters(): p.requires_grad = False
        #   cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
        #                    lora_dropout=lora_dropout, target_modules=[...])
        #   self._model = get_peft_model(base, cfg)
        #   self.hidden_size = base.config.hidden_size

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return pooled (mean over residues) embeddings: (batch, hidden_size)."""
        # TODO: out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        #   masked mean-pool out.last_hidden_state with attention_mask
        raise NotImplementedError("ESM-C backbone forward not implemented")
```

- [ ] **Step 4: Implement `backbone/__init__.py` (registry)**

```python
# src/petabite/model_module/backbone/__init__.py
from petabite.utils import Registry

from .esmc_backbone import ESMCBackbone

BACKBONE_REGISTRY = Registry("backbone")
BACKBONE_REGISTRY.register("esmc")(ESMCBackbone)

__all__ = ["BACKBONE_REGISTRY", "ESMCBackbone"]
```

- [ ] **Step 5: Implement `model.py`**

```python
# src/petabite/model_module/model.py
"""Composed ESM-C activity model: backbone + task head."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch import nn

from .backbone import BACKBONE_REGISTRY
from .heads import HEAD_REGISTRY

logger = logging.getLogger(__name__)

_LOSSES = {
    "regression": nn.MSELoss,
    "classification": nn.CrossEntropyLoss,
}


class ESMCActivityModel(nn.Module):
    """ESM-C backbone + regression/classification head.

    Args:
        backbone_name: registry key for the backbone (e.g. "esmc").
        task: "regression" or "classification".
        model_name: HF checkpoint id passed to the backbone.
        output_dim: head output dim (1 for regression, n_classes for clf).
        lora_r / lora_alpha / lora_dropout: LoRA hyperparameters.
    """

    def __init__(
        self,
        backbone_name: str,
        task: str,
        model_name: str,
        output_dim: int = 1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if task not in _LOSSES:
            raise ValueError(f"Unknown task '{task}'. Valid: {sorted(_LOSSES)}")
        self.task = task
        backbone_cls = BACKBONE_REGISTRY.get(backbone_name)
        self.backbone = backbone_cls(
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        head_cls = HEAD_REGISTRY.get(task)
        # hidden_size is 0 until backbone load is implemented; head built lazily.
        self._head_cls = head_cls
        self._output_dim = output_dim
        self.head: Optional[nn.Module] = None
        self.loss_fn = _LOSSES[task]()

    def _ensure_head(self) -> None:
        if self.head is None:
            self.head = self._head_cls(
                input_dim=self.backbone.hidden_size, output_dim=self._output_dim
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return dict with 'logits' and (if labels given) 'loss'."""
        embeddings = self.backbone(input_ids, attention_mask)
        self._ensure_head()
        assert self.head is not None
        logits = self.head(embeddings)
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            target = labels if self.task == "classification" else labels.float()
            out["loss"] = self.loss_fn(logits.squeeze(-1), target)
        return out
```

- [ ] **Step 6: Implement `uncertainty.py`**

```python
# src/petabite/model_module/uncertainty.py
"""MC-dropout uncertainty estimation."""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)


def enable_dropout(model: nn.Module) -> None:
    """Set all dropout layers to train mode while keeping the rest in eval."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_samples: int = 20,
) -> Dict[str, torch.Tensor]:
    """Run ``n_samples`` stochastic forward passes; return mean and variance.

    Returns:
        dict with 'mean' and 'variance' over the sampled logits.
    """
    # TODO: implement once backbone.forward works. Sketch:
    #   model.eval(); enable_dropout(model)
    #   preds = torch.stack([
    #       model(input_ids, attention_mask)["logits"] for _ in range(n_samples)
    #   ])
    #   return {"mean": preds.mean(0), "variance": preds.var(0)}
    raise NotImplementedError("MC-dropout prediction not implemented")
```

- [ ] **Step 7: Implement `model_module/__init__.py` (factories)**

```python
# src/petabite/model_module/__init__.py
from .backbone import BACKBONE_REGISTRY, ESMCBackbone
from .heads import HEAD_REGISTRY, ClassificationHead, RegressionHead
from .model import ESMCActivityModel
from .uncertainty import enable_dropout, mc_dropout_predict

_MODELS = {"esmc_activity": ESMCActivityModel}


def ModelFactory(name: str):
    """Return the model class registered under ``name``."""
    if name not in _MODELS:
        raise KeyError(f"Unknown model '{name}'. Valid: {sorted(_MODELS)}")
    return _MODELS[name]


def BackboneFactory(name: str):
    """Return the backbone class registered under ``name``."""
    return BACKBONE_REGISTRY.get(name)


__all__ = [
    "ModelFactory",
    "BackboneFactory",
    "ESMCActivityModel",
    "ESMCBackbone",
    "RegressionHead",
    "ClassificationHead",
    "HEAD_REGISTRY",
    "BACKBONE_REGISTRY",
    "mc_dropout_predict",
    "enable_dropout",
]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS (2 pass, 1 xfail).

- [ ] **Step 9: Commit**

```bash
git add src/petabite/model_module tests/test_model.py
git commit -m "feat: add ESM-C LoRA backbone stub, model composition, and MC-dropout"
```

---

## Task 7: Metrics + trainer wrapper

**Files:**
- Create: `src/petabite/trainer_module/metrics.py`
- Create: `src/petabite/trainer_module/trainer.py`
- Create: `src/petabite/trainer_module/__init__.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np

from petabite.trainer_module.metrics import classification_metrics, regression_metrics


def test_regression_metrics_perfect():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    m = regression_metrics(y_true=y, y_pred=y)
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0
    assert "spearman" in m


def test_classification_metrics_perfect():
    y = np.array([0, 1, 1, 0])
    m = classification_metrics(y_true=y, y_pred=y)
    assert m["accuracy"] == 1.0
    assert "f1" in m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `metrics.py`**

```python
# src/petabite/trainer_module/metrics.py
"""Task metrics for regression and classification."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, R^2, and Spearman correlation."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    rho = spearmanr(y_true, y_pred).correlation
    return {"rmse": rmse, "r2": r2, "spearman": float(rho) if rho == rho else 0.0}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy and macro F1."""
    accuracy = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return {"accuracy": accuracy, "f1": f1}
```

- [ ] **Step 4: Implement `trainer.py`**

```python
# src/petabite/trainer_module/trainer.py
"""Training wrapper around HuggingFace Trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from petabite.utils import dump_pip_freeze, log_environment, set_seed

logger = logging.getLogger(__name__)


class ActivityTrainer:
    """Configures and runs training for ESM-C activity models.

    Wraps transformers.Trainer; saves reproducibility artifacts (resolved
    config + pip freeze) to the output directory.

    Args:
        model: an ESMCActivityModel instance.
        output_dir: directory for checkpoints and run artifacts.
        task: "regression" or "classification" (selects metrics).
        seed: random seed.
    """

    def __init__(
        self,
        model: object,
        output_dir: Path,
        task: str,
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.model = model
        self.output_dir = Path(output_dir)
        self.task = task
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Environment: %s", log_environment())

    def train(
        self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> None:
        """Run training. Saves reproducibility artifacts then trains."""
        dump_pip_freeze(self.output_dir / "requirements.txt")
        # TODO: build transformers.TrainingArguments + Trainer with a
        #   data collator that calls ESMCTokenizerWrapper.encode, and
        #   compute_metrics dispatching on self.task. Then trainer.train().
        raise NotImplementedError("HF Trainer wiring not implemented")
```

- [ ] **Step 5: Implement `trainer_module/__init__.py`**

```python
# src/petabite/trainer_module/__init__.py
from .metrics import classification_metrics, regression_metrics
from .trainer import ActivityTrainer

__all__ = ["ActivityTrainer", "regression_metrics", "classification_metrics"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add src/petabite/trainer_module tests/test_metrics.py
git commit -m "feat: add metrics and ActivityTrainer wrapper stub"
```

---

## Task 8: Active learning acquisition + loop

**Files:**
- Create: `src/petabite/active_learning/acquisition/base.py`
- Create: `src/petabite/active_learning/acquisition/uncertainty.py`
- Create: `src/petabite/active_learning/acquisition/random_baseline.py`
- Create: `src/petabite/active_learning/acquisition/__init__.py`
- Create: `src/petabite/active_learning/selection.py`
- Create: `src/petabite/active_learning/loop.py`
- Create: `src/petabite/active_learning/__init__.py`
- Test: `tests/test_acquisition.py`, `tests/test_al_loop.py`

- [ ] **Step 1: Write the failing acquisition test**

```python
# tests/test_acquisition.py
import numpy as np

from petabite.active_learning import AcquisitionFactory


def test_random_acquisition_reproducible():
    cls = AcquisitionFactory("random")
    acq = cls(seed=7)
    scores_a = acq.score(pool_size=10)
    acq_b = cls(seed=7)
    scores_b = acq_b.score(pool_size=10)
    assert np.allclose(scores_a, scores_b)


def test_uncertainty_acquisition_ranks_by_variance():
    cls = AcquisitionFactory("uncertainty")
    acq = cls()
    variances = np.array([0.1, 0.9, 0.5])
    scores = acq.score_from_variance(variances)
    assert np.argmax(scores) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_acquisition.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `acquisition/base.py`**

```python
# src/petabite/active_learning/acquisition/base.py
"""Acquisition function interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AcquisitionFunction(ABC):
    """Scores pool items; higher score == higher priority to label."""

    @abstractmethod
    def score(self, pool_size: int, **kwargs: object) -> np.ndarray:
        """Return a 1-D array of length ``pool_size`` of acquisition scores."""
        raise NotImplementedError
```

- [ ] **Step 4: Implement `acquisition/random_baseline.py`**

```python
# src/petabite/active_learning/acquisition/random_baseline.py
"""Random acquisition baseline."""

from __future__ import annotations

import numpy as np

from .base import AcquisitionFunction


class RandomAcquisition(AcquisitionFunction):
    """Assigns uniform random scores (reproducible under ``seed``)."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def score(self, pool_size: int, **kwargs: object) -> np.ndarray:
        """Return random scores of length ``pool_size``."""
        return self._rng.random(pool_size)
```

- [ ] **Step 5: Implement `acquisition/uncertainty.py`**

```python
# src/petabite/active_learning/acquisition/uncertainty.py
"""Uncertainty (MC-dropout variance) acquisition."""

from __future__ import annotations

import logging

import numpy as np

from .base import AcquisitionFunction

logger = logging.getLogger(__name__)


class UncertaintyAcquisition(AcquisitionFunction):
    """Scores pool items by predictive variance (higher == more uncertain)."""

    def score_from_variance(self, variances: np.ndarray) -> np.ndarray:
        """Return scores directly proportional to per-item variance."""
        return np.asarray(variances, dtype=float)

    def score(self, pool_size: int, **kwargs: object) -> np.ndarray:
        """Score the pool using precomputed ``variances`` (kwarg).

        Raises:
            ValueError: if ``variances`` is missing or wrong length.
        """
        variances = kwargs.get("variances")
        if variances is None:
            raise ValueError(
                "UncertaintyAcquisition.score requires a 'variances' kwarg "
                "(from model_module.mc_dropout_predict)"
            )
        arr = np.asarray(variances, dtype=float)
        if arr.shape[0] != pool_size:
            raise ValueError("variances length must equal pool_size")
        return arr
```

- [ ] **Step 6: Implement `acquisition/__init__.py` (registry)**

```python
# src/petabite/active_learning/acquisition/__init__.py
from petabite.utils import Registry

from .base import AcquisitionFunction
from .random_baseline import RandomAcquisition
from .uncertainty import UncertaintyAcquisition

ACQUISITION_REGISTRY = Registry("acquisition")
ACQUISITION_REGISTRY.register("random")(RandomAcquisition)
ACQUISITION_REGISTRY.register("uncertainty")(UncertaintyAcquisition)

__all__ = [
    "ACQUISITION_REGISTRY",
    "AcquisitionFunction",
    "RandomAcquisition",
    "UncertaintyAcquisition",
]
```

- [ ] **Step 7: Implement `selection.py`**

```python
# src/petabite/active_learning/selection.py
"""Batch selection and wet-lab query export."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

Record = Dict[str, object]


def select_top_k(scores: np.ndarray, k: int) -> List[int]:
    """Return indices of the top-``k`` highest scores (descending)."""
    if k > len(scores):
        raise ValueError("k exceeds number of scored items")
    return list(np.argsort(scores)[::-1][:k])


def export_query(
    pool: List[Record], indices: List[int], out_path: Path, round_idx: int
) -> Path:
    """Write selected pool records to a CSV for wet-lab labeling."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "id", "sequence"])
        for i in indices:
            rec = pool[i]
            writer.writerow([round_idx, rec.get("id", i), rec["sequence"]])
    logger.info("Wrote %d query records to %s", len(indices), out_path)
    return out_path
```

- [ ] **Step 8: Implement `loop.py`**

```python
# src/petabite/active_learning/loop.py
"""Pool-based active learning round controller."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .acquisition import ACQUISITION_REGISTRY
from .selection import export_query, select_top_k

logger = logging.getLogger(__name__)

Record = Dict[str, object]


class ActiveLearningLoop:
    """Runs one active-learning round: score pool, select, export query.

    Args:
        acquisition_name: registry key ("uncertainty" or "random").
        query_size: number of items to select per round.
        output_dir: directory for query CSVs.
        acquisition_kwargs: kwargs forwarded to the acquisition constructor.
    """

    def __init__(
        self,
        acquisition_name: str,
        query_size: int,
        output_dir: Path,
        **acquisition_kwargs: object,
    ) -> None:
        acq_cls = ACQUISITION_REGISTRY.get(acquisition_name)
        self.acquisition = acq_cls(**acquisition_kwargs)
        self.query_size = query_size
        self.output_dir = Path(output_dir)

    def run_round(
        self,
        labeled: List[Record],
        pool: List[Record],
        round_idx: int,
        scores: np.ndarray | None = None,
        **score_kwargs: object,
    ) -> Tuple[List[int], Path]:
        """Score the pool, select top-K, and export a query CSV.

        ``scores`` may be passed precomputed; otherwise the acquisition
        function is asked to score a pool of ``len(pool)``.
        """
        if scores is None:
            scores = self.acquisition.score(pool_size=len(pool), **score_kwargs)
        selected = select_top_k(scores, self.query_size)
        query_path = export_query(
            pool, selected, self.output_dir / f"query_round_{round_idx}.csv", round_idx
        )
        logger.info(
            "AL round %d: %d labeled, %d pool, selected %d",
            round_idx,
            len(labeled),
            len(pool),
            len(selected),
        )
        return selected, query_path
```

- [ ] **Step 9: Implement `active_learning/__init__.py` (factory)**

```python
# src/petabite/active_learning/__init__.py
from .acquisition import (
    ACQUISITION_REGISTRY,
    AcquisitionFunction,
    RandomAcquisition,
    UncertaintyAcquisition,
)
from .loop import ActiveLearningLoop
from .selection import export_query, select_top_k


def AcquisitionFactory(name: str):
    """Return the acquisition class registered under ``name``."""
    return ACQUISITION_REGISTRY.get(name)


__all__ = [
    "AcquisitionFactory",
    "ACQUISITION_REGISTRY",
    "AcquisitionFunction",
    "RandomAcquisition",
    "UncertaintyAcquisition",
    "ActiveLearningLoop",
    "select_top_k",
    "export_query",
]
```

- [ ] **Step 10: Write the failing AL loop test**

```python
# tests/test_al_loop.py
from pathlib import Path

import numpy as np

from petabite.active_learning import ActiveLearningLoop


def _records(n):
    return [{"id": str(i), "sequence": "ACDEFG"} for i in range(n)]


def test_al_loop_run_round_exports_query(tmp_path: Path):
    loop = ActiveLearningLoop(
        acquisition_name="uncertainty", query_size=2, output_dir=tmp_path
    )
    pool = _records(5)
    variances = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    selected, query_path = loop.run_round(
        labeled=_records(2), pool=pool, round_idx=1, variances=variances
    )
    assert selected == [1, 3]
    assert query_path.exists()
    assert query_path.name == "query_round_1.csv"
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `uv run pytest tests/test_acquisition.py tests/test_al_loop.py -v`
Expected: PASS (acquisition: 2 tests; al_loop: 1 test).

- [ ] **Step 12: Commit**

```bash
git add src/petabite/active_learning tests/test_acquisition.py tests/test_al_loop.py
git commit -m "feat: add active-learning acquisition, selection, and round loop"
```

---

## Task 9: Hydra configs

**Files:**
- Create: `run/conf/config.yaml`
- Create: `run/conf/data/petase.yaml`
- Create: `run/conf/model/esmc_lora.yaml`
- Create: `run/conf/trainer/default.yaml`
- Create: `run/conf/active_learning/uncertainty.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from hydra import compose, initialize_config_dir
from pathlib import Path


def test_config_composes():
    conf_dir = str(Path("run/conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config")
    assert cfg.model.task in {"regression", "classification"}
    assert cfg.model.backbone_name == "esmc"
    assert cfg.active_learning.query_size > 0
    assert cfg.data.csv_path.endswith(".csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (config files missing).

- [ ] **Step 3: Create `run/conf/config.yaml`**

```yaml
defaults:
  - data: petase
  - model: esmc_lora
  - trainer: default
  - active_learning: uncertainty
  - _self_

seed: 42
output_dir: outputs
```

- [ ] **Step 4: Create `run/conf/data/petase.yaml`**

```yaml
name: petase
csv_path: data/sample_petase.csv
val_frac: 0.1
test_frac: 0.1
init_labeled: 3
max_length: 1024
```

- [ ] **Step 5: Create `run/conf/model/esmc_lora.yaml`**

```yaml
name: esmc_activity
backbone_name: esmc
# TODO: set to a real ESM-C HF checkpoint id when wiring the backbone.
model_name: EvolutionaryScale/esmc-300m-2024-12
task: regression          # or: classification
output_dim: 1             # 1 for regression; n_classes for classification
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
```

- [ ] **Step 6: Create `run/conf/trainer/default.yaml`**

```yaml
num_train_epochs: 5
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
learning_rate: 1.0e-4
weight_decay: 0.01
logging_steps: 10
```

- [ ] **Step 7: Create `run/conf/active_learning/uncertainty.yaml`**

```yaml
acquisition_name: uncertainty
query_size: 16
n_mc_samples: 20
n_rounds: 5
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add run/conf tests/test_config.py
git commit -m "feat: add Hydra configs for data, model, trainer, and active learning"
```

---

## Task 10: Pipeline entrypoints

**Files:**
- Create: `run/pipeline/train.py`
- Create: `run/pipeline/predict.py`
- Create: `run/pipeline/active_learning.py`
- Test: `tests/test_pipeline_imports.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_imports.py
import importlib.util
from pathlib import Path


def _load(name: str):
    path = Path("run/pipeline") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_modules_expose_main():
    for name in ["train", "predict", "active_learning"]:
        mod = _load(name)
        assert hasattr(mod, "main"), f"{name}.py must define main()"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline_imports.py -v`
Expected: FAIL (pipeline files missing).

- [ ] **Step 3: Create `run/pipeline/train.py`**

```python
# run/pipeline/train.py
"""Train entrypoint: LoRA fine-tune ESM-C on PETase activity data."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from petabite.data_module import DatasetFactory, make_splits
from petabite.data_module.utils import read_activity_csv
from petabite.model_module import ModelFactory
from petabite.trainer_module import ActivityTrainer
from petabite.utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build dataset, model, and trainer from config, then train."""
    setup_logging()
    records = read_activity_csv(Path(cfg.data.csv_path))
    train_recs, val_recs, _ = make_splits(
        records, cfg.data.val_frac, cfg.data.test_frac, cfg.seed
    )
    dataset_cls = DatasetFactory(cfg.data.name)
    train_ds = dataset_cls(records=train_recs)
    val_ds = dataset_cls(records=val_recs)

    model_cls = ModelFactory(cfg.model.name)
    model = model_cls(
        backbone_name=cfg.model.backbone_name,
        task=cfg.model.task,
        model_name=cfg.model.model_name,
        output_dim=cfg.model.output_dim,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
    )
    trainer = ActivityTrainer(
        model=model,
        output_dir=Path(cfg.output_dir),
        task=cfg.model.task,
        seed=cfg.seed,
    )
    trainer.train(train_ds, val_ds)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `run/pipeline/predict.py`**

```python
# run/pipeline/predict.py
"""Predict entrypoint: run inference on a CSV of sequences."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from petabite.data_module import DatasetFactory
from petabite.data_module.utils import read_activity_csv
from petabite.model_module import ModelFactory
from petabite.utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load model and predict activity for each sequence in the input CSV."""
    setup_logging()
    records = read_activity_csv(Path(cfg.data.csv_path))
    dataset_cls = DatasetFactory(cfg.data.name)
    dataset = dataset_cls(records=records)

    model_cls = ModelFactory(cfg.model.name)
    _ = model_cls(
        backbone_name=cfg.model.backbone_name,
        task=cfg.model.task,
        model_name=cfg.model.model_name,
        output_dim=cfg.model.output_dim,
    )
    # TODO: load trained checkpoint, tokenize dataset, run forward, write
    #   predictions CSV to cfg.output_dir.
    raise NotImplementedError(
        f"Prediction loop not implemented (dataset has {len(dataset)} records)"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `run/pipeline/active_learning.py`**

```python
# run/pipeline/active_learning.py
"""Active-learning entrypoint: train, score pool, export wet-lab queries."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from petabite.active_learning import ActiveLearningLoop
from petabite.data_module import make_al_split
from petabite.data_module.utils import read_activity_csv
from petabite.utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one active-learning round and export a query CSV.

    NOTE: scoring with real model uncertainty requires the backbone/trainer
    stubs to be implemented. Until then this wires the loop and selection.
    """
    setup_logging()
    records = read_activity_csv(Path(cfg.data.csv_path))
    labeled, pool = make_al_split(records, cfg.data.init_labeled, cfg.seed)

    loop = ActiveLearningLoop(
        acquisition_name=cfg.active_learning.acquisition_name,
        query_size=min(cfg.active_learning.query_size, len(pool)),
        output_dir=Path(cfg.output_dir),
    )
    # TODO: train model on `labeled`, run mc_dropout_predict over `pool`,
    #   pass variances into loop.run_round(..., variances=variances).
    raise NotImplementedError(
        f"AL scoring needs the model stubs implemented "
        f"({len(labeled)} labeled, {len(pool)} pool ready)"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline_imports.py -v`
Expected: PASS (the modules import and expose `main`; `@hydra.main` does not run at import).

- [ ] **Step 7: Commit**

```bash
git add run/pipeline tests/test_pipeline_imports.py
git commit -m "feat: add train, predict, and active-learning entrypoints"
```

---

## Task 11: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass except the explicitly-marked `xfail` in `tests/test_model.py`.

- [ ] **Step 2: Lint**

Run: `uv run ruff check .`
Expected: no errors. Fix any reported issues, then re-run.

- [ ] **Step 3: Type check**

Run: `uv run mypy src/petabite`
Expected: no errors (note `ignore_missing_imports = true` covers untyped deps). Fix any real type issues.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: pass lint and type checks across petabite scaffold"
```

- [ ] **Step 5: Final status**

Run: `git log --oneline feat/petabite-scaffold` and confirm the branch contains the scaffold commits. The repo now has a complete, importable, tested interfaces-and-stubs skeleton; remaining `NotImplementedError` sites (ESM-C tokenizer/backbone load, MC-dropout, HF Trainer wiring, predict/AL scoring loops) are the next implementation milestone.

---

## Self-Review Notes

- **Spec coverage:** data_module (Tasks 3–4), model_module backbone+heads+model+uncertainty (Tasks 5–6), trainer+metrics (Task 7), active_learning acquisition+loop+selection (Task 8), Hydra configs (Task 9), pipeline entrypoints (Task 10), utils/registry/seed/env (Task 2), reproducibility artifacts (Task 7 trainer), sample CSV (Task 3), tests for dataset/model/acquisition/AL loop (Tasks 4,6,8). All spec sections mapped.
- **Stub scope honored:** ESM-C load, tokenization, MC-dropout, HF Trainer, predict/AL scoring raise `NotImplementedError` with TODO; tests assert contracts and import wiring; one `xfail` for the backbone stub.
- **Type/name consistency:** `Registry.get/register/keys`; `DatasetFactory`, `ModelFactory`, `BackboneFactory`, `AcquisitionFactory`; `ESMCActivityModel(backbone_name, task, model_name, output_dim, lora_*)`; `AcquisitionFunction.score(pool_size, **kwargs)` with `UncertaintyAcquisition.score_from_variance`; `ActiveLearningLoop.run_round(labeled, pool, round_idx, scores=None, **score_kwargs)`; `select_top_k(scores, k)` / `export_query(pool, indices, out_path, round_idx)` — all consistent across tasks.
