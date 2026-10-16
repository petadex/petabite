"""Download the model and corpus into HF_HOME. Run this from a login node.

Compute nodes on this cluster have no outbound network, so anything a job
touches at runtime has to be in the cache first. A job that skips this step
fails inside ``from_pretrained`` after it has already queued and started.

Usage:
    python scripts/prestage_assets.py
    python scripts/prestage_assets.py --model biohub/ESMC-300M --skip-dataset

The ESM-C checkpoint also carries remote modeling code, which transformers
caches separately under ``$HF_HOME/modules``. Loading it once here, rather than
only downloading the files, is what gets that module compiled and cached too.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_MODEL = "biohub/ESMC-300M"
DEFAULT_DATASET = "petadex/catalytic-orfs-90pid"

# The checkpoint's own loader refuses anything older, with a message that is
# easy to miss in a job log.
MIN_ESM_VERSION = "3.4.1"


def check_esm_version() -> None:
    """Fail early if `esm` is too old to load the checkpoint."""
    try:
        import esm
    except ImportError:
        sys.exit(
            "the `esm` package is required to load ESM-C checkpoints.\n"
            f"    pip install 'esm>={MIN_ESM_VERSION}'"
        )

    installed = tuple(int(p) for p in esm.__version__.split(".")[:3] if p.isdigit())
    required = tuple(int(p) for p in MIN_ESM_VERSION.split("."))
    if installed < required:
        sys.exit(
            f"esm {esm.__version__} is installed but the ESM-C checkpoint needs "
            f">={MIN_ESM_VERSION}. Older versions fail with a bare ImportError "
            f"from inside the Hub module loader.\n"
            f"    pip install --upgrade 'esm>={MIN_ESM_VERSION}'"
        )
    print(f"esm {esm.__version__} is new enough")


def stage_model(model_id: str) -> None:
    """Download the checkpoint and load it once, so the remote code is cached."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMaskedLM

    started = time.time()
    path = snapshot_download(repo_id=model_id, max_workers=8)
    print(f"model files at {path} ({time.time() - started:.0f}s)")

    started = time.time()
    model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"loaded {type(model).__name__} with {n_params / 1e6:.1f}M parameters "
        f"({time.time() - started:.0f}s)"
    )

    # The tokenizer is built from tokenizer.json rather than AutoTokenizer, which
    # cannot load this checkpoint. Confirm it works while the network is still up.
    from petabite.data_module.tokenization import load_esmc_tokenizer

    tokenizer = load_esmc_tokenizer(model_id)
    print(
        f"tokenizer ok: {len(tokenizer)} tokens, "
        f"mask={tokenizer.mask_token_id}, pad={tokenizer.pad_token_id}"
    )


def stage_dataset(dataset_id: str) -> None:
    """Download the corpus and report its splits."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    started = time.time()
    path = snapshot_download(repo_id=dataset_id, repo_type="dataset", max_workers=8)
    print(f"dataset files at {path} ({time.time() - started:.0f}s)")

    corpus = load_dataset(dataset_id)
    print(f"splits: { {name: len(split) for name, split in corpus.items()} }")
    first = corpus[next(iter(corpus))]
    print(f"columns: {first.column_names}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()

    if "HF_HOME" not in os.environ:
        sys.exit(
            "HF_HOME is not set, so assets would land in ~/.cache and be "
            "invisible to jobs. Export it first, e.g.\n"
            "    export HF_HOME=/project/6122149/purav/hf_cache"
        )
    print(f"HF_HOME={os.environ['HF_HOME']}")

    check_esm_version()
    if not args.skip_model:
        stage_model(args.model)
    if not args.skip_dataset:
        stage_dataset(args.dataset)

    print("\nprestage complete: jobs can now run with HF_HUB_OFFLINE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
