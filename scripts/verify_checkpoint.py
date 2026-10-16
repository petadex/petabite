"""Verify a saved LoRA checkpoint is complete, reloadable, and actually trained.

This is the check that the original fused-LoRA bug would have failed. That bug
saved only the ``out_proj`` adapters, 12.5% of the trained tensors, and nothing
compared the file against the model, so the loss curve looked healthy and the
checkpoint was useless.

Three questions are asked, because passing the first two is not enough:

  1. Is it complete? Every adapter the configured targets imply must be
     present, with the right shapes.
  2. Does it reload? Loading into a freshly built model must reproduce the
     tensors on disk exactly.
  3. Was it trained? ``lora_B`` is zero-initialised, so an adapter that is
     still all zeros means no gradient ever reached it. A checkpoint can be
     complete and reloadable and still be worthless.

Usage:
    python scripts/verify_checkpoint.py outputs/esmc-mlm-lora
    python scripts/verify_checkpoint.py outputs/esmc-mlm-lora --model biohub/ESMC-300M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from petabite.model_module.lora import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    LoraConfig,
    apply_lora,
    load_lora,
    trainable_parameters,
)

# How many of the 30 blocks must be adapted for a checkpoint to be considered
# full coverage. ESM-C 300M has 30 transformer blocks.
EXPECTED_BLOCKS = 30


def read_manifest(directory: Path) -> dict:
    """Load the manifest written beside the weights."""
    path = directory / MANIFEST_NAME
    if not path.exists():
        sys.exit(f"no manifest at {path}; this is not a petabite LoRA checkpoint")
    return json.loads(path.read_text())


def check_completeness(manifest: dict) -> tuple[bool, list[str]]:
    """Check the manifest covers every target in every block."""
    problems: list[str] = []
    targets = manifest["config"]["targets"]
    tensors = manifest["tensors"]

    # Each adapted projection contributes a down and an up tensor.
    expected_per_block = len(targets) * 2
    expected_total = expected_per_block * EXPECTED_BLOCKS
    if len(tensors) != expected_total:
        problems.append(
            f"expected {expected_total} tensors "
            f"({len(targets)} targets x 2 matrices x {EXPECTED_BLOCKS} blocks), "
            f"found {len(tensors)}"
        )

    # The failure mode being guarded against is one target surviving and the
    # rest being filtered out, so count per target name.
    for target in targets:
        matching = [name for name in tensors if _target_of(name) == target]
        if len(matching) != EXPECTED_BLOCKS * 2:
            problems.append(
                f"target {target!r}: expected {EXPECTED_BLOCKS * 2} tensors, "
                f"found {len(matching)}"
            )

    injection = manifest.get("injection")
    if injection and injection.get("adapters") != EXPECTED_BLOCKS * 3:
        problems.append(
            f"injection report says {injection.get('adapters')} adapted modules, "
            f"expected {EXPECTED_BLOCKS * 3}"
        )
    return not problems, problems


def _target_of(param_name: str) -> str:
    """Map a parameter name back to the projection it adapts.

    Names look like ``...attn.layernorm_qkv.lora.down.weight`` for QKV and
    ``...ffn.lora_fc1.up.weight`` for the FFN halves.
    """
    if "lora_fc1" in param_name:
        return "fc1"
    if "lora_fc2" in param_name:
        return "fc2"
    if "out_proj" in param_name:
        return "out_proj"
    return "qkv"


def check_trained(directory: Path) -> tuple[bool, list[str]]:
    """Check the up-projections moved away from their zero initialisation."""
    from safetensors.torch import load_file

    tensors = load_file(str(directory / WEIGHTS_NAME))
    up_tensors = {n: t for n, t in tensors.items() if n.endswith("up.weight")}
    if not up_tensors:
        return False, ["no up-projection tensors found, so nothing to check"]

    all_zero = [n for n, t in up_tensors.items() if not t.any()]
    problems = []
    if len(all_zero) == len(up_tensors):
        problems.append(
            f"every one of the {len(up_tensors)} up-projections is still exactly "
            "zero, which is the LoRA initialisation. No gradient reached any "
            "adapter, so this checkpoint encodes no training at all."
        )
    elif all_zero:
        problems.append(
            f"{len(all_zero)} of {len(up_tensors)} up-projections are still "
            f"exactly zero, so those adapters were never trained: "
            f"{all_zero[:4]}"
        )

    magnitudes = torch.stack([t.float().abs().mean() for t in up_tensors.values()])
    print(
        f"  up-projection magnitude: mean {magnitudes.mean():.3e}, "
        f"min {magnitudes.min():.3e}, max {magnitudes.max():.3e}"
    )
    return not problems, problems


def check_reload(directory: Path, model_id: str, manifest: dict) -> tuple[bool, list[str]]:
    """Build a fresh model, load the checkpoint, and compare tensor by tensor."""
    from transformers import AutoModelForMaskedLM

    from safetensors.torch import load_file

    cfg = manifest["config"]
    lora_cfg = LoraConfig(
        r=cfg["r"], alpha=cfg["alpha"], dropout=cfg["dropout"],
        targets=tuple(cfg["targets"]), verify=True,
    )

    print(f"  building a fresh {model_id} to load into")
    model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    apply_lora(model, lora_cfg)

    n_loaded = load_lora(model, directory, strict=True)
    on_disk = load_file(str(directory / WEIGHTS_NAME))
    in_model = trainable_parameters(model)

    problems = []
    if set(on_disk) != set(in_model):
        problems.append("key sets differ after load, which load_lora should have caught")
    for name, expected in on_disk.items():
        actual = in_model[name].cpu()
        if not torch.equal(actual, expected):
            diff = (actual.float() - expected.float()).abs().max().item()
            problems.append(f"{name} differs after reload: max |diff| = {diff:.3e}")
            if len(problems) > 4:
                break
    print(f"  reloaded {n_loaded} tensors into a fresh model")
    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", default="biohub/ESMC-300M")
    parser.add_argument(
        "--skip-reload",
        action="store_true",
        help="skip the reload check, which needs to build the full backbone",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        sys.exit(f"{args.directory} is not a directory")

    manifest = read_manifest(args.directory)
    print(f"checkpoint: {args.directory}")
    print(f"  format   : {manifest['format']}")
    print(f"  config   : {manifest['config']}")
    print(
        f"  contents : {manifest['num_tensors']} tensors, "
        f"{manifest['num_parameters']:,} parameters"
    )

    results: list[tuple[str, bool, list[str]]] = []

    print("\n[1/3] completeness")
    ok, problems = check_completeness(manifest)
    results.append(("completeness", ok, problems))

    print("\n[2/3] was it actually trained")
    ok, problems = check_trained(args.directory)
    results.append(("trained", ok, problems))

    print("\n[3/3] reload into a fresh model")
    if args.skip_reload:
        print("  skipped")
    else:
        ok, problems = check_reload(args.directory, args.model, manifest)
        results.append(("reload", ok, problems))

    print("\n" + "=" * 60)
    failed = False
    for name, ok, problems in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        for problem in problems:
            print(f"        {problem}")
        failed |= not ok

    if failed:
        print("\nCHECKPOINT VERIFICATION FAILED")
        return 1
    print("\nCHECKPOINT VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
