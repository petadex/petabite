"""Embed every labelled sequence once, with each backbone, and cache to disk.

The evaluation sweep needs the same embeddings many times, across several
targets and protocols. Computing them is the only part that needs a GPU, so it
happens once here and everything downstream runs on CPU.

Two backbones are cached so the sweep can ask whether the MLM adaptation
changed anything: `stock` is ESM-C as published, `plasticesm` is the same
architecture with the Stage 1 LoRA adapters loaded.

Usage:
    python scripts/embed_labels.py --lora outputs/plasticesm-c60-r32 \\
        --out logs/label_embeddings.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM

from petabite.data_module.tokenization import load_esmc_tokenizer
from petabite.model_module.lora import LoraConfig, apply_lora, load_lora

MODEL_ID = "biohub/ESMC-300M"
BATCH = 16


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    name, parts = None, []
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    sequences[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name:
        sequences[name] = "".join(parts)
    return sequences


@torch.no_grad()
def embed_all(sequences: dict[str, str], lora_dir: Path | None,
              max_length: int) -> tuple[np.ndarray, list[str]]:
    """Mean-pool the final hidden states, masked so padding contributes nothing."""
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    if lora_dir is not None:
        manifest = json.loads((lora_dir / "lora_manifest.json").read_text())
        cfg = manifest["config"]
        apply_lora(model, LoraConfig(
            r=cfg["r"], alpha=cfg["alpha"], dropout=cfg["dropout"],
            targets=tuple(cfg["targets"]), verify=False,
        ))
        load_lora(model, lora_dir)
    model = model.cuda().eval()
    tokenizer = load_esmc_tokenizer(MODEL_ID, max_length)

    names = sorted(sequences)
    vectors = []
    for start in range(0, len(names), BATCH):
        batch = names[start : start + BATCH]
        encoded = tokenizer([sequences[n] for n in batch], padding=True,
                            truncation=True, max_length=max_length,
                            return_tensors="pt")
        # ESM-C's forward takes only these two; the tokenizer also emits
        # token_type_ids.
        encoded = {k: v.cuda() for k, v in encoded.items()
                   if k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(**encoded, compute_sae=False).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vectors.append(pooled.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(vectors), names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path,
                        default=Path("/project/6122149/purav/petai-labels/v2"))
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    sequences = read_fasta(args.labels / "sequences.fasta")
    print(f"{len(sequences)} labelled sequences")

    payload: dict[str, np.ndarray] = {}
    names_ref: list[str] | None = None
    for label, lora_dir in (("stock", None), ("plasticesm", args.lora)):
        print(f"embedding with {label}")
        matrix, names = embed_all(sequences, lora_dir, args.max_length)
        print(f"  {matrix.shape[0]} x {matrix.shape[1]}")
        payload[label] = matrix
        names_ref = names

    assert names_ref is not None
    payload["names"] = np.array(names_ref)
    # A trivial feature, cached alongside, because the probe showed sequence
    # length beating both backbones and every later claim has to clear it.
    payload["length"] = np.array(
        [[len(sequences[n])] for n in names_ref], dtype=np.float32
    )
    # Amino acid composition, the other standard control.
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    composition = np.zeros((len(names_ref), len(alphabet)), dtype=np.float32)
    for row, name in enumerate(names_ref):
        sequence = sequences[name]
        if not sequence:
            continue
        for column, residue in enumerate(alphabet):
            composition[row, column] = sequence.count(residue) / len(sequence)
    payload["composition"] = composition

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
