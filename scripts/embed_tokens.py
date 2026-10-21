"""Cache per-token hidden states, so learned pooling can be tested at all.

Everything downstream of ESM-C so far has used mean-pooled vectors, which
throws away the token axis before any model sees it. Item 7 of the search
space asks whether a learned pooler does better than that mean, and the
question cannot even be posed without the per-token states.

They are cheaper to store than the earlier estimate suggested. The labelled
set is 1137 sequences totalling about 393,000 residues, so one backbone's
states are 393,000 by 960 in float16, around 750MB. Both backbones fit in
about 1.5GB, which is why this is a cache rather than a recomputation inside
every experiment.

Storage is packed rather than padded. A padded array would be 1137 by 1174 by
960, two thirds of it padding, so instead the sequences are concatenated along
the token axis and an offset table says where each one starts and ends. Only
real residue positions are kept: the special tokens at either end are dropped
here so no downstream consumer has to know about them.

Usage:
    python scripts/embed_tokens.py --lora outputs/plasticesm-c60-r32 \\
        --out logs/token_states.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM

from petabite.data_module.tokenization import load_esmc_tokenizer
from petabite.eval_module import labels
from petabite.model_module.lora import LoraConfig, apply_lora, load_lora

MODEL_ID = "biohub/ESMC-300M"
BATCH = 8


def load_backbone(lora_dir: Path | None):
    """Stock ESM-C, or the same with the Stage 1 adapters loaded."""
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    if lora_dir is not None:
        manifest = json.loads((lora_dir / "lora_manifest.json").read_text())
        cfg = manifest["config"]
        apply_lora(model, LoraConfig(
            r=cfg["r"], alpha=cfg["alpha"], dropout=cfg["dropout"],
            targets=tuple(cfg["targets"]), verify=False,
        ))
        load_lora(model, lora_dir)
    return model.cuda().eval()


@torch.no_grad()
def encode_all(model, tokenizer, sequences: dict[str, str], names: list[str],
               max_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Packed per-token states, plus the offset of each sequence.

    Returns an array of shape (total_residues, 960) in float16 and an array of
    shape (n_sequences, 2) giving the start and end row of each sequence.
    """
    special = set(tokenizer.all_special_ids)
    blocks: list[np.ndarray] = []
    offsets = np.zeros((len(names), 2), dtype=np.int64)
    cursor = 0

    for start in range(0, len(names), BATCH):
        batch = names[start : start + BATCH]
        encoded = tokenizer([sequences[name] for name in batch], padding=True,
                            truncation=True, max_length=max_length,
                            return_tensors="pt")
        encoded = {key: value.cuda() for key, value in encoded.items()
                   if key in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(**encoded, compute_sae=False).last_hidden_state

        ids = encoded["input_ids"].cpu().numpy()
        states = hidden.float().cpu().numpy()
        for row, name in enumerate(batch):
            keep = [position for position, token in enumerate(ids[row])
                    if token not in special]
            block = states[row, keep].astype(np.float16)
            blocks.append(block)
            offsets[start + row] = (cursor, cursor + len(block))
            cursor += len(block)

        if (start + len(batch)) % 200 < BATCH:
            print(f"  {start + len(batch)}/{len(names)}  {cursor} residues")

    return np.concatenate(blocks, axis=0), offsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0,
                        help="first n sequences only, for a smoke test")
    args = parser.parse_args()

    sequences = labels.read_sequences()
    names = sorted(sequences)
    if args.limit:
        names = names[: args.limit]
    total = sum(len(sequences[name]) for name in names)
    print(f"{len(names)} sequences, {total} residues")
    print(f"expect about {total * 960 * 2 / 1e9:.2f}GB per backbone in float16")

    tokenizer = load_esmc_tokenizer(MODEL_ID, args.max_length)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"names": np.array(names)}

    for backbone, lora_dir in (("stock", None), ("plasticesm", args.lora)):
        print(f"encoding with {backbone}")
        model = load_backbone(lora_dir)
        states, offsets = encode_all(model, tokenizer, sequences, names,
                                     args.max_length)
        payload[f"{backbone}_states"] = states
        payload[f"{backbone}_offsets"] = offsets
        print(f"  {states.shape} {states.dtype}, "
              f"{states.nbytes / 1e9:.2f}GB")

        # Saved after each backbone, as the zero-shot run learned to do.
        np.savez(args.out, **payload)
        print(f"  wrote {args.out}")

        del model
        torch.cuda.empty_cache()

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
