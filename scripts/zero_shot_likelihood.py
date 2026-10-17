"""Score every labelled sequence by ESM-C's own likelihood, using no labels.

Search-space item 2. Every measurement so far fits a ridge on embeddings,
which means every measurement inherits the choice of readout, the penalty,
the fold structure and the scoring metric. Three of those four have already
produced a retracted result.

A masked language model offers something that avoids all of it. The model
assigns a likelihood to a sequence, and for variant effect prediction that
likelihood is a well established zero-shot predictor: a mutation the model
finds improbable tends to be one evolution avoided, which tends to be one
that breaks the protein. No labels are used, nothing is fitted, and there are
no folds, so the evaluation is a single rank correlation with a component
bootstrap and none of the metric faults can arise.

It is also the baseline this project should have had from the start. If the
pretrained likelihood ranks variants as well as a supervised probe on the
same backbone, the probe is adding nothing.

The quantity computed is the pseudo-log-likelihood: mask each position in
turn, read the log probability of the residue that truly sits there, and
average over positions. Averaging rather than summing, because the sequences
differ in length by a factor of three and a sum would mostly measure length.

Exact masking costs one forward pass per residue. ``--stride`` trades
accuracy for time by masking every nth position in n passes instead, which
is standard practice; stride 1 is exact and is the default.

Usage:
    python scripts/zero_shot_likelihood.py --lora outputs/plasticesm-c60-r32 \\
        --out logs/zero_shot.npz
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
# Masked copies of one sequence are scored together; this bounds the batch so
# a long sequence does not exhaust memory.
MASK_BATCH = 64


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
def pseudo_log_likelihood(model, tokenizer, sequence: str, mask_id: int,
                          stride: int, max_length: int) -> float:
    """Mean log probability of each residue when that residue is masked.

    The special tokens at either end are left alone: only positions holding
    an actual residue are masked and scored.
    """
    encoded = tokenizer([sequence], truncation=True, max_length=max_length,
                        return_tensors="pt")
    input_ids = encoded["input_ids"][0].cuda()
    attention = encoded["attention_mask"][0].cuda()

    # Residue positions are everything the tokenizer did not mark special.
    special = set(tokenizer.all_special_ids)
    positions = [i for i, token in enumerate(input_ids.tolist())
                 if token not in special]
    if not positions:
        return float("nan")

    total, counted = 0.0, 0
    for offset in range(stride):
        chosen = positions[offset::stride]
        if not chosen:
            continue

        # One masked copy per chosen position, processed in bounded batches.
        for start in range(0, len(chosen), MASK_BATCH):
            block = chosen[start : start + MASK_BATCH]
            batch = input_ids.unsqueeze(0).repeat(len(block), 1)
            for row, position in enumerate(block):
                batch[row, position] = mask_id

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch,
                    attention_mask=attention.unsqueeze(0).repeat(len(block), 1),
                    compute_sae=False,
                ).logits

            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for row, position in enumerate(block):
                true_token = int(input_ids[position])
                total += float(log_probs[row, position, true_token])
                counted += 1

    return total / counted if counted else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1,
                        help="mask every nth position in n passes; 1 is exact")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first n sequences, for a smoke test")
    args = parser.parse_args()

    sequences = labels.read_sequences()
    names = sorted(sequences)
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names)} sequences, stride {args.stride}")

    tokenizer = load_esmc_tokenizer(MODEL_ID, args.max_length)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise SystemExit("tokenizer has no mask token; cannot score likelihood")
    print(f"mask token id {mask_id}")

    payload: dict[str, np.ndarray] = {"names": np.array(names)}
    for backbone, lora_dir in (("stock", None), ("plasticesm", args.lora)):
        print(f"scoring with {backbone}")
        model = load_backbone(lora_dir)
        scores = []
        for position, name in enumerate(names, start=1):
            scores.append(pseudo_log_likelihood(
                model, tokenizer, sequences[name], mask_id,
                args.stride, args.max_length))
            if position % 100 == 0:
                print(f"  {position}/{len(names)}")
        payload[f"pll_{backbone}"] = np.array(scores, dtype=np.float32)
        finite = np.isfinite(payload[f"pll_{backbone}"])
        print(f"  {finite.sum()} finite, mean "
              f"{payload[f'pll_{backbone}'][finite].mean():.4f}")
        del model
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
