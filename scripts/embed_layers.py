"""Embed the labelled sequences at every few transformer layers, not just the last.

The evaluation so far has used the final hidden state, which is the obvious
choice and probably the wrong one. A masked-language-model head trains the last
layers hardest to reconstruct residues, a task that rewards local sequence
statistics; representations useful for a global property like catalytic
efficiency tend to peak in the middle of the stack. If that holds here then
every score in the evaluation is from the least informative layer available,
and the stratified result that did appear would appear more strongly elsewhere.

This also makes Stage 1 a fairer test. LoRA adapters sit on all 30 blocks, but
MLM training pushes hardest at the top, so comparing stock against PlasticESM
at the final layer alone measures the adaptation where it is most distorted by
the pretraining objective.

ESM-C's forward returns only the last hidden state, so intermediate layers are
captured with forward hooks on `model.esmc.transformer.blocks`.

One subtlety cost a confusing result before it was found. `TransformerStack`
returns `self.norm(x), x, ...`, so `last_hidden_state` is the final block's
output *after* a LayerNorm while a hook on the last block sees it before.
The two are not interchangeable: the pre-norm residual stream grows with depth
and reaches a column standard deviation of 26.8 against 0.008 post-norm, which
is three orders of magnitude of difference in how hard a fixed ridge penalty
bites. The post-norm tensor is therefore cached as its own key, under
`<backbone>_post`, and scoring standardises every feature set.

Usage:
    python scripts/embed_layers.py --lora outputs/plasticesm-c60-r32 \\
        --out logs/layer_embeddings.npz
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
BATCH = 16
# Every fifth block, plus the last. Thirty layers at 960 floats for 1,137
# sequences is cheap to store, but scoring every layer multiplies the
# evaluation runtime by thirty for no extra insight.
LAYERS = (5, 10, 15, 20, 25, 29)
# Key for the post-final-LayerNorm representation, which is what
# `last_hidden_state` returns and what every earlier script used.
POST_NORM = -1


class LayerCapture:
    """Collects the output of chosen transformer blocks during a forward pass.

    A block returns a tuple on this model, so the hook keeps element zero,
    which is the residual stream. Hooks are registered once and reused for
    every batch; `taken` is cleared between batches by the caller.
    """

    def __init__(self, model, layers: tuple[int, ...]):
        self.taken: dict[int, torch.Tensor] = {}
        self.handles = []
        blocks = model.esmc.transformer.blocks

        for layer in layers:
            def hook(_module, _inputs, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                self.taken[layer] = hidden

            self.handles.append(blocks[layer].register_forward_hook(hook))

    def release(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def embed_all(sequences: dict[str, str], lora_dir: Path | None,
              max_length: int) -> tuple[dict[int, np.ndarray], list[str]]:
    """Mean-pooled embeddings at each chosen layer, padding masked out."""
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
    capture = LayerCapture(model, LAYERS)
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in LAYERS}
    collected[POST_NORM] = []

    names = sorted(sequences)
    for start in range(0, len(names), BATCH):
        batch = names[start : start + BATCH]
        encoded = tokenizer([sequences[n] for n in batch], padding=True,
                            truncation=True, max_length=max_length,
                            return_tensors="pt")
        # ESM-C's forward takes only these two; the tokenizer also emits
        # token_type_ids.
        encoded = {k: v.cuda() for k, v in encoded.items()
                   if k in ("input_ids", "attention_mask")}

        capture.taken.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**encoded, compute_sae=False)

        mask = encoded["attention_mask"].unsqueeze(-1)
        captured = dict(capture.taken)
        captured[POST_NORM] = output.last_hidden_state

        for layer, hidden in captured.items():
            weights = mask.to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
            collected[layer].append(pooled.float().cpu().numpy())

    capture.release()
    del model
    torch.cuda.empty_cache()
    return {layer: np.concatenate(parts) for layer, parts in collected.items()}, names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    sequences = labels.read_sequences()
    print(f"{len(sequences)} labelled sequences")
    print(f"layers: {LAYERS}")

    payload: dict[str, np.ndarray] = {}
    names_ref: list[str] | None = None

    for backbone, lora_dir in (("stock", None), ("plasticesm", args.lora)):
        print(f"embedding with {backbone}")
        per_layer, names = embed_all(sequences, lora_dir, args.max_length)
        for layer, matrix in per_layer.items():
            name = (f"{backbone}_post" if layer == POST_NORM
                    else f"{backbone}_L{layer}")
            payload[name] = matrix
            print(f"  {name:18}: {matrix.shape[0]} x {matrix.shape[1]}  "
                  f"column SD {matrix.std(axis=0).mean():.3f}")
        names_ref = names

    assert names_ref is not None
    payload["names"] = np.array(names_ref)

    # The controls travel with the embeddings so the scoring script needs only
    # one file, exactly as with the single-layer cache.
    payload["length"] = np.array(
        [[len(sequences[name])] for name in names_ref], dtype=np.float32
    )
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
