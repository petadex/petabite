"""Does the MLM adaptation help? Probe stock ESM-C against PlasticESM.

This is the only measurement that says whether Stage 1 bought anything. It
freezes both backbones, mean-pools their embeddings for the labelled
sequences, and fits a ridge regression to PET monomer release. The difference
between the two is the answer.

The target is the Stage 2 evaluated set: PET substrate, quantitative,
concentration units, human-curated only. Values are aggregated to one number
per sequence, the median of log10 micromolar, because a sequence carries a
median of 4 labels measured under different assay conditions and the probe has
no way to represent those conditions. Non-positive values, 16.9% of the set,
are dropped: they are control-subtracted readings and have no logarithm.

Two numbers are reported, because the fixed split is small:

  held-out   fit on the 115 train sequences, alpha chosen on the 27 val
             sequences, scored on the 15 test sequences. Honest but noisy.

  LOCO       leave-one-component-out over all 42 val and test sequences,
             pooled out-of-fold predictions. Same homology guarantee, nearly
             three times the evaluation data, and the bootstrap resamples
             components rather than sequences because sequences within a
             component are not independent.

Usage:
    python scripts/probe_stage2.py --lora outputs/plasticesm-c60-r32
    python scripts/probe_stage2.py --lora ... --out logs/probe.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from transformers import AutoModelForMaskedLM

from petabite.data_module.tokenization import load_esmc_tokenizer
from petabite.model_module.lora import LoraConfig, apply_lora, load_lora

LABELS = Path("/project/6122149/purav/petai-labels/v2")
MODEL_ID = "biohub/ESMC-300M"
TARGET = "pet_release_conc"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
EMBED_BATCH = 16


def read_targets() -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """Per-sequence median log10 concentration, plus split and component."""
    per_sequence: dict[str, list[float]] = collections.defaultdict(list)
    split: dict[str, str] = {}
    with (LABELS / "normalised" / "labels_normalised.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["target"] != TARGET or row["source"] == "llm":
                continue
            if not row["value_canonical"]:
                continue
            value = float(row["value_canonical"])
            if value <= 0:
                continue
            per_sequence[row["seq_md5"]].append(math.log10(value))
            split[row["seq_md5"]] = row["split"]

    component: dict[str, str] = {}
    with (LABELS / "sequences.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            component[row["seq_md5"]] = row["component"]

    targets = {md5: float(np.median(v)) for md5, v in per_sequence.items()}
    return targets, split, component


def read_sequences(wanted: set[str]) -> dict[str, str]:
    """Read the FASTA, keeping only sequences that carry a target."""
    sequences: dict[str, str] = {}
    name, parts = None, []
    with (LABELS / "sequences.fasta").open() as fh:
        for line in fh:
            if line.startswith(">"):
                if name and name in wanted:
                    sequences[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name and name in wanted:
        sequences[name] = "".join(parts)
    return sequences


@torch.no_grad()
def embed(sequences: dict[str, str], lora_dir: Path | None,
          max_length: int = 1024) -> dict[str, np.ndarray]:
    """Mean-pooled embeddings, masked so padding contributes nothing.

    ``lora_dir`` of None gives stock ESM-C. Otherwise the adapters are applied
    and loaded, which is what makes the comparison a like-for-like one: the
    same architecture, the same pooling, only the weights differ.
    """
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
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(names), EMBED_BATCH):
        batch = names[start : start + EMBED_BATCH]
        encoded = tokenizer([sequences[n] for n in batch], padding=True,
                            truncation=True, max_length=max_length,
                            return_tensors="pt")
        # The tokenizer also emits token_type_ids, which ESM-C's forward does
        # not accept. Whitelist rather than drop, for the same reason the
        # collator does: a new tokenizer output should not break this.
        encoded = {k: v.cuda() for k, v in encoded.items()
                   if k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(**encoded, compute_sae=False).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        for name, vector in zip(batch, pooled.float().cpu().numpy()):
            out[name] = vector

    del model
    torch.cuda.empty_cache()
    return out


def fit_and_score(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    """Ridge with alpha chosen on val, scored on test."""
    best_alpha, best_score = None, -np.inf
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(x_train, y_train)
        score = spearmanr(model.predict(x_val), y_val).statistic
        if np.isfinite(score) and score > best_score:
            best_alpha, best_score = alpha, score

    model = Ridge(alpha=best_alpha).fit(x_train, y_train)
    predicted = model.predict(x_test)
    return {
        "alpha": best_alpha,
        "val_spearman": round(float(best_score), 4),
        "test_spearman": round(float(spearmanr(predicted, y_test).statistic), 4),
        "test_pearson": round(float(pearsonr(predicted, y_test).statistic), 4),
        "test_r2": round(float(1 - np.sum((y_test - predicted) ** 2)
                               / np.sum((y_test - y_test.mean()) ** 2)), 4),
        "n_test": len(y_test),
        "predictions": predicted.tolist(),
    }


def leave_one_component_out(x, y, components, alpha: float) -> dict:
    """Pooled out-of-fold predictions, one fold per homology component.

    A component never appears in both the fit and the prediction, so this keeps
    the same guarantee the fixed split has while using every evaluable
    sequence.
    """
    unique = sorted(set(components))
    predicted = np.zeros_like(y)
    for held_out in unique:
        mask = np.array([c == held_out for c in components])
        if mask.all():
            continue
        model = Ridge(alpha=alpha).fit(x[~mask], y[~mask])
        predicted[mask] = model.predict(x[mask])

    return {
        "n_components": len(unique),
        "n_sequences": len(y),
        "spearman": round(float(spearmanr(predicted, y).statistic), 4),
        "pearson": round(float(pearsonr(predicted, y).statistic), 4),
        "predictions": predicted.tolist(),
    }


def bootstrap_over_components(y, predicted, components, n: int = 2000,
                              seed: int = 0) -> tuple[float, float]:
    """Resample components, not sequences: sequences within one are dependent."""
    rng = np.random.default_rng(seed)
    by_component: dict[str, list[int]] = collections.defaultdict(list)
    for index, component in enumerate(components):
        by_component[component].append(index)
    keys = list(by_component)

    scores = []
    for _ in range(n):
        drawn = rng.choice(len(keys), size=len(keys), replace=True)
        rows = [i for d in drawn for i in by_component[keys[d]]]
        if len(set(np.asarray(y)[rows])) < 3:
            continue
        score = spearmanr(np.asarray(predicted)[rows], np.asarray(y)[rows]).statistic
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        return float("nan"), float("nan")
    return (round(float(np.percentile(scores, 2.5)), 4),
            round(float(np.percentile(scores, 97.5)), 4))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", type=Path, required=True,
                        help="directory holding lora_adapters.safetensors")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    targets, split, component = read_targets()
    sequences = read_sequences(set(targets))
    print(f"{len(targets)} sequences carry a {TARGET} target; "
          f"{len(sequences)} found in the FASTA")
    counts = collections.Counter(split[m] for m in sequences)
    print(f"  by split: {dict(counts)}")

    results: dict[str, dict] = {}
    for name, lora_dir in (("stock_esmc", None), ("plasticesm", args.lora)):
        print(f"\n=== embedding with {name} ===")
        vectors = embed(sequences, lora_dir, args.max_length)
        dim = len(next(iter(vectors.values())))
        print(f"  {len(vectors)} embeddings of dim {dim}")

        def block(which: str):
            names = sorted(m for m in sequences if split[m] == which)
            return (np.stack([vectors[m] for m in names]),
                    np.array([targets[m] for m in names]),
                    [component[m] for m in names])

        x_train, y_train, _ = block("train")
        x_val, y_val, c_val = block("val")
        x_test, y_test, c_test = block("test")

        held_out = fit_and_score(x_train, y_train, x_val, y_val, x_test, y_test)
        low, high = bootstrap_over_components(
            y_test, held_out.pop("predictions"), c_test)
        held_out["test_spearman_ci95"] = [low, high]

        # LOCO over everything held out of training, val and test together.
        x_eval = np.concatenate([x_val, x_test])
        y_eval = np.concatenate([y_val, y_test])
        c_eval = c_val + c_test
        loco = leave_one_component_out(x_eval, y_eval, c_eval, held_out["alpha"])
        low, high = bootstrap_over_components(
            y_eval, loco.pop("predictions"), c_eval)
        loco["spearman_ci95"] = [low, high]

        results[name] = {"held_out": held_out, "loco": loco}
        print(f"  held-out test: spearman {held_out['test_spearman']:+.4f} "
              f"CI {held_out['test_spearman_ci95']}, r2 {held_out['test_r2']:+.4f}, "
              f"n={held_out['n_test']}")
        print(f"  LOCO         : spearman {loco['spearman']:+.4f} "
              f"CI {loco['spearman_ci95']}, "
              f"n={loco['n_sequences']} over {loco['n_components']} components")

    stock, adapted = results["stock_esmc"], results["plasticesm"]
    delta_test = adapted["held_out"]["test_spearman"] - stock["held_out"]["test_spearman"]
    delta_loco = adapted["loco"]["spearman"] - stock["loco"]["spearman"]
    results["delta"] = {
        "held_out_test_spearman": round(delta_test, 4),
        "loco_spearman": round(delta_loco, 4),
    }

    print("\n" + "=" * 62)
    print(f"{'':12} {'stock ESM-C':>14} {'PlasticESM':>14} {'delta':>10}")
    print(f"{'test rho':12} {stock['held_out']['test_spearman']:>14.4f} "
          f"{adapted['held_out']['test_spearman']:>14.4f} {delta_test:>+10.4f}")
    print(f"{'LOCO rho':12} {stock['loco']['spearman']:>14.4f} "
          f"{adapted['loco']['spearman']:>14.4f} {delta_loco:>+10.4f}")
    print("=" * 62)
    print("A delta inside the bootstrap intervals is not evidence of a gain.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
