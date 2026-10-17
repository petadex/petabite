"""Search item 7, second half: a learned pooler instead of the mean.

Every result so far reduced a sequence to the mean of its token states before
any model saw it. That is a choice, and a crude one: it weights a catalytic
residue the same as a disordered terminus. A learned pooler decides the
weights instead, which is the strongest version of item 7 and the one the
first pass could not run because only the mean-pooled vectors were cached.

The pooler is gated attention over tokens, the standard formulation: each
token gets a score from two small projections combined multiplicatively, the
scores are softmaxed into weights, and the sequence vector is their weighted
sum. A linear head turns that into a prediction. Everything is trained
end-to-end on the fold's training components.

Three things keep the comparison honest.

The baseline is recomputed from the same tokens rather than taken from the
cached mean-pooled file, so pooler and mean differ only in the pooling. The
cached file averaged over special tokens as well, which would have been a
second difference nobody asked about.

The inner split used for early stopping holds out whole components, not
random rows. The small head in the first pass used a random inner split, which
gave it a within-component leak that was noted as cutting in its favour. There
is no reason to repeat that when the grouping is available.

And every configuration runs several seeds, with the spread reported beside
the mean, because the first pass found a head's score moving by up to 0.181
between initialisations of one configuration.

Usage:
    python scripts/eval_pooling.py --states logs/token_states.npz \\
        --embeddings logs/label_embeddings.npz --target kcat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from torch import nn

from petabite.eval_module import labels, stats

BACKBONES = ("stock", "plasticesm")
HIDDEN = 32
EPOCHS = 300
PATIENCE = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2


class GatedAttentionPooler(nn.Module):
    """Weighted sum over tokens, with the weights learned.

    Deliberately small. The width is 32 against a 960-dimensional input and
    40-odd independent components, because the ridge alpha profile for these
    embeddings peaks at the most regularised end of an eight-order grid, which
    says the data supports very little capacity. A wide pooler would only
    restate that.
    """

    def __init__(self, width: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.attend = nn.Linear(width, hidden)
        self.gate = nn.Linear(width, hidden)
        self.score = nn.Linear(hidden, 1)
        self.head = nn.Linear(width, 1)

    def forward(self, tokens: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """tokens is (batch, length, width); mask is (batch, length)."""
        scores = self.score(torch.tanh(self.attend(tokens))
                            * torch.sigmoid(self.gate(tokens))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1)
        return self.head(pooled).squeeze(-1)


def read_states(path: Path, backbone: str, index: dict[str, int],
                names: list[str]) -> list[np.ndarray]:
    """The per-token states of the chosen sequences, one array each."""
    cached = np.load(path, allow_pickle=False)
    states = cached[f"{backbone}_states"]
    offsets = cached[f"{backbone}_offsets"]
    out = []
    for name in names:
        start, end = offsets[index[name]]
        out.append(np.asarray(states[start:end], dtype=np.float32))
    return out


def pad_batch(blocks: list[np.ndarray], rows: np.ndarray,
              scaler: tuple[np.ndarray, np.ndarray], device
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the chosen sequences into one tensor, standardised and masked."""
    chosen = [blocks[row] for row in rows]
    longest = max(len(block) for block in chosen)
    width = chosen[0].shape[1]

    tokens = np.zeros((len(chosen), longest, width), dtype=np.float32)
    mask = np.zeros((len(chosen), longest), dtype=bool)
    centre, scale = scaler
    for position, block in enumerate(chosen):
        tokens[position, : len(block)] = (block - centre) / scale
        mask[position, : len(block)] = True

    return (torch.from_numpy(tokens).to(device),
            torch.from_numpy(mask).to(device))


def token_scaler(blocks: list[np.ndarray],
                 rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and spread over every token of the fitting rows."""
    stacked = np.concatenate([blocks[row] for row in rows], axis=0)
    centre = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return centre, scale


def train_pooler(blocks: list[np.ndarray], y: np.ndarray,
                 fit_rows: np.ndarray, inner_groups: list[str],
                 held_rows: np.ndarray, seed: int, device) -> np.ndarray:
    """Fit on the fitting rows, predict the held-out ones.

    Early stopping holds out whole components of the fitting data, so the
    stopping point is chosen without seeing any sequence related to one it
    will be judged on.
    """
    torch.manual_seed(seed)
    unique = sorted(set(inner_groups))
    rng = np.random.default_rng(seed)
    validation = set(rng.choice(unique, max(1, len(unique) // 5),
                                replace=False).tolist())

    inner_fit = np.array([row for row, group in zip(fit_rows, inner_groups)
                          if group not in validation])
    inner_val = np.array([row for row, group in zip(fit_rows, inner_groups)
                          if group in validation])
    if len(inner_val) < 3 or len(inner_fit) < 10:
        inner_fit, inner_val = fit_rows, fit_rows

    scaler = token_scaler(blocks, inner_fit)
    width = blocks[0].shape[1]
    model = GatedAttentionPooler(width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    targets = torch.from_numpy(y[inner_fit].astype(np.float32)).to(device)
    validation_targets = torch.from_numpy(
        y[inner_val].astype(np.float32)).to(device)
    fit_tokens, fit_mask = pad_batch(blocks, inner_fit, scaler, device)
    val_tokens, val_mask = pad_batch(blocks, inner_val, scaler, device)

    best, best_state, waited = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        optimiser.zero_grad()
        loss = nn.functional.mse_loss(model(fit_tokens, fit_mask), targets)
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            score = float(nn.functional.mse_loss(
                model(val_tokens, val_mask), validation_targets))
        if score < best - 1e-6:
            best, waited = score, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    held_tokens, held_mask = pad_batch(blocks, held_rows, scaler, device)
    with torch.no_grad():
        return model(held_tokens, held_mask).cpu().numpy()


def pooler_out_of_fold(blocks: list[np.ndarray], y: np.ndarray,
                       groups: list[str], seed: int, device) -> np.ndarray:
    """Grouped out-of-fold predictions from the pooler, fold-centred."""
    predicted = np.zeros(len(y), dtype=float)
    for fit, held in GroupKFold(
            n_splits=min(5, len(set(groups)))).split(np.zeros(len(y)), y,
                                                     groups):
        fold = train_pooler(blocks, y, fit, [groups[i] for i in fit], held,
                            seed, device)
        predicted[held] = stats.centre_fold(fold)
    return predicted


def mean_pool(blocks: list[np.ndarray]) -> np.ndarray:
    """The baseline this is trying to beat, from the same tokens."""
    return np.stack([block.mean(axis=0) for block in blocks])


def report(predicted: np.ndarray, y: np.ndarray, groups: list[str]) -> dict:
    value = spearmanr(predicted, y).statistic
    low, high = stats.jackknife_ci(y, predicted, groups)
    return {
        "spearman": round(float(value), 3) if np.isfinite(value) else None,
        "jackknife95": [low, high],
        "permutation_p": stats.permutation_p(y, predicted, groups),
    }


def evaluate(target: str, states: Path, cached, index: dict[str, int],
             seeds: int, device) -> dict | None:
    """Pooler against mean-pool ridge against the controls, on one target."""
    measurements = labels.read_measurements(target)
    per_sequence = labels.aggregate(
        [m for m in measurements if m.seq_md5 in index])
    components = labels.read_components()

    names = sorted(per_sequence)
    y = np.array([per_sequence[name] for name in names])
    groups = [components[name] for name in names]
    if not labels.is_scoreable(len(names), len(set(groups))):
        print(f"{target}: too few to score")
        return None

    rows = np.array([index[name] for name in names], dtype=int)
    print(f"{target}: {len(names)} sequences, {len(set(groups))} components")
    result: dict[str, dict] = {"n_sequences": len(names),
                               "n_components": len(set(groups))}

    print(f"  {'features':22} {'method':18} {'rho':>7} {'p':>7}  jackknife")
    for control in ("length", "composition"):
        x = np.asarray(cached[control], dtype=np.float64)[rows]
        entry = report(stats.grouped_out_of_fold(x, y, groups, n_splits=5,
                                                 standardise=True),
                       y, groups)
        result[control] = entry
        print(f"  {control:22} {'ridge':18} {str(entry['spearman']):>7} "
              f"{entry['permutation_p']:>7}  {entry['jackknife95']}")

    for backbone in BACKBONES:
        blocks = read_states(states, backbone, index, names)
        lengths = [len(block) for block in blocks]
        print(f"  {backbone}: {min(lengths)} to {max(lengths)} tokens each")

        pooled = mean_pool(blocks)
        entry = report(stats.grouped_out_of_fold(pooled, y, groups,
                                                 n_splits=5,
                                                 standardise=True),
                       y, groups)
        result[f"{backbone}_mean_ridge"] = entry
        print(f"  {backbone:22} {'mean + ridge':18} "
              f"{str(entry['spearman']):>7} {entry['permutation_p']:>7}  "
              f"{entry['jackknife95']}")

        scores = []
        for seed in range(seeds):
            entry = report(pooler_out_of_fold(blocks, y, groups, seed, device),
                           y, groups)
            scores.append(entry)
            print(f"  {backbone:22} {f'learned pool s{seed}':18} "
                  f"{str(entry['spearman']):>7} {entry['permutation_p']:>7}  "
                  f"{entry['jackknife95']}")

        values = [s["spearman"] for s in scores if s["spearman"] is not None]
        result[f"{backbone}_learned_pool"] = {
            "seeds": len(values),
            "mean": round(float(np.mean(values)), 3) if values else None,
            "sd": round(float(np.std(values)), 3) if values else None,
            "best": round(float(np.max(values)), 3) if values else None,
            "per_seed": scores,
        }
        summary = result[f"{backbone}_learned_pool"]
        print(f"  {backbone:22} {'learned pool':18} "
              f"{str(summary['mean']):>7} {'':>7}  "
              f"sd {summary['sd']}, best {summary['best']}")

    print()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--targets", default="kcat,kcat_over_km")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cached = np.load(args.embeddings, allow_pickle=False)
    index = {name: position for position, name in enumerate(cached["names"])}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}, {args.seeds} seeds per pooler")
    print()

    results = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        outcome = evaluate(target, args.states, cached, index, args.seeds,
                           device)
        if outcome is not None:
            results[target] = outcome

    print("=" * 78)
    print("The pooler is only interesting where it beats the mean on the same")
    print("tokens, and then only where that also beats both controls.")
    print("=" * 78)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
