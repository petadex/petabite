"""Tests for the Stage 2 inference path.

The real backbone needs biohub/ESMC-300M, so the model is faked down to the two
things inference touches: a ``forward`` returning ``{"logits": ...}`` and a state
dict shaped like one Trainer would save.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from torch import nn

from petabite.data_module.utils import read_sequence_csv
from petabite.predict import load_checkpoint, predict_scores, write_predictions


class _FakeModel(nn.Module):
    """Scores a sequence as its length; enough to check ordering and plumbing."""

    def __init__(self, output_dim: int = 1) -> None:
        super().__init__()
        self.head = nn.Linear(4, output_dim)

    def forward(self, input_ids, attention_mask):  # noqa: ANN001
        lengths = attention_mask.sum(dim=1).float()
        if self.head.out_features == 1:
            return {"logits": lengths.unsqueeze(-1)}
        logits = lengths.unsqueeze(-1).repeat(1, self.head.out_features)
        logits[:, 0] += 1.0
        return {"logits": logits}


class _FakeTokenizer:
    def encode(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        width = max(len(s) for s in sequences)
        ids = torch.zeros(len(sequences), width, dtype=torch.long)
        mask = torch.zeros(len(sequences), width, dtype=torch.long)
        for i, seq in enumerate(sequences):
            mask[i, : len(seq)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# --- read_sequence_csv ------------------------------------------------------------


def test_read_sequence_csv_without_label(tmp_path: Path):
    """Candidate pools have no measurement yet — a label must not be required."""
    path = tmp_path / "candidates.csv"
    _write_csv(path, [{"id": "a", "sequence": "ACDE"}], ["id", "sequence"])
    records = read_sequence_csv(path)
    assert records == [{"id": "a", "sequence": "ACDE"}]


def test_read_sequence_csv_carries_label_when_present(tmp_path: Path):
    path = tmp_path / "held_out.csv"
    _write_csv(
        path, [{"id": "a", "sequence": "ACDE", "label": "1.5"}], ["id", "sequence", "label"]
    )
    assert read_sequence_csv(path)[0]["label"] == 1.5


def test_read_sequence_csv_defaults_missing_id(tmp_path: Path):
    path = tmp_path / "no_id.csv"
    _write_csv(path, [{"sequence": "AC"}, {"sequence": "DE"}], ["sequence"])
    assert [r["id"] for r in read_sequence_csv(path)] == ["0", "1"]


def test_read_sequence_csv_requires_sequence_column(tmp_path: Path):
    path = tmp_path / "bad.csv"
    _write_csv(path, [{"id": "a"}], ["id"])
    with pytest.raises(ValueError, match="sequence"):
        read_sequence_csv(path)


def test_read_sequence_csv_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_sequence_csv(tmp_path / "nope.csv")


# --- load_checkpoint --------------------------------------------------------------


def test_load_checkpoint_restores_weights(tmp_path: Path):
    from safetensors.torch import save_file

    trained = _FakeModel()
    with torch.no_grad():
        trained.head.weight.fill_(0.5)
    save_file(trained.state_dict(), str(tmp_path / "model.safetensors"))

    fresh = _FakeModel()
    load_checkpoint(fresh, tmp_path)
    assert torch.equal(fresh.head.weight, trained.head.weight)


def test_load_checkpoint_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        load_checkpoint(_FakeModel(), tmp_path)


def test_load_checkpoint_rejects_checkpoint_without_head(tmp_path: Path):
    """A checkpoint lacking trained tensors must fail, not silently predict noise."""
    from safetensors.torch import save_file

    save_file({"backbone.unrelated": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="missing"):
        load_checkpoint(_FakeModel(), tmp_path)


# --- scoring and output -----------------------------------------------------------


def test_predict_scores_regression(tmp_path: Path):
    records = [{"id": "a", "sequence": "AC"}, {"id": "b", "sequence": "ACDEF"}]
    scores = predict_scores(
        _FakeModel(), records, _FakeTokenizer(), torch.device("cpu"), 8, "regression"
    )
    assert scores == [2.0, 5.0]


def test_predict_scores_batches_cover_all_records():
    records = [{"id": str(i), "sequence": "A" * (i + 1)} for i in range(5)]
    scores = predict_scores(
        _FakeModel(), records, _FakeTokenizer(), torch.device("cpu"), 2, "regression"
    )
    assert scores == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_predict_scores_classification_returns_probability():
    records = [{"id": "a", "sequence": "AC"}]
    scores = predict_scores(
        _FakeModel(output_dim=3), records, _FakeTokenizer(), torch.device("cpu"), 8,
        "classification",
    )
    assert 0.0 < scores[0] <= 1.0


def test_write_predictions_ranks_descending(tmp_path: Path):
    records = [{"id": "low", "sequence": "AC"}, {"id": "high", "sequence": "ACDEF"}]
    out = tmp_path / "nested" / "predictions.csv"
    write_predictions(records, [2.0, 5.0], out)

    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["id"] for r in rows] == ["high", "low"]
    assert "label" not in rows[0]


def test_write_predictions_keeps_label_column_when_present(tmp_path: Path):
    records = [{"id": "a", "sequence": "AC", "label": 1.0}]
    out = tmp_path / "predictions.csv"
    write_predictions(records, [0.5], out)
    with out.open() as f:
        assert list(csv.DictReader(f))[0]["label"] == "1.0"
