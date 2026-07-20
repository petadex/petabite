"""Stage 2 inference: score PETadex candidates with the trained activity model.

Takes a CSV of Logan-derived PETase sequences (from the PETadex fragment library
or an active-learning candidate pool) and outputs predicted activity scores.
Top-ranked sequences are handed to the iGEM wet-lab team for experimental
validation, closing the active-learning feedback loop.

Run:
    python -m petabite.predict data.csv_path=candidates.csv
    python -m petabite.predict predict.checkpoint_dir=outputs/esmc-activity
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from petabite.data_module.tokenization import ESMCTokenizerWrapper
from petabite.data_module.utils import read_sequence_csv
from petabite.model_module import ModelFactory
from petabite.utils import setup_logging

logger = logging.getLogger(__name__)

_SAFETENSORS_NAME = "model.safetensors"
_TORCH_BIN_NAME = "pytorch_model.bin"


def _resolve_device(requested: str | None) -> torch.device:
    """Pick the inference device, honouring an explicit config override."""
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model: nn.Module, checkpoint_dir: Path) -> None:
    """Load weights saved by ``Trainer.save_model`` into ``model``, in place.

    ``ESMCActivityModel`` is a plain ``nn.Module``, so Trainer saves a raw state
    dict rather than a PreTrainedModel directory: ``model.safetensors`` normally,
    ``pytorch_model.bin`` when ``save_safetensors=False``.

    Raises:
        FileNotFoundError: if neither weights file is present.
        RuntimeError: if the checkpoint does not match the configured model.
    """
    safetensors_path = checkpoint_dir / _SAFETENSORS_NAME
    bin_path = checkpoint_dir / _TORCH_BIN_NAME

    if safetensors_path.is_file():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path))
        source = safetensors_path
    elif bin_path.is_file():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        source = bin_path
    else:
        raise FileNotFoundError(
            f"No checkpoint in {checkpoint_dir}: expected {_SAFETENSORS_NAME} or "
            f"{_TORCH_BIN_NAME}. Run `python -m petabite.train` first."
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    # The frozen ESM-C base is reloaded from the Hub by the backbone, so it is
    # legitimately absent from the checkpoint; adapter and head weights are not.
    critical = [k for k in missing if "lora_" in k or k.startswith("head.")]
    if critical:
        raise RuntimeError(
            f"Checkpoint {source} is missing {len(critical)} trained tensors "
            f"(e.g. {critical[:3]}). Check that model config matches the run."
        )
    if unexpected:
        logger.warning("Checkpoint has %d unexpected keys, ignored", len(unexpected))
    logger.info("Loaded checkpoint from %s", source)


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    records: list[dict[str, object]],
    tokenizer: ESMCTokenizerWrapper,
    device: torch.device,
    batch_size: int,
    task: str,
) -> list[float]:
    """Return one score per record: activity for regression, P(top class) for clf."""
    model.eval()
    scores: list[float] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        encoded = tokenizer.encode([str(r["sequence"]) for r in batch])
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )["logits"]
        if task == "regression":
            scores.extend(logits.squeeze(-1).float().cpu().tolist())
        else:
            scores.extend(logits.softmax(dim=-1).max(dim=-1).values.float().cpu().tolist())
        logger.info("Scored %d/%d", min(start + batch_size, len(records)), len(records))
    return scores


def write_predictions(
    records: list[dict[str, object]], scores: list[float], path: Path
) -> None:
    """Write id,sequence,prediction[,label] sorted by descending prediction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(zip(records, scores), key=lambda pair: pair[1], reverse=True)
    has_label = any("label" in r for r in records)
    columns = ["id", "sequence", "prediction"] + (["label"] if has_label else [])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record, score in ranked:
            row = {"id": record["id"], "sequence": record["sequence"], "prediction": score}
            if has_label:
                row["label"] = record.get("label", "")
            writer.writerow(row)
    logger.info("Wrote %d predictions to %s", len(ranked), path)


@hydra.main(version_base=None, config_path="../../config", config_name="train")
def main(cfg: DictConfig) -> None:
    """Load model and predict activity for each sequence in the input CSV."""
    setup_logging()
    predict_cfg = cfg.get("predict", {})
    checkpoint_dir = Path(predict_cfg.get("checkpoint_dir") or cfg.output_dir)
    output_csv = Path(predict_cfg.get("output_csv") or (checkpoint_dir / "predictions.csv"))
    # `or` not `get(default)`: the key exists in train.yaml with an explicit null.
    batch_size = predict_cfg.get("batch_size") or cfg.trainer.per_device_eval_batch_size
    device = _resolve_device(predict_cfg.get("device"))

    records = read_sequence_csv(Path(cfg.data.csv_path))
    logger.info("Scoring %d candidates on %s", len(records), device)

    model_cls = ModelFactory(cfg.model.name)
    model = model_cls(
        backbone_name=cfg.model.backbone_name,
        task=cfg.model.task,
        model_id=cfg.model.model_id,
        output_dim=cfg.model.output_dim,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=list(cfg.model.target_modules),
        lora_fused=cfg.model.lora_fused,
        fused_lora_path=cfg.model.get("fused_lora_path"),
        trust_remote_code=cfg.model.trust_remote_code,
    )
    load_checkpoint(model, checkpoint_dir)
    model.to(device)

    tokenizer = ESMCTokenizerWrapper(
        model_name=cfg.model.model_id, max_length=cfg.data.max_length
    )
    scores = predict_scores(
        model, records, tokenizer, device, batch_size, cfg.model.task
    )
    write_predictions(records, scores, output_csv)


if __name__ == "__main__":
    main()
