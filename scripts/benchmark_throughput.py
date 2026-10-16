"""Measure LoRA training throughput on one GPU, and project the epoch cost.

Reports tokens per second rather than steps per second. Steps per second is
misleading here because the knobs being tuned change how many tokens a step
contains: a bigger batch or a shorter truncation length both make steps faster
and mean less per step. Tokens per second is what actually predicts how long an
epoch takes.

Each configuration is timed on the same batches, after a warmup that covers
CUDA context setup, autocast cache population and the first optimizer step.

Usage:
    python scripts/benchmark_throughput.py --preset baseline
    python scripts/benchmark_throughput.py --preset tuned
    python scripts/benchmark_throughput.py --compare
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

from petabite.data_module.tokenization import load_esmc_tokenizer
from petabite.model_module.lora import LoraConfig, apply_lora

MODEL_ID = "biohub/ESMC-300M"

# The corpus this is projected against: the 60%-identity centroids of PETadex.
CORPUS_SEQUENCES = 1_814_296
# Median ESM-C token count for a 60%-centroid, including <cls> and <eos>.
CORPUS_MEDIAN_TOKENS = 308


@dataclass
class Settings:
    """One throughput configuration."""

    name: str
    batch_size: int
    max_length: int
    bf16: bool
    tf32: bool
    fused_optimizer: bool
    compute_sae: bool
    # Pads every sequence to max_length, which is what happens without length
    # bucketing when the batch contains one long sequence.
    pad_to_max: bool
    # Gives every sequence in a batch the same length, which is what
    # `group_by_length` approximates by bucketing similar lengths together.
    # Without it a batch is padded to its longest member, so a batch of 32
    # sequences averaging 308 tokens is padded to nearer 475.
    bucketed: bool = False


PRESETS = {
    # What the repo did before this work: small batch, long truncation, no TF32,
    # unfused optimizer, ESM-C's SAE head left on, and ragged batches padded to
    # the longest member.
    "baseline": Settings(
        name="baseline",
        batch_size=8,
        max_length=1024,
        bf16=False,
        tf32=False,
        fused_optimizer=False,
        compute_sae=True,
        pad_to_max=True,
    ),
    "tuned": Settings(
        name="tuned",
        batch_size=32,
        max_length=512,
        bf16=True,
        tf32=True,
        fused_optimizer=True,
        compute_sae=False,
        pad_to_max=False,
    ),
    # What the shipped config actually produces: the tuned settings plus
    # length-grouped batching, at a batch size that fits the L40S's 46 GB.
    "tuned_bucketed": Settings(
        name="tuned_bucketed",
        batch_size=48,
        max_length=512,
        bf16=True,
        tf32=True,
        fused_optimizer=True,
        compute_sae=False,
        pad_to_max=False,
        bucketed=True,
    ),
}


def build_batches(
    settings: Settings, n_batches: int, seed: int = 0
) -> list[dict[str, torch.Tensor]]:
    """Build masked batches with a realistic length distribution.

    Lengths are drawn around the corpus median rather than fixed, so that the
    cost of padding a ragged batch is represented. With ``pad_to_max`` every
    batch is padded to ``max_length`` instead, which is the no-bucketing case.
    """
    generator = torch.Generator().manual_seed(seed)
    tokenizer = load_esmc_tokenizer(MODEL_ID, settings.max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    vocab_low, vocab_high = 4, 29  # the 20 amino acids, skipping special tokens
    batches = []
    for _ in range(n_batches):
        features = []
        # One length for the whole batch when bucketing, so the batch is not
        # padded out to its longest member.
        batch_spread = torch.randn(1, generator=generator).item() * 80
        batch_length = int(
            min(settings.max_length, max(32, CORPUS_MEDIAN_TOKENS + batch_spread))
        )
        for _ in range(settings.batch_size):
            if settings.pad_to_max:
                length = settings.max_length
            elif settings.bucketed:
                length = batch_length
            else:
                spread = torch.randn(1, generator=generator).item() * 80
                length = int(min(settings.max_length, max(32, CORPUS_MEDIAN_TOKENS + spread)))
            ids = torch.randint(
                vocab_low, vocab_high, (length,), generator=generator
            ).tolist()
            ids[0] = tokenizer.cls_token_id
            ids[-1] = tokenizer.eos_token_id
            features.append({"input_ids": ids})
        batches.append(collator(features))
    return batches


def build_model(settings: Settings) -> torch.nn.Module:
    """Load ESM-C with LoRA adapters, on the GPU."""
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    apply_lora(model, LoraConfig(r=8, alpha=16, verify=False))
    return model.cuda().train()


def run(settings: Settings, n_steps: int, n_warmup: int) -> dict:
    """Time ``n_steps`` optimizer steps and return the measurements."""
    torch.backends.cuda.matmul.allow_tf32 = settings.tf32
    torch.backends.cudnn.allow_tf32 = settings.tf32

    model = build_model(settings)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=1e-4, fused=settings.fused_optimizer
    )

    batches = build_batches(settings, n_warmup + n_steps)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=settings.bf16)

    step_times: list[float] = []
    token_counts: list[int] = []

    for index, batch in enumerate(batches):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        timed = index >= n_warmup

        if timed:
            torch.cuda.synchronize()
            started = time.perf_counter()

        with autocast:
            outputs = model(**batch, compute_sae=settings.compute_sae)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if timed:
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - started)
            # Padding is real work for the GPU, so every position counts.
            token_counts.append(batch["input_ids"].numel())

    median_step = statistics.median(step_times)
    tokens_per_step = statistics.mean(token_counts)
    tokens_per_second = tokens_per_step / median_step

    # Projected on real sequence lengths, not the padded width, because that is
    # the number of tokens an epoch of the corpus actually contains.
    epoch_tokens = CORPUS_SEQUENCES * CORPUS_MEDIAN_TOKENS
    epoch_hours = epoch_tokens / tokens_per_second / 3600

    return {
        "settings": asdict(settings),
        "median_step_seconds": round(median_step, 4),
        "tokens_per_step": int(tokens_per_step),
        "tokens_per_second": int(tokens_per_second),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "projected_epoch_hours_60pct_centroids": round(epoch_hours, 2),
    }


def report(result: dict) -> None:
    settings = result["settings"]
    print(f"\n--- {settings['name']} ---")
    print(
        f"  batch {settings['batch_size']} x {settings['max_length']} tokens"
        f" | bf16={settings['bf16']} tf32={settings['tf32']}"
        f" fused_opt={settings['fused_optimizer']}"
        f" compute_sae={settings['compute_sae']}"
        f" pad_to_max={settings['pad_to_max']}"
    )
    print(f"  median step        : {result['median_step_seconds']:.4f} s")
    print(f"  tokens per step    : {result['tokens_per_step']:,}")
    print(f"  tokens per second  : {result['tokens_per_second']:,}")
    print(f"  peak memory        : {result['peak_memory_gb']} GB")
    print(
        f"  projected epoch    : "
        f"{result['projected_epoch_hours_60pct_centroids']} h "
        f"over the 1.81M 60%-centroids"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tuned")
    parser.add_argument("--compare", action="store_true", help="run every preset")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible; this benchmark only means anything on one")
    print(f"device: {torch.cuda.get_device_name(0)}")

    names = sorted(PRESETS) if args.compare else [args.preset]
    results = []
    for name in names:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = run(PRESETS[name], args.steps, args.warmup)
        report(result)
        results.append(result)

    if len(results) > 1:
        ordered = sorted(results, key=lambda r: r["tokens_per_second"])
        slow, fast = ordered[0], ordered[-1]
        speedup = fast["tokens_per_second"] / slow["tokens_per_second"]
        print(
            f"\n{fast['settings']['name']} is {speedup:.2f}x the token throughput "
            f"of {slow['settings']['name']}: "
            f"{slow['projected_epoch_hours_60pct_centroids']} h -> "
            f"{fast['projected_epoch_hours_60pct_centroids']} h per epoch"
        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
