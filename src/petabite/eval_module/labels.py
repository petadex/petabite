"""Reading the v2 label release, with the assay conditions kept attached.

The earlier evaluation collapsed every measurement of a sequence to its median
and threw the conditions away. That was the only option while the question was
"can a backbone predict this target", but it hides a second question: whether
the target is a property of the sequence at all, or mostly a property of the
protocol it was measured under.

So the loader here works one measurement at a time and keeps pH, temperature,
incubation time and substrate alongside each value. Aggregation to one value
per sequence is still available, but it is now a separate step a caller asks
for rather than something that happens on the way in.
"""

from __future__ import annotations

import collections
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABELS = Path("/project/6122149/purav/petai-labels/v2")

# Which targets are worth asking about, and whether they span enough orders of
# magnitude to need a log transform. The canonical units come from the Stage 3
# unit normalisation.
TARGETS = {
    "pet_release_conc": True,      # uM
    "relative_activity": False,    # percent
    "kcat": True,                  # s-1
    "km": True,                    # uM
    "kcat_over_km": True,          # M-1 s-1
    "weight_loss": False,          # percent
    "melting_temperature": False,  # degC
}

# Below these counts a correlation is noise whatever it reads, so the target is
# reported as unscoreable rather than scored. The component floor matters more
# than the sequence floor: ten sequences from two families carry two
# independent observations.
MIN_SEQUENCES = 25
MIN_COMPONENTS = 8


@dataclass(frozen=True)
class Measurement:
    """One row of the label table, after unit normalisation."""

    seq_md5: str
    target: str
    value: float          # already log10 transformed where the target needs it
    substrate: str
    ph: float | None
    temperature_c: float | None
    time_h: float | None
    source: str
    split: str


def _optional_float(text: str) -> float | None:
    """Parse a condition column, which is blank wherever the paper was silent."""
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_measurements(target: str, include_llm: bool = False) -> list[Measurement]:
    """Every usable measurement of one target, conditions attached.

    Non-positive values are dropped from log-transformed targets. They are
    control-subtracted readings rather than errors, but they have no
    logarithm, and treating them as a floor would invent an ordering between
    sequences that the assay did not measure.
    """
    needs_log = TARGETS[target]
    out: list[Measurement] = []

    with (LABELS / "normalised" / "labels_normalised.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["target"] != target or not row["value_canonical"]:
                continue
            if row["source"] == "llm" and not include_llm:
                continue

            value = float(row["value_canonical"])
            if needs_log:
                if value <= 0:
                    continue
                value = math.log10(value)

            out.append(Measurement(
                seq_md5=row["seq_md5"],
                target=target,
                value=value,
                substrate=row["substrate"].strip().lower(),
                ph=_optional_float(row["ph"]),
                temperature_c=_optional_float(row["temperature_c"]),
                time_h=_optional_float(row["time_h"]),
                source=row["source"],
                split=row["split"],
            ))
    return out


def read_components() -> dict[str, str]:
    """Homology component of every sequence in the release."""
    with (LABELS / "sequences.csv").open(newline="") as fh:
        return {row["seq_md5"]: row["component"] for row in csv.DictReader(fh)}


def read_sequences() -> dict[str, str]:
    """The release FASTA, keyed by the md5 names used throughout."""
    sequences: dict[str, str] = {}
    name, parts = None, []

    with (LABELS / "sequences.fasta").open() as fh:
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


def aggregate(measurements: list[Measurement]) -> dict[str, float]:
    """One value per sequence, the median of its measurements.

    The median rather than the mean because a sequence occasionally carries an
    outlying reading from a single unusual assay, and because the downstream
    scoring is rank based anyway.
    """
    per_sequence: dict[str, list[float]] = collections.defaultdict(list)
    for m in measurements:
        per_sequence[m.seq_md5].append(m.value)
    return {md5: float(np.median(values)) for md5, values in per_sequence.items()}


def substrate_counts(measurements: list[Measurement],
                     components: dict[str, str]) -> list[tuple[str, int, int, int]]:
    """Per substrate: measurements, distinct sequences, distinct components.

    Sorted by component count, because that is what decides whether a
    substrate can be scored, and it is routinely much smaller than the
    sequence count suggests.
    """
    by_substrate: dict[str, list[Measurement]] = collections.defaultdict(list)
    for m in measurements:
        by_substrate[m.substrate].append(m)

    rows = []
    for substrate, group in by_substrate.items():
        sequences = {m.seq_md5 for m in group}
        families = {components.get(md5) for md5 in sequences}
        rows.append((substrate, len(group), len(sequences), len(families)))
    return sorted(rows, key=lambda row: (-row[3], -row[2]))


def is_scoreable(n_sequences: int, n_components: int) -> bool:
    """Does a target have enough independent families to be worth scoring?"""
    return n_sequences >= MIN_SEQUENCES and n_components >= MIN_COMPONENTS
