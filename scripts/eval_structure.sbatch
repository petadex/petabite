#!/bin/bash
# Search-space item 6, done properly.
#
# The first pass recorded this item as unavailable, and that was too quick.
# The precomputed structures the search space referred to genuinely do not
# exist locally: the corpus holds a 308,394-row manifest of AlphaFold S3 keys
# and no coordinates, and the per-sequence annotation layers reach 4.7% of the
# labelled set because 95% of these sequences are engineered variants absent
# from a metagenomic ORF corpus. But ESMFold2 is installed in its own
# virtualenv with its weights cached, so the structures can simply be
# computed, and "unavailable" was a statement about the wrong thing.
#
# Only the 397 sequences carrying a scoreable target are folded, since one
# with no usable label cannot contribute to any score.
#
# Eleven hours, not eight. Job 6354579 timed the three longest structures at
# 246s each, and extrapolating over the length distribution by the pair
# stack's quadratic scaling gives 3.0h to 6.5h for all 397 depending on the
# exponent. Eight hours left no margin at the slow end of that range.
#
# Two virtualenvs are involved and they cannot share a process: esm 3.4.1
# pins torch 2.11 and lives in ~/esmfold2-341, while the evaluation code lives
# in ~/petabite. Each step therefore runs in its own subshell, and the
# selection of which sequences to fold is handed over as a FASTA rather than
# duplicating the label logic into the folding script.
#SBATCH --job-name=petabite-structure
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=11:00:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
cd "${SLURM_SUBMIT_DIR}"

OUT="logs/structure-${SLURM_JOB_ID}"
FASTA="${OUT}/to_fold.fasta"
GATE="${OUT}/gate.npz"
STRUCTURES="${OUT}/structures.npz"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## 1. choose what to fold, where the labels are readable ########"
(
  source activate_petabite.sh >/dev/null 2>&1
  FASTA="${FASTA}" python - <<'PY'
import os
from petabite.eval_module import labels

sequences = labels.read_sequences()
components = labels.read_components()

wanted = set()
for target in labels.TARGETS:
    per_sequence = labels.aggregate(labels.read_measurements(target))
    names = [n for n in per_sequence if n in sequences]
    groups = {components[n] for n in names if n in components}
    if labels.is_scoreable(len(names), len(groups)):
        wanted.update(names)
        print(f"  {target:22} {len(names):>4} sequences, "
              f"{len(groups):>3} components  -> folding")
    else:
        print(f"  {target:22} {len(names):>4} sequences, "
              f"{len(groups):>3} components  -> skipped")

chosen = sorted(n for n in wanted if n in components)
path = os.environ["FASTA"]
with open(path, "w") as handle:
    for name in chosen:
        handle.write(f">{name}\n{sequences[name]}\n")
families = {components[n] for n in chosen}
print(f"\n{len(chosen)} sequences to fold across {len(families)} components")
print(f"wrote {path}")
PY
)

echo
echo "########## 2. fold three, and check the features before folding the rest #"
(
  source /project/6122149/purav/igem-toronto/activate_esmfold2.sh
  export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
  python scripts/fold_labels.py --sequences "${FASTA}" --out "${GATE}" --limit 3
)

GATE="${GATE}" python3 - <<'PY'
import os
import numpy as np

cached = np.load(os.environ["GATE"], allow_pickle=False)
columns = [str(c) for c in cached["columns"]]
scalars = cached["scalars"]
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


print(f"  folded {scalars.shape[0]} structures, {len(columns)} features")
table = {name: scalars[:, position] for position, name in enumerate(columns)}

check("three structures came back", scalars.shape[0] == 3)
check("pLDDT is a probability", bool(np.all((table["plddt_mean"] >= 0)
                                            & (table["plddt_mean"] <= 1))),
      f"  {table['plddt_mean']}")
check("pTM is a probability", bool(np.all((table["ptm"] >= 0)
                                          & (table["ptm"] <= 1))),
      f"  {table['ptm']}")
check("the structures are confident, as real enzymes should be",
      float(table["plddt_mean"].mean()) > 0.6,
      f"  mean pLDDT {table['plddt_mean'].mean():.3f}")
check("radius of gyration is positive and plausible",
      bool(np.all((table["radius_of_gyration"] > 5)
                  & (table["radius_of_gyration"] < 100))),
      f"  {table['radius_of_gyration']}")
check("contacts were found", bool(np.all(table["contact_density"] > 1)),
      f"  {table['contact_density']}")
if "helix" in table and np.isfinite(table["helix"]).all():
    total = table["helix"] + table["sheet"] + table["loop"]
    check("secondary structure fractions sum to one",
          bool(np.allclose(total, 1.0, atol=1e-6)), f"  {total}")
    check("some secondary structure was assigned",
          float((table["helix"] + table["sheet"]).mean()) > 0.1,
          f"  helix+sheet {(table['helix'] + table['sheet']).mean():.3f}")
else:
    print("  NOTE  secondary structure is nan; the backbone was incomplete")
if "pair_embedding" in cached:
    check("the pair embedding came back",
          cached["pair_embedding"].shape[0] == 3
          and cached["pair_embedding"].shape[1] > 1,
          f"  {cached['pair_embedding'].shape}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("the features are sane; folding the rest")
PY

echo
echo "########## 3. fold everything ##########"
(
  source /project/6122149/purav/igem-toronto/activate_esmfold2.sh
  export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
  python scripts/fold_labels.py --sequences "${FASTA}" --out "${STRUCTURES}"
)

echo
echo "########## 4. score the structure features ##########"
(
  source activate_petabite.sh >/dev/null 2>&1
  python scripts/eval_structure.py \
    --structures "${STRUCTURES}" \
    --embeddings logs/label_embeddings.npz \
    --out "${OUT}/structure_scores.json" 2>&1 \
    | grep -vE "Warning|score = spearmanr" || true
)
