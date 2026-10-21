#!/bin/bash
# Search-space item 1: can a backbone rank variants inside one family?
#
# Component 00855c73 holds 546 sequences and carries almost all the dense
# label coverage: 113 PET release values spanning seven orders of magnitude,
# 76 kcat_over_km, 65 kcat, 47 melting temperatures. The earlier evaluation
# treated it as one unit of independence, which discarded every within-family
# comparison in the release.
#
# Step 0 verifies the new alignment features against hand calculations before
# any score is computed from them, because a new metric reported without that
# check is what produced three retracted results.
#SBATCH --job-name=petabite-family
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=02:45:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUT="logs/family-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 0. verify the new features by hand ##########"
python - <<'PY'
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, "scripts")
from eval_within_family import build_aligner, identity, read_clusters
from petabite.eval_module import labels

aligner = build_aligner()
failures = []

def check(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, want {want}")
    if not ok:
        failures.append(name)

# A sequence against itself must be fully identical with no substitutions.
seq = "MKVLAAGIVGLAAWQLSQAHGHP"
frac, subs, gaps = identity(aligner, seq, seq)
check("self identity", frac, 1.0)
check("self substitutions", subs, 0)
check("self gaps", gaps, 0)

# One hand-made substitution at position 3 of an otherwise identical pair.
mutant = seq[:3] + ("A" if seq[3] != "A" else "G") + seq[4:]
frac, subs, gaps = identity(aligner, seq, mutant)
check("one substitution counted", subs, 1)
check("one substitution gaps", gaps, 0)
check("one substitution identity", frac, (len(seq) - 1) / len(seq))

# A single deletion must register as a gap, not a substitution.
deleted = seq[:5] + seq[6:]
frac, subs, gaps = identity(aligner, seq, deleted)
check("one deletion gaps", gaps, 1)

# And the within-family length control, against a direct calculation with no
# model involved: the mean within-fold Spearman of length against the target.
cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
index = {n: i for i, n in enumerate(cached["names"])}
meta = read_clusters()
per_sequence = labels.aggregate(labels.read_measurements("pet_release_conc"))
inside = sorted(m for m in per_sequence
                if m in index and m in meta
                and meta[m]["component"].startswith("00855c73c087597a"))
y = np.array([per_sequence[m] for m in inside])
length = cached["length"][np.array([index[m] for m in inside])].ravel()
groups = [meta[m]["cluster_90"] for m in inside]
folds = list(GroupKFold(n_splits=5).split(length.reshape(-1, 1), y, groups))
direct = [spearmanr(length[held], y[held]).statistic for _, held in folds]
print(f"  reference: direct mean within-fold Spearman(length, y) = "
      f"{np.mean(direct):+.4f}")
print(f"  reference: whole-family Spearman(length, y)            = "
      f"{spearmanr(length, y).statistic:+.4f}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("all feature checks passed")
PY

echo
echo "########## 1. within-family and cross-family ranking ##########"
pids=()
for target in pet_release_conc kcat_over_km kcat relative_activity km melting_temperature; do
  python scripts/eval_within_family.py \
    --embeddings logs/label_embeddings.npz \
    --target "${target}" --power-trials 100 \
    --out "${OUT}/${target}.json" > "${OUT}/${target}.log" 2>&1 &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
for f in "${OUT}"/*.log; do
  echo "##################### $(basename "${f}") #####################"
  cat "${f}"
  echo
done
exit "${failed}"
