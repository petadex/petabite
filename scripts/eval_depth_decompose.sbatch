#!/bin/bash
# Two questions, both about the melting temperature candidate.
#
# Where does its signal live? Job 6352414 showed the stock embeddings scoring
# +0.534 on melting temperature, surviving fold reshuffling at 100% above zero
# and the whole alpha grid, but collapsing to +0.091 when the one component
# holding 97 of the 115 measurements is dropped. A homology-controlled number
# can still be family recognition rather than function prediction, and the two
# look identical in one number. The decomposition separates them.
#
# And is the last layer the right place to read from? The depth sweep was run
# twice before and both passes predate the metric fix, so nothing has ever
# been validly claimed from it. The per-layer embeddings are already cached
# and `stock_post` is bit-identical to the `stock` block that produced the
# candidate, so the same validated path can score all fourteen.
#SBATCH --job-name=petabite-depth
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUT="logs/depth-${SLURM_JOB_ID}"
LAYERS=logs/depth-6343616/layer_embeddings.npz
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 0. verify the decomposition ##########"
python - <<'PY'
import collections
import numpy as np

import sys
sys.path.insert(0, "scripts")
from eval_measurements import decompose, read_rows
from petabite.eval_module import labels

cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
index = {name: position for position, name in enumerate(cached["names"])}
components = labels.read_components()
rows = read_rows("melting_temperature", False, index)

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


# Each sequence's own median, handed back as the prediction, is a perfect
# predictor, so both parts of the decomposition have to be 1.
medians = collections.defaultdict(list)
for m in rows:
    medians[m.seq_md5].append(m.value)
per_sequence = {name: float(np.median(v)) for name, v in medians.items()}
perfect = decompose(rows, np.array([per_sequence[m.seq_md5] for m in rows]),
                    components)
check("a perfect prediction is 1 between families",
      perfect["between_families"] == 1.0, f"  {perfect['between_families']}")
check("a perfect prediction is 1 within families",
      perfect["within_families"] == 1.0, f"  {perfect['within_families']}")

# A prediction that is constant inside each component carries the between
# part and nothing else, which is exactly what family recognition looks like.
by_component = collections.defaultdict(list)
for name, value in per_sequence.items():
    by_component[components[name]].append(value)
component_mean = {group: float(np.mean(values))
                  for group, values in by_component.items()}
recognition = decompose(
    rows, np.array([component_mean[components[m.seq_md5]] for m in rows]),
    components)
check("family recognition alone is 1 between families",
      recognition["between_families"] == 1.0,
      f"  {recognition['between_families']}")
# There is no within-family ordering to score here, and the honest answer is
# that the question does not apply rather than a number. A first pass of this
# gate read +0.056, which was a rank correlation over floating point dust.
check("family recognition alone has no within-family part",
      recognition["within_families"] is None
      and recognition["within_is_degenerate"],
      f"  {recognition['within_families']}, "
      f"degenerate {recognition['within_is_degenerate']}")

# And the mirror image: a prediction holding only each sequence's deviation
# from its own family mean has the within part and no between part.
deviation = {name: per_sequence[name] - component_mean[components[name]]
             for name in per_sequence}
ranking = decompose(rows,
                    np.array([deviation[m.seq_md5] for m in rows]),
                    components)
check("within-family ranking alone is 1 within families",
      ranking["within_families"] == 1.0
      and not ranking["within_is_degenerate"],
      f"  {ranking['within_families']}")
check("within-family ranking alone is near 0 between families",
      abs(ranking["between_families"]) < 0.5,
      f"  {ranking['between_families']}")

print()
print(f"  the target's own shape: {perfect['n_components']} components, "
      f"largest holds {perfect['largest_component_share']:.0%}, "
      f"{perfect['n_within_pairs']} sequences have a relative to rank against")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("decomposition verified")
PY

echo
echo "########## 1. where the candidate's signal lives ##########"
pids=()
for pair in "melting_temperature:stock" "kcat:length"; do
  target=${pair%%:*}
  block=${pair##*:}
  python scripts/eval_measurements.py \
    --embeddings logs/label_embeddings.npz \
    --targets "${target}" --stress "${block}" --stress-trials 40 \
    --out "${OUT}/decompose_${target}_${block}.json" \
    > "${OUT}/decompose_${target}_${block}.log" 2>&1 &
  pids+=($!)
done

echo
echo "########## 2. the depth sweep, on the corrected metric ##########"
BLOCKS="none,length,composition"
for backbone in stock plasticesm; do
  for layer in L5 L10 L15 L20 L25 L29 post; do
    BLOCKS="${BLOCKS},${backbone}_${layer}"
  done
done
echo "blocks: ${BLOCKS}"

for target in melting_temperature kcat kcat_over_km; do
  python scripts/eval_measurements.py \
    --embeddings "${LAYERS}" \
    --targets "${target}" --blocks "${BLOCKS}" --standardise \
    --out "${OUT}/layers_${target}.json" \
    > "${OUT}/layers_${target}.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

for f in "${OUT}"/*.log; do
  echo "##################### $(basename "${f}") #####################"
  grep -vE "Warning|score = spearmanr" "${f}" || true
  echo
done
exit "${failed}"
