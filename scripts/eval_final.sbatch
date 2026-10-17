#!/bin/bash
# The corrected evaluation, after three methodological faults were found in
# the first pass:
#
#   1. A hook on the last transformer block captures the residual stream
#      before the final LayerNorm, while `last_hidden_state` is after it.
#      Different tensors, and the pre-norm one has a column standard
#      deviation of 26.8 against 0.008.
#   2. A ridge penalty is not scale invariant, so one alpha grid across
#      feature sets differing 3,000-fold in scale compares penalties rather
#      than representations. Everything here is scored standardised, with the
#      raw version kept alongside for the strata so the difference is visible.
#   3. The grid was truncated at 10,000, which is where the standardised
#      optimum sat, and plain argmax selection over roughly 17 training
#      components swung the headline between 0.30 and 0.40. The grid now
#      reaches 1e6 and selection uses the one-standard-error rule.
#
# Everything runs here rather than on the login node, where three concurrent
# analyses starved each other.
#SBATCH --job-name=petabite-final
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=02:45:00
#SBATCH --output=logs/%x-%j.out
set -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONWARNINGS=ignore::UserWarning
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUT="logs/final-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
LAYERS=logs/depth-6343616/layer_embeddings.npz
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 0. is the raw fit even deterministic? ##########"
# The raw fits on the post-norm embeddings hit singular-matrix fallbacks, and
# a score that moves with the BLAS thread count is not a measurement. Run the
# same stratum under one thread and under sixteen.
for threads in 1 16; do
  echo "--- OMP_NUM_THREADS=${threads} ---"
  OMP_NUM_THREADS=${threads} MKL_NUM_THREADS=${threads} \
    OPENBLAS_NUM_THREADS=${threads} \
    python scripts/eval_determinism.py \
      --embeddings logs/label_embeddings.npz \
      --target kcat_over_km --substrate p-np-c4
done

echo
echo "########## 1. corrected stratified sweep, standardised ##########"
for target in kcat kcat_over_km km pet_release_conc; do
  python scripts/eval_conditions.py \
    --embeddings logs/label_embeddings.npz \
    --targets "${target}" --standardise --power-trials 100 \
    --out "${OUT}/conditions_${target}_std.json" \
    > "${OUT}/conditions_${target}_std.log" 2>&1 &
done
wait
for f in "${OUT}"/conditions_*_std.log; do
  echo "##### $(basename "${f}") #####"
  sed -n '/-- 3. one substrate/,$p' "${f}"
  echo
done

echo
echo "########## 2. robustness, raw against standardised ##########"
for pair in "kcat_over_km:p-np-c4" "kcat:p-np-c4" "km:p-np-c4"; do
  target=${pair%%:*}
  substrate=${pair##*:}
  for mode in raw std; do
    flag=""
    if [ "${mode}" = "std" ]; then flag="--standardise"; fi
    python scripts/eval_robust.py \
      --embeddings logs/label_embeddings.npz \
      --target "${target}" --substrate "${substrate}" ${flag} \
      --out "${OUT}/robust_${target}_${mode}.json" \
      > "${OUT}/robust_${target}_${mode}.log" 2>&1 &
  done
done
wait
for f in "${OUT}"/robust_*.log; do
  echo "##### $(basename "${f}") #####"; cat "${f}"; echo
done

echo
echo "########## 3. layer profile, standardised, wider grid ##########"
for pair in "kcat_over_km:p-np-c4" "kcat:p-np-c4" "km:p-np-c4"; do
  target=${pair%%:*}
  substrate=${pair##*:}
  python scripts/eval_layers.py --embeddings "${LAYERS}" \
    --target "${target}" --substrate "${substrate}" \
    --out "${OUT}/layers_${target}.json" \
    > "${OUT}/layers_${target}.log" 2>&1 &
done
for target in kcat kcat_over_km km pet_release_conc; do
  python scripts/eval_layers.py --embeddings "${LAYERS}" --target "${target}" \
    --out "${OUT}/layers_${target}_pooled.json" \
    > "${OUT}/layers_${target}_pooled.log" 2>&1 &
done
wait
for f in "${OUT}"/layers_*.log; do
  echo "##### $(basename "${f}") #####"; cat "${f}"; echo
done
