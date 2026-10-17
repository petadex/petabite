#!/bin/bash
# The whole evaluation rerun on the corrected metric.
#
# Everything reported before this job used pooled out-of-fold predictions,
# which is invalid under a strong penalty: each fold's predictions shrink
# toward that fold's own training mean, so the concatenated vector ranks
# sequences partly by which fold they landed in. On the p-np-c4 stratum the
# pooled score for sequence length ran from +0.04 to -0.29 across the penalty
# grid while the truth, computed directly within folds, is +0.4157.
#
# Predictions are now z-scored within their fold before pooling, which keeps
# each model's own ordering and discards only the fold-specific level. The
# per-fold mean is reported alongside as the check. Leave-one-component-out is
# dropped: a one-sequence fold has no within-fold ordering to preserve.
#SBATCH --job-name=petabite-corrected
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
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUT="logs/corrected-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 1. main sweep ##########"
python scripts/eval_sweep.py --embeddings logs/label_embeddings.npz \
  --standardise --out "${OUT}/sweep.json" > "${OUT}/sweep.log" 2>&1 &
python scripts/eval_sweep.py --embeddings logs/label_embeddings.npz \
  --standardise --include-llm --out "${OUT}/sweep_llm.json" \
  > "${OUT}/sweep_llm.log" 2>&1 &

echo "########## 2. leakage ##########"
python scripts/eval_leakage.py --embeddings logs/label_embeddings.npz \
  --standardise --out "${OUT}/leakage.json" > "${OUT}/leakage.log" 2>&1 &

echo "########## 3. strata and conditions ##########"
for target in kcat kcat_over_km km pet_release_conc relative_activity; do
  python scripts/eval_conditions.py \
    --embeddings logs/label_embeddings.npz \
    --targets "${target}" --standardise --power-trials 100 \
    --out "${OUT}/conditions_${target}.json" \
    > "${OUT}/conditions_${target}.log" 2>&1 &
done

echo "########## 4. robustness of whatever wins ##########"
for pair in "kcat_over_km:p-np-c4" "kcat:p-np-c4" "km:p-np-c4"; do
  target=${pair%%:*}
  substrate=${pair##*:}
  python scripts/eval_robust.py --embeddings logs/label_embeddings.npz \
    --target "${target}" --substrate "${substrate}" --standardise \
    --out "${OUT}/robust_${target}.json" \
    > "${OUT}/robust_${target}.log" 2>&1 &
done
wait
echo "all finished"

for section in sweep leakage conditions robust; do
  for f in "${OUT}"/${section}*.log; do
    [ -e "${f}" ] || continue
    echo "##################### $(basename "${f}") #####################"
    cat "${f}"
    echo
  done
done
