#!/bin/bash
# Two questions in one job, because both need the same node and the first
# needs the GPU only briefly.
#
#   1. Is the one positive result robust? Pure CPU, four stresses per feature
#      set, run on the strata where a backbone cleared the controls.
#   2. Does a middle layer of ESM-C know more about activity than the last
#      one? Needs the GPU to embed at six depths, then CPU to score.
#SBATCH --job-name=petabite-depth
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
# Standardising removes most of the ill-conditioning, but a near-singular fold
# can still warn, and 89 warnings in one log made the last run's tables
# unreadable.
export PYTHONWARNINGS=ignore::UserWarning
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

LORA=outputs/plasticesm-c60-r32
OUT="logs/depth-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"

echo
echo "########## 1. multi-layer embeddings ##########"
python scripts/embed_layers.py --lora "${LORA}" \
  --out "${OUT}/layer_embeddings.npz"

echo
echo "########## 2. robustness of the stratified result ##########"
# One process per stratum: the strata are independent and each runs dozens of
# refits.
# Each stratum is run twice, raw and standardised. The alpha grid means
# something different for a 960-column embedding of scale 0.008 than for a
# single length column, so a result that survives only one of the two is a
# result about the ridge penalty.
for pair in "kcat_over_km:p-np-c4" "kcat:p-np-c4" "km:p-np-c4" "kcat:p-np-c2"; do
  target=${pair%%:*}
  substrate=${pair##*:}
  for mode in raw std; do
    flag=""
    if [ "${mode}" = "std" ]; then flag="--standardise"; fi
    python scripts/eval_robust.py \
      --embeddings logs/label_embeddings.npz \
      --target "${target}" --substrate "${substrate}" ${flag} \
      --out "${OUT}/robust_${target}_${substrate}_${mode}.json" \
      > "${OUT}/robust_${target}_${substrate}_${mode}.log" 2>&1 &
  done
done
wait
for f in "${OUT}"/robust_*.log; do
  echo "##### ${f} #####"; cat "${f}"; echo
done

echo
echo "########## 3. layer profile ##########"
for pair in "kcat_over_km:p-np-c4" "kcat:p-np-c4" "km:p-np-c4" "pet_release_conc:pet"; do
  target=${pair%%:*}
  substrate=${pair##*:}
  python scripts/eval_layers.py \
    --embeddings "${OUT}/layer_embeddings.npz" \
    --target "${target}" --substrate "${substrate}" \
    --out "${OUT}/layers_${target}_${substrate}.json" \
    > "${OUT}/layers_${target}_${substrate}.log" 2>&1 &
done
# The pooled kinetics targets too, since a mid-layer win there would change
# the headline nulls rather than only the stratified result.
for target in kcat km kcat_over_km pet_release_conc; do
  python scripts/eval_layers.py \
    --embeddings "${OUT}/layer_embeddings.npz" --target "${target}" \
    --out "${OUT}/layers_${target}_pooled.json" \
    > "${OUT}/layers_${target}_pooled.log" 2>&1 &
done
wait
for f in "${OUT}"/layers_*.log; do
  echo "##### ${f} #####"; cat "${f}"; echo
done
