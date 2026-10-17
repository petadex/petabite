#!/bin/bash
# Pooled against fold-averaged scoring, across the whole alpha grid. Decides
# whether the stratified result is a model result or a fold-mean artefact.
#SBATCH --job-name=petabite-metric
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUT="logs/metric-${SLURM_JOB_ID}"
mkdir -p "${OUT}"

for pair in "kcat_over_km:p-np-c4" "kcat:p-np-c4" "km:p-np-c4"; do
  target=${pair%%:*}
  substrate=${pair##*:}
  python scripts/eval_metric.py --embeddings logs/label_embeddings.npz \
    --target "${target}" --substrate "${substrate}" --raw \
    --out "${OUT}/${target}.json" > "${OUT}/${target}.log" 2>&1 &
done
wait
for f in "${OUT}"/*.log; do echo "##### $(basename "${f}") #####"; cat "${f}"; echo; done
