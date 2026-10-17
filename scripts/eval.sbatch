#!/bin/bash
# The evaluation sweep and power analysis. CPU-bound but heavy enough that the
# login node starves it, and running two of them there starved both.
#SBATCH --job-name=petabite-eval
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_interac
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

echo "########## sweep ##########"
python scripts/eval_sweep.py --embeddings logs/label_embeddings.npz \
  --out "logs/eval_sweep-${SLURM_JOB_ID}.json"

echo "########## sweep, with LLM labels folded in ##########"
python scripts/eval_sweep.py --embeddings logs/label_embeddings.npz --include-llm \
  --out "logs/eval_sweep_llm-${SLURM_JOB_ID}.json"

echo "########## power ##########"
for t in pet_release_conc kcat km kcat_over_km; do
  python scripts/eval_power.py --target "$t" --trials 120 \
    --out "logs/power_${t}-${SLURM_JOB_ID}.json"
done
