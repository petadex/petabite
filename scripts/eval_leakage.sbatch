#!/bin/bash
# The homology leakage measurement: how much a conventional random split
# flatters the same models on the same targets. Cheaper than the other
# analyses, since it needs no bootstrap or permutation null, but still too
# heavy for the login node.
#SBATCH --job-name=petabite-leakage
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
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

echo "########## human-curated labels ##########"
python scripts/eval_leakage.py --embeddings logs/label_embeddings.npz \
  --out "logs/eval_leakage-${SLURM_JOB_ID}.json"

echo
echo "########## with the LLM labels folded in ##########"
python scripts/eval_leakage.py --embeddings logs/label_embeddings.npz --include-llm \
  --out "logs/eval_leakage_llm-${SLURM_JOB_ID}.json"
