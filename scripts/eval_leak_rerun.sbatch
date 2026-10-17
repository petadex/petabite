#!/bin/bash
#SBATCH --job-name=petabite-leak2
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUT=logs/corrected-6344807
python scripts/eval_leakage.py --embeddings logs/label_embeddings.npz \
  --standardise --out "${OUT}/leakage.json" 2>&1 | tee "${OUT}/leakage.log"
echo
python scripts/eval_leakage.py --embeddings logs/label_embeddings.npz \
  --standardise --include-llm --out "${OUT}/leakage_llm.json" 2>&1 \
  | tee "${OUT}/leakage_llm.log"
