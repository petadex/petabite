#!/bin/bash
#SBATCH --job-name=petabite-embed
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_interac
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1
python scripts/embed_labels.py --lora outputs/plasticesm-c60-r32 \
  --out logs/label_embeddings.npz
