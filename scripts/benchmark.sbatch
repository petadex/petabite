#!/bin/bash
# Throughput benchmark: baseline settings against the tuned ones, same GPU.
#
# Writes a JSON next to the log so the numbers in the PR can be traced back to
# a specific run on specific hardware.
#
# Submit from the repo root:
#     mkdir -p logs && sbatch scripts/benchmark.sbatch
#SBATCH --job-name=petabite-bench
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -u -o pipefail

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/project/6122149/purav/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/benchmark_throughput.py --compare \
  --steps 15 --warmup 4 \
  --json "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.json"
