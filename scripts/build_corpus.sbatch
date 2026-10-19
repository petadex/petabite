#!/bin/bash
# Build the 60%-identity pretraining corpus. CPU work, but this is a GPU-only
# cluster so it takes the smallest GPU allocation available.
#
# First a scan of md5 and c60_id over the corpus spine (orfs/, 307M rows) to find
# the 60% clusters holding a labelled sequence, then two passes over the
# 18,172,960-row centroid parquet: one to draw a member per 60% cluster, one to
# tokenize the chosen rows. Must not run on a login node, which OOM-kills it.
#
#     mkdir -p logs && sbatch scripts/build_corpus.sbatch
#SBATCH --job-name=petabite-corpus
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_interac
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -u -o pipefail

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache

cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

SEED=${SEED:-0}
OUT=${OUT:-/project/6122149/purav/petabite-tokenized/c60-seed${SEED}}
ORFS=${ORFS:-/project/6122149/purav/petadex-corpus/full/orfs}

echo "host $(hostname) | seed ${SEED} | orfs ${ORFS} | out ${OUT}"
python scripts/build_c60_corpus.py --out "${OUT}" --seed "${SEED}" --orfs "${ORFS}" --max-length 512
BUILD_RC=$?
echo "exit=${BUILD_RC}"
exit "${BUILD_RC}"
