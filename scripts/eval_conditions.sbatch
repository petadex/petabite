#!/bin/bash
# The protocol-confounding test: is each target a property of the sequence or
# of the assay it was measured under?
#
# Pure CPU work, but the login node starves it and this cluster has no CPU-only
# partition, so it takes a GPU node and leaves the GPU idle.
#
# One process per target, because the targets are independent and the whole
# sweep is otherwise an hour of serial ridge fits. Each process is pinned to a
# single BLAS thread: ten processes each spawning sixteen would oversubscribe
# the node and run slower than the serial version.
#SBATCH --job-name=petabite-conditions
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

TARGETS="pet_release_conc kcat km kcat_over_km relative_activity"
OUT="logs/conditions-${SLURM_JOB_ID}"
mkdir -p "${OUT}"

echo "job ${SLURM_JOB_ID} on $(hostname)"
echo "targets: ${TARGETS}"
echo

# Human-curated labels, and the same again with the LLM-extracted labels
# folded in, since those changed the family structure of kcat in the first
# sweep and may change it here too.
for target in ${TARGETS}; do
  python scripts/eval_conditions.py \
    --embeddings logs/label_embeddings.npz \
    --targets "${target}" \
    --power-trials 100 \
    --out "${OUT}/${target}.json" > "${OUT}/${target}.log" 2>&1 &

  python scripts/eval_conditions.py \
    --embeddings logs/label_embeddings.npz \
    --targets "${target}" \
    --include-llm \
    --power-trials 100 \
    --out "${OUT}/${target}_llm.json" > "${OUT}/${target}_llm.log" 2>&1 &
done

wait
echo "all processes finished"
echo

for f in "${OUT}"/*.log; do
  echo "##################### ${f} #####################"
  cat "${f}"
  echo
done
