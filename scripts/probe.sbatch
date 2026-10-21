#!/bin/bash
# Probe stock ESM-C against PlasticESM on the Stage 2 labels.
# Small and quick: 157 sequences embedded twice, then ridge regressions.
#SBATCH --job-name=petabite-probe
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_interac
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1
LORA=${LORA:-outputs/plasticesm-c60-r32}
echo "host $(hostname) | lora ${LORA}"
python scripts/probe_stage2.py --lora "${LORA}" --out "logs/probe-${SLURM_JOB_ID}.json"
