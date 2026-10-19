#!/bin/bash
# Stage 1 production run: one epoch of MLM LoRA over the 60%-identity corpus.
#
# The corpus is about 1.81M sequences and 545M tokens (the first build, which
# excluded only centroid-level label matches, was 1,814,215 and 544,580,912; the
# exact figures are in the corpus manifest.json). At the measured 27.5k tokens/s
# on one L40S that is about 5.5 h, so this CANNOT go to gpubase_l40s_b1 or
# gpubase_interac, whose limit is 3 h. b2 allows 12 h.
#
# Build the corpus first:
#     python scripts/build_c60_corpus.py --out /project/.../c60-seed0 --seed 0
#
# Submit:
#     mkdir -p logs && sbatch scripts/train_mlm.sbatch
#
# To resume after a timeout, resubmit with:
#     sbatch --export=ALL,RESUME=true scripts/train_mlm.sbatch
# The adapters are restored from the last checkpoint by LoraSaveCallback, which
# refuses rather than silently restarting them from zero if the file is absent.
#SBATCH --job-name=plasticesm
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -u -o pipefail

export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=${SLURM_SUBMIT_DIR}/wandb

cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

CORPUS=${CORPUS:-/project/6122149/purav/petabite-tokenized/c60-seed0}
OUTDIR=${OUTDIR:-outputs/plasticesm-c60-r32}
RESUME=${RESUME:-null}

echo "=================== environment ==================="
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "corpus  : ${CORPUS}"
echo "outdir  : ${OUTDIR}"
echo "resume  : ${RESUME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, esm, transformers; print('torch', torch.__version__, '| esm', esm.__version__, '| transformers', transformers.__version__)"
test -f "${CORPUS}/manifest.json" && cat "${CORPUS}/manifest.json"

echo "=================== train ==================="
python -m petabite.mlm_finetune \
  output_dir="${OUTDIR}" \
  data.tokenized_path="${CORPUS}" \
  trainer.resume_from_checkpoint="${RESUME}" \
  wandb.run_name="plasticesm-c60-r32-${SLURM_JOB_ID}"
TRAIN_RC=$?
echo "training exit code: ${TRAIN_RC}"
if [ "${TRAIN_RC}" -ne 0 ]; then
  echo "RUN FAILED"
  exit "${TRAIN_RC}"
fi

echo "=================== verify checkpoint ==================="
python scripts/verify_checkpoint.py "${OUTDIR}"
VERIFY_RC=$?

echo "=================== saved files ==================="
ls -la "${OUTDIR}"
du -sh "${OUTDIR}"

if [ "${VERIFY_RC}" -eq 0 ]; then
  echo "RUN COMPLETE AND CHECKPOINT VERIFIED"
else
  echo "RUN COMPLETE BUT CHECKPOINT VERIFICATION FAILED"
fi
exit "${VERIFY_RC}"
