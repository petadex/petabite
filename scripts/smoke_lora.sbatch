#!/bin/bash
# GPU smoke test for the Stage 1 LoRA trainer.
#
# Trains for a handful of steps on one L40S, saves the adapters, then runs
# scripts/verify_checkpoint.py to prove the checkpoint is complete, reloadable
# and actually trained. The point is not the loss value, it is that the whole
# path works end to end on a GPU and that the weights survive the save.
#
# gpubase_interac is used rather than gpubase_l40s_b*: it has 25 nodes of 4x
# L40S and is usually almost empty, so these start in seconds instead of
# queueing behind thousands of pending jobs.
#
# Submit from the repo root:
#     mkdir -p logs && sbatch scripts/smoke_lora.sbatch
#SBATCH --job-name=petabite-smoke
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_interac
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -u -o pipefail

export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export HF_HOME=/project/6122149/purav/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Compute nodes here do have outbound network (verified: api.wandb.ai answers
# from kn089), so the run streams live. The config sets the mode; this only
# pins where an offline run would land if it is ever switched back.
export WANDB_DIR=${SLURM_SUBMIT_DIR}/wandb

cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUTDIR=outputs/smoke-${SLURM_JOB_ID}

echo "=================== environment ==================="
echo "host     : $(hostname)"
echo "job      : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "outdir   : ${OUTDIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0))"
python -c "import esm, transformers; print('esm', esm.__version__, '| transformers', transformers.__version__)"

# Settle the question of whether compute nodes can reach the internet. The
# answer decides whether W&B can ever run online from a job.
echo "=================== network probe ==================="
if curl -sS --max-time 8 -o /dev/null -w "api.wandb.ai reachable, HTTP %{http_code}\n" https://api.wandb.ai 2>/dev/null; then
  echo "  compute node HAS outbound network"
else
  echo "  compute node has NO outbound network (expected; W&B stays offline)"
fi

echo "=================== train ==================="
# Small but real: enough steps for the optimizer to move every adapter away
# from its zero initialisation, which is what verify_checkpoint.py checks.
python -m petabite.mlm_finetune \
  output_dir="${OUTDIR}" \
  data.tokenized_path="${TOKENIZED:-/project/6122149/purav/petabite-tokenized/smoke}" \
  data.max_train_samples=4096 \
  data.max_eval_samples=256 \
  data.max_length=512 \
  trainer.max_steps=30 \
  trainer.per_device_train_batch_size=16 \
  trainer.gradient_accumulation_steps=1 \
  trainer.eval_steps=30 \
  trainer.save_steps=1000 \
  trainer.logging_steps=5 \
  wandb.run_name="smoke-${SLURM_JOB_ID}"
TRAIN_RC=$?
echo "training exit code: ${TRAIN_RC}"
if [ "${TRAIN_RC}" -ne 0 ]; then
  echo "SMOKE FAILED: training did not complete"
  exit "${TRAIN_RC}"
fi

echo "=================== saved files ==================="
ls -la "${OUTDIR}"

echo "=================== verify checkpoint ==================="
python scripts/verify_checkpoint.py "${OUTDIR}"
VERIFY_RC=$?

echo "=================== wandb ==================="
# Online runs print their URL above and need nothing further. Offline runs land
# here and have to be uploaded from a login node.
if ls -d "${WANDB_DIR}"/offline-run-* >/dev/null 2>&1; then
  ls -d "${WANDB_DIR}"/offline-run-* | tail -3
  echo "offline run found, upload with:  wandb sync ${WANDB_DIR}/offline-run-*"
else
  echo "run streamed online; see the W&B URL above"
fi

if [ "${VERIFY_RC}" -eq 0 ]; then
  echo "SMOKE PASSED"
else
  echo "SMOKE FAILED: checkpoint verification failed"
fi
exit "${VERIFY_RC}"
