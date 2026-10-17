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

# Colon-separated, NOT comma: sbatch --export uses commas to separate variables,
# so CORPUS="a,b" silently arrives as CORPUS="a" and the second path vanishes.
# That cost job 6334352, which trained one corpus while reporting success.
CORPUS=${CORPUS:-/project/6122149/purav/petabite-tokenized/c60-seed0}
CORPUS_LIST=${CORPUS//:/,}
N_CORPORA=$(echo "${CORPUS_LIST}" | awk -F, '{print NF}')
RUNNAME=${RUNNAME:-plasticesm-c60-r32}
# The job id goes in the path by default. Two runs sharing an output directory
# is not hypothetical: on 2026-10-16 jobs 6318367 and 6318656 both wrote
# outputs/plasticesm-c60-r32 for five hours, leaving intermediate checkpoints
# that belonged to whichever job happened to get there last.
OUTDIR=${OUTDIR:-outputs/${RUNNAME}-${SLURM_JOB_ID}}
RESUME=${RESUME:-null}

# Refuse to start alongside another training job, which catches the same
# mistake at submission instead of after five hours of duplicated GPU.
OTHERS=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null \
         | awk -v me="${SLURM_JOB_ID}" '$1 != me && $2 ~ /^plasticesm/ {print $1}' | tr '\n' ' ')
if [ -n "${OTHERS// /}" ] && [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
  echo "REFUSING TO START: another plasticesm job is queued or running: ${OTHERS}"
  echo "If deliberate, resubmit with --export=ALL,ALLOW_CONCURRENT=1 and a distinct OUTDIR."
  exit 2
fi

for c in ${CORPUS//:/ }; do
  if [ ! -d "$c" ]; then echo "MISSING CORPUS: $c"; exit 3; fi
done

echo "=================== environment ==================="
echo "host    : $(hostname)"
echo "job     : ${SLURM_JOB_ID} on ${SLURM_JOB_PARTITION}"
echo "corpus  : ${N_CORPORA} corpus/corpora -> ${CORPUS_LIST}"
echo "outdir  : ${OUTDIR}"
echo "resume  : ${RESUME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, esm, transformers; print('torch', torch.__version__, '| esm', esm.__version__, '| transformers', transformers.__version__)"
# CORPUS may be a comma-separated list of corpora, so report each manifest.
for c in ${CORPUS//:/ }; do
  echo "--- ${c} ---"
  test -f "${c}/manifest.json" && cat "${c}/manifest.json" || echo "  no manifest"
done

echo "=================== train ==================="
python -m petabite.mlm_finetune \
  output_dir="${OUTDIR}" \
  data.tokenized_path="[${CORPUS_LIST}]" \
  trainer.resume_from_checkpoint="${RESUME}" \
  wandb.run_name="${RUNNAME}-${SLURM_JOB_ID}"
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
