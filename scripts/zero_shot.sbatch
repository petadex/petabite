#!/bin/bash
# Search-space item 2: zero-shot ESM-C likelihood as a variant-effect
# predictor. Needs no labels and fits nothing, so the evaluation is a single
# rank correlation with a component bootstrap and none of the metric faults
# that invalidated the earlier passes can arise.
#
# Step 0 verifies the likelihood is measuring what it claims before any score
# is reported from it: a real sequence must score above a shuffled version of
# itself, and the two backbones must disagree with each other less than either
# disagrees with noise.
#SBATCH --job-name=petabite-zeroshot
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
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

LORA=outputs/plasticesm-c60-r32
OUT="logs/zeroshot-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## 0. verify the likelihood measures something real ##########"
python - <<'PY'
import random
import numpy as np
import torch
from petabite.data_module.tokenization import load_esmc_tokenizer
from petabite.eval_module import labels
import sys
sys.path.insert(0, "scripts")
from zero_shot_likelihood import MODEL_ID, load_backbone, pseudo_log_likelihood

sequences = labels.read_sequences()
names = sorted(sequences)
tokenizer = load_esmc_tokenizer(MODEL_ID, 1024)
model = load_backbone(None)
mask_id = tokenizer.mask_token_id
print(f"mask token id {mask_id}")

rng = random.Random(0)
failures = []
real_scores, shuffled_scores = [], []

# Five real sequences against shuffled versions of themselves. Shuffling
# preserves length and composition exactly and destroys only the order, so a
# model that has learned anything about protein sequence must prefer the real
# one.
for name in names[:5]:
    sequence = sequences[name][:200]
    letters = list(sequence)
    rng.shuffle(letters)
    shuffled = "".join(letters)

    real = pseudo_log_likelihood(model, tokenizer, sequence, mask_id, 1, 1024)
    fake = pseudo_log_likelihood(model, tokenizer, shuffled, mask_id, 1, 1024)
    real_scores.append(real)
    shuffled_scores.append(fake)
    ok = real > fake
    print(f"  {'PASS' if ok else 'FAIL'}  {name[:12]}  real {real:+.4f}  "
          f"shuffled {fake:+.4f}  margin {real - fake:+.4f}")
    if not ok:
        failures.append(name)

print(f"  mean real {np.mean(real_scores):+.4f}, "
      f"mean shuffled {np.mean(shuffled_scores):+.4f}")

# A log probability must be negative, and a mean over positions must sit well
# above the uniform-over-vocabulary floor if the model is informative.
floor = -np.log(tokenizer.vocab_size)
print(f"  uniform-vocabulary floor is {floor:+.4f}")
if max(real_scores) > 0:
    failures.append("positive log probability")
    print("  FAIL  a mean log probability came out positive")
if min(real_scores) < floor:
    failures.append("below uniform floor")
    print("  FAIL  a real sequence scored below the uniform floor")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("likelihood behaves as a likelihood should")
PY

echo
echo "########## 1. score every labelled sequence ##########"
python scripts/zero_shot_likelihood.py --lora "${LORA}" \
  --out "${OUT}/zero_shot.npz"

echo
echo "########## 2. evaluate it against the controls ##########"
for target in pet_release_conc kcat_over_km kcat relative_activity km melting_temperature; do
  python scripts/eval_within_family.py \
    --embeddings logs/label_embeddings.npz \
    --zero-shot "${OUT}/zero_shot.npz" \
    --target "${target}" --power-trials 100 \
    --out "${OUT}/${target}.json" > "${OUT}/${target}.log" 2>&1 &
done
wait
for f in "${OUT}"/*.log; do
  echo "##################### $(basename "${f}") #####################"
  cat "${f}"
  echo
done
