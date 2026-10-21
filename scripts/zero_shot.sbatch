#!/bin/bash
# Search-space item 2: zero-shot ESM-C likelihood as a variant-effect
# predictor. Needs no labels and fits nothing, so the evaluation is a single
# rank correlation with a component bootstrap and none of the metric faults
# that invalidated the earlier passes can arise.
#
# Twelve-hour partition because the exact calculation needs one forward pass
# per residue per sequence, about 400,000 per backbone. Job 6346834 was run
# on the three-hour partition and was heading for a timeout partway through
# the second backbone, which at the time would have discarded the first one
# as well; the script now saves after each backbone.
#SBATCH --job-name=petabite-zeroshot
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

LORA=outputs/plasticesm-c60-r32
OUT="logs/zeroshot-${SLURM_JOB_ID}"
# The gate below verifies this exact batch size, so the two stay in step.
MASK_BATCH=256
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## 0. verify the likelihood before trusting a number from it ##########"
MASK_BATCH="${MASK_BATCH}" python - <<'PY'
import os
import random
import numpy as np
import torch
import sys
sys.path.insert(0, "scripts")
from petabite.data_module.tokenization import load_esmc_tokenizer
from petabite.eval_module import labels
from zero_shot_likelihood import MODEL_ID, load_backbone, pseudo_log_likelihood

PRODUCTION_BATCH = int(os.environ["MASK_BATCH"])

sequences = labels.read_sequences()
names = sorted(sequences)
tokenizer = load_esmc_tokenizer(MODEL_ID, 1024)
mask_id = tokenizer.mask_token_id
model = load_backbone(None)
print(f"mask token id {mask_id}")
failures = []

def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)

# 1. A real sequence must beat a shuffled copy of itself. Shuffling preserves
#    length and composition exactly, so only sequence knowledge can separate
#    them. This is the test that the likelihood means anything at all.
rng = random.Random(0)
real_scores, shuffled_scores = [], []
for name in names[:5]:
    sequence = sequences[name][:200]
    letters = list(sequence)
    rng.shuffle(letters)
    real = pseudo_log_likelihood(model, tokenizer, sequence, mask_id, 1, 1024)
    fake = pseudo_log_likelihood(model, tokenizer, "".join(letters), mask_id,
                                 1, 1024)
    real_scores.append(real)
    shuffled_scores.append(fake)
    check(f"{name[:12]} beats its shuffle",
          real > fake, f"  real {real:+.4f}  shuffled {fake:+.4f}")

# 2. The vectorised log-probability gather replaced a hand-written loop, so it
#    has to agree with a one-at-a-time calculation. The comparison runs at the
#    batch size the scoring step will actually use, because checking one batch
#    size and scoring at another leaves the path that produces the numbers
#    unchecked, which is what happened in job 6348599.
#
#    Agreement is approximate rather than exact. Batching changes the order
#    the matrix multiplications reduce in, and the forward pass runs under
#    bf16 autocast, so a mean log probability moves slightly. Job 6352095
#    measured that movement over 40 length-stratified sequences: at most
#    0.0054, mean +0.0004 so unbiased, 0.28% of the spread of the scores, and
#    uncorrelated with length at +0.10. The tolerance below is set from that
#    measurement with headroom, rather than guessed.
#
#    What must not move is the ordering of one sequence against another, since
#    every number downstream is a rank correlation. That is the binding check,
#    and the same job put the rank correlation between the two batch sizes at
#    0.9989 over the full length range.
probes = [sequences[name][:60] for name in names[:4]]
one_at_a_time, in_batches = [], []
for short in probes:
    one_at_a_time.append(pseudo_log_likelihood(model, tokenizer, short,
                                               mask_id, 1, 1024, mask_batch=1))
    in_batches.append(pseudo_log_likelihood(model, tokenizer, short, mask_id,
                                            1, 1024,
                                            mask_batch=PRODUCTION_BATCH))
worst = max(abs(a - b) for a, b in zip(one_at_a_time, in_batches))
check(f"batch {PRODUCTION_BATCH} agrees with one-at-a-time",
      worst < 2e-2, f"  worst gap {worst:.6f}")
check(f"batch {PRODUCTION_BATCH} preserves the ordering",
      list(np.argsort(one_at_a_time)) == list(np.argsort(in_batches)),
      f"  {list(np.argsort(one_at_a_time))} vs {list(np.argsort(in_batches))}")

# 3. An independent hand calculation of the same quantity, written out
#    longhand here so it shares no code with the function under test.
probe, single = probes[0], one_at_a_time[0]
encoded = tokenizer([probe], truncation=True, max_length=1024,
                    return_tensors="pt")
ids = encoded["input_ids"][0].cuda()
attention = encoded["attention_mask"][0].cuda()
special = set(tokenizer.all_special_ids)
positions = [i for i, t in enumerate(ids.tolist()) if t not in special]
total = 0.0
with torch.no_grad():
    for position in positions:
        altered = ids.clone()
        altered[position] = mask_id
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=altered.unsqueeze(0),
                           attention_mask=attention.unsqueeze(0),
                           compute_sae=False).logits
        total += float(torch.log_softmax(logits[0, position].float(), dim=-1)[
            int(ids[position])])
hand = total / len(positions)
check("matches an independent longhand calculation",
      abs(hand - single) < 2e-3, f"  longhand {hand:+.6f} vs {single:+.6f}")

# 4. Sanity bounds: a mean log probability is negative and, if the model is
#    informative, sits well above the uniform-over-vocabulary floor.
floor = -np.log(tokenizer.vocab_size)
check("log probabilities are negative", max(real_scores) < 0)
check("real sequences beat the uniform floor", min(real_scores) > floor,
      f"  min {min(real_scores):+.4f}  floor {floor:+.4f}")
print(f"  mean real {np.mean(real_scores):+.4f}, "
      f"mean shuffled {np.mean(shuffled_scores):+.4f}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("likelihood verified; proceeding to score the label set")
PY

echo
echo "########## 1. score every labelled sequence ##########"
python scripts/zero_shot_likelihood.py --lora "${LORA}" \
  --mask-batch "${MASK_BATCH}" --out "${OUT}/zero_shot.npz"

echo
echo "########## 2. evaluate it against the controls ##########"
pids=()
for target in pet_release_conc kcat_over_km kcat relative_activity km melting_temperature; do
  python scripts/eval_within_family.py \
    --embeddings logs/label_embeddings.npz \
    --zero-shot "${OUT}/zero_shot.npz" \
    --target "${target}" --power-trials 100 \
    --out "${OUT}/${target}.json" > "${OUT}/${target}.log" 2>&1 &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
for f in "${OUT}"/*.log; do
  echo "##################### $(basename "${f}") #####################"
  grep -vE "Warning|score = spearmanr" "${f}" || true
  echo
done
exit "${failed}"
