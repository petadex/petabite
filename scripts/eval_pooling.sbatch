#!/bin/bash
# Search-space item 7, the half the first pass could not run.
#
# Learned pooling needs per-token hidden states and only the mean-pooled
# vectors had been cached, so the first pass tested a small head on the mean
# and recorded learned pooling as untested. The storage estimate that made it
# look out of reach was wrong: the labelled set is about 393,000 residues, so
# one backbone's states are 750MB in float16 and both fit in 1.5GB.
#
# So this caches them and tests the pooler properly. Gated attention over
# tokens, trained end-to-end, against the mean of the same tokens, against
# length and composition.
#
# Two things the first pass got wrong are fixed here. The baseline is
# recomputed from the same tokens instead of read from the cached mean-pooled
# file, which had averaged over special tokens too. And early stopping holds
# out whole components rather than random rows, so the pooler does not get the
# within-component leak the small head was given.
#SBATCH --job-name=petabite-pooling
#SBATCH --account=aip-qmorris
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out
set -e -u -o pipefail
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export HF_HOME=/project/6122149/purav/hf_cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "${SLURM_SUBMIT_DIR}"
source activate_petabite.sh >/dev/null 2>&1

OUT="logs/pooling-${SLURM_JOB_ID}"
STATES="${OUT}/token_states.npz"
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "########## 1. cache the per-token states ##########"
# Job 6354123 already computed these and they are identical input to
# identical code, so they are reused when present rather than recomputed.
CACHED=logs/pooling-6354123/token_states.npz
if [ -f "${CACHED}" ]; then
  echo "reusing ${CACHED} from job 6354123"
  STATES="${CACHED}"
else
  python scripts/embed_tokens.py --lora outputs/plasticesm-c60-r32 \
    --out "${STATES}"
fi

echo
echo "########## 2. verify the pooler before trusting a number from it #######"
STATES="${STATES}" python - <<'PY'
import os
import numpy as np
import torch
from scipy.stats import spearmanr

import sys
sys.path.insert(0, "scripts")
from eval_pooling import (GatedAttentionPooler, mean_pool, pad_batch,
                          read_states, token_scaler)
from petabite.eval_module import labels

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    if not ok:
        failures.append(name)


states_path = os.environ["STATES"]
cached = np.load("logs/label_embeddings.npz", allow_pickle=False)
index = {n: i for i, n in enumerate(cached["names"])}
names = sorted(index)[:40]
blocks = read_states(states_path, "stock", index, names)

# 1. The packed states must be the states of the right sequences. Their mean
#    has to track the independently cached mean-pooled vector almost exactly;
#    the only difference is that the cached one averaged special tokens in.
rows = np.array([index[n] for n in names])
recomputed = mean_pool(blocks)
reference = np.asarray(cached["stock"], dtype=np.float32)[rows]
per_sequence = [float(np.corrcoef(a, b)[0, 1])
                for a, b in zip(recomputed, reference)]
check("the packed states recover the cached mean-pooled vectors",
      min(per_sequence) > 0.99,
      f"  worst correlation {min(per_sequence):.5f}")

sequences = labels.read_sequences()
check("each sequence has one row per residue",
      all(len(block) == len(sequences[name])
          for block, name in zip(blocks, names)),
      f"  example {len(blocks[0])} rows for {len(sequences[names[0]])} residues")

# 2. Padding must not change a prediction, which is the one way a masked
#    attention pooler quietly goes wrong.
torch.manual_seed(0)
model = GatedAttentionPooler(blocks[0].shape[1]).eval()
scaler = token_scaler(blocks, np.arange(len(blocks)))
alone, alone_mask = pad_batch(blocks, np.array([0]), scaler, "cpu")
with torch.no_grad():
    unpadded = float(model(alone, alone_mask)[0])
together, together_mask = pad_batch(
    blocks, np.array([0, int(np.argmax([len(b) for b in blocks]))]), scaler,
    "cpu")
with torch.no_grad():
    padded = float(model(together, together_mask)[0])
check("padding does not change a prediction",
      abs(unpadded - padded) < 1e-4,
      f"  alone {unpadded:+.6f} vs padded to "
      f"{together.shape[1]} tokens {padded:+.6f}")

# 3. The pooler must beat the mean where attention is the right answer, or a
#    loss against the mean on real data would say nothing about pooling. The
#    task has to be one a mean genuinely cannot solve. A first version put the
#    signal in an otherwise quiet dimension, where averaging it over 45 tokens
#    still ranked the sequences at 0.762, so the comparison was empty.
#
#    Here the signal dimension is loud everywhere: one token carries the
#    target and every other token carries noise of ten times its scale, so the
#    mean is dominated by the distractors. A second dimension marks the
#    informative token, which is the cue attention can key on and a mean
#    cannot use at all.
generator = np.random.default_rng(0)
width, count = 16, 200
y = generator.normal(size=count)
synthetic = []
for position in range(count):
    length = int(generator.integers(30, 60))
    block = generator.normal(scale=10.0, size=(length, width)).astype(np.float32)
    block[:, 1] = 0.0
    carrier = int(generator.integers(0, length))
    block[carrier, 0] = y[position] * 10.0   # the informative token
    block[carrier, 1] = 10.0                 # the cue marking it
    synthetic.append(block)

split = 150
means = mean_pool(synthetic)
mean_rho = max(abs(float(spearmanr(means[split:, column],
                                   y[split:]).statistic))
               for column in range(width))

torch.manual_seed(0)
model = GatedAttentionPooler(width, hidden=16)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
scaler = token_scaler(synthetic, np.arange(split))
tokens, mask = pad_batch(synthetic, np.arange(split), scaler, "cpu")
targets = torch.from_numpy(y[:split].astype(np.float32))
for _ in range(3000):
    optimiser.zero_grad()
    loss = torch.nn.functional.mse_loss(model(tokens, mask), targets)
    loss.backward()
    optimiser.step()

model.eval()
held_tokens, held_mask = pad_batch(synthetic, np.arange(split, count), scaler,
                                   "cpu")
with torch.no_grad():
    predicted = model(held_tokens, held_mask).numpy()
pooler_rho = abs(float(spearmanr(predicted, y[split:]).statistic))

check("a mean cannot solve the synthetic task",
      mean_rho < 0.4, f"  best mean-pooled dimension {mean_rho:.3f}")
check("the pooler can solve it",
      pooler_rho > 0.7 and pooler_rho > mean_rho + 0.2,
      f"  pooler {pooler_rho:.3f} against mean {mean_rho:.3f}")

print()
if failures:
    print(f"VERIFICATION FAILED: {failures}")
    raise SystemExit(1)
print("the pooler reads the right tokens, ignores padding, and can beat a mean")
PY

echo
echo "########## 3. pooler against the mean, on the best-powered targets #####"
python scripts/eval_pooling.py \
  --states "${STATES}" --embeddings logs/label_embeddings.npz \
  --targets kcat,kcat_over_km --seeds 3 \
  --out "${OUT}/pooling.json" 2>&1 \
  | grep -vE "Warning|score = spearmanr|warnings.warn" || true
