# Environment bootstrap. Source it, do not execute it:
#     source activate_petabite.sh
#
# This file is tracked deliberately. It used to be gitignored, which meant a
# fresh clone had no way to build the environment at all.

module load python/3.12.4
module load scipy-stack/2024a      # numpy, scipy, packaging
module load gcc/12.3
module load cuda
module load cudnn
module load arrow/21.0.0           # pyarrow, for reading the corpus parquet
# rdkit, rust, diamond and seqkit are not imported anywhere in src/. They stay
# because the data-prep notebooks shell out to seqkit and diamond.
module load rdkit
module load rust
module load diamond
module load seqkit

source ~/petabite/bin/activate

# Both of these are required, and leaving them to the caller was a reliable way
# to lose an hour. Without PYTHONNOUSERSITE a stray ~/.local package shadows
# the venv; without HF_HOME the model resolves to ~/.cache and is invisible to
# jobs, which then fail on a compute node running with HF_HUB_OFFLINE=1.
export PYTHONNOUSERSITE=1
export HF_HOME=/project/6122149/purav/hf_cache

# Never set PYTHONPATH on this cluster. Doing so drops the module-provided
# site-packages from sys.path and numpy, packaging and pyarrow all disappear.
# The package is importable because it is installed with `pip install -e .`.
