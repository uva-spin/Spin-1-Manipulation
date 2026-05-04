#!/bin/bash
# Submit with:
#   sbatch TensorStudies/train_single_bin_array.slurm
# Override the array range if needed:
#   sbatch --array=0-499 TensorStudies/train_single_bin_array.slurm

#SBATCH --job-name=bin-train
#SBATCH --array=0-499
#SBATCH --output=TensorStudies/slurm_logs/bin_%A_%a.out
#SBATCH --error=TensorStudies/slurm_logs/bin_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-TensorStudies/single_bin_results}"
NUM_BINS="${NUM_BINS:-500}"

cd "${REPO_ROOT}"

mkdir -p TensorStudies/slurm_logs
mkdir -p "${OUTPUT_DIR}"

if [[ -n "${VENV_PATH:-}" ]]; then
  # Optional virtualenv activation for clusters that do not preload Python deps.
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]] && command -v conda >/dev/null 2>&1; then
  # Optional conda activation when the cluster uses conda-based environments.
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

echo "Running bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" TensorStudies/train_single_bin_model.py \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  "$@"
