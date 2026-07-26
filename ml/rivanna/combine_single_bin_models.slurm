#!/bin/bash
#SBATCH --job-name=bin-combine
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --output=Train_Out/slurm_logs/bin_combine_%j.out
#SBATCH --error=Train_Out/slurm_logs/bin_combine_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL_DIR="${MODEL_DIR:-Train_Out/binning-model-v1}"
OUTPUT="${OUTPUT:-${MODEL_DIR}/combined_bin_model.pth}"
NUM_BINS="${NUM_BINS:-500}"
STRICT="${STRICT:-1}"

cd "${REPO_ROOT}"

mkdir -p Train_Out/slurm_logs

module purge
module load apptainer pytorch/2.9.0

if [[ -z "${CONTAINERDIR:-}" ]]; then
  echo "CONTAINERDIR is unset after loading pytorch module" >&2
  exit 1
fi

echo "Combining single-bin models on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}"
echo "Model dir: ${MODEL_DIR}"
echo "Output: ${OUTPUT}"
echo "Num bins: ${NUM_BINS}"

COMBINE_ARGS=(
  --model-dir "${MODEL_DIR}"
  --output "${OUTPUT}"
  --num-bins "${NUM_BINS}"
)
if [[ "${STRICT}" == "1" ]]; then
  COMBINE_ARGS+=(--strict)
fi

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" ml/rivanna/combine_single_bin_models.py \
  "${COMBINE_ARGS[@]}" \
  "$@"
