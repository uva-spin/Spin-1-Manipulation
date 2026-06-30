#!/bin/bash
# Combine per-bin model checkpoints after the array job finishes.
#
# Submit with dependency on a completed array job:
#   ARRAY_JOB_ID=12345 sbatch ml/combine_single_bin_models.slurm
#
# Submit standalone once all per-bin checkpoints exist:
#   sbatch ml/combine_single_bin_models.slurm
#
# Submit array + combine in one step:
#   ARRAY_JOB_ID=$(sbatch --parsable ml/train_single_bin_array.slurm)
#   ARRAY_JOB_ID=${ARRAY_JOB_ID} sbatch --dependency=afterok:${ARRAY_JOB_ID} ml/combine_single_bin_models.slurm
#
# Override directories:
#   MODEL_DIR=TensorStudies/single_bin_results_v2 OUTPUT=TensorStudies/combined/combined_bin_model.pth sbatch ml/combine_single_bin_models.slurm

#SBATCH --job-name=bin-combine
#SBATCH --partition=gpu
#SBATCH -A spinquest_standard
#SBATCH --output=TensorStudies/slurm_logs/bin_combine.out
#SBATCH --error=TensorStudies/slurm_logs/bin_combine.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL_DIR="${MODEL_DIR:-TensorStudies/single_bin_results_v2}"
OUTPUT="${OUTPUT:-${MODEL_DIR}/combined_bin_model.pth}"
NUM_BINS="${NUM_BINS:-249}"
STRICT="${STRICT:-1}"

cd "${REPO_ROOT}"

mkdir -p TensorStudies/slurm_logs

module purge
module load apptainer pytorch/2.9.0

echo "Combining single-bin models on $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "Model dir: ${MODEL_DIR}"
echo "Output: ${OUTPUT}"
echo "Num bins: ${NUM_BINS}"
echo "Strict: ${STRICT}"

COMBINE_ARGS=(
  --model-dir "${MODEL_DIR}"
  --output "${OUTPUT}"
  --num-bins "${NUM_BINS}"
)
if [[ "${STRICT}" == "1" ]]; then
  COMBINE_ARGS+=(--strict)
fi

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" ml/combine_single_bin_models.py \
  "${COMBINE_ARGS[@]}" \
  "$@"
