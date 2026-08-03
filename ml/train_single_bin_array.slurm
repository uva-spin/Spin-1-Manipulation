#!/bin/bash
# Train one MLP per spectral bin (single_bin.py) on GPU array.
#
# Submit from the directory that contains combined_train_all/ and single_bin.py:
#   sbatch train_single_bin_array.slurm
#
# From the repo instead:
#   DATA_DIR=Data_Creation/dulya_fit_v5/data/combined_train_all \
#   TRAIN_SCRIPT=ml/single_bin.py \
#     sbatch ml/train_single_bin_array.slurm
#
# Override:
#   OUTPUT_DIR=single_bin_models \
#   sbatch --array=125-374%32 train_single_bin_array.slurm

#SBATCH --job-name=bin-train
#SBATCH --array=0-499
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A spinquest_standard
#SBATCH --output=slurm_logs/bin_%A_%a.out
#SBATCH --error=slurm_logs/bin_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/combined_train_all}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/single_bin_models}"
FEATURE_CLIP_Z="${FEATURE_CLIP_Z:-0.0}"
TRAIN_POLARIZATION_FRACTION="${TRAIN_POLARIZATION_FRACTION:-0.8}"

if [[ -z "${TRAIN_SCRIPT:-}" ]]; then
  if [[ -f "${WORK_DIR}/single_bin.py" ]]; then
    TRAIN_SCRIPT="${WORK_DIR}/single_bin.py"
  elif [[ -f "${WORK_DIR}/ml/single_bin.py" ]]; then
    TRAIN_SCRIPT="${WORK_DIR}/ml/single_bin.py"
  else
    echo "No single_bin.py found under ${WORK_DIR}" >&2
    exit 1
  fi
fi

cd "${WORK_DIR}"

mkdir -p slurm_logs
mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing data dir: ${DATA_DIR}" >&2
  exit 1
fi

TRAIN_NPZ="${DATA_DIR}/train_bin_$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")".npz
if [[ ! -f "${TRAIN_NPZ}" ]]; then
  echo "Missing ${TRAIN_NPZ}; skipping empty bin ${SLURM_ARRAY_TASK_ID}" >&2
  exit 0
fi

module purge
module load apptainer pytorch/2.9.0

if [[ -z "${CONTAINERDIR:-}" ]]; then
  echo "CONTAINERDIR is unset after loading pytorch module" >&2
  exit 1
fi

echo "Running bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Work dir: ${WORK_DIR}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Features: p0 + ps -> P, Q"
echo "Feature clip z: ${FEATURE_CLIP_Z}"

apptainer run --nv \
  --bind "${WORK_DIR}:${WORK_DIR}" \
  --pwd "${WORK_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  "${TRAIN_SCRIPT}" \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --feature-clip-z "${FEATURE_CLIP_Z}" \
  --train-polarization-fraction "${TRAIN_POLARIZATION_FRACTION}" \
  "$@"
