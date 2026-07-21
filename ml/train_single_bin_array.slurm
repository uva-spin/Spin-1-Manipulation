#!/bin/bash
# Submit from repo root:
#   sbatch ml/train_single_bin_array.slurm
#
# Expects per-bin NPZs from the organize/combine pipeline, e.g.:
#   physics/lineshape/rate_eqs_test/combined_train/train_bin_XXXX.npz
#
# Override:
#   DATA_DIR=physics/lineshape/rate_eqs_test/combined_train \
#   OUTPUT_DIR=TensorStudies/single_bin_results_v2 \
#   FEATURE_SET=ps_p0 \
#     sbatch --array=0-499 ml/train_single_bin_array.slurm

#SBATCH --job-name=bin-train
#SBATCH --array=0-499
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A spinquest_standard
#SBATCH --output=TensorStudies/slurm_logs/bin_%A_%a.out
#SBATCH --error=TensorStudies/slurm_logs/bin_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-physics/lineshape/rate_eqs_test/combined_train_all}"
OUTPUT_DIR="${OUTPUT_DIR:-TensorStudies/single_bin_results_v2}"
FEATURE_SET="${FEATURE_SET:-ps_p0}"
FEATURE_CLIP_Z="${FEATURE_CLIP_Z:-0.0}"

cd "${REPO_ROOT}"

mkdir -p TensorStudies/slurm_logs
mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing data dir: ${DATA_DIR}" >&2
  exit 1
fi

module purge
module load apptainer pytorch/2.9.0

if [[ -z "${CONTAINERDIR:-}" ]]; then
  echo "CONTAINERDIR is unset after loading pytorch module" >&2
  exit 1
fi

echo "Running bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Feature set: ${FEATURE_SET}"
echo "Feature clip z: ${FEATURE_CLIP_Z}"

apptainer run --nv "${CONTAINERDIR}/pytorch-2.9.0.sif" ml/single_bin.py \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --feature-set "${FEATURE_SET}" \
  --feature-clip-z "${FEATURE_CLIP_Z}" \
  "$@"
