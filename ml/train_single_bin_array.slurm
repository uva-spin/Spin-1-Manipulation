#!/bin/bash
# Submit from repo root:
#   sbatch ml/train_single_bin_array.slurm
# Override lookup table or output directory:
#   LOOKUP_TABLE=Data_Creation/burn_lookup_table.pkl OUTPUT_DIR=TensorStudies/single_bin_results_v2 sbatch ml/train_single_bin_array.slurm

#SBATCH --job-name=bin-train
#SBATCH --array=1-247
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A spinquest_standard
#SBATCH --output=TensorStudies/slurm_logs/bin_%a.out
#SBATCH --error=TensorStudies/slurm_logs/bin_%a.err
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
LOOKUP_TABLE="${LOOKUP_TABLE:-Data_Creation/burn_lookup_table.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-TensorStudies/single_bin_results_v2}"
FEATURE_SET="${FEATURE_SET:-burn_context}"
FEATURE_CLIP_Z="${FEATURE_CLIP_Z:-8.0}"

cd "${REPO_ROOT}"

mkdir -p TensorStudies/slurm_logs
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${LOOKUP_TABLE}" ]]; then
  echo "Missing lookup table: ${LOOKUP_TABLE}" >&2
  exit 1
fi

module purge
module load apptainer pytorch/2.9.0

echo "Running bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "Lookup table: ${LOOKUP_TABLE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Feature set: ${FEATURE_SET}"
echo "Feature clip z: ${FEATURE_CLIP_Z}"

apptainer run --nv "${CONTAINERDIR}/pytorch-2.9.0.sif" ml/single_bin.py \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --lookup-table "${LOOKUP_TABLE}" \
  --output-dir "${OUTPUT_DIR}" \
  --feature-set "${FEATURE_SET}" \
  --feature-clip-z "${FEATURE_CLIP_Z}" \
  "$@"
