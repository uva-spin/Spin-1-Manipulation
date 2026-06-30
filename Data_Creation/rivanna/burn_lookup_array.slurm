#!/bin/bash
# Submit from repo root:
#   sbatch Data_Creation/rivanna/burn_lookup_array.slurm
# Override shard/output directories, polarization grid, or bin count:
#   P_STEP=0.005 NUM_BINS=500 OUTPUT_DIR=Data_Creation/burn_lookup_shards sbatch Data_Creation/rivanna/burn_lookup_array.slurm
# If NUM_BINS changes, update --array below to 0-(NUM_BINS-1).
# After the array completes, combine shards:
#   sbatch Data_Creation/rivanna/combine_burn_lookup.slurm

#SBATCH --job-name=burn-lookup
#SBATCH --array=0-499
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH -A spinquest_standard
#SBATCH --output=Data_Creation/slurm_logs/burn_lookup_%a.out
#SBATCH --error=Data_Creation/slurm_logs/burn_lookup_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-Data_Creation/burn_lookup_shards}"
P_STEP="${P_STEP:-0.005}"
MAX_STEPS="${MAX_STEPS:-1000}"
NUM_BINS="${NUM_BINS:-500}"

cd "${REPO_ROOT}"

mkdir -p Data_Creation/slurm_logs
mkdir -p "${OUTPUT_DIR}"

module purge
module load apptainer pytorch/2.9.0

echo "Running burn bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "P step: ${P_STEP}"
echo "Max steps: ${MAX_STEPS}"
echo "Num bins: ${NUM_BINS}"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" Data_Creation/burn_lookup_table.py \
  --burn-bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --p-step "${P_STEP}" \
  --max-steps "${MAX_STEPS}" \
  --num-bins "${NUM_BINS}" \
  --no-validate \
  --skip-if-exists \
  "$@"
