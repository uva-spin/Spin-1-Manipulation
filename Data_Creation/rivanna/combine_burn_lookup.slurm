#!/bin/bash
# Combine per-bin burn lookup shards after the array job finishes.
#
# Submit with dependency on a completed array job:
#   ARRAY_JOB_ID=12345 sbatch Data_Creation/rivanna/combine_burn_lookup.slurm
# Or submit standalone once all shards exist:
#   sbatch Data_Creation/rivanna/combine_burn_lookup.slurm
#
# Submit array + combine in one step:
#   ARRAY_JOB_ID=$(sbatch --parsable Data_Creation/rivanna/burn_lookup_array.slurm)
#   ARRAY_JOB_ID=${ARRAY_JOB_ID} sbatch --dependency=afterok:${ARRAY_JOB_ID} Data_Creation/rivanna/combine_burn_lookup.slurm

#SBATCH --job-name=burn-combine
#SBATCH --partition=gpu
#SBATCH -A spinquest_standard
#SBATCH --output=Data_Creation/slurm_logs/burn_lookup_combine.out
#SBATCH --error=Data_Creation/slurm_logs/burn_lookup_combine.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_DIR="${SHARD_DIR:-Data_Creation/burn_lookup_shards}"
OUTPUT="${OUTPUT:-Data_Creation/burn_lookup_table.pkl}"
NUM_BINS="${NUM_BINS:-500}"

cd "${REPO_ROOT}"

mkdir -p Data_Creation/slurm_logs

module purge
module load apptainer pytorch/2.9.0

echo "Combining burn lookup shards on $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "Shard dir: ${SHARD_DIR}"
echo "Output: ${OUTPUT}"
echo "Num bins: ${NUM_BINS}"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" combine_burn_lookup_table.py \
  --shard-dir "${SHARD_DIR}" \
  --output "${OUTPUT}" \
  --num-bins "${NUM_BINS}" \
  --strict \
  "$@"
