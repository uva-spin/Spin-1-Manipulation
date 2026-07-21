#!/bin/bash
# Organize AFP trajectory shards into one training NPZ per bin (after array job).
#
# Submit with dependency on a completed array job (from repo root):
#   ARRAY_JOB_ID=$(sbatch --parsable physics/lineshape/rate_eqs_test/afp_traj_array.slurm)
#   sbatch --dependency=afterok:${ARRAY_JOB_ID} \
#       physics/lineshape/rate_eqs_test/afp_traj_combine.slurm
#
# Or submit standalone once all shards exist:
#   sbatch physics/lineshape/rate_eqs_test/afp_traj_combine.slurm
#
# Writes afp_train/afp_train_bin_XXXX.npz — center and mirror amplitudes are
# routed to their respective bin files (no single combined NPZ).

#SBATCH --job-name=afp-traj-organize
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=physics/lineshape/rate_eqs_test/slurm_logs/afp_traj_organize_%j.out
#SBATCH --error=physics/lineshape/rate_eqs_test/slurm_logs/afp_traj_organize_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_DIR="${SHARD_DIR:-physics/lineshape/rate_eqs_test/afp_shards}"
OUTPUT_DIR="${OUTPUT_DIR:-physics/lineshape/rate_eqs_test/afp_train}"
NUM_BINS="${NUM_BINS:-500}"

cd "${REPO_ROOT}"

mkdir -p physics/lineshape/rate_eqs_test/slurm_logs
mkdir -p "${OUTPUT_DIR}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
export PYTHONUNBUFFERED=1

module purge
module load apptainer pytorch/2.9.0

if [[ -z "${CONTAINERDIR:-}" ]]; then
  echo "CONTAINERDIR is unset after loading pytorch module" >&2
  exit 1
fi

n_shards="$(find "${SHARD_DIR}" -maxdepth 1 -name 'afp_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
echo "Organizing AFP traj shards on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}"
echo "Shard dir: ${SHARD_DIR} (${n_shards} shards found)"
echo "Output dir: ${OUTPUT_DIR} (${NUM_BINS} per-bin training files)"
echo "Container: ${CONTAINERDIR}/pytorch-2.9.0.sif"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  physics/lineshape/rate_eqs_test/afp_bin_traj.py \
  --organize \
  --strict \
  --shard-dir "${SHARD_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  "$@"
