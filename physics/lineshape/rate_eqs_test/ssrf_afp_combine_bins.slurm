#!/bin/bash
# Merge ssRF + AFP trajectory shards into one combined training NPZ per bin.
#
# Run after both array jobs finish (from repo root):
#   SSRF_ID=$(sbatch --parsable physics/lineshape/rate_eqs_test/ssrf_traj_array.slurm)
#   AFP_ID=$(sbatch --parsable physics/lineshape/rate_eqs_test/afp_traj_array.slurm)
#   sbatch --dependency=afterok:${SSRF_ID}:${AFP_ID} \
#       physics/lineshape/rate_eqs_test/ssrf_afp_combine_bins.slurm
#
# Or submit standalone once both shard dirs are complete:
#   sbatch physics/lineshape/rate_eqs_test/ssrf_afp_combine_bins.slurm
#
# Writes combined_train/train_bin_XXXX.npz with fields:
#   ps, iplus, iminus, amp, p0, step, center_bin, is_mirror, source
#   source: 0=ssrf, 1=afp
#
# Override paths with env vars:
#   SSRF_SHARD_DIR=... AFP_SHARD_DIR=... OUTPUT_DIR=... NUM_BINS=500 \
#     sbatch physics/lineshape/rate_eqs_test/ssrf_afp_combine_bins.slurm

#SBATCH --job-name=ssrf-afp-combine
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=physics/lineshape/rate_eqs_test/slurm_logs/ssrf_afp_combine_%j.out
#SBATCH --error=physics/lineshape/rate_eqs_test/slurm_logs/ssrf_afp_combine_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SSRF_SHARD_DIR="${SSRF_SHARD_DIR:-physics/lineshape/rate_eqs_test/ssrf_shards}"
AFP_SHARD_DIR="${AFP_SHARD_DIR:-physics/lineshape/rate_eqs_test/afp_shards}"
OUTPUT_DIR="${OUTPUT_DIR:-physics/lineshape/rate_eqs_test/combined_train}"
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

n_ssrf="$(find "${SSRF_SHARD_DIR}" -maxdepth 1 -name 'ssrf_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
n_afp="$(find "${AFP_SHARD_DIR}" -maxdepth 1 -name 'afp_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"

echo "Combining ssRF + AFP shards into per-bin training files on $(hostname)"
echo "Job: ${SLURM_JOB_ID:-?}"
echo "ssRF shards: ${SSRF_SHARD_DIR} (${n_ssrf} files)"
echo "AFP shards:  ${AFP_SHARD_DIR} (${n_afp} files)"
echo "Output dir:  ${OUTPUT_DIR} (${NUM_BINS} train_bin_XXXX.npz)"
echo "Container: ${CONTAINERDIR}/pytorch-2.9.0.sif"

apptainer run "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  physics/lineshape/rate_eqs_test/combine_ssrf_afp_train.py \
  --ssrf-shard-dir "${SSRF_SHARD_DIR}" \
  --afp-shard-dir "${AFP_SHARD_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  --strict \
  "$@"
