#!/bin/bash
# Dulya-fit v2 ssRF full-spectrum array (500 bins, ps_full at each timestep).
# Writes ssrf_spectrum_bin_XXXX.npz into data/ssrf_shards/ (same dir as traj shards).
#
# Submit from THIS directory (dulya_fit_v2/):
#   sbatch ssrf_spectrum_array.slurm
#
# Then merge for SpectrumSplitNet training:
#   sbatch combine_spectrum_train.slurm

#SBATCH --job-name=dulya-v2-ssrf-spectrum
#SBATCH --array=0-499
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/ssrf_spectrum_%A_%a.out
#SBATCH --error=slurm_logs/ssrf_spectrum_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARD_DIR="${SHARD_DIR:-data/ssrf_shards}"
NUM_BINS="${NUM_BINS:-500}"
P_MIN="${P_MIN:--0.9}"
P_MAX="${P_MAX:-0.9}"
P_STEP="${P_STEP:-0.05}"
DT="${DT:-0.0015}"
GAMMA_MIN="${GAMMA_MIN:-1.0}"
GAMMA_MAX="${GAMMA_MAX:-5.0}"
GAMMA_STEP="${GAMMA_STEP:-0.25}"
STEPS_MIN="${STEPS_MIN:-200}"
STEPS_MAX="${STEPS_MAX:-20000}"
STEPS_STEP="${STEPS_STEP:-200}"
RF_MODE="${RF_MODE:-physical_voigt}"

cd "${PKG_DIR}"

mkdir -p slurm_logs
mkdir -p "${SHARD_DIR}"

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
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is unset; submit as an array job" >&2
  exit 1
fi

echo "Dulya v2 ssRF spectrum bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Shard dir: ${SHARD_DIR}"

apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  ssrf_bin_traj.py \
  --spectrum-mode \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --shard-dir "${SHARD_DIR}" \
  --num-bins "${NUM_BINS}" \
  --p-min "${P_MIN}" \
  --p-max "${P_MAX}" \
  --p-step "${P_STEP}" \
  --dt "${DT}" \
  --gamma-min "${GAMMA_MIN}" \
  --gamma-max "${GAMMA_MAX}" \
  --gamma-step "${GAMMA_STEP}" \
  --steps-min "${STEPS_MIN}" \
  --steps-max "${STEPS_MAX}" \
  --steps-step "${STEPS_STEP}" \
  --rf-mode "${RF_MODE}" \
  --skip-if-exists \
  "$@"
