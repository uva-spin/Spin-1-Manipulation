#!/bin/bash
# Dulya-fit v5 ssRF full-spectrum array (burn-window bins, R in (-3, 3)).
# Each task writes ssrf_spectrum_bin_XXXX.npz with ps_full / iplus_full / iminus_full
# at every burn timestep (500-bin spectra).
#
# Submit from THIS directory:
#   sbatch ssrf_spectrum_array.slurm
#
# Then flatten + build per-bin train NPZs (includes calibrated P, Q):
#   SPEC_JOB=$(sbatch --parsable ssrf_spectrum_array.slurm)
#   sbatch --dependency=afterok:${SPEC_JOB} combine_spectrum_train_bins.slurm
#
# Throttle concurrency: --array=125-374%32
#
# Spectrum shards are RAM-heavy; raise --mem if tasks OOM.

#SBATCH --job-name=dulya-v5-ssrf-spec
#SBATCH --array=125-374
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/ssrf_spectrum_%A_%a.out
#SBATCH --error=slurm_logs/ssrf_spectrum_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARD_DIR="${SHARD_DIR:-data/ssrf_shards}"
NUM_BINS="${NUM_BINS:-500}"
P_MIN="${P_MIN:--0.9}"
P_MAX="${P_MAX:-0.9}"
P_STEP="${P_STEP:-0.05}"
DT="${DT:-0.00005}"
GAMMA_MIN="${GAMMA_MIN:-50.0}"
GAMMA_MAX="${GAMMA_MAX:-50.0}"
GAMMA_STEP="${GAMMA_STEP:-50.0}"
STEPS_MIN="${STEPS_MIN:-50}"
STEPS_MAX="${STEPS_MAX:-50}"
STEPS_STEP="${STEPS_STEP:-50}"
RF_MODE="${RF_MODE:-physical_voigt}"
RANDOM_SAMPLES="${RANDOM_SAMPLES:-0}"
DENSE_BIN_STRIDE="${DENSE_BIN_STRIDE:-5}"
MC_DRAWS_PER_BIN="${MC_DRAWS_PER_BIN:-2}"

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

EXTRA_SSRF_ARGS=()
if [[ "${MULTI_BURN:-0}" == "1" ]]; then
  EXTRA_SSRF_ARGS+=(--multi-burn)
fi

echo "Dulya v5 ssRF spectrum bin ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Pkg dir: ${PKG_DIR}"
echo "Shard dir: ${SHARD_DIR}"
echo "P: [${P_MIN}, ${P_MAX}] step=${P_STEP}"
echo "gamma_rf: [${GAMMA_MIN}, ${GAMMA_MAX}] step=${GAMMA_STEP}"
echo "burn_steps: [${STEPS_MIN}, ${STEPS_MAX}] step=${STEPS_STEP}"
echo "dt=${DT}  rf_mode=${RF_MODE}  random_samples=${RANDOM_SAMPLES}"
echo "dense_bin_stride=${DENSE_BIN_STRIDE}  mc_draws_per_bin=${MC_DRAWS_PER_BIN}"

apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  ssrf_bin_traj.py \
  --bin-idx "${SLURM_ARRAY_TASK_ID}" \
  --spectrum-mode \
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
  --random-samples "${RANDOM_SAMPLES}" \
  --dense-bin-stride "${DENSE_BIN_STRIDE}" \
  --mc-draws-per-bin "${MC_DRAWS_PER_BIN}" \
  --skip-if-exists \
  "${EXTRA_SSRF_ARGS[@]}" \
  "$@"
