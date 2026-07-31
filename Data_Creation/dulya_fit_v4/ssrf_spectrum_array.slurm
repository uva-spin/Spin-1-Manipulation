#!/bin/bash
# Dulya-fit v2 ssRF full-spectrum array (burn-window bins only).
# Writes ssrf_spectrum_bin_XXXX.npz (or batched _partYYYY.npz) into data/ssrf_shards/.
#
# Hybrid sampling (see common.py / ssrf_bin_traj.py):
#   - every SPECTRUM_DENSE_BIN_STRIDE-th burn bin: full Cartesian P x gamma x steps
#   - remaining burn bins: P x SPECTRUM_MC_DRAWS_PER_BIN on-grid (gamma, steps) draws
#
# Defaults: R burn in (-3,3) -> bins 125-374; gamma 5.0-10.0/2.5; steps 20-100/20.
#
# Submit from THIS directory (dulya_fit_v2/):
#   sbatch ssrf_spectrum_array.slurm
#
# Full pipeline (spectrum mode + pre-flattened rows):
#   UNMANIP_ID=$(sbatch --parsable unmanipulated_bin_array.slurm)
#   SSRF_ID=$(sbatch --parsable ssrf_spectrum_array.slurm)
#   AFP_ID=$(sbatch --parsable afp_spectrum_array.slurm)
#   UNMANIP_FLAT=$(sbatch --parsable --dependency=afterok:${UNMANIP_ID} flatten_unmanip_spectrum_rows.slurm)
#   SSRF_FLAT=$(sbatch --parsable --dependency=afterok:${SSRF_ID} flatten_ssrf_spectrum_rows.slurm)
#   AFP_FLAT=$(sbatch --parsable --dependency=afterok:${AFP_ID} flatten_afp_spectrum_rows.slurm)
#   sbatch --dependency=afterok:${UNMANIP_FLAT}:${SSRF_FLAT}:${AFP_FLAT} combine_spectrum_rows.slurm
#   # or: combine_spectrum_train.slurm (shards + unmanip_dir -> sharded spectrum_train)

#SBATCH --job-name=dulya-v2-ssrf-spectrum
#SBATCH --array=125-374
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/ssrf_spectrum_%A_%a.out
#SBATCH --error=slurm_logs/ssrf_spectrum_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SHARD_DIR="${SHARD_DIR:-data/ssrf_shards}"
NUM_BINS="${NUM_BINS:-500}"
P_MIN="${P_MIN:--0.9}"
P_MAX="${P_MAX:-0.9}"
P_STEP="${P_STEP:-0.05}"
DT="${DT:-0.0015}"
GAMMA_MIN="${GAMMA_MIN:-5.0}"
GAMMA_MAX="${GAMMA_MAX:-10.0}"
GAMMA_STEP="${GAMMA_STEP:-2.5}"
STEPS_MIN="${STEPS_MIN:-20}"
STEPS_MAX="${STEPS_MAX:-100}"
STEPS_STEP="${STEPS_STEP:-20}"
RF_MODE="${RF_MODE:-physical_voigt}"
COMBO_BATCH_SIZE="${COMBO_BATCH_SIZE:-64}"
DENSE_BIN_STRIDE="${DENSE_BIN_STRIDE:-5}"
MC_DRAWS_PER_BIN="${MC_DRAWS_PER_BIN:-2}"
UNMANIP_FRACTION="${UNMANIP_FRACTION:-0.0}"

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
echo "Shard dir: ${SHARD_DIR}  combo_batch_size=${COMBO_BATCH_SIZE}"
echo "gamma=[${GAMMA_MIN},${GAMMA_MAX}]/${GAMMA_STEP}  steps=[${STEPS_MIN},${STEPS_MAX}]/${STEPS_STEP}"
echo "dense_stride=${DENSE_BIN_STRIDE}  mc_draws=${MC_DRAWS_PER_BIN}  unmanip_fraction=${UNMANIP_FRACTION}"

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
  --combo-batch-size "${COMBO_BATCH_SIZE}" \
  --dense-bin-stride "${DENSE_BIN_STRIDE}" \
  --mc-draws-per-bin "${MC_DRAWS_PER_BIN}" \
  --unmanip-fraction "${UNMANIP_FRACTION}" \
  --skip-if-exists \
  "$@"
