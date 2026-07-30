#!/bin/bash
# Merge ssRF + AFP shards + unmanipulated bins into spectrum_train.npz
# (one full-spectrum row per sample: ps/iplus/iminus shape (n_samples, 500)).
#
# Expects ssrf_bin_*.npz and afp_bin_*.npz under data/ssrf_shards and data/afp_shards.
#
# Submit from THIS directory (dulya_fit_v2/):
#   sbatch combine_spectrum_train.slurm

#SBATCH --job-name=dulya-v2-combine-spectrum
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/combine_spectrum_train_%j.out
#SBATCH --error=slurm_logs/combine_spectrum_train_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SSRF_SHARD_DIR="${SSRF_SHARD_DIR:-data/ssrf_shards}"
AFP_SHARD_DIR="${AFP_SHARD_DIR:-data/afp_shards}"
UNMANIP_DIR="${UNMANIP_DIR:-data/unmanip_train}"
OUTPUT="${OUTPUT:-data/spectrum_train/spectrum_train.npz}"
NUM_BINS="${NUM_BINS:-500}"
AFP_STEP_SUBSAMPLE="${AFP_STEP_SUBSAMPLE:-50}"
SHARD_SIZE="${SHARD_SIZE:-200000}"

cd "${PKG_DIR}"

mkdir -p slurm_logs
mkdir -p "$(dirname "${OUTPUT}")"

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
n_unmanip="$(find "${UNMANIP_DIR}" -maxdepth 1 -name 'unmanip_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"

echo "Combining Dulya v2 shards -> ${OUTPUT}"
echo "ssRF: ${SSRF_SHARD_DIR} (${n_ssrf} ssrf_bin_*.npz)"
echo "AFP:  ${AFP_SHARD_DIR} (${n_afp} afp_bin_*.npz)"
echo "Unmanip: ${UNMANIP_DIR} (${n_unmanip})"
echo "AFP step subsample: ${AFP_STEP_SUBSAMPLE}"
echo "Output shard size: ${SHARD_SIZE}"

COMBINE_ARGS=(
  --ssrf-shard-dir "${SSRF_SHARD_DIR}"
  --afp-shard-dir "${AFP_SHARD_DIR}"
  --unmanip-dir "${UNMANIP_DIR}"
  --output "${OUTPUT}"
  --num-bins "${NUM_BINS}"
  --afp-step-subsample "${AFP_STEP_SUBSAMPLE}"
  --strict
)
if [[ "${SHARD_SIZE}" != "0" ]]; then
  COMBINE_ARGS+=(--shard-size "${SHARD_SIZE}")
fi

apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  combine_spectrum_train.py \
  "${COMBINE_ARGS[@]}" \
  "$@"
