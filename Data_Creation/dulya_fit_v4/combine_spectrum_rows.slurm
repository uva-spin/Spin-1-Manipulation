#!/bin/bash
# Concatenate pre-flattened spectrum row NPZs into spectrum_train output.
#
# Run after flatten_ssrf_spectrum_rows.slurm, flatten_afp_spectrum_rows.slurm,
# and flatten_unmanip_spectrum_rows.slurm:
#   sbatch --dependency=afterok:${SSRF_FLAT}:${AFP_FLAT}:${UNMANIP_FLAT} combine_spectrum_rows.slurm
#
# Submit from THIS directory (dulya_fit_v2/):
#   sbatch combine_spectrum_rows.slurm

#SBATCH --job-name=dulya-v2-combine-rows
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/combine_spectrum_rows_%j.out
#SBATCH --error=slurm_logs/combine_spectrum_rows_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SSRF_ROWS_DIR="${SSRF_ROWS_DIR:-data/ssrf_spectrum_rows}"
AFP_ROWS_DIR="${AFP_ROWS_DIR:-data/afp_spectrum_rows}"
UNMANIP_ROWS_DIR="${UNMANIP_ROWS_DIR:-data/unmanip_spectrum_rows}"
OUTPUT="${OUTPUT:-data/spectrum_train/spectrum_train.npz}"
NUM_BINS="${NUM_BINS:-500}"
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

n_ssrf="$(find "${SSRF_ROWS_DIR}" -maxdepth 1 -name 'ssrf_spectrum_rows_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
n_afp="$(find "${AFP_ROWS_DIR}" -maxdepth 1 -name 'afp_spectrum_rows_*.npz' 2>/dev/null | wc -l | tr -d ' ')"
n_unmanip="$([[ -f "${UNMANIP_ROWS_DIR}/unmanip_spectrum_rows.npz" ]] && echo 1 || echo 0)"
echo "Combining pre-flattened rows -> ${OUTPUT}"
echo "ssRF rows: ${SSRF_ROWS_DIR} (${n_ssrf} files)"
echo "AFP rows:  ${AFP_ROWS_DIR} (${n_afp} files)"
echo "Unmanip rows: ${UNMANIP_ROWS_DIR} (${n_unmanip} file)"
echo "Output shard size: ${SHARD_SIZE}"

COMBINE_ARGS=(
  --ssrf-rows-dir "${SSRF_ROWS_DIR}"
  --afp-rows-dir "${AFP_ROWS_DIR}"
  --unmanip-rows-dir "${UNMANIP_ROWS_DIR}"
  --output "${OUTPUT}"
  --num-bins "${NUM_BINS}"
  --strict
)
if [[ "${SHARD_SIZE}" != "0" ]]; then
  COMBINE_ARGS+=(--shard-size "${SHARD_SIZE}")
fi

apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  combine_spectrum_rows.py \
  "${COMBINE_ARGS[@]}" \
  "$@"
