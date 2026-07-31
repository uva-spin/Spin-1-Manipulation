#!/bin/bash
# Merge ssRF shards + AFP shards + unmanipulated bins into
# data/combined_train_all/train_bin_XXXX.npz (source 0/1/2).
#
# Submit from THIS directory (dulya_fit_v2/) once all three inputs exist:
#   sbatch combine_all_train.slurm

#SBATCH --job-name=dulya-v2-combine-all
#SBATCH --partition=standard
#SBATCH -A spinquest_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/combine_all_train_%j.out
#SBATCH --error=slurm_logs/combine_all_train_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PKG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SSRF_SHARD_DIR="${SSRF_SHARD_DIR:-data/ssrf_shards}"
AFP_SHARD_DIR="${AFP_SHARD_DIR:-data/afp_shards}"
UNMANIP_DIR="${UNMANIP_DIR:-data/unmanip_train}"
OUTPUT_DIR="${OUTPUT_DIR:-data/combined_train_all}"
NUM_BINS="${NUM_BINS:-500}"

cd "${PKG_DIR}"

mkdir -p slurm_logs
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
n_unmanip="$(find "${UNMANIP_DIR}" -maxdepth 1 -name 'unmanip_bin_*.npz' 2>/dev/null | wc -l | tr -d ' ')"

echo "Combining Dulya v2 ssRF + AFP + unmanip -> ${OUTPUT_DIR}"
echo "ssRF shards: ${SSRF_SHARD_DIR} (${n_ssrf})"
echo "AFP shards:  ${AFP_SHARD_DIR} (${n_afp})"
echo "Unmanip:     ${UNMANIP_DIR} (${n_unmanip})"

apptainer run --bind "${PKG_DIR}:${PKG_DIR}" --pwd "${PKG_DIR}" \
  "${CONTAINERDIR}/pytorch-2.9.0.sif" \
  combine_all_train.py \
  --ssrf-shard-dir "${SSRF_SHARD_DIR}" \
  --afp-shard-dir "${AFP_SHARD_DIR}" \
  --unmanip-dir "${UNMANIP_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-bins "${NUM_BINS}" \
  --strict \
  "$@"
