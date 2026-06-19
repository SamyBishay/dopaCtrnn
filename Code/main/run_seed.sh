#!/usr/bin/env bash
# Runs ONE seed of the experiment plus its untrained control (for H7).
# Invoked by submit_oar.sh as an OAR array job: the array index becomes the seed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HERE}/venv"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${HERE}"

# Seed: OAR array index if present, else first CLI arg, else 0.
SEED="${OAR_ARRAY_INDEX:-${1:-0}}"
OUTDIR="${OUTDIR:-results}"

echo "[$(hostname)] seed=${SEED} -> ${OUTDIR}"
python run_experiment.py --seed "${SEED}" --outdir "${OUTDIR}"
python run_experiment.py --seed "${SEED}" --outdir "${OUTDIR}" --untrained
echo "[$(hostname)] seed=${SEED} done"
