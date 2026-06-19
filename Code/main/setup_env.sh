#!/usr/bin/env bash
# Set up the Python environment on Grid5000.
# RUN THIS ON A SITE FRONTEND (e.g. fnancy, frennes): frontends have internet via the
# HTTP proxy; compute nodes usually do NOT. Do it once; the venv lives on your NFS home
# and is visible from the nodes you reserve.
set -euo pipefail

# Grid5000 frontends expose the web through a proxy; export it so pip can reach PyPI.
export http_proxy="http://proxy:3128"
export https_proxy="http://proxy:3128"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the dopactrnn_stage1 dir
VENV="${HERE}/venv"

module load python/3.11 2>/dev/null || true   # adjust/remove if your site differs
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade pip
pip install -r "${HERE}/requirements.txt"

echo "venv ready at ${VENV}"
echo "these nets are tiny -> CPU only, no GPU reservation needed."
