#!/usr/bin/env bash
# Submit N seeds to Grid5000 as an OAR array job (each array task = one seed).
# Grid5000 uses OAR (not SLURM). Run this from a site frontend AFTER setup_env.sh.
#
#   bash g5k/submit_oar.sh            # 5 seeds, 2h walltime (defaults)
#   NSEEDS=10 WALLTIME=03:00:00 bash g5k/submit_oar.sh
#
# After the jobs finish, aggregate across seeds:
#   source venv/bin/activate && python aggregate.py --root results --out results/summary
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NSEEDS="${NSEEDS:-5}"
WALLTIME="${WALLTIME:-02:00:00}"
CORES="${CORES:-1}"            # 1 core/seed is plenty (single tiny CTRNN, CPU)

# --array N            : run the script N times; task i exposes $OAR_ARRAY_INDEX = i
# -l core=CORES,walltime=...  : per-task resources
# stdout/stderr land in OAR.<jobid>.<arrayidx>.std{out,err} in $HERE
oarsub \
  --array "${NSEEDS}" \
  -l "core=${CORES},walltime=${WALLTIME}" \
  -d "${HERE}" \
  -O "OAR.%jobid%.%array_index%.stdout" \
  -E "OAR.%jobid%.%array_index%.stderr" \
  "bash ${HERE}/g5k/run_seed.sh"

echo "submitted ${NSEEDS} seeds. check status with:  oarstat -u \$USER"
echo "when done:  python aggregate.py --root results --out results/summary"
