#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec python3 run.py "$@"
