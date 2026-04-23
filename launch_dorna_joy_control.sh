#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Could not find conda activation script at: $CONDA_SH" >&2
  echo "Set CONDA_ROOT if your Miniconda/Anaconda install lives elsewhere." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate dorna-bridge

cd "$ROOT_DIR"
exec python dorna_joy_control.py "$@"
