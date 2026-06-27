#!/bin/bash
set -euo pipefail

VENV_PATH="/home/hep/jcc525/gan_particle_physics/.venv"
WORK_DIR="/home/hep/jcc525/gan_particle_physics"
SCRATCH_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}"

source "$VENV_PATH/bin/activate"

mkdir -p "$SCRATCH_DIR/gan_run"
rsync -a "$WORK_DIR/" "$SCRATCH_DIR/gan_run/"
cd "$SCRATCH_DIR/gan_run"

# Enable GPU debugging if CUDA device is available
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    export CUDA_VISIBLE_DEVICES=0
fi

python src/main.py "$@"

rsync -a "$SCRATCH_DIR/gan_run/gan_results/" "$WORK_DIR/gan_results/" || true