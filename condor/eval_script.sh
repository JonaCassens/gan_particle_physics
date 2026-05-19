#!/bin/bash
set -euo pipefail

cd /home/hep/jcc525/gan_particle_physics
source /home/hep/jcc525/venv/bin/activate

python -u src/evaluate_saved_generator.py "$@"