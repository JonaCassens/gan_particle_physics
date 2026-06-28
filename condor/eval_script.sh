#!/bin/bash
set -euo pipefail

cd /home/hep/jcc525/gan_particle_physics
source /home/hep/jcc525/gan_particle_physics/.venv/bin/activate

python -u "$@"