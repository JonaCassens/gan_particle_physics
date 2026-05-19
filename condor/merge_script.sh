#!/bin/bash

cd /home/hep/jcc525/gan_particle_physics
source /home/hep/jcc525/venv/bin/activate

python -u src/merge_batches.py "$@"