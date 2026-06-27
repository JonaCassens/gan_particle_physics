#!/bin/bash
# filepath: /home/hep/jcc525/gan_particle_physics/condor/preprocess_job_script.sh

cd /home/hep/jcc525/gan_particle_physics
source /home/hep/jcc525/gan_particle_physics/.venv/bin/activate

python -u src/preprocess_rootfiles.py "$@"