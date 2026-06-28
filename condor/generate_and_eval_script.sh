#!/bin/bash
set -euo pipefail

cd /home/hep/jcc525/gan_particle_physics
source /home/hep/jcc525/gan_particle_physics/.venv/bin/activate

OUTPUT_PARQUET="${1:-/home/hep/jcc525/gan_particle_physics/synthetic_data/optimal_mix_6pdg.parquet}"
EVAL_OUTPUT_DIR="${2:-/home/hep/jcc525/gan_particle_physics/eval_results/optimal_mix_6pdg}"

echo "=== Step 1: generate_by_pdg_distribution ==="
python -u src/generate_by_pdg_distribution.py \
    --folders \
        electron_no_x \
        positron_bounded_r_2 \
        e_theta_constraint \
        muon_minus_bounded_r_2 \
        neutron_x_constrained_dropped \
        photon_bounded_r_2 \
    --pdg-codes 11 -11 13 -13 2112 22 \
    --gan-results-root /home/hep/jcc525/gan_particle_physics/gan_results_optimal \
    --n-particles 1000000 \
    --output-parquet "$OUTPUT_PARQUET" \
    --device cuda

echo "=== Step 2: evaluate_saved_data ==="
python -u src/evaluate_saved_data.py \
    --synthetic-parquet "$OUTPUT_PARQUET" \
    --truth-parquet /home/hep/jcc525/cleaned_data/pdgNone_monitor4.parquet \
    --output-dir "$EVAL_OUTPUT_DIR" \
    --truth-pdg-allowlist "11,-11,13,-13,2112,22" \
    --n-truth 500000 \
    --n-synthetic 500000 \
    --per-pdg \
    --device cuda

echo "=== Done. Results in $EVAL_OUTPUT_DIR ==="
