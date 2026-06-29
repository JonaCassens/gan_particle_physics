# Week 26: 25/06/26 – 01/07/26

## Previous Week Recap
- Trained front face WGAN-GP across 6 key particle types (n, μ, e⁻, γ, π⁻, e⁺)
- Individual generators all <0.01 MMD and ~0.52 C2ST — positron still needs attention
- Cluster connectivity issues blocked combined evaluation
- Awaiting Yuki's config file details for downstream data

---

## 1. Weekly Objectives

- Talk to Yuki about generating data for downstream
- Compare distributions between generated data and downstream data
- Evaluate combined 6-PDG generator on full mix
- Reorganise codebase for cleaner model imports

## 2. Technical Progress

- Reorganised `src/` into `src/models/` package (wgan_gp_model, cwgan_gp_model, wgan_model, gan_model)
  - All imports updated across pipeline scripts
- Added `--per-pdg` flag to `evaluate_saved_data.py` for per-particle-type metric breakdown
- Added `--mmd-max-samples` and `--c2st-max-samples` CLI args to `evaluate_saved_data.py`, then hardcoded to 200k / 500k for reproducibility
- Generated and committed eval results for optimal 6-PDG bounded-r mix (`eval_results/optimal_mix_bounded_r_6pdg/`)
  - **Overall:** MMD=0.034, C2ST acc=0.535, ROC-AUC=0.558
  - **Per-PDG summary:**
    | PDG | Particle | MMD | C2ST acc | ROC-AUC | Status |
    |-----|----------|-----|----------|---------|--------|
    | 13 | μ⁻ | ≈0.000 | 0.500 | 0.500 | Excellent |
    | -211 | π⁻ | ≈0.000 | 0.537 | 0.540 | Good |
    | 11 | e⁻ | 0.005 | 0.510 | 0.517 | Good |
    | -11 | e⁺ | 0.008 | 0.494 | 0.491 | Good |
    | 22 | γ | 0.022 | 0.527 | 0.539 | Acceptable |
    | 2112 | n | 0.148 | 0.667 | 0.724 | Poor |
  - Neutron generator failing significantly — phi_p (KS=0.237, W=0.667) and t (W=1.876) badly mismatched
  - phi_p and t are top feature importance drivers for neutron C2ST separation
- Also evaluated `bounded_r_6pdg_r_rejection` variant (radius rejection sampling experiment)
- Identified that r-clipping applied *inside* the WGAN generator training loop was likely saturating particles at the boundary, distorting gradients and introducing training errors
  - Fix: removed the hard clip from the training cycle (unbounded r during training)
  - Mitigation for already-trained generators: apply accept-reject filtering post-generation to enforce r bounds without retraining

**On-shell log_t coupling (2026-06-28)**
- Added explicit coupling of `log_t` to `log1p_p_mag` via the on-shell relation `t = sqrt(p² + m²)` in MeV
- Implemented as a per-sample penalty term during training + generation-time projection
- This directly addresses the worst C2ST feature separation seen in previous runs

**v2 generator results — retrained with rejection sampling + on-shell coupling**
| Particle | Model | MMD | C2ST acc | ROC-AUC | Status |
|----------|-------|-----|----------|---------|--------|
| μ⁻ | muon_minus_bounded_r_2 | 0.00306 | 0.514 | 0.524 | Excellent |
| e⁺ | positron_bounded_r_2 | 0.00573 | 0.504 | 0.511 | Excellent |
| γ | photon_bounded_r_2 | 0.00989 | 0.554 | 0.576 | Acceptable |
| e⁻ | e_theta_constraint | 0.00767 | 0.511 | 0.519 | Good |

- Photon v2: `log_t` (KS=0.045) and `log1p_p_mag` (KS=0.053) still the weakest features — photon is massless so the on-shell coupling is trivial; C2ST driven mainly by phi_p and angle distributions
- Positron v2: resolved prior concern — C2ST near 0.504, all Wasserstein distances < 0.017
- Muon-minus v2: all features excellent; no Wasserstein distance > 0.005

**Infrastructure / tooling (2026-06-28)**
- CPU MMD computation capped at 50k samples to prevent O(n²) memory/time hang
- Fixed train/test split clamping in `evaluate_saved_generator` when dataset is smaller than requested
- Eval condor scripts updated: CPU queue for bounded_r evals (no GPU), added device/timing info
- Metrics now saved as both `.json` and human-readable `.md` report

**`confix_*` training runs — new best results (2026-06-29)**

New naming convention (`confix_<particle>`) for runs combining all fixes: rejection-sampling r-bound + on-shell log_t projection + GPU capability filter. Trained on 10M samples, 70 epochs.

| Particle | Run | C2ST acc | Balanced acc | ROC-AUC | MMD | Notes |
|----------|-----|----------|--------------|---------|-----|-------|
| μ⁻ | confix_muon | 0.5055 | 0.5066 | 0.5140 | 0.00958 | New muon best |
| e⁻ | confix_electron | **0.5037** | **0.5045** | **0.5098** | **0.00509** | **Milestone: all three C2ST metrics <0.51** |

- *confix_electron* is the first run where C2ST accuracy, balanced accuracy, and ROC-AUC all fall below 0.51. Neither model's accuracy is statistically distinguishable from chance: muon z≈1.6 (p≈0.12), electron z≈1.0 (p≈0.31) at n=20 000 test pairs.
- *Muon* per-feature: `phi_p` (W=0.0147) and `log1p_r` (W=0.0138) are the largest residuals; `log_t` has *negative* C2ST importance (−0.0036), confirming on-shell projection eliminates it as a discriminating axis.
- *Electron* per-feature: `log_t` (W=0.0159) and `log1p_p_mag` (W=0.0125) are the largest residuals; `cos_phi_s` leads C2ST importance (0.0054). The tiny electron mass (~0.511 MeV) means rounding in `p_mag` propagates visibly into the derived `log_t`.

**HTCondor fixes (2026-06-29)**
- Added `GPUs_Capability >= 7.0` requirement to training submit file — P100 nodes (capability 6.0) are incompatible with PyTorch 2.8 and were silently failing jobs
- Changed `queue` to single-statement form to resolve HTCondor deprecation warning

## 3. Selected Plots & Visualisations

- `eval_results/optimal_mix_bounded_r_6pdg/eval_2d_comparisons.png`
- `eval_results/bounded_r_6pdg_r_rejection/` (per-PDG subdirectory plots)
- `gan_results_optimal/muon_minus_bounded_r_2/gan_comparison.png`
- `gan_results_optimal/positron_bounded_r_2/gan_comparison.png`
- `gan_results_optimal/photon_bounded_r_2/gan_comparison.png`
- `gan_results_optimal/e_theta_constraint/gan_comparison.png`
- `gan_results/confix_muon/gan_comparison.png`
- `gan_results/confix_electron/gan_comparison.png`

## 4. Challenges & Solutions

- Neutron generator produces poor phi_p and t distributions in the combined mix — needs retraining or additional constraints specific to neutron kinematics
- Positron results from last week: now showing good C2ST (0.494) — appears resolved in the optimal bounded-r model
- Photon remains acceptable but not excellent: massless particle means on-shell coupling doesn't constrain log_t vs log1p_p_mag; largest residual KS in log_t (0.045) and log1p_p_mag (0.053)

## 5. Administrative Tracking

- 

## 6. Plan for Next Week

- Retrain neutron generator with all confix fixes — focus on phi_p and t distributions (currently KS=0.237, W=0.667 for phi_p)
- Run `confix_*` training for remaining particles: photon, positron, antimuon, proton — use submit_wgan_gp_confix.sub
- Re-evaluate combined 6-PDG mix using confix generators once available, to see if overall mix MMD/C2ST improves from 0.034/0.535
- Talk to Yuki about downstream data config and comparison pipeline
- Consider photon-specific constraint to improve log_t / log1p_p_mag matching (massless: on-shell coupling is trivial)

## 7. Random Questions / Comments

- 
