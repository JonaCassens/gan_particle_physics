# Optimal Generators

One generator per PDG, selected by 100k-sample training-time C2ST from `metrics.json`.
Two runs are additionally verified at 200k samples (noted below).

| PDG | Particle | Folder | MMD (100k) | C2ST (100k) | AUC (100k) | 200k C2ST |
|-----|----------|--------|------------|-------------|------------|-----------|
| 11 | electron | `electron_no_x` | 0.00676 | 0.5047 | 0.5147 | **0.5141** ✓ |
| -11 | positron | `positron_bounded_r_2` | 0.00574 | 0.5043 | 0.5107 | not tested |
| 13 | muon | `e_theta_constraint` | 0.00767 | 0.5110 | 0.5194 | **0.5126** ✓ |
| -13 | muon minus | `muon_minus_bounded_r_2` | 0.00306 | 0.5139 | 0.5237 | not tested |
| 2112 | neutron | `neutron_x_constrained_dropped` | 0.00918 | 0.5065 | 0.5127 | not tested |
| 22 | photon | `photon_bounded_r_2` | 0.00989 | 0.5538 | 0.5762 | not tested |

## Why these runs

**electron_no_x** and **e_theta_constraint** are the only two confirmed to hold up at 200k samples.
All bounded_r v1 runs in the previous optimal set failed at 200k (C2ST 0.76–0.87), caused by
undertrained generators: epochs=50/patience=20 vs 100+/patience=40 for the good runs. Early
stopping on val_wasserstein also selected checkpoints as early as epoch 6–16 while val_mmd kept
improving for many more epochs.

**positron_bounded_r_2** wins all positron runs by both MMD and C2ST at 100k.

**muon_minus_bounded_r_2** is the only muon minus run that passes at 100k (v1 scored C2ST=0.80).
Needs 200k verification.

**neutron_x_constrained_dropped** has the best C2ST among neutron runs with the current feature set
(x dropped). neutron_no_x had lower MMD but was trained on the old feature set including x.
Needs 200k verification.

**photon_bounded_r_2** is the best available photon run but C2ST ~0.55 at 100k is noticeably
weaker than other particles. The v1 bounded_r_photon failed at 200k (C2ST=0.81). Photon likely
needs a dedicated retraining with longer epochs.
