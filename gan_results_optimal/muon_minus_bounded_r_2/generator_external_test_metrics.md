# Evaluation Metrics Report

_Generated: 2026-06-28 19:40:32_

## Wasserstein Distance (per feature)

| Feature | Wasserstein |
|---------|-------------|
| sin_phi_s | 0.1048 |
| cos_phi_s | 0.0935 |
| log1p_r | 0.0283 |
| phi_p | 0.0197 |
| log1p_p_mag | 0.0152 |
| cos_theta | 0.0083 |
| log_t | 0.0014 |
| sin_theta | 0.0012 |

## MMD (RBF kernel)

**MMD:** 0.096095

## C2ST

| Metric | Value |
|--------|-------|
| Accuracy | 0.7765 |
| Balanced Accuracy | 0.7766 |
| ROC-AUC | 0.8656 |

### Feature Importance

| Feature | Importance (mean ± std) |
|---------|------------------------|
| cos_phi_s | 0.2110 ± 0.0059 |
| sin_phi_s | 0.1588 ± 0.0069 |
| log1p_r | 0.0664 ± 0.0027 |
| cos_theta | 0.0494 ± 0.0055 |
| phi_p | 0.0377 ± 0.0046 |
| log1p_p_mag | 0.0320 ± 0.0028 |
| log_t | 0.0271 ± 0.0030 |
| sin_theta | 0.0219 ± 0.0012 |

## Real-vs-Real Baseline

| Metric | Synth vs Real | Real vs Real |
|--------|--------------|-------------|
| MMD | 0.096095 | 0.007829 |
| C2ST Accuracy | 0.7765 | 0.5018 |
| ROC-AUC | 0.8656 | 0.5181 |

