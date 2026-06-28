# Evaluation Metrics Report

_Generated: 2026-06-28 19:50:55_

## Wasserstein Distance (per feature)

| Feature | Wasserstein |
|---------|-------------|
| log_t | 0.0183 |
| sin_phi_s | 0.0156 |
| phi_p | 0.0097 |
| cos_theta | 0.0066 |
| log1p_r | 0.0064 |
| log1p_p_mag | 0.0052 |
| cos_phi_s | 0.0013 |
| sin_theta | 0.0009 |

## MMD (RBF kernel)

**MMD:** 0.009852

## C2ST

| Metric | Value |
|--------|-------|
| Accuracy | 0.5236 |
| Balanced Accuracy | 0.5237 |
| ROC-AUC | 0.5351 |

### Feature Importance

| Feature | Importance (mean ± std) |
|---------|------------------------|
| log1p_r | 0.0131 ± 0.0009 |
| phi_p | 0.0116 ± 0.0011 |
| sin_phi_s | 0.0105 ± 0.0008 |
| log_t | 0.0095 ± 0.0010 |
| cos_phi_s | 0.0089 ± 0.0015 |
| log1p_p_mag | 0.0081 ± 0.0012 |
| cos_theta | 0.0063 ± 0.0016 |
| sin_theta | 0.0032 ± 0.0009 |

## Real-vs-Real Baseline

| Metric | Synth vs Real | Real vs Real |
|--------|--------------|-------------|
| MMD | 0.009852 | 0.003693 |
| C2ST Accuracy | 0.5236 | 0.4975 |
| ROC-AUC | 0.5351 | 0.4987 |

