# Evaluation Metrics Report

_Generated: 2026-06-28 19:53:20_

## Wasserstein Distance (per feature)

| Feature | Wasserstein |
|---------|-------------|
| phi_p | 0.0164 |
| sin_phi_s | 0.0070 |
| log1p_r | 0.0062 |
| log_t | 0.0036 |
| cos_phi_s | 0.0034 |
| log1p_p_mag | 0.0030 |
| cos_theta | 0.0027 |
| sin_theta | 0.0013 |

## MMD (RBF kernel)

**MMD:** 0.006881

## C2ST

| Metric | Value |
|--------|-------|
| Accuracy | 0.5145 |
| Balanced Accuracy | 0.5146 |
| ROC-AUC | 0.5241 |

### Feature Importance

| Feature | Importance (mean ± std) |
|---------|------------------------|
| log1p_r | 0.0088 ± 0.0016 |
| cos_phi_s | 0.0076 ± 0.0008 |
| sin_phi_s | 0.0070 ± 0.0016 |
| phi_p | 0.0047 ± 0.0017 |
| log_t | 0.0029 ± 0.0008 |
| log1p_p_mag | 0.0014 ± 0.0012 |
| cos_theta | 0.0010 ± 0.0015 |
| sin_theta | 0.0009 ± 0.0012 |

## Real-vs-Real Baseline

| Metric | Synth vs Real | Real vs Real |
|--------|--------------|-------------|
| MMD | 0.006881 | 0.002615 |
| C2ST Accuracy | 0.5145 | 0.4966 |
| ROC-AUC | 0.5241 | 0.4964 |

