# GAN-Based Particle Physics Data Generation

Workflow for generating synthetic COMET experiment particle physics data using a Wasserstein GAN with Gradient Penalty (WGAN-GP). This pipeline loads ROOT files containing particle trajectories, trains a GAN, and generates statistically similar synthetic samples.

> **🏆 Current Best Result — `e_theta_constraint` (WGAN-GP, muons, 10M training samples)**
> 
> | Metric | Synthetic vs Real | Real vs Real Baseline |
> |--------|:-----------------:|:---------------------:|
> | C2ST Accuracy | 0.513 | 0.500 |
> | C2ST ROC-AUC | 0.521 | 0.500 |
> | MMD | 0.00634 | ~0.0 |
> | 1-NN LOO Accuracy | 51.0% | ~50% |
> 
> All metrics are close to the real-vs-real baseline, meaning the generator produces synthetic muon data that is **nearly statistically indistinguishable from real data**. C2ST accuracy near 0.5 and ROC-AUC near 0.5 are ideal — a classifier cannot reliably separate real from synthetic.

## Project Structure

```
gan_particle_physics/
├── src/
│   ├── main.py                      # Main pipeline orchestration
│   ├── data_loader.py               # ROOT file loading, feature extraction, parquet I/O
│   ├── gan_model.py                 # Vanilla GAN model definition and training
│   ├── wgan_model.py                # WGAN model definition and training
│   ├── wgan_gp_model.py             # WGAN-GP model with gradient penalty
│   ├── cwgan_gp_model.py            # Conditional WGAN-GP conditioned on PDG code
│   ├── evaluate_saved_generator.py  # Offline evaluation of saved generator checkpoints
│   ├── preprocess_rootfiles.py      # Convert ROOT files to Parquet format
│   ├── merge_batches.py             # Merge batched Parquet files into one
│   └── utils.py                     # Metrics, visualization, output handling
├── condor/                          # HTCondor batch submission files
│   ├── submit_gan.sub
│   ├── job_script.sh
│   └── ...
├── gan_results/                     # Timestamped output directories (auto-created)
├── requirements.txt                 # Python dependencies
└── README.md
```

## Data Pipeline

The workflow processes COMET experiment ROOT files (`.rootracker` format) containing:
- **Tree**: `RooTrackerTree`
- **Branches**: `StdHepPdg` (particle codes), `StdHepP4` (4-momentum), `StdHepX4` (4-position), `MonitorID`
- **Default filter**: All particles at MonitorID=4 (PDG filter optional via `--pdg` or `--pdg-allowlist`)
- **Detector centering**: Coordinates centered on the midstream MonitorID=4 plane origin `(x=3259, y=0, z=7655.529) mm`
- **Features extracted** (cylindrical frame, after centering):
  - `y`, `z` — transverse Cartesian position
  - `r` → `log1p_r` — radial distance (log1p-transformed)
  - `sin_phi_s`, `cos_phi_s` — spatial azimuthal angle (trig-encoded)
  - `sin_theta`, `cos_theta` — polar angle (trig-encoded)
  - `phi_p` — momentum azimuthal angle
  - `p_mag` → `log1p_p_mag` — momentum magnitude (log1p-transformed)
  - `pdg` — PDG particle code (kept only for `cwgan-gp`)

**Preprocessing**: ROOT files are first converted to Parquet via `preprocess_rootfiles.py` and optionally batched with `merge_batches.py`.

**Data location (hardcoded)**: `/home/hep/jcc525/comet_data/midstream_merged*.rootracker`
**Preprocessed data**: `/home/hep/jcc525/cleaned_data/pdgNone_monitor<ID>.parquet`

## Setup Instructions

1. **Activate virtual environment**:
   ```bash
   source /home/hep/jcc525/venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Run (standard config with C2ST)
```bash
python src/main.py --entries 5000000 --test-entries 2000000 --model wgan-gp
```

### Skip C2ST for Faster Testing
```bash
python src/main.py --entries 100000 --test-entries 50000 --skip-c2st
```

### Custom C2ST Configuration
```bash
python src/main.py \
  --entries 5000000 \
  --test-entries 2000000 \
  --c2st-max-samples 50000 \
  --c2st-epochs 100 \
  --c2st-hidden-dim 128 \
  --c2st-seed 123
```

### Control Training
```bash
python src/main.py --epochs 500 --batch-size 512 --patience 100 --n-critic 5
```

### cWGAN-GP Tolerance Test (Muon- + Neutron)
```bash
python src/main.py \
  --model cwgan-gp \
  --entries 500000 \
  --test-entries 100000 \
  --pdg-allowlist 13,2112
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--entries` | int | None | Number of samples for training |
| `--test-entries` | int | None | Number of samples for test/synthetic evaluation |
| `--output-name` | str | None | Custom output folder name (default: timestamped) |
| `--results-dir` | str | `./gan_results` | Base directory for outputs |
| `--pdg` | int | None | PDG code filter (default: all particles) |
| `--pdg-allowlist` | str | None | Comma-separated PDG codes to keep for `cwgan-gp` (e.g. `13,2112`) |
| `--monitor-id` | int | 4 | MonitorID filter |
| `--model` | str | `gan` | Model type: `gan`, `wgan`, `wgan-gp`, or `cwgan-gp` |
| `--epochs` | int | 100 | Training epochs |
| `--batch-size` | int | 512 | Batch size |
| `--latent-dim` | int | 256 | Latent vector dimension |
| `--lr-g` | float | 5e-5 | Generator learning rate |
| `--lr-c` | float | 5e-5 | Critic/discriminator learning rate |
| `--gp` | int | 10 | Gradient penalty coefficient (WGAN-GP / cWGAN-GP) |
| `--trig-constraint-weight` | float | 0.01 | Unit-circle penalty weight for sin/cos features |
| `--device` | str | `cuda` (if available) | Device: `cpu` or `cuda` |
| `--patience` | int | 15 | Early stopping patience (WGAN-GP / cWGAN-GP) |
| `--n-critic` | int | 5 | Critic training steps per generator step (WGAN/WGAN-GP) |
| `--log-interval` | int | 1 | Print loss every N epochs |
| `--split-seed` | int | 42 | Random seed for shuffled train/test split |
| **C2ST Options** | | | |
| `--skip-c2st` | flag | False | Skip C2ST evaluation (faster for debugging) |
| `--c2st-max-samples` | int | 50000 | Max samples per class for C2ST |
| `--c2st-epochs` | int | 30 | C2ST classifier training epochs |
| `--c2st-hidden-dim` | int | 64 | C2ST classifier hidden layer dimension |
| `--c2st-seed` | int | 42 | Random seed for C2ST reproducibility |

## Output Files

Each run creates a timestamped folder (e.g., `gan_results/run_20260131_143022_entries5000000/`) containing:

| File | Description |
|------|-------------|
| `synthetic_samples.parquet` | Generated synthetic particle data (gzip-compressed Parquet) |
| `metrics.json` | Quality metrics: KS, Wasserstein, correlation diff, MMD, C2ST, training curves |
| `gan_comparison.png` | Pairwise 2D histograms (real test vs synthetic) |
| `training_loss.png` | Generator/critic loss history and validation Wasserstein/MMD curves |
| `generator.pth` | Saved generator state dict |
| `critic.pth` / `discriminator.pth` | Saved critic or discriminator state dict |
| `per_pdg/` | Per-PDG metrics and plots (cWGAN-GP only) |

## Pipeline Steps

1. **Load data**: Load preprocessed parquet; split into train and test
2. **Train GAN**: Learns feature distribution on training data using chosen model (GAN, WGAN, or WGAN-GP)
3. **Generate samples**: Sample latent vectors, generate synthetic test set of same size as test data
4. **Compute metrics**: 
   - **Univariate**: KS tests and Wasserstein distance per feature
   - **Multivariate**: Correlation matrix differences  
   - **C2ST**: Train a classifier to distinguish real from synthetic (unless `--skip-c2st`)
5. **Create visualizations**: 2D histograms with custom white-to-viridis colormap
6. **Save outputs**: Synthetic samples (parquet), metrics (JSON), plots (PNG), and trained models

## Module Details

### `data_loader.py`
- Loads ROOT files using `uproot`, `awkward`, and `vector` (LorentzVector support)
- Centers coordinates on the midstream MonitorID=4 detector plane origin
- Converts to cylindrical frame; applies log1p transforms to `r` and `p_mag`
- Encodes azimuthal/polar angles as sin/cos pairs for periodicity
- Streams preprocessed Parquet files in batches to avoid large memory footprints
- Exposes `load_preprocessed_data` and `load_root_files` APIs

### `gan_model.py`
- **Dataset**: `ParticleDataset` for PyTorch training
- **Model**: Generator + Discriminator (vanilla GAN with BCE loss)
- **Training**: Standard min-max adversarial training
- **Sampling**: Generates synthetic data and rescales to original feature space

### `wgan_model.py`
- WGAN variant using weight-clipping to enforce the Lipschitz constraint
- Uses RMSProp optimizer as recommended by the original WGAN paper

### `wgan_gp_model.py`
- WGAN with gradient penalty (Gulrajani et al. 2017)
- Supports trig unit-circle constraint penalty for sin/cos features
- Early stopping on validation Wasserstein distance
- Tracks MMD (median-heuristic RBF kernel) alongside Wasserstein for monitoring

### `cwgan_gp_model.py`
- Conditional WGAN-GP conditioned on PDG particle code via a learned embedding
- PDG code is mapped to a dense embedding concatenated into both Generator and Critic
- Drop-in replacement for `train_wgan_gp`; expects a `pdg` column in the dataframe
- Enforces unit-circle constraints on trig features and per-PDG early stopping

### `evaluate_saved_generator.py`
- Offline evaluation of a saved `generator.pth` checkpoint from any prior run
- Auto-loads run configuration from matching Condor `.out` log files
- Computes: KS/Wasserstein, MMD, C2ST (with per-feature permutation importance), FPD (optional, requires TorchScript embedder), 1-NN LOO accuracy
- Computes a real-vs-real baseline for contextualising synth-vs-real metrics
- Saves metrics JSON alongside the original run directory

### `preprocess_rootfiles.py`
- Converts raw `.rootracker` ROOT files to Parquet via `load_root_files`
- Supports batching (`--batch-num`, `--total-batches`) for HTCondor array jobs
- Output path: `/home/hep/jcc525/cleaned_data/pdg<PDG>_monitor<ID>_batch<N>.parquet`

### `merge_batches.py`
- Merges multiple batched Parquet files matching a glob pattern into a single file

### `utils.py`
- **Metrics**:
  - `compute_metrics`: KS tests, Wasserstein distance, correlation matrix differences
  - `compute_mmd_rbf`: Maximum Mean Discrepancy with median-heuristic RBF kernel (chunked, GPU-optional)
  - `compute_c2st_metrics`: Classifier Two-Sample Test (MLP) with permutation feature importance
  - `compute_fpd`: Fréchet Physics Distance using an external TorchScript embedder
  - `compute_1nn_loo`: 1-Nearest-Neighbour leave-one-out accuracy
- **Visualization**: Pairwise 2D histograms with stats boxes, viridis colormap
- **Training plots**: Loss history with validation Wasserstein and MMD curves
- **I/O**: JSON and Parquet export helpers

## Understanding Metrics

The `metrics.json` file contains the following quality indicators:

### 1. Univariate Metrics (per feature)
```json
{ "variable": "y", "ks_stat": 0.027, "ks_p": 1.5e-15, "wasserstein": 2.99 }
```
- **KS statistic**: Max vertical distance between empirical CDFs; near 0 is better
- **KS p-value**: Values << 0.05 reject identical distributions
- **Wasserstein distance**: Earth mover's distance; smaller is better

### 2. Correlation Structure
```json
{ "mean_abs_diff": 0.0107, "max_abs_diff": 0.0387 }
```
- Mean/max absolute differences in correlation matrix (upper triangle); smaller is better

### 3. Maximum Mean Discrepancy (MMD)
```json
{ "mmd": 0.000312, "mmd2": 9.7e-8, "sigma": 14.2, "n_real": 200000, "unbiased": true }
```
- Multivariate kernel test using a median-heuristic RBF kernel
- **MMD ≈ 0** → distributions are close; **MMD >> 0** → poor fit
- Unbiased estimator; computed with chunked matrix operations (GPU optional)

### 4. Classifier Two-Sample Test (C2ST)
```json
{
  "accuracy": 0.52, "balanced_accuracy": 0.51, "roc_auc": 0.53,
  "feature_importance": [{"feature": "log1p_p_mag", "importance_mean": 0.031, ...}],
  "confusion_matrix": { "true_negatives": ..., "false_positives": ..., ... }
}
```
- An MLP classifier trained to distinguish real from synthetic
  - **Accuracy / AUC ≈ 0.5** → synthetic indistinguishable → **Good fit** ✓
  - **Accuracy / AUC >> 0.5** → easily separated → **Poor fit** ✗
- **Permutation feature importance**: identifies which features drive the classifier's decisions

### 5. 1-NN LOO Accuracy (`evaluate_saved_generator.py` only)
- 1-nearest-neighbour leave-one-out test across pooled real + synthetic
- **Accuracy ≈ 50%** → good fit; **>> 50%** → synthetic easily separated

### 6. Real-vs-Real Baseline (`evaluate_saved_generator.py` only)
- Computes the same MMD and C2ST metrics on two disjoint slices of real data
- Provides a reference floor: synthetic quality can be compared against this baseline via `delta_vs_real_baseline`

## Offline Evaluation of Saved Checkpoints

Use `evaluate_saved_generator.py` to re-evaluate any previously saved run without retraining:

```bash
python src/evaluate_saved_generator.py --run-dir run_20260519_120000_entries5000000
```

The script auto-loads the run configuration from matching Condor log files. Key options:

| Argument | Description |
|----------|-------------|
| `--run-dir` | Folder name under `gan_results/` |
| `--pdg` | Override PDG filter (auto-loaded from logs when possible) |
| `--monitor-id` | Override MonitorID (default: 4) |
| `--test-entries` | Override test set size |
| `--fpd-model` | Path to TorchScript physics embedder for FPD metric |
| `--exclude-features` | Comma-separated features to drop from evaluation (e.g. `x`) |

Outputs a `generator_external_test_metrics.json` file alongside the run directory.

## HTCondor Batch Processing

1. **Edit** `condor/job_script.sh` to set desired arguments
2. **Submit job**:
   ```bash
   condor_submit /home/hep/jcc525/gan_particle_physics/condor/submit_gan.sub
   ```
3. **Monitor**:
   ```bash
   condor_q
   ```

## Requirements

- Python 3.9+
- `uproot`, `awkward`, `vector` — ROOT file reading and LorentzVector support
- `scipy`, `numpy` — statistics and linear algebra
- `pandas`, `pyarrow` — data manipulation and Parquet I/O
- `matplotlib`, `seaborn` — visualization
- `torch`, `scikit-learn` — GAN training and evaluation metrics

## License

MIT License - See LICENSE file for details.