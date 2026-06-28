# Utility functions for visualization and output handling

import numpy as np
import pandas as pd
import torch
import os
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from scipy import stats
import json
import gc
import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import Optional, Any
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix


def _as_finite_float32_2d(data):
    """Convert data to finite float32 2D array by dropping rows with NaN/Inf."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    finite_mask = np.all(np.isfinite(arr), axis=1)
    return arr[finite_mask]


def _subsample_rows(data, max_samples=None, seed=42):
    """Subsample rows without replacement."""
    if max_samples is None or len(data) <= max_samples:
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=max_samples, replace=False)
    return data[idx]


def _estimate_rbf_sigma_median(x, y, max_points=2000, seed=42):
    """Estimate RBF sigma using the median heuristic on pooled samples."""
    pooled = np.vstack([x, y])
    pooled = _subsample_rows(pooled, max_samples=max_points, seed=seed)

    if len(pooled) < 2:
        return 1.0

    diff = pooled[:, None, :] - pooled[None, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    tri = np.triu_indices_from(dists_sq, k=1)
    positive = dists_sq[tri]
    positive = positive[positive > 0]

    if len(positive) == 0:
        return 1.0
    sigma = float(np.sqrt(np.median(positive)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return sigma


def _rbf_kernel_sum(x, y, gamma, chunk_size=1024, exclude_diag=False):
    """Compute sum of RBF kernel values with chunking to limit memory."""
    total = 0.0
    n_x = len(x)
    n_y = len(y)

    for i in range(0, n_x, chunk_size):
        x_chunk = x[i:i + chunk_size]
        for j in range(0, n_y, chunk_size):
            y_chunk = y[j:j + chunk_size]
            diff = x_chunk[:, None, :] - y_chunk[None, :, :]
            dists_sq = np.sum(diff * diff, axis=2)
            k_block = np.exp(-gamma * dists_sq)

            if exclude_diag and (x is y) and (i == j):
                np.fill_diagonal(k_block, 0.0)

            total += float(np.sum(k_block))

    return total


def _subsample_rows_torch(data: torch.Tensor, max_samples=None, seed: int = 42) -> torch.Tensor:
    if max_samples is None or data.shape[0] <= max_samples:
        return data
    generator = torch.Generator(device=data.device)
    generator.manual_seed(seed)
    indices = torch.randperm(data.shape[0], generator=generator, device=data.device)[:max_samples]
    return data[indices]


def _estimate_rbf_sigma_median_torch(x: torch.Tensor, y: torch.Tensor, max_points: int = 2000, seed: int = 42) -> float:
    pooled = torch.cat([x, y], dim=0)
    pooled = _subsample_rows_torch(pooled, max_samples=max_points, seed=seed)

    if pooled.shape[0] < 2:
        return 1.0

    dists = torch.pdist(pooled, p=2)
    positive = dists[dists > 0]

    if positive.numel() == 0:
        return 1.0

    sigma = float(torch.median(positive).item())
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return sigma


def _rbf_kernel_sum_torch(x: torch.Tensor, y: torch.Tensor, gamma: float, chunk_size: int = 1024, exclude_diag: bool = False) -> float:
    total = torch.zeros((), dtype=torch.float32, device=x.device)
    n_x = x.shape[0]
    n_y = y.shape[0]

    same_tensor = x.data_ptr() == y.data_ptr() and n_x == n_y
    gamma_tensor = torch.tensor(float(gamma), dtype=torch.float32, device=x.device)

    for i in range(0, n_x, chunk_size):
        x_chunk = x[i:i + chunk_size]
        for j in range(0, n_y, chunk_size):
            y_chunk = y[j:j + chunk_size]
            dists_sq = torch.cdist(x_chunk, y_chunk, p=2).pow(2)
            k_block = torch.exp(-gamma_tensor * dists_sq)

            if exclude_diag and same_tensor and i == j:
                diag_len = min(x_chunk.shape[0], y_chunk.shape[0])
                diag_idx = torch.arange(diag_len, device=x.device)
                k_block[diag_idx, diag_idx] = 0.0

            total = total + k_block.sum()

    return float(total.item())


def compute_mmd_rbf(real_data, synthetic_data, sigma="median", sigma_scale=1.0,
                    max_samples=5000, chunk_size=1024, unbiased=True, seed=42,
                    device: str = "cpu"):
    """
    Compute full multivariate MMD using an RBF kernel.

    Parameters
    ----------
    real_data, synthetic_data : array-like
        2D arrays with shape (n_samples, n_features).
    sigma : float or "median"
        RBF bandwidth. If "median", use pooled median heuristic.
    sigma_scale : float
        Multiplicative factor for sigma.
    max_samples : int or None
        Maximum rows per dataset for runtime control.
    chunk_size : int
        Kernel chunk size to control memory.
    unbiased : bool
        If True, uses unbiased estimator for within-sample terms.
    seed : int
        Random seed for subsampling.

    Returns
    -------
    dict
        mmd, mmd2, sigma, gamma, sample counts and estimator config.
    """
    x_np = _as_finite_float32_2d(real_data)
    y_np = _as_finite_float32_2d(synthetic_data)

    use_gpu = "cuda" in str(device).lower() and torch.cuda.is_available()

    if use_gpu:
        x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y_np, dtype=torch.float32, device=device)

        x_t = _subsample_rows_torch(x_t, max_samples=max_samples, seed=seed)
        y_t = _subsample_rows_torch(y_t, max_samples=max_samples, seed=seed + 1)

        n = int(x_t.shape[0])
        m = int(y_t.shape[0])
        if n < 2 or m < 2:
            return {
                "mmd": float("nan"),
                "mmd2": float("nan"),
                "sigma": float("nan"),
                "gamma": float("nan"),
                "n_real": int(n),
                "n_synthetic": int(m),
                "max_samples": max_samples,
                "unbiased": bool(unbiased),
                "backend": "torch",
                "device": str(device),
            }

        pooled = torch.cat([x_t, y_t], dim=0)
        pooled_mean = pooled.mean(dim=0, keepdim=True)
        pooled_std = pooled.std(dim=0, keepdim=True) + 1e-8
        x_t = (x_t - pooled_mean) / pooled_std
        y_t = (y_t - pooled_mean) / pooled_std

        if sigma == "median":
            sigma_value = _estimate_rbf_sigma_median_torch(x_t, y_t, seed=seed)
        else:
            sigma_value = float(sigma)

        sigma_value = max(sigma_value * float(sigma_scale), 1e-8)
        gamma = 1.0 / (2.0 * sigma_value * sigma_value)

        sum_xy = _rbf_kernel_sum_torch(x_t, y_t, gamma=gamma, chunk_size=chunk_size, exclude_diag=False)

        if unbiased:
            sum_xx = _rbf_kernel_sum_torch(x_t, x_t, gamma=gamma, chunk_size=chunk_size, exclude_diag=True)
            sum_yy = _rbf_kernel_sum_torch(y_t, y_t, gamma=gamma, chunk_size=chunk_size, exclude_diag=True)
            mmd2 = (sum_xx / (n * (n - 1))) + (sum_yy / (m * (m - 1))) - (2.0 * sum_xy / (n * m))
        else:
            sum_xx = _rbf_kernel_sum_torch(x_t, x_t, gamma=gamma, chunk_size=chunk_size, exclude_diag=False)
            sum_yy = _rbf_kernel_sum_torch(y_t, y_t, gamma=gamma, chunk_size=chunk_size, exclude_diag=False)
            mmd2 = (sum_xx / (n * n)) + (sum_yy / (m * m)) - (2.0 * sum_xy / (n * m))

        mmd2 = float(mmd2)
        mmd = float(np.sqrt(max(mmd2, 0.0)))

        return {
            "mmd": mmd,
            "mmd2": mmd2,
            "sigma": float(sigma_value),
            "gamma": float(gamma),
            "n_real": int(n),
            "n_synthetic": int(m),
            "max_samples": max_samples,
            "chunk_size": int(chunk_size),
            "unbiased": bool(unbiased),
            "backend": "torch",
            "device": str(device),
        }

    x = _subsample_rows(x_np, max_samples=max_samples, seed=seed)
    y = _subsample_rows(y_np, max_samples=max_samples, seed=seed + 1)

    n = len(x)
    m = len(y)
    if n < 2 or m < 2:
        return {
            "mmd": float("nan"),
            "mmd2": float("nan"),
            "sigma": float("nan"),
            "gamma": float("nan"),
            "n_real": int(n),
            "n_synthetic": int(m),
            "max_samples": max_samples,
            "unbiased": bool(unbiased),
            "backend": "numpy",
            "device": "cpu",
        }

    pooled = np.vstack([x, y])
    pooled_mean = pooled.mean(axis=0, keepdims=True)
    pooled_std = pooled.std(axis=0, keepdims=True) + 1e-8
    x = (x - pooled_mean) / pooled_std
    y = (y - pooled_mean) / pooled_std

    if sigma == "median":
        sigma_value = _estimate_rbf_sigma_median(x, y, seed=seed)
    else:
        sigma_value = float(sigma)

    sigma_value = max(sigma_value * float(sigma_scale), 1e-8)
    gamma = 1.0 / (2.0 * sigma_value * sigma_value)

    sum_xy = _rbf_kernel_sum(x, y, gamma=gamma, chunk_size=chunk_size, exclude_diag=False)

    if unbiased:
        sum_xx = _rbf_kernel_sum(x, x, gamma=gamma, chunk_size=chunk_size, exclude_diag=True)
        sum_yy = _rbf_kernel_sum(y, y, gamma=gamma, chunk_size=chunk_size, exclude_diag=True)
        mmd2 = (sum_xx / (n * (n - 1))) + (sum_yy / (m * (m - 1))) - (2.0 * sum_xy / (n * m))
    else:
        sum_xx = _rbf_kernel_sum(x, x, gamma=gamma, chunk_size=chunk_size, exclude_diag=False)
        sum_yy = _rbf_kernel_sum(y, y, gamma=gamma, chunk_size=chunk_size, exclude_diag=False)
        mmd2 = (sum_xx / (n * n)) + (sum_yy / (m * m)) - (2.0 * sum_xy / (n * m))

    mmd2 = float(mmd2)
    mmd = float(np.sqrt(max(mmd2, 0.0)))

    return {
        "mmd": mmd,
        "mmd2": mmd2,
        "sigma": float(sigma_value),
        "gamma": float(gamma),
        "n_real": int(n),
        "n_synthetic": int(m),
        "max_samples": max_samples,
        "chunk_size": int(chunk_size),
        "unbiased": bool(unbiased),
        "backend": "numpy",
        "device": "cpu",
    }


def _infer_generator_architecture_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    linear_weights = []
    for key, tensor in state_dict.items():
        if tensor.ndim == 2 and key.startswith("model.") and key.endswith(".weight"):
            try:
                layer_index = int(key.split(".")[1])
            except (ValueError, IndexError):
                continue
            linear_weights.append((layer_index, key, (int(tensor.shape[0]), int(tensor.shape[1]))))

    linear_weights.sort(key=lambda item: item[0])
    if not linear_weights:
        raise ValueError("Could not infer generator architecture from checkpoint state_dict.")

    hidden_dims = [shape[0] for _, _, shape in linear_weights[:-1]]
    latent_dim = linear_weights[0][2][1]
    output_dim = linear_weights[-1][2][0]
    normalization = "batchnorm" if any("running_mean" in key for key in state_dict) else "layernorm"
    if "embedding.weight" in state_dict:
        model_type = "cwgan-gp"
    elif normalization == "layernorm":
        model_type = "wgan-gp"
    else:
        model_type = "gan-or-wgan"

    return {
        "latent_dim": int(latent_dim),
        "output_dim": int(output_dim),
        "hidden_dims": [int(v) for v in hidden_dims],
        "normalization": normalization,
        "model_type": model_type,
    }


def _build_generator_from_checkpoint(
    model_type: str,
    inferred: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> nn.Module:
    from models.gan_model import Generator as GANGenerator
    from models.wgan_model import Generator as WGANGenerator
    from models.wgan_gp_model import Generator as WGANGPGenerator
    from models.cwgan_gp_model import CGenerator as CWGANGPGenerator

    latent_dim = int(inferred["latent_dim"])
    output_dim = int(inferred["output_dim"])
    hidden_dims = list(inferred["hidden_dims"])

    if model_type == "cwgan-gp":
        embedding_weight = state_dict.get("embedding.weight")
        if embedding_weight is None:
            raise ValueError("cwgan-gp checkpoint missing generator embedding.weight")
        vocab_size = int(embedding_weight.shape[0])
        embed_dim = int(embedding_weight.shape[1])
        generator = CWGANGPGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
        )
    elif model_type == "wgan-gp":
        generator = WGANGPGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )
    elif model_type == "wgan":
        generator = WGANGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )
    else:
        generator = GANGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )

    generator.load_state_dict(state_dict, strict=True)
    return generator


def _compute_normalization_stats_for_model(train_df: pd.DataFrame, model_type: str) -> tuple[np.ndarray, np.ndarray]:
    from models.gan_model import ParticleDataset as GANParticleDataset
    from models.wgan_model import ParticleDataset as WGANParticleDataset
    from models.wgan_gp_model import ParticleDataset as WGANGPParticleDataset

    if model_type in {"wgan-gp", "cwgan-gp"}:
        dataset = WGANGPParticleDataset(train_df)
    elif model_type == "wgan":
        dataset = WGANParticleDataset(train_df)
    else:
        dataset = GANParticleDataset(train_df)

    mean = dataset.mean.detach().cpu().numpy().astype(np.float32)
    std = dataset.std.detach().cpu().numpy().astype(np.float32)
    return mean, std


def _resolve_generation_feature_names(
    eval_feature_names: list[str],
    generated_width: int,
    generator_output_columns: Optional[list[str]] = None,
) -> list[str]:
    if generator_output_columns is not None:
        if len(generator_output_columns) != generated_width:
            raise ValueError(
                f"generator_output_columns has length {len(generator_output_columns)}, "
                f"but generator produced {generated_width} features."
            )
        return list(generator_output_columns)

    if generated_width == len(eval_feature_names):
        return list(eval_feature_names)

    if generated_width == len(eval_feature_names) + 1 and "x" not in eval_feature_names:
        return ["x", *eval_feature_names]

    if generated_width == len(eval_feature_names) - 1 and "x" in eval_feature_names:
        return [column for column in eval_feature_names if column != "x"]

    raise ValueError(
        f"Generator produced {generated_width} features, but evaluation data has "
        f"{len(eval_feature_names)} columns: {eval_feature_names}."
    )


def generate_synthetic_from_checkpoint(
    generator_path: str,
    train_df: pd.DataFrame,
    n_samples: int,
    device: str = "cpu",
    batch_size: int = 32768,
    model_type: Optional[str] = None,
    latent_dim: Optional[int] = None,
    apply_angle_clipping: Optional[bool] = None,
    conditional_pdg_codes: Optional[np.ndarray] = None,
    generator_output_columns: Optional[list[str]] = None,
    onshell_mass_mev: Optional[float] = None,
) -> dict[str, Any]:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {generator_path}")

    checkpoint = torch.load(generator_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format; expected dict-like state_dict.")

    inferred = _infer_generator_architecture_from_state_dict(state_dict)
    resolved_model_type = (model_type or inferred["model_type"]).strip().lower()
    if resolved_model_type == "gan-or-wgan":
        resolved_model_type = "gan"

    if latent_dim is not None:
        inferred["latent_dim"] = int(latent_dim)

    mean, std = _compute_normalization_stats_for_model(train_df, resolved_model_type)
    feature_names = list(train_df.columns)
    if "pdg" in feature_names and len(mean) == len(feature_names) - 1:
        feature_names = [column for column in feature_names if column != "pdg"]
    elif len(mean) != len(feature_names):
        raise RuntimeError(
            "Normalization shape mismatch after schema alignment: "
            f"len(mean)={len(mean)}, len(std)={len(std)}, len(feature_names)={len(feature_names)}"
        )

    generator = _build_generator_from_checkpoint(
        model_type=resolved_model_type,
        inferred=inferred,
        state_dict=state_dict,
    ).to(device)
    generator.eval()

    use_amp = "cuda" in str(device).lower() and torch.cuda.is_available()

    pdg_to_idx: dict[int, int] = {}
    unknown_idx = 0
    cond_codes = None
    if conditional_pdg_codes is not None:
        cond_codes = np.asarray(conditional_pdg_codes, dtype=np.int64)
        unique_codes = sorted(set(int(code) for code in cond_codes.tolist()))
        embedding = getattr(generator, "embedding", None)
        if embedding is None or not hasattr(embedding, "weight"):
            raise ValueError("conditional_pdg_codes provided but generator has no embedding layer")
        vocab_capacity = int(embedding.weight.shape[0])
        if vocab_capacity < 2:
            raise ValueError("cwgan embedding vocab size must be at least 2")
        max_known = vocab_capacity - 1
        pdg_to_idx = {code: idx for idx, code in enumerate(unique_codes[:max_known])}
        unknown_idx = max_known

    if apply_angle_clipping is None:
        apply_angle_clipping = resolved_model_type in {"wgan-gp", "cwgan-gp"}

    generation_feature_names = None
    mean_gen = None
    std_gen = None
    out = None

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start)
            latent = torch.randn(current_batch_size, int(inferred["latent_dim"]), device=device)

            autocast_context = getattr(torch.cuda.amp, "autocast")(dtype=torch.float16) if use_amp else nullcontext()
            with autocast_context:
                if cond_codes is not None:
                    pdg_chunk = cond_codes[start:start + current_batch_size]
                    pdg_idx_np = np.array([pdg_to_idx.get(int(code), unknown_idx) for code in pdg_chunk], dtype=np.int64)
                    pdg_idx = torch.from_numpy(pdg_idx_np).to(device)
                    chunk = generator(latent, pdg_idx)
                else:
                    chunk = generator(latent)

            chunk = chunk.detach().float().cpu().numpy()

            if generation_feature_names is None:
                generation_feature_names = _resolve_generation_feature_names(
                    eval_feature_names=feature_names,
                    generated_width=int(chunk.shape[1]),
                    generator_output_columns=generator_output_columns,
                )
                if generation_feature_names == feature_names:
                    mean_gen = mean
                    std_gen = std
                else:
                    feat_index = {name: idx for idx, name in enumerate(feature_names)}
                    gen_indices = [feat_index[name] for name in generation_feature_names if name in feat_index]
                    mean_gen = mean[gen_indices]
                    std_gen = std[gen_indices]
                out = np.empty((n_samples, len(generation_feature_names)), dtype=np.float32)

            if chunk.shape[1] != len(generation_feature_names):
                raise ValueError(
                    f"Generator produced {chunk.shape[1]} features, but generation_feature_names has "
                    f"{len(generation_feature_names)} entries."
                )

            chunk = chunk * (std_gen + 1e-8) + mean_gen

            if apply_angle_clipping:
                from models.wgan_gp_model import (
                    _apply_generation_bounds,
                    BOUNDED_CLIP_FEATURES,
                    ONSHELL_CLIP_FEATURES,
                )

                clip_feature_indices = {
                    name: idx
                    for idx, name in enumerate(generation_feature_names)
                    if name in BOUNDED_CLIP_FEATURES + ONSHELL_CLIP_FEATURES
                }
                chunk = _apply_generation_bounds(
                    chunk, clip_feature_indices, mass_mev=onshell_mass_mev
                )

            out[start:start + current_batch_size] = chunk.astype(np.float32, copy=False)

    if out is None or generation_feature_names is None:
        raise RuntimeError("No synthetic samples were generated.")

    return {
        "samples": out,
        "feature_names": generation_feature_names,
        "inferred_architecture": inferred,
        "model_type": resolved_model_type,
        "latent_dim": int(inferred["latent_dim"]),
        "mean": mean,
        "std": std,
    }

def create_white_to_viridis_cmap():
    """Create custom colormap: white (zero) -> viridis colors."""
    viridis = plt.cm.viridis
    colors = ['white'] + [viridis(i) for i in range(1, 256)]
    return LinearSegmentedColormap.from_list('white_viridis', colors, N=256)

def plot_2d_hist_with_stats(ax, x, y, x_label, y_label, title,
                            bins=100, cmap="viridis",
                            x_range=None, y_range=None):
    """
    Plot a 2D histogram (heatmap) of (x, y) with colorbar and stats box.
    """
    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.float32)
    
    # Ensure numeric conversion and remove NaN
    mask = np.isfinite(x_np) & np.isfinite(y_np)
    x_np = x_np[mask]
    y_np = y_np[mask]

    entries = len(x_np)
    mean_x = float(np.mean(x_np)) if entries else float("nan")
    mean_y = float(np.mean(y_np)) if entries else float("nan")
    std_x = float(np.std(x_np)) if entries else float("nan")
    std_y = float(np.std(y_np)) if entries else float("nan")

    kwargs = dict(bins=bins, cmap=cmap)
    if x_range is not None and y_range is not None:
        kwargs["range"] = [x_range, y_range]

    h = ax.hist2d(x_np, y_np, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    cb = plt.colorbar(h[3], ax=ax)
    cb.set_label("Counts")

    stats_text = (
        f"Entries: {entries}\n"
        f"Mean x: {mean_x:.2f}\n"
        f"Mean y: {mean_y:.2f}\n"
        f"Std Dev x: {std_x:.2f}\n"
        f"Std Dev y: {std_y:.2f}"
    )
    at = AnchoredText(stats_text, loc="upper right", prop=dict(size=8), frameon=True)
    at.patch.set_alpha(0.85)
    ax.add_artist(at)
    
    # Free memory
    del x_np, y_np, mask
    
def compute_metrics(original_df, synthetic_df):
    """
    Compute quality metrics:
    - KS test for each variable
    - Wasserstein distance for each variable
    - Correlation matrix difference
    """
    metrics = {}
    
    # Univariate metrics (process one at a time)
    univariate = []
    for col in original_df.columns:
        x = original_df[col].dropna().values.astype(np.float32)
        y = synthetic_df[col].dropna().values.astype(np.float32)
        
        ks_stat, ks_p = stats.ks_2samp(x, y)
        w_dist = stats.wasserstein_distance(x, y)
        
        univariate.append({
            "variable": col,
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "wasserstein": float(w_dist)
        })
        del x, y
    
    metrics["univariate"] = univariate
    gc.collect()
    
    # Correlation structure (use only upper triangle to save memory)
    corr_orig = original_df.corr().values.astype(np.float32)
    corr_sim = synthetic_df.corr().values.astype(np.float32)
    
    # Only compute diff for upper triangle
    upper_idx = np.triu_indices_from(corr_orig, k=1)
    corr_diff_upper = np.abs(corr_orig[upper_idx] - corr_sim[upper_idx])
    
    metrics["correlation_diff"] = {
        "mean_abs_diff": float(np.mean(corr_diff_upper)),
        "max_abs_diff": float(np.max(corr_diff_upper)),
        "median_abs_diff": float(np.median(corr_diff_upper))
    }
    
    del corr_orig, corr_sim, corr_diff_upper, upper_idx
    gc.collect()
    
    return metrics

def save_metrics_json(metrics, filename):
    """Save metrics dict to JSON file."""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filename}")

def save_dataframe_to_csv(df, filename):
    """Save DataFrame to CSV."""
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def save_plot(fig, filename, dpi=150):
    """Save figure to file."""
    fig.savefig(filename, bbox_inches='tight', dpi=dpi)
    print(f"Plot saved to {filename}")
    plt.close(fig)

def get_histogram_range(real_data, synthetic_data, percentile=99):
    """
    Compute histogram range based on percentiles of real data.
    Apply the same range to both real and synthetic for consistent comparison.
    
    Parameters:
    - real_data: Array of real data values
    - synthetic_data: Array of synthetic data values
    - percentile: Percentile to use for range (default 99 to exclude 1% outliers)
    
    Returns:
    - tuple: (min_val, max_val) for histogram range
    """
    real_data = np.asarray(real_data, dtype=np.float32)
    real_data = real_data[np.isfinite(real_data)]
    
    if len(real_data) == 0:
        return None
    
    # Compute percentile range from real data only
    p_low = np.percentile(real_data, 100 - percentile)
    p_high = np.percentile(real_data, percentile)
    
    # Add small margin for better visualization (5% padding)
    margin = 0.05 * (p_high - p_low)
    range_min = p_low - margin
    range_max = p_high + margin
    
    return [range_min, range_max]

def plot_training_history(history, output_path, title="Training Loss"):
    """Plot discriminator/critic and generator loss vs epoch."""
    if not history:
        return
    epochs = history.get("epoch", [])
    d_loss = history.get("d_loss", [])
    g_loss = history.get("g_loss", [])
    train_wd = history.get("train_wasserstein", [])
    val_wd = history.get("val_wasserstein", [])
    train_mmd = history.get("train_mmd", [])
    val_mmd = history.get("val_mmd", [])

    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Left plot: D/C and G loss
    if d_loss:
        ax1.plot(epochs, d_loss, label="D/C loss")
    if g_loss:
        ax1.plot(epochs, g_loss, label="G loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Train + Validation distribution metrics
    if train_wd:
        ax2.plot(epochs, train_wd, label="Train WD", color="tab:green", linestyle="--")
    if val_wd:
        ax2.plot(epochs, val_wd, label="Val WD", color="green")
    if train_mmd:
        ax2.plot(epochs, train_mmd, label="Train MMD", color="tab:purple", linestyle="--")
    if val_mmd:
        ax2.plot(epochs, val_mmd, label="Val MMD", color="purple")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Train/Validation Metrics")
    if train_wd or val_wd or train_mmd or val_mmd:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to {output_path}")

def compute_c2st_metrics(real_df, synthetic_df, device="cpu", max_samples=100000,
                         epochs=50, hidden_dim=64, batch_size=1024, seed=42,
                         feature_names=None, importance_repeats=5,
                         importance_max_samples=20000):
    """
    Compute Classifier Two-Sample Test (C2ST) metrics.
    
    Trains a simple MLP to distinguish real from synthetic samples. If the classifier 
    achieves ~0.5 accuracy (random guessing), the synthetic data is indistinguishable 
    from real data (good fit). If accuracy >> 0.5, the classifier easily separates them 
    (poor fit).
    
    Parameters:
    -----------
    real_df : pd.DataFrame
        Real test data
    synthetic_df : pd.DataFrame
        Synthetic generated data
    device : str
        "cpu" or "cuda"
    max_samples : int
        Max samples per class for training (to control runtime on large jobs)
    epochs : int
        Classifier training epochs
    hidden_dim : int
        Hidden layer dimension for the MLP
    batch_size : int
        Batch size for classifier training
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict : C2ST metrics including accuracy, balanced_accuracy, roc_auc, 
           sample counts, and config details
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    requested_device = str(device)
    resolved_device = requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ C2ST requested CUDA but no GPU driver/device is available; falling back to CPU.")
        resolved_device = "cpu"
    
    # Convert DataFrames to numpy and cap samples
    real_data = real_df.values.astype(np.float32)
    synthetic_data = synthetic_df.values.astype(np.float32)
    
    n_real = min(len(real_data), max_samples)
    n_synthetic = min(len(synthetic_data), max_samples)
    
    real_data = real_data[:n_real]
    synthetic_data = synthetic_data[:n_synthetic]
    
    # Combine and create labels (0 = real, 1 = synthetic)
    combined_data = np.vstack([real_data, synthetic_data])
    labels = np.hstack([np.zeros(n_real, dtype=np.int64), 
                        np.ones(n_synthetic, dtype=np.int64)])
    
    # Shuffle
    idx = np.random.permutation(len(combined_data))
    combined_data = combined_data[idx]
    labels = labels[idx]
    
    # Split: 80% train, 20% test (to avoid optimistic bias)
    split_idx = int(0.8 * len(combined_data))
    train_data = torch.from_numpy(combined_data[:split_idx]).to(resolved_device)
    train_labels = torch.from_numpy(labels[:split_idx]).to(resolved_device)
    test_data = torch.from_numpy(combined_data[split_idx:]).to(resolved_device)
    test_labels = torch.from_numpy(labels[split_idx:]).to(resolved_device)
    
    # Standardize using train stats only
    train_mean = train_data.mean(dim=0)
    train_std = train_data.std(dim=0) + 1e-8
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std
    
    # Build simple MLP classifier
    n_features = real_data.shape[1]
    
    class Classifier(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    classifier = Classifier(n_features, hidden_dim).to(resolved_device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    # Train classifier
    classifier.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size].float().unsqueeze(1)
            
            optimizer.zero_grad()
            logits = classifier(batch_data)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_data)
        
        epoch_loss /= len(train_data)
    
    # Evaluate on test set
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_data).cpu().numpy().squeeze()
        test_labels_np = test_labels.cpu().numpy()
    
    # Compute metrics
    test_preds = (test_logits >= 0.5).astype(int)
    
    accuracy = float(accuracy_score(test_labels_np, test_preds))
    balanced_accuracy = float(balanced_accuracy_score(test_labels_np, test_preds))
    try:
        roc_auc = float(roc_auc_score(test_labels_np, test_logits))
    except ValueError:
        roc_auc = float("nan")
    
    tn, fp, fn, tp = confusion_matrix(test_labels_np, test_preds).ravel()

    # Permutation feature importance on held-out C2ST test set
    if feature_names is None:
        feature_names = list(real_df.columns) if hasattr(real_df, "columns") else [f"feature_{i}" for i in range(n_features)]

    test_data_np = test_data.detach().cpu().numpy().astype(np.float32)
    n_test = len(test_data_np)
    if importance_max_samples is not None and n_test > importance_max_samples:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_test, size=importance_max_samples, replace=False)
        imp_data_np = test_data_np[sel]
        imp_labels_np = test_labels_np[sel]
    else:
        imp_data_np = test_data_np
        imp_labels_np = test_labels_np

    rng = np.random.default_rng(seed)
    feature_importance = []

    for feature_idx in range(n_features):
        score_drops = []
        for _ in range(max(1, int(importance_repeats))):
            permuted = imp_data_np.copy()
            perm_idx = rng.permutation(len(permuted))
            permuted[:, feature_idx] = permuted[perm_idx, feature_idx]

            with torch.no_grad():
                perm_tensor = torch.from_numpy(permuted).to(resolved_device)
                perm_logits = classifier(perm_tensor).cpu().numpy().squeeze()
            perm_preds = (perm_logits >= 0.5).astype(int)
            perm_bal_acc = float(balanced_accuracy_score(imp_labels_np, perm_preds))
            score_drops.append(balanced_accuracy - perm_bal_acc)

        feature_importance.append({
            "feature": str(feature_names[feature_idx]) if feature_idx < len(feature_names) else f"feature_{feature_idx}",
            "importance_mean": float(np.mean(score_drops)),
            "importance_std": float(np.std(score_drops)),
            "importance_repeats": int(max(1, int(importance_repeats)))
        })

    feature_importance.sort(key=lambda item: item["importance_mean"], reverse=True)
    
    # Clean up
    del train_data, train_labels, test_data, test_labels, classifier
    del combined_data, labels
    gc.collect()
    torch.cuda.empty_cache() if str(resolved_device).startswith("cuda") else None
    
    c2st_metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "sample_counts": {
            "real": int(n_real),
            "synthetic": int(n_synthetic)
        },
        "config": {
            "max_samples_per_class": max_samples,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "train_test_split": 0.8,
            "importance_repeats": int(max(1, int(importance_repeats))),
            "importance_max_samples": importance_max_samples,
            "device_requested": requested_device,
            "device_used": resolved_device,
        },
        "feature_importance": feature_importance,
        "interpretation": (
            "Accuracy near 0.5 indicates synthetic and real data are indistinguishable (good fit). "
            "Accuracy near 1.0 indicates classifier easily separates them (poor fit). "
            "ROC-AUC near 0.5 is better (synthetic ≈ real); ROC-AUC near 1.0 is worse."
        )
    }
    
    return c2st_metrics

def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @ (sigma2 + eps * np.eye(sigma2.shape[0])), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def _batched_embeddings(df: pd.DataFrame, embedder: torch.nn.Module, device: str, batch_size: int) -> np.ndarray:
    x = torch.tensor(df.values, dtype=torch.float32, device=device)
    out = []
    embedder.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            z = embedder(x[i:i + batch_size])
            if isinstance(z, (tuple, list)):
                z = z[0]
            out.append(z.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_fpd(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    embedder: torch.nn.Module,
    device: str = "cpu",
    max_samples: int = 50000,
    batch_size: int = 4096,
    seed: int = 42,
) -> dict:
    n = min(len(real_df), len(synthetic_df), max_samples)
    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real_df), size=n, replace=False)
    synth_idx = rng.choice(len(synthetic_df), size=n, replace=False)

    real_emb = _batched_embeddings(real_df.iloc[real_idx], embedder, device, batch_size)
    synth_emb = _batched_embeddings(synthetic_df.iloc[synth_idx], embedder, device, batch_size)

    mu_r, mu_s = real_emb.mean(axis=0), synth_emb.mean(axis=0)
    cov_r, cov_s = np.cov(real_emb, rowvar=False), np.cov(synth_emb, rowvar=False)

    return {
        "fpd": _frechet_distance(mu_r, cov_r, mu_s, cov_s),
        "n_real": int(n),
        "n_synthetic": int(n),
        "embedding_dim": int(real_emb.shape[1]),
        "seed": int(seed),
    }


def compute_1nn_loo(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    max_samples: int = 50000,
    seed: int = 42,
    metric: str = "euclidean",
    device: str = "cpu",
    chunk_size: int = 2048,
) -> dict:
    real_arr = _as_finite_float32_2d(real_df.to_numpy(dtype=np.float32, copy=False))
    synth_arr = _as_finite_float32_2d(synthetic_df.to_numpy(dtype=np.float32, copy=False))

    n = min(len(real_arr), len(synth_arr), max_samples)
    if n < 1:
        return {
            "accuracy": float("nan"),
            "accuracy_percent": float("nan"),
            "real_accuracy": float("nan"),
            "synthetic_accuracy": float("nan"),
            "target_for_good_generation": 0.5,
            "interpretation": "Closer to 0.5 means real and synthetic are less distinguishable by 1-NN.",
            "n_per_class": int(n),
            "metric": metric,
            "seed": int(seed),
            "backend": "numpy",
            "device": "cpu",
        }

    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real_arr), size=n, replace=False)
    synth_idx = rng.choice(len(synth_arr), size=n, replace=False)

    use_gpu = "cuda" in str(device).lower() and torch.cuda.is_available() and metric == "euclidean"

    if use_gpu:
        x_real_t = torch.as_tensor(real_arr[real_idx], dtype=torch.float32, device=device)
        x_synth_t = torch.as_tensor(synth_arr[synth_idx], dtype=torch.float32, device=device)

        x_t = torch.cat([x_real_t, x_synth_t], dim=0)
        y_t = torch.cat(
            [
                torch.zeros(n, dtype=torch.long, device=device),
                torch.ones(n, dtype=torch.long, device=device),
            ],
            dim=0,
        )

        total_samples = int(x_t.shape[0])
        nn_idx = torch.empty(total_samples, dtype=torch.long, device=device)

        for start in range(0, total_samples, chunk_size):
            stop = min(total_samples, start + chunk_size)
            query = x_t[start:stop]
            dists_sq = torch.cdist(query, x_t, p=2).pow(2)
            row_idx = torch.arange(stop - start, device=device)
            dists_sq[row_idx, start + row_idx] = float("inf")
            nn_idx[start:stop] = torch.argmin(dists_sq, dim=1)

        nn_label = y_t[nn_idx]
        acc = float((nn_label == y_t).float().mean().item())
        acc_real = float((nn_label[:n] == 0).float().mean().item())
        acc_synth = float((nn_label[n:] == 1).float().mean().item())

        return {
            "accuracy": acc,
            "accuracy_percent": 100.0 * acc,
            "real_accuracy": acc_real,
            "synthetic_accuracy": acc_synth,
            "target_for_good_generation": 0.5,
            "interpretation": "Closer to 0.5 means real and synthetic are less distinguishable by 1-NN.",
            "n_per_class": int(n),
            "metric": metric,
            "seed": int(seed),
            "backend": "torch",
            "device": str(device),
            "chunk_size": int(chunk_size),
        }

    x_real = real_arr[real_idx]
    x_synth = synth_arr[synth_idx]
    x = np.vstack([x_real, x_synth])
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    nn = NearestNeighbors(n_neighbors=2, metric=metric)
    nn.fit(x)
    idx = nn.kneighbors(x, return_distance=False)
    nn_label = y[idx[:, 1]]

    acc = float((nn_label == y).mean())
    acc_real = float((nn_label[:n] == 0).mean())
    acc_synth = float((nn_label[n:] == 1).mean())

    return {
        "accuracy": acc,
        "accuracy_percent": 100.0 * acc,
        "real_accuracy": acc_real,
        "synthetic_accuracy": acc_synth,
        "target_for_good_generation": 0.5,
        "interpretation": "Closer to 0.5 means real and synthetic are less distinguishable by 1-NN.",
        "n_per_class": int(n),
        "metric": metric,
        "seed": int(seed),
        "backend": "numpy",
        "device": "cpu",
    }