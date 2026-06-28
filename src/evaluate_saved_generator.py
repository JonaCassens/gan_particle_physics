import argparse
import glob
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from contextlib import nullcontext

from data_loader import load_preprocessed_data
from models.gan_model import Generator as GANGenerator, ParticleDataset as GANParticleDataset
from utils import (
    compute_c2st_metrics,
    compute_metrics,
    compute_mmd_rbf,
    save_metrics_json,
    compute_fpd,
    compute_1nn_loo,
    generate_synthetic_from_checkpoint,
)
from models.wgan_model import Generator as WGANGenerator, ParticleDataset as WGANParticleDataset
from models.wgan_gp_model import (
    Generator as WGANGPGenerator,
    ParticleDataset as WGANGPParticleDataset,
    _apply_generation_bounds,
    BOUNDED_CLIP_FEATURES,
    PDG_MASS_MEV,
)
from models.cwgan_gp_model import CGenerator as CWGANGPGenerator


CONFIG_LINE_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\.*\s+(.*?)\s*$")

GAN_RESULTS_ROOT = "/home/hep/jcc525/gan_particle_physics/gan_results"
LOGS_DIR = "/home/hep/jcc525/gan_particle_physics/condor/logs"
GENERATOR_FILENAME = "generator.pth"

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_MODE = "auto"
DEFAULT_BATCH_SIZE = 32768

DEFAULT_SKIP_C2ST = False
DEFAULT_C2ST_MAX_SAMPLES = 200000
DEFAULT_C2ST_EPOCHS = 20
DEFAULT_C2ST_HIDDEN_DIM = 128
DEFAULT_C2ST_SEED = 42
DEFAULT_C2ST_EXCLUDE_FEATURES = "x"
DEFAULT_C2ST_IMPORTANCE_REPEATS = 5
DEFAULT_C2ST_IMPORTANCE_MAX_SAMPLES = 50000

DEFAULT_MMD_MAX_SAMPLES = 200000
DEFAULT_MMD_CHUNK_SIZE = 512
DEFAULT_MMD_SEED = 42

DEFAULT_FPD_MAX_SAMPLES = 50000
DEFAULT_FPD_BATCH_SIZE = 4096
DEFAULT_1NN_MAX_SAMPLES = 200000

BASELINE_ENABLED = True
BASELINE_SLICE_SEED_A = 101
BASELINE_SLICE_SEED_B = 202

# Set this only for checkpoints whose generator output schema differs from the
# currently loaded parquet schema. The order must match the generator output.
# Example:
# GENERATOR_OUTPUT_COLUMNS = ["x", "r", "sin_phi_s", "cos_phi_s", "sin_theta", "cos_theta", "phi_p", "p_mag", "z"]
GENERATOR_OUTPUT_COLUMNS: Optional[list[str]] = None


def _validate_run_dir_name(run_dir_name: str) -> str:
    normalized = run_dir_name.strip()
    if not normalized:
        raise ValueError("`--run-dir` must be a non-empty folder name under gan_results.")
    if normalized in {".", ".."}:
        raise ValueError("`--run-dir` must be a folder name, not '.' or '..'.")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("`--run-dir` must be only a folder name (no path separators).")
    return normalized


def _clear_memory(device: str = "cpu") -> None:
    import gc

    gc.collect()
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _convert_config_value(raw_value: str):
    value = raw_value.strip()
    if value in {"True", "False"}:
        return value == "True"
    if value in {"None", "null"}:
        return None

    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _find_matching_run_log(run_name: str, logs_dir: str) -> Optional[str]:
    log_paths = sorted(glob.glob(os.path.join(logs_dir, "*.out")), key=os.path.getmtime, reverse=True)
    needle = f"output_name.............. {run_name}"

    for log_path in log_paths:
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
                if needle in handle.read():
                    return log_path
        except OSError:
            continue
    return None


def _parse_run_log_config(log_path: str) -> Tuple[dict, Optional[str]]:
    config: dict[str, object] = {}
    split_mode = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    in_config = False
    for line in lines:
        stripped = line.rstrip("\n")

        if stripped.strip() == "Configuration:":
            in_config = True
            continue

        if in_config:
            if stripped.startswith("============================================================") or not stripped.strip():
                if config:
                    in_config = False
                continue

            match = CONFIG_LINE_RE.match(stripped)
            if match:
                key, raw_value = match.groups()
                config[key] = _convert_config_value(raw_value)

        if stripped.startswith("Testing on "):
            split_mode = "shuffled" if "(shuffled)" in stripped else "sequential"

    return config, split_mode


def _sequential_train_test_split(df: pd.DataFrame, entries: Optional[int], test_entries: Optional[int]):
    train_end = entries if entries is not None else len(df)
    test_start = train_end
    default_test_entries = int(train_end * 0.2)
    test_end = test_start + (test_entries if test_entries is not None else default_test_entries)

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _shuffled_train_test_split(df: pd.DataFrame, n_train: int, n_test: int, seed: int = 42):
    n_total = len(df)
    if n_train + n_test > n_total:
        raise ValueError(f"Requested n_train+n_test={n_train + n_test} > total rows={n_total}")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_total)
    train_idx = permutation[:n_train]
    test_idx = permutation[n_train:n_train + n_test]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def _resolve_split(
    df: pd.DataFrame,
    split_mode: str,
    entries: Optional[int],
    test_entries: Optional[int],
    split_seed: int,
):
    if split_mode == "sequential":
        return _sequential_train_test_split(df, entries=entries, test_entries=test_entries)

    n_total = len(df)
    requested_n_train = entries if entries is not None else int(0.9 * n_total)
    requested_n_test = test_entries if test_entries is not None else max(1, n_total - requested_n_train)

    if requested_n_train + requested_n_test > n_total:
        n_test = max(1, int(0.1 * n_total)) if n_total > 0 else 0
        n_train = max(0, n_total - n_test)
        print(
            f"[split-adjusted] Requested train+test={requested_n_train + requested_n_test:,} exceeds "
            f"available rows={n_total:,}. Using fallback split train={n_train:,}, test={n_test:,} (10% test)."
        )
    else:
        n_train = min(max(0, requested_n_train), n_total)
        n_test = min(max(0, requested_n_test), n_total - n_train)

    return _shuffled_train_test_split(df, n_train=n_train, n_test=n_test, seed=split_seed)


def _normalize_model_type(raw_model_type: object, inferred_model_type: str) -> str:
    value = str(raw_model_type).strip().lower() if raw_model_type is not None else ""
    if value in {"gan", "wgan", "wgan-gp", "cwgan-gp"}:
        return value

    if inferred_model_type == "cwgan-gp":
        return "cwgan-gp"

    if inferred_model_type == "wgan-gp":
        return "wgan-gp"

    # Ambiguous between gan and wgan when inferred from batchnorm-only architecture.
    # Default to gan for backward compatibility if logs are unavailable.
    return "gan"


def _compute_normalization_stats(train_df: pd.DataFrame, model_type: str) -> tuple[np.ndarray, np.ndarray]:
    if model_type in {"wgan-gp", "cwgan-gp"}:
        dataset = WGANGPParticleDataset(train_df)
    elif model_type == "wgan":
        dataset = WGANParticleDataset(train_df)
    else:
        dataset = GANParticleDataset(train_df)

    mean = dataset.mean.detach().cpu().numpy().astype(np.float32)
    std = dataset.std.detach().cpu().numpy().astype(np.float32)
    return mean, std


def _create_generator(
    model_type: str,
    latent_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    normalization: str,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> nn.Module:
    if model_type == "cwgan-gp":
        if state_dict is None:
            raise ValueError("state_dict is required to construct cwgan-gp generator")

        embedding_weight = state_dict.get("embedding.weight")
        if embedding_weight is None:
            raise ValueError("cwgan-gp checkpoint missing generator embedding.weight")

        vocab_size = int(embedding_weight.shape[0])
        embed_dim = int(embedding_weight.shape[1])
        return CWGANGPGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
        )

    if model_type == "wgan-gp":
        return WGANGPGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )

    if model_type == "wgan":
        return WGANGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )

    if model_type == "gan":
        return GANGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )

    # Fallback path for unknown/legacy model labels
    return InferredGenerator(
        latent_dim=latent_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        normalization=normalization,
    )


def _infer_generator_architecture(state_dict: dict[str, torch.Tensor]) -> Dict[str, Any]:
    linear_weights: List[Tuple[int, str, Tuple[int, int]]] = []
    for key, tensor in state_dict.items():
        if tensor.ndim == 2 and key.startswith("model.") and key.endswith(".weight"):
            layer_index = int(key.split(".")[1])
            out_dim = int(tensor.shape[0])
            in_dim = int(tensor.shape[1])
            linear_weights.append((layer_index, key, (out_dim, in_dim)))

    linear_weights.sort(key=lambda item: item[0])
    if not linear_weights:
        raise ValueError("Could not infer generator architecture from checkpoint state_dict.")

    hidden_dims = [shape[0] for _, _, shape in linear_weights[:-1]]
    latent_dim = linear_weights[0][2][1]
    output_dim = linear_weights[-1][2][0]

    normalization = "batchnorm" if any("running_mean" in key for key in state_dict) else "layernorm"
    has_embedding = "embedding.weight" in state_dict
    if has_embedding:
        model_type = "cwgan-gp"
    else:
        model_type = "wgan-gp" if normalization == "layernorm" else "gan-or-wgan"

    return {
        "latent_dim": latent_dim,
        "output_dim": output_dim,
        "hidden_dims": hidden_dims,
        "normalization": normalization,
        "model_type": model_type,
    }


class InferredGenerator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list[int], normalization: str):
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model(latent)


def _resolve_generation_feature_names(
    eval_feature_names: list[str],
    generated_width: int,
) -> list[str]:
    if GENERATOR_OUTPUT_COLUMNS is not None:
        if len(GENERATOR_OUTPUT_COLUMNS) != generated_width:
            raise ValueError(
                f"GENERATOR_OUTPUT_COLUMNS has length {len(GENERATOR_OUTPUT_COLUMNS)}, "
                f"but generator produced {generated_width} features."
            )
        return list(GENERATOR_OUTPUT_COLUMNS)

    if generated_width == len(eval_feature_names):
        return list(eval_feature_names)

    if generated_width == len(eval_feature_names) + 1 and "x" not in eval_feature_names:
        return ["x", *eval_feature_names]

    # Generator was trained without `x`; eval data includes it — drop it from the name list.
    if generated_width == len(eval_feature_names) - 1 and "x" in eval_feature_names:
        names_without_x = [c for c in eval_feature_names if c != "x"]
        print(
            f"[INFO] Generator produced {generated_width} features; eval schema has "
            f"{len(eval_feature_names)} columns including 'x'. "
            f"Assuming generator does not output 'x'. Using: {names_without_x}"
        )
        return names_without_x

    raise ValueError(
        f"Generator produced {generated_width} features, but evaluation data has "
        f"{len(eval_feature_names)} columns: {eval_feature_names}. "
        f"If this checkpoint uses a different schema, set GENERATOR_OUTPUT_COLUMNS."
    )


def _align_generated_samples(
    samples: np.ndarray,
    generation_feature_names: list[str],
    eval_feature_names: list[str],
) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError(f"Expected 2D generated samples, got shape {samples.shape}")

    if len(generation_feature_names) != samples.shape[1]:
        raise ValueError(
            f"generation_feature_names has length {len(generation_feature_names)}, "
            f"but generated samples have width {samples.shape[1]}."
        )

    if generation_feature_names == eval_feature_names:
        return samples.astype(np.float32, copy=False)

    generated_df = pd.DataFrame(samples, columns=generation_feature_names)
    missing = [c for c in eval_feature_names if c not in generated_df.columns]
    if missing:
        raise ValueError(
            f"Evaluation columns missing from generator output mapping: {missing}. "
            f"generation_feature_names={generation_feature_names}"
        )

    return generated_df.loc[:, eval_feature_names].to_numpy(dtype=np.float32, copy=False)


def _generate_synthetic_samples(
    generator: nn.Module,
    n_samples: int,
    latent_dim: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int,
    feature_names: list[str],
    apply_angle_clipping: bool,
    conditional_pdg_codes: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, list[str]]:
    generator.eval()
    generation_feature_names: Optional[list[str]] = None
    # will be set after first batch once generation_feature_names is resolved
    mean_gen: Optional[np.ndarray] = None
    std_gen: Optional[np.ndarray] = None
    out: Optional[np.ndarray] = None

    use_amp = "cuda" in str(device).lower() and torch.cuda.is_available()

    pdg_to_idx: dict[int, int] = {}
    unknown_idx = 0
    if conditional_pdg_codes is not None:
        cond_codes = np.asarray(conditional_pdg_codes, dtype=np.int64)
        unique_codes = sorted(set(int(code) for code in cond_codes.tolist()))
        embedding_weight = getattr(generator, "embedding", None)
        if embedding_weight is None or not hasattr(embedding_weight, "weight"):
            raise ValueError("conditional_pdg_codes provided but generator has no embedding layer")

        vocab_capacity = int(embedding_weight.weight.shape[0])
        if vocab_capacity < 2:
            raise ValueError("cwgan embedding vocab size must be at least 2")

        max_known = vocab_capacity - 1
        known_codes = unique_codes[:max_known]
        pdg_to_idx = {code: idx for idx, code in enumerate(known_codes)}
        unknown_idx = max_known

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - start)
            latent = torch.randn(current_batch_size, latent_dim, device=device)

            autocast_context = getattr(torch.cuda.amp, "autocast")(dtype=torch.float16) if use_amp else nullcontext()
            with autocast_context:
                if conditional_pdg_codes is not None:
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
                )
                # Build mean/std aligned to what the generator actually outputs.
                # If the generator omits a feature (e.g. 'x'), slice it out.
                if generation_feature_names == feature_names:
                    mean_gen = mean
                    std_gen = std
                else:
                    feat_index = {name: i for i, name in enumerate(feature_names)}
                    gen_indices = [feat_index[n] for n in generation_feature_names if n in feat_index]
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
                clip_feature_indices = {name: idx for idx, name in enumerate(generation_feature_names) if name in BOUNDED_CLIP_FEATURES}
                chunk = _apply_generation_bounds(chunk, clip_feature_indices)

            out[start:start + current_batch_size] = chunk.astype(np.float32, copy=False)

    assert out is not None and generation_feature_names is not None, "No samples were generated."
    return out, generation_feature_names


def _default_output_json(run_dir: str, generator_path: str) -> str:
    generator_name = Path(generator_path).stem
    return os.path.join(run_dir, f"{generator_name}_external_test_metrics.json")


def _make_real_vs_real_slices(
    df: pd.DataFrame,
    target_size: int,
    seed_a: int,
    seed_b: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    n_total = len(df)
    if n_total < 2:
        raise ValueError("Need at least 2 rows to compute real-vs-real baseline.")

    max_disjoint_size = max(1, n_total // 2)
    slice_size = max(1, min(target_size, max_disjoint_size))

    max_start = n_total - slice_size
    start_a = 0 if max_start == 0 else int(0.1 * max_start)
    start_b = 0 if max_start == 0 else int(0.7 * max_start)

    if start_b < start_a + slice_size:
        start_b = min(max_start, start_a + slice_size)

    if start_b < start_a + slice_size:
        start_a = 0
        start_b = n_total - slice_size

    if start_b < start_a + slice_size:
        slice_size = max(1, n_total // 2)
        start_a = 0
        start_b = n_total - slice_size

    real_a = df.iloc[start_a:start_a + slice_size].copy().reset_index(drop=True)
    real_b = df.iloc[start_b:start_b + slice_size].copy().reset_index(drop=True)

    perm_a = np.random.default_rng(seed_a).permutation(len(real_a))
    perm_b = np.random.default_rng(seed_b).permutation(len(real_b))
    real_a = real_a.iloc[perm_a].reset_index(drop=True)
    real_b = real_b.iloc[perm_b].reset_index(drop=True)

    metadata = {
        "n_total": n_total,
        "slice_size": slice_size,
        "slice_a_start": int(start_a),
        "slice_a_end": int(start_a + slice_size),
        "slice_b_start": int(start_b),
        "slice_b_end": int(start_b + slice_size),
        "shuffle_seed_a": seed_a,
        "shuffle_seed_b": seed_b,
        "construction": "two unshuffled slices at different offsets, then shuffled independently",
    }
    return real_a, real_b, metadata


def _build_delta_vs_baseline(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    baseline = metrics.get("real_vs_real_baseline")
    if not isinstance(baseline, dict):
        return None

    synth_mmd = ((metrics.get("mmd") or {}).get("mmd"))
    base_mmd = ((baseline.get("mmd") or {}).get("mmd"))

    synth_c2st = metrics.get("c2st") or {}
    base_c2st = baseline.get("c2st") or {}

    synth_acc = synth_c2st.get("accuracy")
    base_acc = base_c2st.get("accuracy")
    synth_bal = synth_c2st.get("balanced_accuracy")
    base_bal = base_c2st.get("balanced_accuracy")
    synth_auc = synth_c2st.get("roc_auc")
    base_auc = base_c2st.get("roc_auc")

    mmd_delta = None
    mmd_ratio = None
    if synth_mmd is not None and base_mmd is not None:
        mmd_delta = float(synth_mmd) - float(base_mmd)
        if abs(float(base_mmd)) > 1e-12:
            mmd_ratio = float(synth_mmd) / float(base_mmd)

    result = {
        "synth_vs_real": {
            "mmd": synth_mmd,
            "c2st_accuracy": synth_acc,
            "c2st_balanced_accuracy": synth_bal,
            "c2st_roc_auc": synth_auc,
        },
        "real_vs_real_baseline": {
            "mmd": base_mmd,
            "c2st_accuracy": base_acc,
            "c2st_balanced_accuracy": base_bal,
            "c2st_roc_auc": base_auc,
        },
        "deltas": {
            "mmd_delta": mmd_delta,
            "mmd_ratio": mmd_ratio,
            "c2st_accuracy_delta": None if (synth_acc is None or base_acc is None) else float(synth_acc) - float(base_acc),
            "c2st_balanced_accuracy_delta": None if (synth_bal is None or base_bal is None) else float(synth_bal) - float(base_bal),
            "c2st_roc_auc_delta": None if (synth_auc is None or base_auc is None) else float(synth_auc) - float(base_auc),
        },
        "interpretation": "Deltas near 0 indicate synthetic-vs-real separability is close to real-vs-real baseline; larger positive deltas indicate synthetic data is easier to distinguish.",
    }
    return result


def _print_section(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}")
    print(title)
    print(line)


def _print_key_value_block(title: str, values: Dict[str, Any]) -> None:
    _print_section(title)
    if not values:
        print("  (no values)")
        return

    key_width = max(len(str(key)) for key in values.keys()) + 2
    for key, value in values.items():
        print(f"  {str(key):.<{key_width + 10}} {value}")


def _fmt_float(value: Any, digits: int = 6) -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _print_univariate_summary(univariate_rows: List[Dict[str, Any]]) -> None:
    _print_section("STEP 4A: Univariate Metrics")
    if not univariate_rows:
        print("No univariate metrics available.")
        return

    df_uni = pd.DataFrame(univariate_rows)
    preferred_columns = ["variable", "ks_stat", "ks_p", "wasserstein"]
    visible_columns = [col for col in preferred_columns if col in df_uni.columns]
    print(df_uni[visible_columns].to_string(index=False))

    if "wasserstein" in df_uni.columns:
        worst = df_uni.sort_values("wasserstein", ascending=False).head(3)
        best = df_uni.sort_values("wasserstein", ascending=True).head(3)

        print("\nTop 3 largest Wasserstein features:")
        for _, row in worst.iterrows():
            print(f"  - {row['variable']}: {_fmt_float(row['wasserstein'])}")

        print("\nTop 3 smallest Wasserstein features:")
        for _, row in best.iterrows():
            print(f"  - {row['variable']}: {_fmt_float(row['wasserstein'])}")


def _print_mmd_summary(mmd_metrics: Dict[str, Any], label: str = "Synth-vs-Real") -> None:
    if not mmd_metrics:
        return

    print(f"\n{label} MMD summary:")
    print(f"  - MMD:      {_fmt_float(mmd_metrics.get('mmd'))}")
    print(f"  - MMD^2:    {_fmt_float(mmd_metrics.get('mmd2'))}")
    print(f"  - Sigma:    {_fmt_float(mmd_metrics.get('sigma'))}")
    print(f"  - Gamma:    {_fmt_float(mmd_metrics.get('gamma'))}")
    print(f"  - n_real:   {mmd_metrics.get('n_real')}")
    print(f"  - n_synth:  {mmd_metrics.get('n_synthetic')}")
    print(f"  - max_samp: {mmd_metrics.get('max_samples')}")
    print(f"  - chunk:    {mmd_metrics.get('chunk_size')}")
    print(f"  - unbiased: {mmd_metrics.get('unbiased')}")


def _print_c2st_summary(c2st_metrics: Dict[str, Any], title: str = "C2ST") -> None:
    if not c2st_metrics:
        return

    print(f"\n{title} summary:")
    print(f"  - Accuracy:          {_fmt_float(c2st_metrics.get('accuracy'), digits=4)}")
    print(f"  - Balanced Accuracy: {_fmt_float(c2st_metrics.get('balanced_accuracy'), digits=4)}")
    print(f"  - ROC-AUC:           {_fmt_float(c2st_metrics.get('roc_auc'), digits=4)}")

    confusion = c2st_metrics.get("confusion_matrix") or {}
    if confusion:
        print(
            "  - Confusion Matrix: "
            f"TN={confusion.get('true_negatives')}, "
            f"FP={confusion.get('false_positives')}, "
            f"FN={confusion.get('false_negatives')}, "
            f"TP={confusion.get('true_positives')}"
        )

    counts = c2st_metrics.get("sample_counts") or {}
    if counts:
        print(f"  - Sample Counts: real={counts.get('real')}, synthetic={counts.get('synthetic')}")

    importances = c2st_metrics.get("feature_importance") or []
    if importances:
        top = importances[:5]
        print("  - Top Feature Importances:")
        for item in top:
            print(
                "    * "
                f"{item.get('feature')}: "
                f"mean={_fmt_float(item.get('importance_mean'), digits=4)}, "
                f"std={_fmt_float(item.get('importance_std'), digits=4)}"
            )

    interpretation = c2st_metrics.get("interpretation")
    if interpretation:
        print(f"  - Interpretation: {interpretation}")


def _print_baseline_and_delta_summary(metrics: Dict[str, Any]) -> None:
    baseline = metrics.get("real_vs_real_baseline") or {}
    delta = metrics.get("delta_vs_real_baseline") or {}

    if not baseline and not delta:
        return

    _print_section("STEP 4C: Real-vs-Real Baseline and Delta Analysis")

    if baseline:
        meta = baseline.get("slice_metadata") or {}
        if meta:
            print("Baseline slice metadata:")
            print(
                "  "
                f"n_total={meta.get('n_total')}, "
                f"slice_size={meta.get('slice_size')}, "
                f"A:[{meta.get('slice_a_start')}, {meta.get('slice_a_end')}), "
                f"B:[{meta.get('slice_b_start')}, {meta.get('slice_b_end')})"
            )

        _print_mmd_summary(baseline.get("mmd") or {}, label="Real-vs-Real")
        _print_c2st_summary(baseline.get("c2st") or {}, title="Real-vs-Real C2ST")

    if delta:
        deltas = delta.get("deltas") or {}
        if deltas:
            print("\nDelta vs baseline:")
            print(f"  - mmd_delta:                 {_fmt_float(deltas.get('mmd_delta'))}")
            print(f"  - mmd_ratio:                 {_fmt_float(deltas.get('mmd_ratio'))}")
            print(f"  - c2st_accuracy_delta:       {_fmt_float(deltas.get('c2st_accuracy_delta'))}")
            print(f"  - c2st_bal_accuracy_delta:   {_fmt_float(deltas.get('c2st_balanced_accuracy_delta'))}")
            print(f"  - c2st_roc_auc_delta:        {_fmt_float(deltas.get('c2st_roc_auc_delta'))}")

        interpretation = delta.get("interpretation")
        if interpretation:
            print(f"  - Interpretation: {interpretation}")

def _print_fpd_summary(fpd_metrics: Dict[str, Any]) -> None:
    if not fpd_metrics:
        return

    print("\nFPD summary:")
    print(f"  - FPD:           {_fmt_float(fpd_metrics.get('fpd'))}")
    print(f"  - n_real:        {fpd_metrics.get('n_real')}")
    print(f"  - n_synthetic:   {fpd_metrics.get('n_synthetic')}")
    print(f"  - embedding_dim: {fpd_metrics.get('embedding_dim')}")
    print(f"  - seed:          {fpd_metrics.get('seed')}")


def _print_1nn_summary(one_nn_metrics: Dict[str, Any]) -> None:
    if not one_nn_metrics:
        return

    print("\n1-NN LOO summary:")
    print(f"  - Accuracy:            {_fmt_float(one_nn_metrics.get('accuracy'), digits=4)}")
    print(f"  - Accuracy (%):        {_fmt_float(one_nn_metrics.get('accuracy_percent'), digits=2)}")
    print(f"  - Real accuracy:       {_fmt_float(one_nn_metrics.get('real_accuracy'), digits=4)}")
    print(f"  - Synthetic accuracy:  {_fmt_float(one_nn_metrics.get('synthetic_accuracy'), digits=4)}")
    print(f"  - Target (ideal):      {one_nn_metrics.get('target_for_good_generation')}")
    print(f"  - n_per_class:         {one_nn_metrics.get('n_per_class')}")
    print(f"  - metric:              {one_nn_metrics.get('metric')}")
    print(f"  - backend/device:      {one_nn_metrics.get('backend')} / {one_nn_metrics.get('device')}")


def _load_torchscript_embedder(model_path: str, device: str) -> nn.Module:
    model = getattr(torch.jit, "load")(model_path, map_location=device)
    model.eval()
    return model


def _fmt_md(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _save_metrics_markdown(metrics: Dict[str, Any], output_path: str) -> None:
    lines = [
        "# Evaluation Metrics Report",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]

    # --- Wasserstein distances ---
    univariate = metrics.get("univariate") or []
    if univariate:
        lines += [
            "## Wasserstein Distance (per feature)",
            "",
            "| Feature | Wasserstein |",
            "|---------|-------------|",
        ]
        for row in sorted(univariate, key=lambda r: r.get("wasserstein", 0), reverse=True):
            lines.append(f"| {row['variable']} | {_fmt_md(row.get('wasserstein'))} |")
        lines.append("")

    # --- MMD (dict form: {"mmd": float, ...}) ---
    mmd_dict = metrics.get("mmd") or {}
    mmd_val = mmd_dict.get("mmd") if isinstance(mmd_dict, dict) else mmd_dict
    if mmd_val is not None:
        lines += [
            "## MMD (RBF kernel)",
            "",
            f"**MMD:** {_fmt_md(mmd_val, digits=6)}",
            "",
        ]

    # --- C2ST (nested dict) ---
    c2st = metrics.get("c2st") or {}
    acc = c2st.get("accuracy")
    bal_acc = c2st.get("balanced_accuracy")
    roc_auc = c2st.get("roc_auc")
    if any(v is not None for v in [acc, bal_acc, roc_auc]):
        lines += [
            "## C2ST",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {_fmt_md(acc)} |",
            f"| Balanced Accuracy | {_fmt_md(bal_acc)} |",
            f"| ROC-AUC | {_fmt_md(roc_auc)} |",
            "",
        ]
        feature_importance = c2st.get("feature_importance") or []
        if feature_importance:
            lines += [
                "### Feature Importance",
                "",
                "| Feature | Importance (mean ± std) |",
                "|---------|------------------------|",
            ]
            for fi in feature_importance:
                mean = _fmt_md(fi.get("importance_mean"))
                std = _fmt_md(fi.get("importance_std"))
                lines.append(f"| {fi['feature']} | {mean} ± {std} |")
            lines.append("")

    # --- Baseline comparison ---
    baseline = metrics.get("real_vs_real_baseline") or {}
    delta = (metrics.get("delta_vs_real_baseline") or {}).get("delta") or {}
    if baseline:
        b_mmd = (baseline.get("mmd") or {}).get("mmd")
        b_c2st = baseline.get("c2st") or {}
        lines += [
            "## Real-vs-Real Baseline",
            "",
            "| Metric | Synth vs Real | Real vs Real |",
            "|--------|--------------|-------------|",
            f"| MMD | {_fmt_md(mmd_val, digits=6)} | {_fmt_md(b_mmd, digits=6)} |",
            f"| C2ST Accuracy | {_fmt_md(acc)} | {_fmt_md(b_c2st.get('accuracy'))} |",
            f"| ROC-AUC | {_fmt_md(roc_auc)} | {_fmt_md(b_c2st.get('roc_auc'))} |",
            "",
        ]

    # --- Per-PDG summary (cwgan-gp) ---
    per_pdg_summary = metrics.get("per_pdg_summary") or {}
    if per_pdg_summary:
        pdg_codes = sorted(per_pdg_summary.keys(), key=lambda x: int(x) if str(x).lstrip("-").isdigit() else 0)
        lines += [
            "## Per-PDG Summary",
            "",
            "| PDG | N Real | N Synth | MMD |",
            "|-----|--------|---------|-----|",
        ]
        for pdg in pdg_codes:
            m = per_pdg_summary[pdg]
            if m.get("status") == "skipped_empty_slice":
                lines.append(f"| {pdg} | {m.get('n_real', '—')} | {m.get('n_synthetic', '—')} | skipped |")
            else:
                lines.append(
                    f"| {pdg} | {m.get('n_real', '—')} | {m.get('n_synthetic', '—')} "
                    f"| {_fmt_md(m.get('mmd'), digits=6)} |"
                )
        lines.append("")

        # Load per-PDG C2ST feature importance from individual metric files if available
        per_pdg_root = metrics.get("per_pdg_root")
        if per_pdg_root:
            any_importance = False
            pdg_importance_data: Dict[str, List] = {}
            for pdg in pdg_codes:
                pdg_json = os.path.join(per_pdg_root, f"pdg_{pdg}", "metrics.json")
                if os.path.isfile(pdg_json):
                    try:
                        with open(pdg_json) as f:
                            pdg_full = json.load(f)
                        fi = (pdg_full.get("c2st") or {}).get("feature_importance") or []
                        if fi:
                            pdg_importance_data[str(pdg)] = fi
                            any_importance = True
                    except Exception:
                        pass
            if any_importance:
                lines.append("### Per-PDG Feature Importance")
                lines.append("")
                for pdg in pdg_codes:
                    fi_list = pdg_importance_data.get(str(pdg)) or []
                    if not fi_list:
                        continue
                    lines += [
                        f"#### PDG {pdg}",
                        "",
                        "| Feature | Importance (mean ± std) |",
                        "|---------|------------------------|",
                    ]
                    for fi in fi_list:
                        mean = _fmt_md(fi.get("importance_mean"))
                        std = _fmt_md(fi.get("importance_std"))
                        lines.append(f"| {fi['feature']} | {mean} ± {std} |")
                    lines.append("")

    Path(output_path).write_text("\n".join(lines) + "\n")
    print(f"Saved metrics report to: {output_path.replace('.json', '.md') if output_path.endswith('.json') else output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Externally evaluate a saved generator checkpoint on test metrics.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Folder name under /home/hep/jcc525/gan_particle_physics/gan_results.",
    )
    parser.add_argument("--pdg", type=int, default=None, help="PDG code; auto-loaded from logs when possible.")
    parser.add_argument("--monitor-id", type=int, default=4, help="MonitorID; auto-loaded from logs when possible.")
    parser.add_argument(
        "--test-entries",
        type=int,
        default=None,
        help="Override test entries from logs. If omitted, keeps log/default behavior.",
    )
    parser.add_argument(
        "--fpd-model",
        type=str,
        default=None,
        help="Path to TorchScript physics embedder (e.g. ParticleNet feature extractor).",
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        default=None,
        help="Comma-separated list of feature columns to exclude from ALL metric evaluation (e.g. 'x' or 'x,log_t').",
    )
    args = parser.parse_args()

    run_dir_name = _validate_run_dir_name(args.run_dir)
    run_dir = os.path.abspath(os.path.join(GAN_RESULTS_ROOT, run_dir_name))
    generator_path = os.path.join(run_dir, GENERATOR_FILENAME)

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found under gan_results: {run_dir}")
    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {generator_path}")

    run_name = os.path.basename(run_dir)

    log_path = _find_matching_run_log(run_name, LOGS_DIR)
    log_config, log_split_mode = ({}, None) if log_path is None else _parse_run_log_config(log_path)

    state_dict = torch.load(generator_path, map_location="cpu")
    inferred = _infer_generator_architecture(state_dict)

    model_type = _normalize_model_type(log_config.get("model"), inferred["model_type"])
    latent_dim = int(log_config.get("latent_dim") or inferred["latent_dim"])
    resolved_pdg = args.pdg if args.pdg is not None else log_config.get("pdg", None)
    pdg = None if resolved_pdg is None else int(resolved_pdg)
    monitor_id = int(args.monitor_id if args.monitor_id is not None else log_config.get("monitor_id", 4))
    entries = log_config.get("entries")
    log_test_entries = log_config.get("test_entries")

    if entries is not None:
        entries = int(entries)
    if log_test_entries is not None:
        log_test_entries = int(log_test_entries)

    test_entries = int(args.test_entries) if args.test_entries is not None else log_test_entries

    split_mode = DEFAULT_SPLIT_MODE
    if split_mode == "auto":
        split_mode = log_split_mode or "shuffled"

    output_json = os.path.abspath(_default_output_json(run_dir, generator_path))
    data_file = f"/home/hep/jcc525/cleaned_data/pdgNone_monitor{monitor_id}.parquet"

    _print_key_value_block(
        "STEP 0: External Evaluation Configuration",
        {
            "device": DEFAULT_DEVICE,
            "run_dir_name": run_dir_name,
            "resolved_run_dir": run_dir,
            "generator_checkpoint": generator_path,
            "matched_log": log_path or "(not found)",
            "resolved_model_type": model_type,
            "resolved_latent_dim": latent_dim,
            "resolved_split_mode": split_mode,
            "resolved_entries": entries,
            "resolved_test_entries": test_entries,
        },
    )

    _print_section("STEP 1: Loading Data")
    print(f"Data file: {data_file}")
    loader_pdg_code = None if model_type == "cwgan-gp" else pdg
    df = load_preprocessed_data(
        data_file,
        pdg_code=loader_pdg_code,
        entries=entries,
        test_entries=test_entries,
    )
    train_df_raw, test_df_raw = _resolve_split(
        pd.DataFrame(df),
        split_mode=split_mode,
        entries=entries,
        test_entries=test_entries,
        split_seed=DEFAULT_SPLIT_SEED,
    )
    train_df = pd.DataFrame(train_df_raw)
    test_df = pd.DataFrame(test_df_raw)

    if len(test_df) == 0:
        raise ValueError("Resolved test split is empty; provide `--test-entries` or adjust split settings.")

    print(f"Total loaded rows: {len(df):,}")
    print(f"Train rows for normalization: {len(train_df):,}")
    print(f"Test rows for evaluation: {len(test_df):,}")
    print(f"Feature columns ({len(train_df.columns)}): {list(train_df.columns)}")

    _print_section("STEP 2: Preparing Generator and Normalization")
    apply_angle_clipping = model_type in {"wgan-gp", "cwgan-gp"}
    print(f"Apply generation bounds clipping: {apply_angle_clipping}")

    _print_section("STEP 3: Generating Synthetic Samples")
    print(f"Requested synthetic rows: {len(test_df):,}")
    print(f"Generation batch size: {DEFAULT_BATCH_SIZE:,}")
    generation_start_time = time.perf_counter()
    # On-shell projection mass: only for the single-PDG (non-conditional) WGAN path.
    # cwgan mass varies per sample by PDG, so projection is skipped (mass=None).
    onshell_mass_mev = (
        None
        if (model_type == "cwgan-gp" or pdg is None)
        else PDG_MASS_MEV.get(abs(int(pdg)))
    )
    if onshell_mass_mev is not None:
        print(f"On-shell log_t projection enabled: pdg={pdg}, mass={onshell_mass_mev:.6f} MeV")
    generation_result = generate_synthetic_from_checkpoint(
        generator_path=generator_path,
        train_df=train_df,
        n_samples=len(test_df),
        device=DEFAULT_DEVICE,
        batch_size=DEFAULT_BATCH_SIZE,
        model_type=model_type,
        latent_dim=latent_dim,
        apply_angle_clipping=apply_angle_clipping,
        conditional_pdg_codes=(test_df["pdg"].to_numpy(dtype=np.int64) if (model_type == "cwgan-gp" and "pdg" in test_df.columns) else None),
        onshell_mass_mev=onshell_mass_mev,
    )
    synthetic_data = generation_result["samples"]
    generation_feature_names = generation_result["feature_names"]
    inferred = generation_result.get("inferred_architecture", inferred)
    mean = generation_result.get("mean")
    std = generation_result.get("std")
    generation_elapsed_seconds = time.perf_counter() - generation_start_time
    print(f"Normalization mode: {'robust median/IQR' if model_type == 'wgan-gp' else 'mean/std'}")
    if mean is not None and std is not None:
        print(f"Mean vector shape: {mean.shape}")
        print(f"Std vector shape: {std.shape}")
        print(f"Std min/max: {_fmt_float(np.min(std))} / {_fmt_float(np.max(std))}")
    print("Generator loaded and samples generated via shared utility.")
    print(f"Inferred architecture: {inferred}")
    synthetic_df = pd.DataFrame(synthetic_data, columns=generation_feature_names)
    if model_type == "cwgan-gp" and "pdg" in test_df.columns and "pdg" not in synthetic_df.columns:
        synthetic_df.insert(0, "pdg", test_df["pdg"].to_numpy(dtype=np.int64))
    # If the generator omitted features present in the parquet (e.g. 'x'), drop them
    # from the eval dataframes so all downstream comparisons use a common schema.
    if generation_feature_names != list(train_df.columns):
        dropped = [c for c in train_df.columns if c not in generation_feature_names]
        print(f"[INFO] Generator did not produce columns {dropped}; dropping from eval DataFrames.")
        test_df = test_df.drop(columns=dropped, errors="ignore")
        train_df = train_df.drop(columns=dropped, errors="ignore")
    print(f"Generated array shape: {synthetic_data.shape}")
    print(f"Synthetic DataFrame shape: {synthetic_df.shape}")
    print(f"Generation time: {generation_elapsed_seconds:.2f} seconds")

    # Apply user-supplied --exclude-features on top of any schema-driven drops
    if args.exclude_features:
        user_excluded = [f.strip() for f in args.exclude_features.split(",") if f.strip()]
        existing = [c for c in user_excluded if c in test_df.columns or c in synthetic_df.columns]
        if existing:
            print(f"[INFO] --exclude-features: dropping {existing} from all eval DataFrames.")
            test_df = test_df.drop(columns=existing, errors="ignore")
            synthetic_df = synthetic_df.drop(columns=existing, errors="ignore")
            train_df = train_df.drop(columns=existing, errors="ignore")
        missing = [c for c in user_excluded if c not in existing]
        if missing:
            print(f"[WARN] --exclude-features: columns not found and skipped: {missing}")

    _clear_memory(DEFAULT_DEVICE)

    _print_section("STEP 4: Computing Metrics")
    test_metrics: Dict[str, Any] = {}

    if model_type == "cwgan-gp" and "pdg" in test_df.columns:
        print("Computing per-PDG metrics for cwgan-gp...")
        per_pdg_root = os.path.join(run_dir, "per_pdg")
        os.makedirs(per_pdg_root, exist_ok=True)

        pdg_summary: Dict[str, Any] = {}
        unique_pdg_codes = sorted(pd.Series(test_df["pdg"]).dropna().astype(int).unique().tolist())

        for pdg_code in unique_pdg_codes:
            real_slice = test_df[test_df["pdg"] == pdg_code].copy()
            synth_slice = synthetic_df[synthetic_df["pdg"] == pdg_code].copy()

            if len(real_slice) == 0 or len(synth_slice) == 0:
                pdg_summary[str(pdg_code)] = {
                    "status": "skipped_empty_slice",
                    "n_real": int(len(real_slice)),
                    "n_synthetic": int(len(synth_slice)),
                }
                continue

            real_eval = real_slice.drop(columns=["pdg"], errors="ignore")
            synth_eval = synth_slice.drop(columns=["pdg"], errors="ignore")

            pdg_metrics = compute_metrics(real_eval, synth_eval)
            pdg_metrics["mmd"] = compute_mmd_rbf(
                real_eval.values,
                synth_eval.values,
                sigma="median",
                max_samples=DEFAULT_MMD_MAX_SAMPLES,
                chunk_size=DEFAULT_MMD_CHUNK_SIZE,
                unbiased=True,
                seed=DEFAULT_MMD_SEED,
                device=DEFAULT_DEVICE,
            )

            if not DEFAULT_SKIP_C2ST:
                excluded_features = {
                    f.strip() for f in DEFAULT_C2ST_EXCLUDE_FEATURES.split(",") if f.strip()
                }
                c2st_feature_names = [c for c in real_eval.columns if c not in excluded_features]
                if c2st_feature_names:
                    pdg_metrics["c2st"] = compute_c2st_metrics(
                        real_eval[c2st_feature_names],
                        synth_eval[c2st_feature_names],
                        device=DEFAULT_DEVICE,
                        max_samples=DEFAULT_C2ST_MAX_SAMPLES,
                        epochs=DEFAULT_C2ST_EPOCHS,
                        hidden_dim=DEFAULT_C2ST_HIDDEN_DIM,
                        seed=DEFAULT_C2ST_SEED,
                        feature_names=c2st_feature_names,
                        importance_repeats=DEFAULT_C2ST_IMPORTANCE_REPEATS,
                        importance_max_samples=DEFAULT_C2ST_IMPORTANCE_MAX_SAMPLES,
                    )

            if args.fpd_model:
                fpd_embedder = _load_torchscript_embedder(args.fpd_model, DEFAULT_DEVICE)
                pdg_metrics["fpd"] = compute_fpd(
                    real_eval,
                    synth_eval,
                    embedder=fpd_embedder,
                    device=DEFAULT_DEVICE,
                    max_samples=DEFAULT_FPD_MAX_SAMPLES,
                    batch_size=DEFAULT_FPD_BATCH_SIZE,
                    seed=DEFAULT_MMD_SEED,
                )

            pdg_metrics["one_nn_loo"] = compute_1nn_loo(
                real_eval,
                synth_eval,
                max_samples=DEFAULT_1NN_MAX_SAMPLES,
                seed=DEFAULT_MMD_SEED,
                device=DEFAULT_DEVICE,
            )

            pdg_dir = os.path.join(per_pdg_root, f"pdg_{pdg_code}")
            os.makedirs(pdg_dir, exist_ok=True)
            save_metrics_json(pdg_metrics, os.path.join(pdg_dir, "metrics.json"))

            pdg_summary[str(pdg_code)] = {
                "status": "ok",
                "n_real": int(len(real_slice)),
                "n_synthetic": int(len(synth_slice)),
                "mmd": float((pdg_metrics.get("mmd") or {}).get("mmd", float("nan"))),
            }
            print(
                f"PDG {pdg_code}: n_real={len(real_slice):,}, n_synth={len(synth_slice):,}, "
                f"MMD={_fmt_float((pdg_metrics.get('mmd') or {}).get('mmd'))}"
            )

            _clear_memory(DEFAULT_DEVICE)

        save_metrics_json(pdg_summary, os.path.join(per_pdg_root, "summary.json"))
        test_metrics["per_pdg_summary"] = pdg_summary
        test_metrics["per_pdg_root"] = per_pdg_root
    else:
        print("Computing univariate + correlation metrics...")
        test_metrics = compute_metrics(test_df, synthetic_df)
        print("Computing multivariate MMD...")
        test_metrics["mmd"] = compute_mmd_rbf(
            test_df.values,
            synthetic_df.values,
            sigma="median",
            max_samples=DEFAULT_MMD_MAX_SAMPLES,
            chunk_size=DEFAULT_MMD_CHUNK_SIZE,
            unbiased=True,
            seed=DEFAULT_MMD_SEED,
            device=DEFAULT_DEVICE,
        )

    if model_type != "cwgan-gp" and not DEFAULT_SKIP_C2ST:
        print("Computing C2ST metrics and feature importances...")
        excluded_features = {
            f.strip() for f in DEFAULT_C2ST_EXCLUDE_FEATURES.split(",") if f.strip()
        }
        c2st_feature_names = [c for c in test_df.columns if c not in excluded_features]
        if not c2st_feature_names:
            raise ValueError("No features left for C2ST after exclusions.")

        # Run C2ST + feature importance on reduced feature set (excluding x by default)
        c2st_real_df = test_df[c2st_feature_names]
        c2st_synth_df = synthetic_df[c2st_feature_names]

        test_metrics["c2st"] = compute_c2st_metrics(
            c2st_real_df,
            c2st_synth_df,
            device=DEFAULT_DEVICE,
            max_samples=DEFAULT_C2ST_MAX_SAMPLES,
            epochs=DEFAULT_C2ST_EPOCHS,
            hidden_dim=DEFAULT_C2ST_HIDDEN_DIM,
            seed=DEFAULT_C2ST_SEED,
            feature_names=c2st_feature_names,
            importance_repeats=DEFAULT_C2ST_IMPORTANCE_REPEATS,
            importance_max_samples=DEFAULT_C2ST_IMPORTANCE_MAX_SAMPLES,
        )
    elif model_type != "cwgan-gp":
        print("C2ST skipped by configuration.")

    if model_type != "cwgan-gp" and BASELINE_ENABLED:
        print("Computing real-vs-real baseline metrics...")
        real_a_df, real_b_df, baseline_meta = _make_real_vs_real_slices(
            df=df,
            target_size=len(test_df),
            seed_a=BASELINE_SLICE_SEED_A,
            seed_b=BASELINE_SLICE_SEED_B,
        )

        baseline_metrics: Dict[str, Any] = {
            "slice_metadata": baseline_meta,
            "mmd": compute_mmd_rbf(
                real_a_df.values,
                real_b_df.values,
                sigma="median",
                max_samples=DEFAULT_MMD_MAX_SAMPLES,
                chunk_size=DEFAULT_MMD_CHUNK_SIZE,
                unbiased=True,
                seed=DEFAULT_MMD_SEED,
                device=DEFAULT_DEVICE,
            ),
        }

        if not DEFAULT_SKIP_C2ST:
            excluded_features = {
                f.strip() for f in DEFAULT_C2ST_EXCLUDE_FEATURES.split(",") if f.strip()
            }
            c2st_feature_names = [c for c in real_a_df.columns if c not in excluded_features]
            if not c2st_feature_names:
                raise ValueError("No features left for real-vs-real C2ST after exclusions.")

            baseline_metrics["c2st"] = compute_c2st_metrics(
                real_a_df[c2st_feature_names],
                real_b_df[c2st_feature_names],
                device=DEFAULT_DEVICE,
                max_samples=DEFAULT_C2ST_MAX_SAMPLES,
                epochs=DEFAULT_C2ST_EPOCHS,
                hidden_dim=DEFAULT_C2ST_HIDDEN_DIM,
                seed=DEFAULT_C2ST_SEED,
                feature_names=c2st_feature_names,
                importance_repeats=DEFAULT_C2ST_IMPORTANCE_REPEATS,
                importance_max_samples=DEFAULT_C2ST_IMPORTANCE_MAX_SAMPLES,
            )

        test_metrics["real_vs_real_baseline"] = baseline_metrics

    if model_type != "cwgan-gp":
        delta_summary = _build_delta_vs_baseline(test_metrics)
        if delta_summary is not None:
            test_metrics["delta_vs_real_baseline"] = delta_summary

        _print_univariate_summary(test_metrics.get("univariate") or [])
        _print_section("STEP 4B: Distribution and Separability Summary")
        _print_mmd_summary(test_metrics.get("mmd") or {}, label="Synth-vs-Real")
        _print_c2st_summary(test_metrics.get("c2st") or {}, title="Synth-vs-Real C2ST")
        _print_baseline_and_delta_summary(test_metrics)
    else:
        _print_section("STEP 4B: Per-PDG Summary")
        for pdg_code, stats in (test_metrics.get("per_pdg_summary") or {}).items():
            print(
                f"PDG {pdg_code}: status={stats.get('status')}, "
                f"n_real={stats.get('n_real')}, n_synthetic={stats.get('n_synthetic')}, "
                f"mmd={_fmt_float(stats.get('mmd'))}"
            )

    test_metrics["external_evaluation"] = {
        "generator_path": generator_path,
        "run_dir": run_dir,
        "data_file": data_file,
        "matched_log": log_path,
        "resolved_config": {
            "model": model_type,
            "latent_dim": latent_dim,
            "entries": entries,
            "test_entries": test_entries,
            "pdg": pdg,
            "monitor_id": monitor_id,
            "split_mode": split_mode,
            "split_seed": DEFAULT_SPLIT_SEED,
            "batch_size": DEFAULT_BATCH_SIZE,
            "c2st_exclude_features": DEFAULT_C2ST_EXCLUDE_FEATURES,
            "generator_filename": GENERATOR_FILENAME,
            "gan_results_root": GAN_RESULTS_ROOT,
            "run_dir_name": run_dir_name,
        },
        "inferred_architecture": inferred,
    }

    if model_type != "cwgan-gp":
        _print_section("STEP 4D: FPD and 1-NN Summary")

        if args.fpd_model:
            print("Computing FPD (Fréchet Physics Distance)...")
            fpd_embedder = _load_torchscript_embedder(args.fpd_model, DEFAULT_DEVICE)
            test_metrics["fpd"] = compute_fpd(
                test_df,
                synthetic_df,
                embedder=fpd_embedder,
                device=DEFAULT_DEVICE,
                max_samples=DEFAULT_FPD_MAX_SAMPLES,
                batch_size=DEFAULT_FPD_BATCH_SIZE,
                seed=DEFAULT_MMD_SEED,
            )

        print("Computing 1-NN leave-one-out two-sample metric...")
        test_metrics["one_nn_loo"] = compute_1nn_loo(
            test_df,
            synthetic_df,
            max_samples=DEFAULT_1NN_MAX_SAMPLES,
            seed=DEFAULT_MMD_SEED,
            device=DEFAULT_DEVICE,
        )

        _print_fpd_summary(test_metrics.get("fpd") or {})
        _print_1nn_summary(test_metrics.get("one_nn_loo") or {})

    _print_section("STEP 5: Saving Metrics JSON")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    save_metrics_json(test_metrics, output_json)
    print(f"Saved external metrics to: {output_json}")

    output_md = output_json.replace(".json", ".md")
    _save_metrics_markdown(test_metrics, output_md)
    print(f"Saved metrics report to: {output_md}")


if __name__ == "__main__":
    main()