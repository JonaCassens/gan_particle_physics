#!/usr/bin/env python3
"""
Evaluate pre-generated synthetic parquet against truth data.
Computes MMD, C2ST, and generates comparison plots.
"""
import argparse
import json
from pathlib import Path
from typing import Optional
from numbers import Number

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from data_loader import _transform_preprocessed_batch
from utils import (
    compute_metrics,
    compute_c2st_metrics,
    compute_mmd_rbf,
    plot_2d_hist_with_stats,
    get_histogram_range,
)

DEFAULT_TRUTH_PARQUET = "/home/hep/jcc525/cleaned_data/pdgNone_monitor4.parquet"
DEFAULT_SYNTHETIC_PARQUET = "/home/hep/jcc525/gan_particle_physics/synthetic_data/optimal_mix_3pdg.parquet"
DEFAULT_MAX_SAMPLES = 200_000
FEATURE_MARKERS = {"log1p_r", "log1p_p_mag", "log_t"}


def parse_pdg_allowlist(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return []
    parsed = []
    for token in tokens:
        try:
            parsed.append(int(token))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid PDG code '{token}' in --truth-pdg-allowlist"
            ) from exc
    return sorted(set(parsed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-generated synthetic parquet against truth data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--truth-parquet",
        type=str,
        default=DEFAULT_TRUTH_PARQUET,
        help="Path to truth data parquet",
    )
    parser.add_argument(
        "--synthetic-parquet",
        type=str,
        default=DEFAULT_SYNTHETIC_PARQUET,
        help="Path to synthetic data parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/hep/jcc525/gan_particle_physics/eval_results",
        help="Directory to save metrics and plots",
    )
    parser.add_argument(
        "--n-truth",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Max truth samples to load (memory-safe default)",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Max synthetic samples to load (memory-safe default)",
    )
    parser.add_argument(
        "--mmd-sigma",
        type=str,
        default="median",
        choices=["median", "scott"],
        help="MMD kernel bandwidth selection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--truth-pdg-allowlist",
        type=str,
        default=None,
        help="Comma-separated PDG codes to keep in truth data only (e.g. '-11,11,2112')",
    )
    return parser.parse_args()


def read_parquet_limited(
    path: str,
    n_rows: Optional[int],
    seed: int,
    columns: Optional[list[str]] = None,
    pdg_allowlist: Optional[list[int]] = None,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    pf = pq.ParquetFile(path)

    if pdg_allowlist is not None and "pdg" not in pf.schema.names:
        raise ValueError(f"PDG allowlist requested, but 'pdg' is missing in parquet schema: {path}")

    if n_rows is not None and n_rows <= 0:
        return pd.DataFrame(columns=columns if columns is not None else pf.schema.names)

    chunks: list[pd.DataFrame] = []
    remaining = n_rows
    rng = np.random.default_rng(seed)

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        if df.empty:
            continue

        df = _transform_preprocessed_batch(df, keep_pdg=True)
        if df.empty:
            continue

        if pdg_allowlist is not None:
            if "pdg" not in df.columns:
                raise ValueError(
                    "PDG allowlist requested, but 'pdg' is missing after preprocessing."
                )
            df = df.loc[df["pdg"].isin(pdg_allowlist)].copy()
            if df.empty:
                continue

        if remaining is None:
            chunks.append(df)
            continue

        if len(df) <= remaining:
            chunks.append(df)
            remaining -= len(df)
        else:
            idx = rng.choice(len(df), size=remaining, replace=False)
            chunks.append(df.iloc[idx])
            remaining = 0

        if remaining == 0:
            break

    if not chunks:
        return pd.DataFrame(columns=columns if columns is not None else pf.schema.names)

    out = pd.concat(chunks, ignore_index=True)
    if len(out) > 1:
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def restore_feature_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Convert transformed features back to raw-space names for comparisons."""
    restored = df.copy()

    if "log_t" in restored.columns and "t" not in restored.columns:
        t_vals = np.exp(restored["log_t"].to_numpy(dtype=np.float64)) - 1e-10
        restored["t"] = np.clip(t_vals, a_min=0.0, a_max=None)
        restored = restored.drop(columns=["log_t"])

    transformed_columns = list(restored.columns)
    for column in transformed_columns:
        if column.startswith("log1p_"):
            base_name = column[len("log1p_"):]
            if base_name and base_name not in restored.columns:
                restored[base_name] = np.expm1(restored[column].to_numpy(dtype=np.float64))
                restored = restored.drop(columns=[column])
        elif column.startswith("log10_"):
            base_name = column[len("log10_"):]
            if base_name and base_name not in restored.columns:
                restored[base_name] = np.power(10.0, restored[column].to_numpy(dtype=np.float64))
                restored = restored.drop(columns=[column])
        elif column.startswith("log_"):
            base_name = column[len("log_"):]
            if base_name and base_name not in restored.columns:
                restored[base_name] = np.exp(restored[column].to_numpy(dtype=np.float64))
                restored = restored.drop(columns=[column])

    return restored


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_pdg_allowlist = parse_pdg_allowlist(args.truth_pdg_allowlist)

    truth_schema_cols = set(pq.ParquetFile(args.truth_parquet).schema.names)
    synth_schema_cols = set(pq.ParquetFile(args.synthetic_parquet).schema.names)
    common_schema_cols = sorted(truth_schema_cols & synth_schema_cols)
    if not common_schema_cols:
        raise ValueError("No common columns between truth and synthetic parquet files.")

    truth_load_cols = set(common_schema_cols)
    if truth_pdg_allowlist is not None:
        truth_load_cols.add("pdg")
    truth_load_cols = sorted(col for col in truth_load_cols if col in truth_schema_cols)

    synth_load_cols = set(common_schema_cols)
    if "pdg" in synth_schema_cols:
        synth_load_cols.add("pdg")
    synth_load_cols = sorted(col for col in synth_load_cols if col in synth_schema_cols)

    print(f"[INFO] Loading truth from: {args.truth_parquet}")
    truth_df = read_parquet_limited(
        args.truth_parquet,
        n_rows=args.n_truth,
        seed=args.seed,
        columns=truth_load_cols,
        pdg_allowlist=truth_pdg_allowlist,
    )
    print(f"       Loaded {len(truth_df):,} truth samples")
    if truth_pdg_allowlist is not None:
        print(f"[INFO] Truth PDG allowlist applied during loading: {truth_pdg_allowlist}")

    print(f"[INFO] Loading synthetic from: {args.synthetic_parquet}")
    synthetic_df = read_parquet_limited(
        args.synthetic_parquet,
        n_rows=args.n_synthetic,
        seed=args.seed,
        columns=synth_load_cols,
    )
    print(f"       Loaded {len(synthetic_df):,} synthetic samples")

    truth_df = restore_feature_markers(truth_df)
    synthetic_df = restore_feature_markers(synthetic_df)

    if truth_df.empty:
        raise ValueError("Truth dataset is empty after loading/filtering.")

    common_cols = sorted(set(truth_df.columns) & set(synthetic_df.columns))
    if not common_cols:
        raise ValueError("No common columns between truth and synthetic data after preprocessing.")
    print(f"[INFO] Using {len(common_cols)} common columns after preprocessing: {common_cols}")

    metric_cols = [col for col in common_cols if col != "pdg"]
    if not metric_cols:
        raise ValueError("No metric columns available after excluding 'pdg'.")
    print(f"[INFO] Metrics computed on {len(metric_cols)} columns: {metric_cols}")

    if "pdg" in common_cols:
        truth_pdg_counts = truth_df["pdg"].value_counts(dropna=False).sort_index().to_dict()
        synthetic_pdg_counts = synthetic_df["pdg"].value_counts(dropna=False).sort_index().to_dict()
        print(f"[INFO] Truth PDG counts: {truth_pdg_counts}")
        print(f"[INFO] Synthetic PDG counts: {synthetic_pdg_counts}")

    truth_metrics_df = truth_df[metric_cols]
    synthetic_metrics_df = synthetic_df[metric_cols]

    # Compute metrics
    print("\n[INFO] Computing metrics...")
    metrics = compute_metrics(truth_metrics_df, synthetic_metrics_df)

    mmd_out = compute_mmd_rbf(
        truth_metrics_df.values,
        synthetic_metrics_df.values,
        sigma=args.mmd_sigma,
        seed=args.seed,
    )

    # Handle both return types: scalar or dict
    if isinstance(mmd_out, dict):
        for k, v in mmd_out.items():
            metrics[k] = float(v) if isinstance(v, Number) else v
        if "mmd_rbf" not in metrics:
            if "mmd" in mmd_out and isinstance(mmd_out["mmd"], Number):
                metrics["mmd_rbf"] = float(mmd_out["mmd"])
            else:
                raise ValueError(
                    "compute_mmd_rbf returned a dict without a numeric 'mmd' or 'mmd_rbf'."
                )
    else:
        metrics["mmd_rbf"] = float(mmd_out)

    c2st_dict = compute_c2st_metrics(truth_metrics_df, synthetic_metrics_df, seed=args.seed)
    metrics.update(c2st_dict)

    if "pdg" in common_cols:
        metrics["pdg_context"] = {
            "truth_allowlist": truth_pdg_allowlist,
            "truth_counts": truth_df["pdg"].value_counts(dropna=False).sort_index().to_dict(),
            "synthetic_counts": synthetic_df["pdg"].value_counts(dropna=False).sort_index().to_dict(),
        }

    print(f"\n[METRICS]")
    for key, val in sorted(metrics.items()):
        if isinstance(val, Number):
            print(f"  {key}: {float(val):.6f}")
        else:
            print(f"  {key}: {val}")

    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Saved metrics to: {metrics_path}")

    # Plot 2D comparisons
    print(f"\n[INFO] Generating 2D comparison plots...")
    features = metric_cols
    n_features = len(features)
    n_pairs = n_features * (n_features - 1) // 2
    pairs_per_row = 4

    import matplotlib.pyplot as plt
    n_cols = pairs_per_row * 2
    n_rows = (n_pairs + pairs_per_row - 1) // pairs_per_row
    figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    axes = np.atleast_1d(axes).flatten()

    pair_idx = 0
    for i, fx in enumerate(features):
        for j, fy in enumerate(features):
            if i >= j:
                continue
            ax_truth = axes[pair_idx * 2]
            ax_synth = axes[pair_idx * 2 + 1]
            try:
                x_range = get_histogram_range(
                    truth_df[fx].values,
                    synthetic_df[fx].values,
                )
                y_range = get_histogram_range(
                    truth_df[fy].values,
                    synthetic_df[fy].values,
                )
                if x_range is None or y_range is None:
                    x_range, y_range = None, None

                plot_2d_hist_with_stats(
                    ax_truth,
                    truth_df[fx].values,
                    truth_df[fy].values,
                    fx,
                    fy,
                    f"Truth: {fx} vs {fy}",
                    cmap="viridis",
                    x_range=x_range,
                    y_range=y_range,
                )

                plot_2d_hist_with_stats(
                    ax_synth,
                    synthetic_df[fx].values,
                    synthetic_df[fy].values,
                    fx,
                    fy,
                    f"Synthetic: {fx} vs {fy}",
                    cmap="viridis",
                    x_range=x_range,
                    y_range=y_range,
                )
            except Exception as e:
                ax_truth.text(0.5, 0.5, f"Error:\n{str(e)}", ha="center", va="center")
                ax_synth.text(0.5, 0.5, f"Error:\n{str(e)}", ha="center", va="center")
            pair_idx += 1

    for idx in range(pair_idx * 2, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plot_path = output_dir / "eval_2d_comparisons.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    print(f"[INFO] Saved plot to: {plot_path}")
    plt.close()

    print(f"\n[INFO] Evaluation complete. Results in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())