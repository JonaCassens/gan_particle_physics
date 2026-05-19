import argparse
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import gc

from data_loader import load_preprocessed_data
from gan_model import train_gan
from wgan_model import train_wgan
from wgan_gp_model import train_wgan_gp
from cwgan_gp_model import train_cwgan_gp
from utils import (
    compute_metrics,
    compute_c2st_metrics,
    compute_mmd_rbf,
    save_metrics_json,
    create_white_to_viridis_cmap,
    plot_2d_hist_with_stats,
    get_histogram_range,
    plot_training_history
)

def _clear_memory(device: str = "cpu") -> None:
    """Free Python + CUDA cached memory between heavy steps."""
    gc.collect()
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Optional on some systems; guarded to avoid runtime issues
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

def _shuffled_train_test_split(df, n_train: int, n_test: int, seed: int = 42):
    n_total = len(df)
    if n_train + n_test > n_total:
        raise ValueError(f"Requested n_train+n_test={n_train+n_test} > total rows={n_total}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    train_idx = perm[:n_train]
    test_idx = perm[n_train:n_train + n_test]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def _prepare_frames_for_plotting(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert transformed features back to plotting space for side-by-side views."""
    real_plot = real_df.copy()
    synth_plot = synthetic_df.copy()

    if "log_t" in real_plot.columns and "log_t" in synth_plot.columns:
        real_plot["t"] = np.exp(real_plot["log_t"])
        synth_plot["t"] = np.exp(synth_plot["log_t"])
        real_plot = real_plot.drop(columns=["log_t"])
        synth_plot = synth_plot.drop(columns=["log_t"])

    if "log1p_p_mag" in real_plot.columns and "log1p_p_mag" in synth_plot.columns:
        real_plot["p_mag"] = np.expm1(real_plot["log1p_p_mag"])
        synth_plot["p_mag"] = np.expm1(synth_plot["log1p_p_mag"])
        real_plot = real_plot.drop(columns=["log1p_p_mag"])
        synth_plot = synth_plot.drop(columns=["log1p_p_mag"])

    if "log1p_r" in real_plot.columns and "log1p_r" in synth_plot.columns:
        real_plot["r"] = np.expm1(real_plot["log1p_r"])
        synth_plot["r"] = np.expm1(synth_plot["log1p_r"])
        real_plot = real_plot.drop(columns=["log1p_r"])
        synth_plot = synth_plot.drop(columns=["log1p_r"])

    return real_plot, synth_plot


def _save_comparison_plot(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, output_path: str, model_label: str):
    """Save pairwise 2D comparison panel for real vs synthetic samples."""
    cmap = "viridis"
    features = list(real_df.columns)
    pairs = [(features[i], features[j]) for i in range(len(features)) for j in range(len(features)) if i != j]

    subplot_size = 4
    pairs_per_row = 4
    n_cols = pairs_per_row * 2
    n_rows = (len(pairs) + pairs_per_row - 1) // pairs_per_row
    figsize = ((subplot_size + 1) * n_cols, subplot_size * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=150)

    for idx, (fx, fy) in enumerate(pairs):
        row_idx = idx // pairs_per_row
        col_idx = (idx % pairs_per_row) * 2

        x_range = get_histogram_range(real_df[fx].values, synthetic_df[fx].values, percentile=99)
        y_range = get_histogram_range(real_df[fy].values, synthetic_df[fy].values, percentile=99)

        ax_real = axes[row_idx, col_idx]
        ax_syn = axes[row_idx, col_idx + 1]

        plot_2d_hist_with_stats(
            ax_real,
            real_df[fx].values,
            real_df[fy].values,
            fx,
            fy,
            f"Test (Real): {fx} vs {fy}",
            cmap=cmap,
            x_range=x_range,
            y_range=y_range,
        )
        plot_2d_hist_with_stats(
            ax_syn,
            synthetic_df[fx].values,
            synthetic_df[fy].values,
            fx,
            fy,
            f"{model_label} (Synthetic): {fx} vs {fy}",
            cmap=cmap,
            x_range=x_range,
            y_range=y_range,
        )

        ax_real.set_box_aspect(1)
        ax_syn.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="GAN-based particle physics data generation")
    parser.add_argument("--entries", type=int, default=None, help="Number of entries to use for training")
    parser.add_argument("--test-entries", type=int, default=None, help="Number of additional entries for test set")
    parser.add_argument("--output-name", type=str, default=None, help="Custom output folder name")
    parser.add_argument("--results-dir", type=str, default="./gan_results", help="Base results directory")
    parser.add_argument("--pdg", type=int, default=None, help="PDG code")
    parser.add_argument("--monitor-id", type=int, default=4, help="MonitorID")
    parser.add_argument("--gp", type=int, default=10, help="Gradient penalty (for WGAN-GP)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--lr-g", type=float, default=5e-5, help="Generator learning rate")
    parser.add_argument("--lr-c", type=float, default=5e-5, help="Critic/Discriminator learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        type=str,
        default="gan",
        choices=["gan", "wgan", "wgan-gp", "cwgan-gp"],
        help="Model type: gan, wgan, wgan-gp, or cwgan-gp"
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--log-interval", type=int, default=1, help="Print loss every N epochs")
    parser.add_argument("--n-critic", type=int, default=5, help="Critic steps per generator step")
    parser.add_argument(
        "--trig-constraint-weight",
        type=float,
        default=0.01,
        help="Weight for trig unit-circle penalty on sin/cos features (WGAN-GP only)"
    )
    parser.add_argument("--c2st-max-samples", type=int, default=50000, 
                        help="Max samples per class for C2ST classifier training (for large jobs)")
    parser.add_argument("--c2st-epochs", type=int, default=30, 
                        help="C2ST classifier training epochs")
    parser.add_argument("--c2st-hidden-dim", type=int, default=64, 
                        help="C2ST classifier hidden layer dimension")
    parser.add_argument("--c2st-seed", type=int, default=42, 
                        help="Random seed for C2ST reproducibility")
    parser.add_argument("--skip-c2st", action="store_true", 
                        help="Skip C2ST evaluation (faster for testing)")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for shuffled train/test split")
    parser.add_argument(
        "--pdg-allowlist",
        type=str,
        default=None,
        help="Comma-separated PDG codes to keep in loader (e.g. 13,2112)",
    )
    args = parser.parse_args()

    # preprocessed_file = f"/home/hep/jcc525/cleaned_data/pdg{args.pdg}_monitor{args.monitor_id}.parquet"
    preprocessed_file = f"/home/hep/jcc525/cleaned_data/pdgNone_monitor{args.monitor_id}.parquet"


    print(f"Using device: {args.device}")

    # Show selected args
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    for arg, val in vars(args).items():
        print(f"  {arg:.<25} {val}")

    # Load data (timing split + cylindrical conversion happen here)
    print("\n" + "="*60)
    print("STEP 1: Loading data...")
    print("="*60)
    
    keep_pdg = args.model == "cwgan-gp"
    df = load_preprocessed_data(
        preprocessed_file,
        pdg_code=args.pdg,
        pdg_allowlist=args.pdg_allowlist,
        entries=args.entries,
        test_entries=args.test_entries,
        keep_pdg=keep_pdg,
    )
    
    # Split into train and test
    train_end = args.entries if args.entries else len(df)
    test_start = train_end
    test_end = test_start + (args.test_entries if args.test_entries else int(train_end * 0.2))
    
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    n_total = len(df)
    n_train = args.entries if args.entries is not None else int(0.9 * n_total)
    n_test = args.test_entries if args.test_entries is not None else max(1, n_total - n_train)

    train_df, test_df = _shuffled_train_test_split(
        df,
        n_train=n_train,
        n_test=n_test,
        seed=args.split_seed,
    )
    
    print(f"Training on {len(train_df):,} samples (shuffled)")
    print(f"Testing on {len(test_df):,} samples (shuffled)")
    
    # Create output directory with timestamp or custom name
    if args.output_name:
        output_dir = os.path.join(args.results_dir, args.output_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.results_dir, f"run_{timestamp}_entries{len(train_df)}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Train GAN on train_df only
    print("\n" + "="*60)
    print(f"STEP 2: Training {args.model.upper()}...")
    print("="*60)
    
    if args.model == "wgan":
        gan, mean, std, history = train_wgan(
            train_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            device=args.device,
            log_interval=args.log_interval
        )
    elif args.model == "wgan-gp":
        gan, mean, std, history = train_wgan_gp(
            train_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            device=args.device,
            n_critic=args.n_critic,
            lambda_gp=args.gp,
            log_interval=args.log_interval,
            patience=args.patience,
            lr_g=args.lr_g,
            lr_c=args.lr_c,
            trig_constraint_weight=args.trig_constraint_weight,
        )
    elif args.model == "cwgan-gp":
        gan, mean, std, history = train_cwgan_gp(
            train_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            device=args.device,
            n_critic=args.n_critic,
            lambda_gp=args.gp,
            log_interval=args.log_interval,
            patience=args.patience,
            lr_g=args.lr_g,
            lr_c=args.lr_c,
            trig_constraint_weight=args.trig_constraint_weight,
        )
    else:
        gan, mean, std, history = train_gan(
            train_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            device=args.device,
        )

    _clear_memory(args.device)  # <-- after training

    plot_training_history(
        history,
        os.path.join(output_dir, "training_loss.png"),
        title=f"{args.model.upper()} Loss"
    )

    # Generate synthetic samples
    print("\n" + "="*60)
    print(f"STEP 3: Generating {len(test_df):,} synthetic samples...")
    print("="*60)

    _clear_memory(args.device)  # <-- before generation
    if args.model == "cwgan-gp":
        if "pdg" in test_df.columns:
            condition_pdg = test_df["pdg"].to_numpy(dtype=np.int64)
        elif args.pdg is not None:
            condition_pdg = np.full(len(test_df), int(args.pdg), dtype=np.int64)
        else:
            raise ValueError("cwgan-gp requires PDG conditioning, but no `pdg` column or --pdg value was provided.")

        synthetic_data = getattr(gan, "generate")(len(test_df), condition_pdg, mean, std)
        feature_columns = [col for col in train_df.columns if col != "pdg"]
        synthetic_df = pd.DataFrame(synthetic_data, columns=feature_columns)
        if "pdg" in test_df.columns:
            synthetic_df.insert(0, "pdg", condition_pdg)
    else:
        synthetic_data = getattr(gan, "generate")(len(test_df), mean, std)
        synthetic_df = pd.DataFrame(synthetic_data, columns=train_df.columns)
    _clear_memory(args.device)  # <-- after generation

    if args.model == "cwgan-gp":
        print("\n" + "="*60)
        print("STEP 4-5: Computing per-PDG metrics and plots...")
        print("="*60)

        if "pdg" not in test_df.columns:
            raise ValueError("cwgan-gp per-PDG evaluation requires `pdg` column in the test dataframe.")

        per_pdg_root = os.path.join(output_dir, "per_pdg")
        os.makedirs(per_pdg_root, exist_ok=True)

        summary: dict[str, dict] = {}
        unique_pdg_codes = sorted(pd.Series(test_df["pdg"]).dropna().astype(int).unique().tolist())

        for pdg_code in unique_pdg_codes:
            real_slice = test_df[test_df["pdg"] == pdg_code].copy()
            synth_slice = synthetic_df[synthetic_df["pdg"] == pdg_code].copy()

            if len(real_slice) == 0 or len(synth_slice) == 0:
                summary[str(pdg_code)] = {
                    "status": "skipped_empty_slice",
                    "n_real": int(len(real_slice)),
                    "n_synthetic": int(len(synth_slice)),
                }
                continue

            real_eval = real_slice.drop(columns=["pdg"], errors="ignore")
            synth_eval = synth_slice.drop(columns=["pdg"], errors="ignore")

            pdg_metrics = compute_metrics(real_eval, synth_eval)
            pdg_metrics["training_curves"] = {
                "epoch": history.get("epoch", []),
                "d_loss": history.get("d_loss", []),
                "g_loss": history.get("g_loss", []),
                "train_wasserstein": history.get("train_wasserstein", []),
                "val_wasserstein": history.get("val_wasserstein", []),
                "train_mmd": history.get("train_mmd", []),
                "val_mmd": history.get("val_mmd", []),
                "best_epoch": history.get("best_epoch"),
                "best_val_wasserstein": history.get("best_val_wasserstein"),
                "best_val_mmd": history.get("best_val_mmd"),
            }

            pdg_mmd = compute_mmd_rbf(
                real_eval.values,
                synth_eval.values,
                sigma="median",
                max_samples=100000,
                chunk_size=512,
                unbiased=True,
                seed=42,
            )
            pdg_metrics["mmd"] = pdg_mmd

            if not args.skip_c2st:
                _clear_memory(args.device)
                c2st_metrics = compute_c2st_metrics(
                    real_eval,
                    synth_eval,
                    device=args.device,
                    max_samples=args.c2st_max_samples,
                    epochs=args.c2st_epochs,
                    hidden_dim=args.c2st_hidden_dim,
                    seed=args.c2st_seed,
                    feature_names=list(real_eval.columns),
                    importance_repeats=5,
                    importance_max_samples=50000,
                )
                pdg_metrics["c2st"] = c2st_metrics

            pdg_dir = os.path.join(per_pdg_root, f"pdg_{pdg_code}")
            os.makedirs(pdg_dir, exist_ok=True)
            save_metrics_json(pdg_metrics, os.path.join(pdg_dir, "metrics.json"))

            real_plot, synth_plot = _prepare_frames_for_plotting(real_eval, synth_eval)
            _save_comparison_plot(
                real_plot,
                synth_plot,
                os.path.join(pdg_dir, "gan_comparison.png"),
                model_label=f"{args.model.upper()} PDG={pdg_code}",
            )
            _clear_memory(args.device)

            summary[str(pdg_code)] = {
                "status": "ok",
                "n_real": int(len(real_slice)),
                "n_synthetic": int(len(synth_slice)),
                "mmd": float(pdg_mmd.get("mmd", float("nan"))),
                "mmd2": float(pdg_mmd.get("mmd2", float("nan"))),
            }

            print(
                f"PDG {pdg_code}: n_real={len(real_slice):,}, n_synth={len(synth_slice):,}, "
                f"MMD={pdg_mmd.get('mmd', float('nan')):.6f}"
            )

        save_metrics_json(summary, os.path.join(per_pdg_root, "summary.json"))
        print(f"Saved per-PDG evaluation outputs in: {per_pdg_root}")

    else:
        # Compute metrics ONLY on test set
        print("\n" + "="*60)
        print("STEP 4: Computing metrics (test set)...")
        print("="*60)

        test_metrics = compute_metrics(test_df, synthetic_df)

        # Save training curves into metrics JSON
        test_metrics["training_curves"] = {
            "epoch": history.get("epoch", []),
            "d_loss": history.get("d_loss", []),
            "g_loss": history.get("g_loss", []),
            "train_wasserstein": history.get("train_wasserstein", []),
            "val_wasserstein": history.get("val_wasserstein", []),
            "train_mmd": history.get("train_mmd", []),
            "val_mmd": history.get("val_mmd", []),
            "best_epoch": history.get("best_epoch"),
            "best_val_wasserstein": history.get("best_val_wasserstein"),
            "best_val_mmd": history.get("best_val_mmd"),
        }

        # Full multivariate MMD on test set
        mmd_metrics = compute_mmd_rbf(
            test_df.values,
            synthetic_df.values,
            sigma="median",
            max_samples=100000,
            chunk_size=512,
            unbiased=True,
            seed=42
        )
        test_metrics["mmd"] = mmd_metrics

        print("Test set metrics:")
        print(pd.DataFrame(test_metrics['univariate']))
        print(f"Multivariate MMD: {mmd_metrics['mmd']:.6f} (MMD^2={mmd_metrics['mmd2']:.6f})")

        # Compute C2ST metrics (unless skipped)
        _clear_memory(args.device)

        if not args.skip_c2st:
            print("\n" + "="*60)
            print("STEP 4b: Computing C2ST metrics...")
            print("="*60)

            c2st_metrics = compute_c2st_metrics(
                test_df,
                synthetic_df,
                device=args.device,
                max_samples=args.c2st_max_samples,
                epochs=args.c2st_epochs,
                hidden_dim=args.c2st_hidden_dim,
                seed=args.c2st_seed,
                feature_names=list(test_df.columns),
                importance_repeats=5,
                importance_max_samples=50000
            )
            test_metrics["c2st"] = c2st_metrics
            print(f"C2ST Accuracy: {c2st_metrics['accuracy']:.4f}")
            print(f"C2ST Balanced Accuracy: {c2st_metrics['balanced_accuracy']:.4f}")
            print(f"C2ST ROC-AUC: {c2st_metrics['roc_auc']:.4f}")
            if c2st_metrics.get("feature_importance"):
                top = c2st_metrics["feature_importance"][:5]
                top_str = ", ".join([f"{f['feature']}:{f['importance_mean']:.4f}" for f in top])
                print(f"Top C2ST permutation features: {top_str}")
            print(f"(Accuracy ≈ 0.5 is better; AUC ≈ 0.5 is better)")

        _clear_memory(args.device)

        print("\n" + "="*60)
        print("STEP 5: Generating plots...")
        print("="*60)

        test_plot_df, synthetic_plot_df = _prepare_frames_for_plotting(pd.DataFrame(test_df), pd.DataFrame(synthetic_df))
        _save_comparison_plot(
            test_plot_df,
            synthetic_plot_df,
            os.path.join(output_dir, "gan_comparison.png"),
            model_label=args.model.upper(),
        )
        _clear_memory(args.device)
    
    # Save outputs
    synthetic_df.to_parquet(os.path.join(output_dir, "synthetic_samples.parquet"), compression='gzip', index=False)
    if args.model != "cwgan-gp":
        save_metrics_json(test_metrics, os.path.join(output_dir, "metrics.json"))
    
    # Save model
    torch.save(gan.generator.state_dict(), os.path.join(output_dir, "generator.pth"))
    critic = getattr(gan, "critic", None)
    discriminator = getattr(gan, "discriminator", None)
    if critic is not None:
        torch.save(critic.state_dict(), os.path.join(output_dir, "critic.pth"))
    elif discriminator is not None:
        torch.save(discriminator.state_dict(), os.path.join(output_dir, "discriminator.pth"))
    
    print(f"\n✅ Training complete! Results in: {output_dir}")

if __name__ == "__main__":
    main()