#!/usr/bin/env python3
"""
Export synthetic particles from a WGAN-GP generator to ROOTRacker format.

This exporter is intentionally minimal and template-free:
- Loads one WGAN-GP generator checkpoint
- Generates N synthetic particles
- Converts features to cartesian 4-vectors
- Writes only these ROOT branches (capitalized exactly):
    - StdHepPdg
    - StdHepX4
    - StdHepP4
    - MonitorID (always 4)

Each tree entry stores exactly one particle.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import uproot
import awkward as ak
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_saved_generator import (
    _infer_generator_architecture,
    _create_generator,
    _compute_normalization_stats,
    _generate_synthetic_samples,
)
DEFAULT_TRAIN_PARQUET = "/home/hep/jcc525/cleaned_data/pdg13_monitor4.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Template-free WGAN-GP exporter to ROOTRacker branches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--generator", type=str, required=True, help="Path to generator checkpoint (.pth)")
    parser.add_argument("--output", type=str, required=True, help="Output .rootracker path")
    parser.add_argument("--n-particles", type=int, required=True, help="Number of synthetic particles to generate")
    parser.add_argument("--train-parquet", type=str, default=DEFAULT_TRAIN_PARQUET, help="Parquet used for normalization stats")
    parser.add_argument("--batch-size", type=int, default=512, help="Generation batch size")
    parser.add_argument("--pdg", type=int, default=13, help="PDG value to write to StdHepPdg")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device")
    return parser.parse_args()


def resolve_device(device_flag: str) -> str:
    if device_flag == "cpu":
        return "cpu"
    if device_flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_wgangp_generator(generator_path: str, train_parquet: str, device: str):
    print(f"Loading generator checkpoint: {generator_path}")
    checkpoint = torch.load(generator_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    arch = _infer_generator_architecture(state_dict)
    latent_dim = int(arch["latent_dim"])
    output_dim = int(arch["output_dim"])
    hidden_dims = list(arch.get("hidden_dims", [256, 256]))

    if not all(k.startswith("model.") for k in state_dict.keys()):
        raise RuntimeError("Expected WGAN-GP checkpoint with keys prefixed by 'model.'")

    model_state = {k: v for k, v in state_dict.items() if k.startswith("model.")}
    generator = _create_generator(
        model_type="wgan-gp",
        latent_dim=latent_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        normalization="layernorm",
        state_dict=model_state,
    )
    generator.load_state_dict(model_state, strict=True)
    generator = generator.to(device)
    generator.eval()

    print(f"Loading normalization parquet: {train_parquet}")
    train_df = pd.read_parquet(train_parquet)
    all_columns = list(train_df.columns)
    mean, std = _compute_normalization_stats(train_df, model_type="wgan-gp")

    if "pdg" in all_columns:
        pdg_idx = all_columns.index("pdg")
        feature_names = [column for column in all_columns if column != "pdg"]
        if len(mean) == len(all_columns):
            keep_idx = [idx for idx in range(len(all_columns)) if idx != pdg_idx]
            mean = mean[keep_idx]
            std = std[keep_idx]
    else:
        feature_names = all_columns

    if len(mean) != len(feature_names) or len(std) != len(feature_names):
        raise RuntimeError(
            "Normalization shape mismatch after schema alignment: "
            f"len(mean)={len(mean)}, len(std)={len(std)}, len(feature_names)={len(feature_names)}"
        )

    return generator, latent_dim, mean, std, feature_names


def generate_particles(
    generator,
    n_particles: int,
    latent_dim: int,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    print(f"Generating {n_particles} particles")
    samples, generated_feature_names = _generate_synthetic_samples(
        generator=generator,
        n_samples=n_particles,
        latent_dim=latent_dim,
        mean=mean,
        std=std,
        device=device,
        batch_size=batch_size,
        feature_names=feature_names,
        apply_angle_clipping=True,
        conditional_pdg_codes=None,
    )
    df = pd.DataFrame(samples, columns=generated_feature_names)
    print(f"Generated columns: {list(df.columns)}")
    return df


def to_cartesian(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    work = df.copy()

    if "log1p_r" in work.columns and "r" not in work.columns:
        work["r"] = np.expm1(work["log1p_r"])
    if "log1p_p_mag" in work.columns and "p_mag" not in work.columns:
        work["p_mag"] = np.expm1(work["log1p_p_mag"])

    if "x" not in work.columns:
        work["x"] = 0.0

    if not {"r", "sin_phi_s", "cos_phi_s"}.issubset(work.columns):
        raise RuntimeError("Missing spatial columns; expected r, sin_phi_s, cos_phi_s")
    work["y"] = work["r"] * work["cos_phi_s"]
    work["z"] = work["r"] * work["sin_phi_s"]

    if not {"p_mag", "sin_theta", "cos_theta", "phi_p"}.issubset(work.columns):
        raise RuntimeError("Missing momentum columns; expected p_mag, sin_theta, cos_theta, phi_p")

    # Inverse of data_loader transform:
    # cos_theta = px / p_mag   (theta measured from x-axis)
    # phi_p = atan2(pz, py)    (azimuth in y-z plane)
    transverse = work["p_mag"] * work["sin_theta"]
    work["px"] = work["p_mag"] * work["cos_theta"]
    work["py"] = transverse * np.cos(work["phi_p"])
    work["pz"] = transverse * np.sin(work["phi_p"])

    if "log_t" in work.columns:
        t_values = np.exp(work["log_t"].to_numpy(dtype=np.float32))
    else:
        t_values = np.zeros(len(work), dtype=np.float32)

    e_values = np.sqrt(
        work["px"].to_numpy(dtype=np.float32) ** 2
        + work["py"].to_numpy(dtype=np.float32) ** 2
        + work["pz"].to_numpy(dtype=np.float32) ** 2
    ).astype(np.float32)

    x4 = np.column_stack(
        [
            work["x"].to_numpy(dtype=np.float32),
            work["y"].to_numpy(dtype=np.float32),
            work["z"].to_numpy(dtype=np.float32),
            t_values,
        ]
    ).astype(np.float32)

    p4 = np.column_stack(
        [
            work["px"].to_numpy(dtype=np.float32),
            work["py"].to_numpy(dtype=np.float32),
            work["pz"].to_numpy(dtype=np.float32),
            e_values,
        ]
    ).astype(np.float32)

    return x4, p4


def write_root(output_path: str, x4: np.ndarray, p4: np.ndarray, pdg_value: int) -> None:
    n_particles = int(x4.shape[0])
    if n_particles == 0:
        raise RuntimeError("No particles to write")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # One particle per tree entry to avoid problematic jagged fixed-width serialization.
    stdhep_x4 = x4.reshape(n_particles, 1, 4).astype(np.float32)
    stdhep_p4 = p4.reshape(n_particles, 1, 4).astype(np.float32)
    stdhep_pdg = np.full((n_particles, 1), int(pdg_value), dtype=np.int32)
    monitor_id = np.full((n_particles, 1), 4, dtype=np.int32)

    branches = {
        "StdHepPdg": stdhep_pdg,
        "StdHepX4": stdhep_x4,
        "StdHepP4": stdhep_p4,
        "MonitorID": monitor_id,
    }

    with uproot.recreate(output_path) as outfile:
        outfile["RooTrackerTree"] = branches

    print(f"Wrote {n_particles} entries to {output_path}")


def validate_output(output_path: str, expected_n: int) -> None:
    with uproot.open(output_path) as root_file:
        tree = root_file["RooTrackerTree"]
        arrays = tree.arrays(["StdHepPdg", "StdHepP4", "StdHepX4", "MonitorID"], library="ak")
        n_entries = int(tree.num_entries)
        if n_entries != expected_n:
            raise RuntimeError(f"Entry mismatch: expected {expected_n}, got {n_entries}")
        print("Validation OK")
        print(f"  Entries: {n_entries}")
        print(f"  StdHepP4 type: {ak.type(arrays['StdHepP4'])}")


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    try:
        generator, latent_dim, mean, std, feature_names = load_wgangp_generator(
            generator_path=args.generator,
            train_parquet=args.train_parquet,
            device=device,
        )
        generated_df = generate_particles(
            generator=generator,
            n_particles=int(args.n_particles),
            latent_dim=latent_dim,
            mean=mean,
            std=std,
            feature_names=feature_names,
            batch_size=int(args.batch_size),
            device=device,
        )
        x4, p4 = to_cartesian(generated_df)
        write_root(args.output, x4=x4, p4=p4, pdg_value=int(args.pdg))
        validate_output(args.output, expected_n=int(args.n_particles))
        print("✓ Export completed")
        return 0
    except Exception as exc:
        print(f"✗ Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
