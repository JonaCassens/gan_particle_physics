#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Optional
import time

import numpy as np
import pandas as pd
import torch

from utils import generate_synthetic_from_checkpoint
from data_loader import load_preprocessed_data


GAN_RESULTS_ROOT = "/home/hep/jcc525/gan_particle_physics/gan_results"
DEFAULT_SYNTHETIC_OUTPUT_DIR = "/home/hep/jcc525/gan_particle_physics/synthetic_data"
DEFAULT_REFERENCE_PARQUET = "/home/hep/jcc525/cleaned_data/pdgNone_monitor4.parquet"

# Hardcoded counts from `/home/hep/jcc525/cleaned_data/pdgNone_monitor4.parquet`
# analyzed on 2026-06-09 (full file, 215,409,934 rows).
HARD_CODED_PDG_COUNTS: Dict[int, int] = {
    22: 123866619,
    11: 33142675,
    2112: 31035471,
    13: 20545008,
    -11: 5157240,
    -211: 1535441,
    -13: 98208,
    2212: 27471,
    1000010020: 965,
    211: 601,
    1000010030: 102,
    1000020040: 80,
    1000010040: 16,
    1000020030: 14,
    1000030040: 5,
    1000220480: 3,
    1000050110: 2,
    1000240500: 2,
    1000080160: 2,
    1000250530: 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute target PDG ratios for specified PDGs and optionally generate mixed synthetic "
            "samples from folder-specific generator checkpoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="Folder names under gan_results (each must contain generator.pth)",
    )
    parser.add_argument(
        "--pdg-codes",
        nargs="+",
        required=True,
        type=int,
        help="PDG code assigned to each folder (same order as --folders)",
    )
    parser.add_argument(
        "--reference-parquet",
        type=str,
        default=DEFAULT_REFERENCE_PARQUET,
        help="Kept for compatibility; ratios now use hardcoded counts by default.",
    )
    parser.add_argument(
        "--preprocessed-file",
        type=str,
        default=DEFAULT_REFERENCE_PARQUET,
        help="Preprocessed parquet used for normalization stats (loaded via data_loader logic).",
    )
    parser.add_argument(
        "--norm-entries",
        type=int,
        default=500000,
        help="Optional entries limit when loading per-PDG normalization data.",
    )
    parser.add_argument(
        "--norm-test-entries",
        type=int,
        default=None,
        help="Optional test_entries add-on when loading per-PDG normalization data.",
    )
    parser.add_argument("--n-particles", type=int, default=None, help="If set, generate this many total particles")
    parser.add_argument(
        "--output-parquet",
        type=str,
        default=None,
        help="If set with --n-particles, write merged synthetic parquet output",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gan-results-root",
        type=str,
        default=GAN_RESULTS_ROOT,
        help="Root directory containing GAN result subfolders (overrides hardcoded default)",
    )
    return parser.parse_args()


def _allocate_counts(total: int, ratios: list[float]) -> list[int]:
    raw = np.array(ratios, dtype=np.float64) * float(total)
    base = np.floor(raw).astype(int)
    remainder = int(total - int(base.sum()))
    if remainder > 0:
        frac = raw - base
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            base[int(idx)] += 1
    return base.tolist()


def _load_normalization_frame_for_pdg(
    pdg_code: int,
    preprocessed_file: str,
    norm_entries: Optional[int],
    norm_test_entries: Optional[int],
) -> pd.DataFrame:
    frame = load_preprocessed_data(
        preprocessed_file,
        pdg_code=int(pdg_code),
        entries=norm_entries,
        test_entries=norm_test_entries,
        keep_pdg=False,
    )
    if len(frame) == 0:
        raise RuntimeError(
            f"No normalization rows found for pdg={pdg_code} in {preprocessed_file}. "
            "Adjust --preprocessed-file / --norm-entries / --norm-test-entries."
        )
    if "pdg" in frame.columns:
        frame = frame.drop(columns=["pdg"], errors="ignore")
    return frame


def _default_output_parquet(folders: list[str], pdg_codes: list[int], n_particles: int) -> str:
    folder_tag = "-".join(folders)
    pdg_tag = "-".join(str(code) for code in pdg_codes)
    filename = f"synthetic_mix_pdg{pdg_tag}_{folder_tag}_n{int(n_particles)}.parquet"
    return str(Path(DEFAULT_SYNTHETIC_OUTPUT_DIR) / filename)


def _resolve_device(device_flag: str) -> str:
    if device_flag == "cpu":
        return "cpu"
    if device_flag == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    overall_start = time.perf_counter()
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(args.folders) != len(args.pdg_codes):
        raise ValueError("--folders and --pdg-codes must have the same length")

    folder_specs = []
    for folder_name, pdg_code in zip(args.folders, args.pdg_codes):
        generator_path = Path(args.gan_results_root) / folder_name / "generator.pth"
        if not generator_path.exists():
            raise FileNotFoundError(f"generator.pth not found for folder '{folder_name}': {generator_path}")
        folder_specs.append({
            "folder": folder_name,
            "pdg": int(pdg_code),
            "generator_path": str(generator_path),
        })

    print("Resolved generator checkpoints:")
    for spec in folder_specs:
        print(f"  - folder={spec['folder']}, pdg={spec['pdg']}, checkpoint={spec['generator_path']}")

    target_counts = [int(HARD_CODED_PDG_COUNTS.get(spec["pdg"], 0)) for spec in folder_specs]
    target_sum = int(sum(target_counts))
    if target_sum <= 0:
        raise RuntimeError(
            "None of the requested PDG codes were found in HARD_CODED_PDG_COUNTS. "
            f"Requested PDGs: {[spec['pdg'] for spec in folder_specs]}. "
            f"Available keys: {sorted(HARD_CODED_PDG_COUNTS.keys())}"
        )

    missing_pdgs = [spec["pdg"] for spec, count in zip(folder_specs, target_counts) if count <= 0]
    if missing_pdgs:
        print(f"[WARN] Missing hardcoded counts for PDGs {missing_pdgs}; they receive ratio 0.")

    ratios = [float(c) / float(target_sum) for c in target_counts]

    print("Specified PDG distribution ratios")
    print("ratio_source: hardcoded_counts")
    print(f"reference_parquet: {args.reference_parquet} (ignored for ratio computation)")
    print(f"subset_total_for_requested_pdgs: {target_sum}")
    print("-")
    print(f"{'folder':>28} {'pdg':>8} {'count':>14} {'ratio':>12}")
    for spec, count, ratio in zip(folder_specs, target_counts, ratios):
        print(f"{spec['folder']:>28} {spec['pdg']:>8d} {count:>14d} {ratio:>11.6f}")

    if args.n_particles is None:
        return 0

    if args.n_particles < 1:
        raise ValueError("--n-particles must be >= 1")
    output_parquet = args.output_parquet or _default_output_parquet(args.folders, args.pdg_codes, args.n_particles)
    if not args.output_parquet:
        print(f"[INFO] --output-parquet not set; using default: {output_parquet}")

    generated_counts = _allocate_counts(int(args.n_particles), ratios)
    device = _resolve_device(args.device)
    print("-")
    print(f"Generating mixed synthetic dataset: n_particles={args.n_particles}, device={device}")
    print("Allocation by folder:")
    for spec, count in zip(folder_specs, generated_counts):
        print(f"  - folder={spec['folder']}, pdg={spec['pdg']}, n_generate={count}")

    frames = []
    total_generated = 0
    for spec, n_gen in zip(folder_specs, generated_counts):
        if n_gen <= 0:
            continue
        step_start = time.perf_counter()
        pdg_code = int(spec["pdg"])
        normalization_df = _load_normalization_frame_for_pdg(
            pdg_code=pdg_code,
            preprocessed_file=args.preprocessed_file,
            norm_entries=args.norm_entries,
            norm_test_entries=args.norm_test_entries,
        )
        print(f"\n[STEP] Starting generation for folder={spec['folder']} (pdg={pdg_code}, n={n_gen})")
        print(f"       using loader source: {args.preprocessed_file}")
        print(f"       normalization rows={len(normalization_df)}, columns={list(normalization_df.columns)}")

        result = generate_synthetic_from_checkpoint(
            generator_path=spec["generator_path"],
            train_df=normalization_df,
            n_samples=n_gen,
            device=device,
            batch_size=int(args.batch_size),
            apply_angle_clipping=True,
        )
        df = pd.DataFrame(result["samples"], columns=result["feature_names"])
        df.insert(0, "pdg", pdg_code)
        frames.append(df)
        total_generated += len(df)
        elapsed = time.perf_counter() - step_start
        pct = 100.0 * float(total_generated) / float(args.n_particles)
        print(
            f"[DONE] folder={spec['folder']}, pdg={pdg_code}, n={n_gen}, elapsed={elapsed:.2f}s"
        )
        print(f"       cumulative_generated={total_generated}/{args.n_particles} ({pct:.1f}%)")

    if not frames:
        raise RuntimeError("No samples generated after allocation.")

    mixed_df = pd.concat(frames, ignore_index=True)
    mixed_df = mixed_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Apply physical solenoid radius bound: discard any row with r >= 350.5
    R_MAX = 350.5
    if "r" in mixed_df.columns:
        before = len(mixed_df)
        mixed_df = mixed_df[mixed_df["r"] < R_MAX].reset_index(drop=True)
        after = len(mixed_df)
        if before != after:
            print(f"[INFO] r-clip: removed {before - after:,} rows with r >= {R_MAX} ({100.0*(before-after)/before:.2f}%)")
    else:
        print("[WARN] Column 'r' not found in merged DataFrame; r-clip skipped.")

    print(f"\nMerged DataFrame ready: rows={len(mixed_df)}, columns={list(mixed_df.columns)}")

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_df.to_parquet(str(output_path), index=False)
    print(f"Saved merged synthetic parquet: {output_path}")
    print(f"Total elapsed: {time.perf_counter() - overall_start:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
