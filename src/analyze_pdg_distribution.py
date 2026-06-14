#!/usr/bin/env python3

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import pyarrow.dataset as ds


DEFAULT_PARQUET = "/home/hep/jcc525/cleaned_data/pdgNone_monitor4.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PDG code distribution in a parquet file without loading all columns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET, help="Input parquet path")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on processed rows for very large files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="If set, print only top-K PDG codes by count",
    )
    return parser.parse_args()


def analyze_pdg_distribution(parquet_path: str, max_rows: Optional[int] = None) -> Tuple[int, Counter]:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    dataset = ds.dataset(str(path), format="parquet")
    if "pdg" not in dataset.schema.names:
        raise RuntimeError(f"Column 'pdg' not found in parquet schema: {dataset.schema.names}")

    scanner = dataset.scanner(columns=["pdg"], use_threads=True)

    counts: Counter = Counter()
    processed = 0

    for batch in scanner.to_batches():
        pdg_values = batch.column("pdg").to_pylist()

        if max_rows is not None:
            remaining = max_rows - processed
            if remaining <= 0:
                break
            pdg_values = pdg_values[:remaining]

        counts.update(int(v) for v in pdg_values if v is not None)
        processed += len(pdg_values)

        if max_rows is not None and processed >= max_rows:
            break

    return processed, counts


def main() -> int:
    args = parse_args()

    n_rows, counts = analyze_pdg_distribution(args.parquet, max_rows=args.max_rows)
    if n_rows == 0:
        raise RuntimeError("No rows processed from parquet.")

    items = counts.most_common(args.top_k) if args.top_k else counts.most_common()

    print("PDG distribution")
    print(f"parquet: {args.parquet}")
    print(f"rows_processed: {n_rows}")
    print(f"unique_pdg: {len(counts)}")
    print("-")
    print(f"{'pdg':>12} {'count':>14} {'ratio':>12}")
    for pdg_code, count in items:
        ratio = float(count) / float(n_rows)
        print(f"{pdg_code:>12d} {count:>14d} {ratio:>11.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
