import pandas as pd
import glob
import os
import argparse

def merge_batch_files(batch_pattern, output_file):
    """Merge batched parquet files into a single file."""
    
    # Find all batch files
    files = sorted(glob.glob(batch_pattern))
    
    if not files:
        print(f"❌ No files found matching: {batch_pattern}")
        return
    
    print(f"Found {len(files)} batch files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # Read and merge
    dfs = []
    total_entries = 0
    
    for f in files:
        print(f"\nReading {os.path.basename(f)}...")
        df = pd.read_parquet(f)
        entries = len(df)
        total_entries += entries
        print(f"  {entries:,} entries")
        dfs.append(df)
    
    print(f"\nConcatenating {len(dfs)} dataframes...")
    combined = pd.concat(dfs, ignore_index=True)
    
    print(f"Saving {len(combined):,} total entries to {output_file}...")
    combined.to_parquet(output_file, compression='gzip', index=False)
    
    file_size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"\n✅ Merge complete!")
    print(f"   Total entries: {len(combined):,}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {file_size_gb:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge batched parquet files")
    parser.add_argument("--pdg", type=int, default=None, help="PDG code (default: all)")
    parser.add_argument("--monitor-id", type=int, default=None, help="MonitorID (default: all)")
    args = parser.parse_args()
    
    # Build file pattern and output path
    batch_pattern = f"/home/hep/jcc525/cleaned_data/pdg{args.pdg}_monitor{args.monitor_id}_batch*.parquet"
    output_file = f"/home/hep/jcc525/cleaned_data/pdg{args.pdg}_monitor{args.monitor_id}.parquet"
    
    merge_batch_files(batch_pattern, output_file)

# cd gan_particle_physics/src/
# python merge_batches.py --pdg 13 --monitor-id 4