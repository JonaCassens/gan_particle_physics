import os
import glob
import pandas as pd
from data_loader import load_root_files

def preprocess_all_rootfiles(file_pattern, output_file, max_files=None, pdg_code=13, monitor_id_value=4, batch_num=None, total_batches=None):
    """Convert ROOT files to Parquet (optionally batched)."""
    all_files = sorted(glob.glob(file_pattern))
    total_files = len(all_files)
    
    print(f"Total files found: {total_files}")
    
    # Handle batching
    if batch_num is not None and total_batches is not None:
        files_per_batch = total_files // total_batches
        start_idx = batch_num * files_per_batch
        end_idx = start_idx + files_per_batch if batch_num < total_batches - 1 else total_files
        all_files = all_files[start_idx:end_idx]
        print(f"Processing batch {batch_num}/{total_batches}: files {start_idx}-{end_idx-1} ({len(all_files)} files)")
    elif max_files:
        all_files = all_files[:max_files]
        print(f"Processing {max_files} files")
    
    print(f"Loading {len(all_files)} ROOT files...")
    
    dfs = []
    for i, f in enumerate(all_files):
        print(f"  [{i+1}/{len(all_files)}] {os.path.basename(f)}")
        try:
            df = load_root_files(f, max_files=1, pdg_code=pdg_code, monitor_id_value=monitor_id_value)
            dfs.append(df)
        except Exception as e:
            print(f"    ⚠️  Error: {e}")
            continue
    
    if not dfs:
        print("❌ No files loaded")
        return
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(df)} entries total")
    
    print(f"Saving to {output_file}")
    df.to_parquet(output_file, compression='gzip', index=False)
    
    file_size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"✅ Complete: {file_size_gb:.2f} GB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--pdg", type=int, default=None, help="PDG code to filter")
    parser.add_argument("--monitor-id", type=int, default=None, help="MonitorID to filter")
    parser.add_argument("--batch-num", type=int, default=None, help="Batch number (0-indexed)")
    parser.add_argument("--total-batches", type=int, default=None, help="Total number of batches")
    args = parser.parse_args()
    
    if args.output is None:
        if args.batch_num is not None:
            args.output = f"/home/hep/jcc525/cleaned_data/pdg{args.pdg}_monitor{args.monitor_id}_batch{args.batch_num}.parquet"
        else:
            args.output = f"/home/hep/jcc525/cleaned_data/pdg{args.pdg}_monitor{args.monitor_id}.parquet"
    
    file_pattern = "/home/hep/jcc525/comet_data/midstream_merged*.rootracker"

    preprocess_all_rootfiles(file_pattern, args.output, args.max_files, args.pdg, args.monitor_id, args.batch_num, args.total_batches)
