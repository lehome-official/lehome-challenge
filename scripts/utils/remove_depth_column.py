#!/usr/bin/env python3
"""
LeRobot V3 Dataset Cleaner (Deep Clean Version)

Features:
1. Data: Remove depth column and merge files.
2. Episodes:
   - Remove all depth-related statistics columns (stats/observation.top_depth/...)
   - Force reset all data pointers (data/file_index, videos/.../file_index) to 0
   - Merge files
3. Meta: Sync and clean JSON files

#### 🛠️ Usage Example
```bash
# Clean a single dataset
python scripts/utils/remove_depth_column.py \
    --dataset_root Datasets/001 \
    --output_root Datasets/001_no_depth

# Clean another dataset
python scripts/utils/remove_depth_column.py \
    --dataset_root Datasets/002 \
    --output_root Datasets/002_no_depth

# Merge cleaned datasets (optional)
python -m scripts.dataset merge \
    --source_roots "['Datasets/001_no_depth', 'Datasets/002_no_depth']" \
    --output_root "Datasets/merged" \
    --output_repo_id "merged_dataset"
```
"""

import argparse
import json
import shutil
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


def clean_episodes_table(table: pa.Table, rm_col_name: str) -> pa.Table:
    """
    Deep clean the Episodes table by removing depth columns and resetting file indices.
    """
    # 1. Find and remove all columns containing rm_col_name (e.g., stats/observation.top_depth/min)
    cols_to_drop = [c for c in table.column_names if rm_col_name in c]
    if cols_to_drop:
        print(f"   ✂️  [Episodes] Removing {len(cols_to_drop)} depth-related statistics columns...")
        table = table.drop(cols_to_drop)

    # 2. Find all file index columns (ending with /file_index or /chunk_index)
    # Since we merge all data into file-000, these must all be reset to 0
    index_cols = [
        c
        for c in table.column_names
        if c.endswith("/file_index") or c.endswith("/chunk_index")
    ]

    if index_cols:
        print(f"   🔧 [Episodes] Resetting {len(index_cols)} file index columns to 0...")
        for col in index_cols:
            # Remove old column
            col_idx = table.column_names.index(col)
            table = table.remove_column(col_idx)
            # Add new column with all zeros
            zero_array = pa.array([0] * table.num_rows, type=pa.int64())
            table = table.add_column(col_idx, col, zero_array)

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--column_to_remove", type=str, default="observation.top_depth")
    args = parser.parse_args()

    src_root = Path(args.dataset_root).resolve()
    dst_root = Path(args.output_root).resolve()
    rm_col = args.column_to_remove

    if dst_root.exists():
        shutil.rmtree(dst_root)

    print(f"🚀 Processing: {src_root.name} -> {dst_root.name}")

    # ==========================================
    # 1. DATA Processing
    # ==========================================
    print("📊 [1/4] Merging Data...")
    dst_data_chunk = dst_root / "data" / "chunk-000"
    dst_data_chunk.mkdir(parents=True)

    data_files = sorted(list((src_root / "data").rglob("*.parquet")))
    tables = []
    for f in data_files:
        t = pq.read_table(f)
        if rm_col in t.column_names:
            t = t.drop([rm_col])
        tables.append(t)

    full_table = pa.concat_tables(tables)
    pq.write_table(full_table, dst_data_chunk / "file-000.parquet")
    total_rows = full_table.num_rows

    # ==========================================
    # 2. EPISODES Processing (Deep Clean)
    # ==========================================
    print("🧹 [2/4] Cleaning & Merging Episodes...")
    dst_ep_chunk = dst_root / "meta" / "episodes" / "chunk-000"
    dst_ep_chunk.mkdir(parents=True)

    ep_files = sorted(list((src_root / "meta" / "episodes").rglob("*.parquet")))
    if ep_files:
        ep_tables = [pq.read_table(f) for f in ep_files]
        full_ep_table = pa.concat_tables(ep_tables)

        # Call cleaning function
        full_ep_table = clean_episodes_table(full_ep_table, rm_col)

        pq.write_table(full_ep_table, dst_ep_chunk / "file-000.parquet")
    else:
        print("⚠️  Warning: No episodes found!")

    # ==========================================
    # 3. META Synchronization
    # ==========================================
    print("📝 [3/4] Syncing Metadata...")
    for item in (src_root / "meta").glob("*"):
        if item.name == "episodes":
            continue
        dst_item = dst_root / "meta" / item.name

        if item.name == "info.json":
            info = json.loads(item.read_text())
            if "features" in info and rm_col in info["features"]:
                del info["features"][rm_col]
            info["chunks"] = 1
            info["total_frames"] = total_rows
            dst_item.write_text(json.dumps(info, indent=4))
        elif item.name == "stats.json":
            stats = json.loads(item.read_text())
            if rm_col in stats:
                del stats[rm_col]
            dst_item.write_text(json.dumps(stats, indent=4))
        elif item.is_dir():
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)

    # ==========================================
    # 4. VIDEOS Synchronization
    # ==========================================
    print("🎥 [4/4] Copying Videos...")
    if (src_root / "videos").exists():
        shutil.copytree(
            src_root / "videos",
            dst_root / "videos",
            ignore=shutil.ignore_patterns(f"{rm_col}.mp4"),
        )

    print("✨ Done!")


if __name__ == "__main__":
    main()
