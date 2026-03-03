"""
src/data_collection/assign_splits.py
======================================
Assign train/val/test split labels to video_manifest.csv.

Split strategy (matching Koivu et al., 2026):
  - ld_test source: always 'test'  (independent NCT05216770 dataset)
  - clinical + youtube sources: 90% 'train', 10% 'val'
    stratified by condition so each condition is proportionally represented.

The split is recorded at the VIDEO level.  Frame-level 90/10 split
(1254 train / 140 val from 57 videos) is achieved later during frame
extraction (Phase 3) by proportionally sampling frames per video.

Usage:
    python -m src.data_collection.assign_splits
    python -m src.data_collection.assign_splits --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "video_manifest.csv"


def stratified_split(
    df: pd.DataFrame,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()

    train_val_mask = df["source"].isin(["clinical", "youtube"])
    test_mask = df["source"] == "ld_test"

    # LD test set is always 'test'
    df.loc[test_mask, "split"] = "test"

    tv_df = df[train_val_mask].copy()
    conditions = tv_df["condition"].unique()

    val_indices = []
    for cond in conditions:
        cond_idx = tv_df[tv_df["condition"] == cond].index.tolist()
        n_val = max(1, round(len(cond_idx) * val_fraction))
        chosen = rng.choice(cond_idx, size=n_val, replace=False).tolist()
        val_indices.extend(chosen)

    # Assign splits
    df.loc[train_val_mask, "split"] = "train"
    df.loc[val_indices, "split"] = "val"

    return df


def print_split_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  SPLIT ASSIGNMENT SUMMARY")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        print(f"\n  {split.upper()} ({len(sub)} videos):")
        for cond, count in sub["condition"].value_counts().items():
            print(f"    {cond:<20} {count:>3}")
    print()
    total_ann = df["annotated_frames"].sum()
    train_ann = df[df["split"] == "train"]["annotated_frames"].sum()
    val_ann = df[df["split"] == "val"]["annotated_frames"].sum()
    test_ann = df[df["split"] == "test"]["annotated_frames"].sum()
    print(f"  Annotated frame targets:")
    print(f"    train  : {train_ann}  (paper: ~1,254)")
    print(f"    val    : {val_ann}   (paper:   ~140)")
    print(f"    test   : {test_ann}    (paper:    ~50)")
    print(f"    total  : {total_ann}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assign train/val/test splits to video manifest.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--val-fraction", type=float, default=0.10,
                        help="Fraction of train_val videos assigned to validation (default 0.10).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 1

    df = pd.read_csv(manifest_path)
    df = stratified_split(df, val_fraction=args.val_fraction, seed=args.seed)
    df.to_csv(manifest_path, index=False)
    print(f"  Split column updated in: {manifest_path}")
    print_split_summary(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
