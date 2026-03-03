"""
src/data_collection/video_validator.py
=======================================
Scan raw_videos/ for .mp4/.avi/.mov files and update video_manifest.csv
with actual per-video metadata (fps, resolution, duration, frame_count).

Usage:
    python -m src.data_collection.video_validator
    python -m src.data_collection.video_validator --manifest data/video_manifest.csv
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "video_manifest.csv"
RAW_VIDEO_ROOT = PROJECT_ROOT / "data" / "raw_videos"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

REQUIRED_COLUMNS = [
    "video_id", "filename", "path", "source", "split", "condition",
    "fps", "width", "height", "duration_s", "frame_count",
    "annotated_frames", "status", "url", "notes",
]


def probe_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"ok": False, "error": f"cv2 could not open {video_path.name}"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "ok": True,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_s": round(duration_s, 2),
    }


def scan_raw_videos(root: Path) -> list[dict]:
    found = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            info = probe_video(p)
            info["path_abs"] = str(p)
            info["filename"] = p.name
            rel = p.relative_to(PROJECT_ROOT)
            info["path_rel"] = str(rel)
            source = "youtube" if "youtube" in str(p) else "clinical"
            info["source_guess"] = source
            found.append(info)
    return found


def update_manifest(manifest_path: Path, scanned: list[dict]) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    path_to_row = {row["path"]: i for i, row in df.iterrows()}
    filename_to_row = {row["filename"]: i for i, row in df.iterrows()}

    updated = 0
    added = 0

    for info in scanned:
        if not info.get("ok"):
            print(f"  [SKIP] {info.get('filename','?')}: {info.get('error','unknown error')}")
            continue

        idx = path_to_row.get(info["path_rel"]) or filename_to_row.get(info["filename"])

        if idx is not None:
            df.at[idx, "fps"] = info["fps"]
            df.at[idx, "width"] = info["width"]
            df.at[idx, "height"] = info["height"]
            df.at[idx, "frame_count"] = info["frame_count"]
            df.at[idx, "duration_s"] = info["duration_s"]
            df.at[idx, "status"] = "downloaded"
            updated += 1
        else:
            stem = Path(info["filename"]).stem
            new_row = {
                "video_id": stem,
                "filename": info["filename"],
                "path": info["path_rel"],
                "source": info["source_guess"],
                "split": "train_val",
                "condition": "unknown",
                "fps": info["fps"],
                "width": info["width"],
                "height": info["height"],
                "duration_s": info["duration_s"],
                "frame_count": info["frame_count"],
                "annotated_frames": 0,
                "status": "downloaded",
                "url": "",
                "notes": "Auto-detected by video_validator.py",
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            added += 1

    print(f"  Updated {updated} existing rows, added {added} new rows.")
    df.to_csv(manifest_path, index=False)
    return df


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    downloaded = (df["status"] == "downloaded").sum()
    pending = (df["status"] == "pending_download").sum()
    unavailable = (df["status"] == "not_available").sum()

    print("\n" + "=" * 60)
    print("  VIDEO MANIFEST SUMMARY")
    print("=" * 60)
    print(f"  Total rows        : {total}")
    print(f"  Downloaded        : {downloaded}")
    print(f"  Pending download  : {pending}")
    print(f"  Not available     : {unavailable}")
    print()

    print("  Condition breakdown (all rows):")
    for cond, count in df["condition"].value_counts().items():
        print(f"    {cond:<20} {count:>3}")
    print()

    print("  Source breakdown:")
    for src, count in df["source"].value_counts().items():
        print(f"    {src:<20} {count:>3}")
    print()

    if downloaded > 0:
        dl = df[df["status"] == "downloaded"]
        print("  Downloaded — resolution breakdown:")
        resolutions = dl.apply(lambda r: f"{int(r.width)}x{int(r.height)}", axis=1)
        for res, count in resolutions.value_counts().items():
            print(f"    {res:<20} {count:>3}")
        print()

        bad_fps = dl[(dl["fps"] < 29) | (dl["fps"] > 31)]
        if len(bad_fps):
            print(f"  [WARN] {len(bad_fps)} video(s) outside 29-31 fps range:")
            for _, row in bad_fps.iterrows():
                print(f"    {row.video_id}  fps={row.fps}")

        short = dl[dl["duration_s"] < 5]
        if len(short):
            print(f"  [WARN] {len(short)} video(s) shorter than 5 s:")
            for _, row in short.iterrows():
                print(f"    {row.video_id}  duration={row.duration_s}s")

    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw laryngoscopy videos and update manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to video_manifest.csv")
    parser.add_argument("--scan-only", action="store_true", help="Print scan results without updating manifest")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 1

    print(f"  Scanning: {RAW_VIDEO_ROOT}")
    scanned = scan_raw_videos(RAW_VIDEO_ROOT)
    print(f"  Found {len(scanned)} video file(s) on disk.\n")

    if args.scan_only:
        for info in scanned:
            status = "OK" if info.get("ok") else "FAIL"
            print(f"  [{status}]  {info.get('filename','?')}  {info.get('fps','?')}fps  "
                  f"{info.get('width','?')}x{info.get('height','?')}  "
                  f"{info.get('duration_s','?')}s")
        return 0

    df = update_manifest(manifest_path, scanned)
    print_summary(df)
    print(f"\n  Manifest saved to: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
