"""
src/data_collection/youtube_downloader.py
==========================================
Download YouTube laryngoscopy videos using yt-dlp and update video_manifest.csv.

The paper (Koivu et al., 2026) used 10 YouTube videos (mean 183.1 s each, ~8.5 min
total) showing in-office flexible white-light laryngoscopy.  These videos were
downloaded under fair use for academic research.

Usage:
    # Download a single URL, auto-assign the next YT_NNN slot
    python -m src.data_collection.youtube_downloader https://www.youtube.com/watch?v=XXXXX

    # Download all pending YouTube entries from the manifest
    python -m src.data_collection.youtube_downloader --from-manifest

    # Dry run (print yt-dlp command but do not execute)
    python -m src.data_collection.youtube_downloader --dry-run https://www.youtube.com/watch?v=XXXXX

Selection criteria (matching the paper):
    - White-light flexible transnasal/transoral laryngoscopy (NOT stroboscopy, NOT rigid)
    - Glottis (vocal folds) clearly visible throughout most of the video
    - Duration 60–360 s preferred (paper mean 183 s)
    - Content: vowel phonation, breathing, sniffing, speech tasks
    - Resolution ≥ 320×240 (any quality accepted; paper range 320×240–1280×720)

Recommended YouTube search terms:
    "flexible laryngoscopy normal vocal folds demonstration"
    "in-office nasolaryngoscopy tutorial ENT vocal folds"
    "videolaryngoscopy vocal fold abduction adduction phonation"
    "laryngoscopy examination vocal fold normal movement breathing"
    "transnasal laryngoscopy procedure demonstration glottis"
    "fiberoptic laryngoscopy tutorial ENT clinic"
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOUTUBE_DIR = PROJECT_ROOT / "data" / "raw_videos" / "youtube"
MANIFEST_PATH = PROJECT_ROOT / "data" / "video_manifest.csv"

# yt-dlp format selector: best video+audio ≤ 1280p, prefer mp4
FORMAT_SELECTOR = (
    "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
    "bestvideo[height<=720]+bestaudio/"
    "best[height<=720]/"
    "best"
)


def _ytdlp_available() -> bool:
    return shutil.which("yt-dlp") is not None


def _next_yt_id(df: pd.DataFrame) -> str:
    yt_rows = df[df["source"] == "youtube"]
    existing = set(yt_rows["video_id"].tolist())
    for i in range(1, 100):
        candidate = f"YT_{i:03d}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("Could not allocate a new YT_NNN slot — manifest may be full.")


def _pending_urls(df: pd.DataFrame) -> list[tuple[str, str]]:
    mask = (df["source"] == "youtube") & (df["status"] == "pending_download")
    rows = df[mask]
    results = []
    for _, row in rows.iterrows():
        url = str(row.get("url", ""))
        if url.startswith("http"):
            results.append((row["video_id"], url))
    return results


def download_video(url: str, video_id: str, dry_run: bool = False) -> dict:
    YOUTUBE_DIR.mkdir(parents=True, exist_ok=True)
    out_template = str(YOUTUBE_DIR / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", FORMAT_SELECTOR,
        "--output", out_template,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--write-info-json",
        "--quiet",
        "--no-warnings",
        url,
    ]

    print(f"  Downloading {video_id}: {url}")
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return {"ok": True, "dry_run": True, "video_id": video_id, "url": url}

    if not _ytdlp_available():
        print("  [ERROR] yt-dlp not found. Install with:  pip install yt-dlp")
        return {"ok": False, "error": "yt-dlp not installed"}

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] yt-dlp exited {result.returncode}: {result.stderr[:200]}")
        return {"ok": False, "error": result.stderr[:200]}

    # Find the downloaded file
    mp4_path = YOUTUBE_DIR / f"{video_id}.mp4"
    if not mp4_path.exists():
        # Search for any file with this video_id stem
        matches = list(YOUTUBE_DIR.glob(f"{video_id}.*"))
        mp4_path = matches[0] if matches else None

    # Extract title from info json if present
    info_json = YOUTUBE_DIR / f"{video_id}.info.json"
    title = ""
    if info_json.exists():
        try:
            with open(info_json) as f:
                meta = json.load(f)
            title = meta.get("title", "")
            info_json.unlink()  # clean up
        except Exception:
            pass

    print(f"  [OK]   {video_id} -> {mp4_path.name if mp4_path else '?'}")
    return {
        "ok": True,
        "video_id": video_id,
        "filename": mp4_path.name if mp4_path else f"{video_id}.mp4",
        "path": str(mp4_path.relative_to(PROJECT_ROOT)) if mp4_path else "",
        "title": title,
        "url": url,
    }


def update_manifest_after_download(result: dict) -> None:
    if not result.get("ok") or result.get("dry_run"):
        return
    df = pd.read_csv(MANIFEST_PATH)
    mask = df["video_id"] == result["video_id"]

    if mask.any():
        df.loc[mask, "filename"] = result["filename"]
        df.loc[mask, "path"] = result["path"]
        df.loc[mask, "status"] = "downloaded"
        if result.get("url"):
            df.loc[mask, "url"] = result["url"]
        if result.get("title"):
            df.loc[mask, "notes"] = df.loc[mask, "notes"].astype(str) + f"; title: {result['title']}"
    else:
        new_row = {
            "video_id": result["video_id"],
            "filename": result["filename"],
            "path": result["path"],
            "source": "youtube",
            "split": "train_val",
            "condition": "mixed",
            "fps": 0,
            "width": 0,
            "height": 0,
            "duration_s": 0,
            "frame_count": 0,
            "annotated_frames": 20,
            "status": "downloaded",
            "url": result.get("url", ""),
            "notes": result.get("title", "Downloaded via youtube_downloader.py"),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"  Manifest updated for {result['video_id']}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download YouTube laryngoscopy videos for the RTVF dataset."
    )
    parser.add_argument("urls", nargs="*", help="One or more YouTube URLs to download.")
    parser.add_argument(
        "--from-manifest", action="store_true",
        help="Download all pending_download YouTube entries that have real http URLs."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    if not args.urls and not args.from_manifest:
        parser.print_help()
        return 1

    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest not found: {MANIFEST_PATH}")
        return 1

    df = pd.read_csv(MANIFEST_PATH)
    errors = 0

    if args.from_manifest:
        pending = _pending_urls(df)
        if not pending:
            print("  No pending YouTube entries with real URLs found in manifest.")
            print("  Edit video_manifest.csv: replace SEARCH: placeholders with real YouTube URLs,")
            print("  then re-run with --from-manifest.")
            return 0
        for vid_id, url in pending:
            result = download_video(url, vid_id, dry_run=args.dry_run)
            update_manifest_after_download(result)
            if not result["ok"]:
                errors += 1

    for url in args.urls:
        df_fresh = pd.read_csv(MANIFEST_PATH)
        vid_id = _next_yt_id(df_fresh)
        result = download_video(url, vid_id, dry_run=args.dry_run)
        update_manifest_after_download(result)
        if not result["ok"]:
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
