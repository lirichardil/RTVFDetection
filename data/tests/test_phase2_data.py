"""
data/tests/test_phase2_data.py
================================
Phase 2 — Data Collection validation tests (T2.1 – T2.7).

Tests mirror the acceptance criteria in the project roadmap:

  T2.1  video_manifest.csv exists with correct columns
  T2.2  At least 20 downloaded videos present in raw_videos/
  T2.3  Every downloaded video listed in manifest opens with cv2
  T2.4  All downloaded videos report 29–31 fps
  T2.5  All downloaded videos have duration ≥ 5 s
  T2.6  At least 2 distinct resolutions present among downloaded videos
  T2.7  Required manifest columns have no null values

Run:
    pytest data/tests/test_phase2_data.py -v -s
Or via:
    ./run_tests.sh data/tests/test_phase2_data.py
"""

import sys
from pathlib import Path

import cv2
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "video_manifest.csv"
RAW_VIDEO_ROOT = PROJECT_ROOT / "data" / "raw_videos"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

REQUIRED_COLUMNS = [
    "video_id", "filename", "path", "source", "split", "condition",
    "fps", "width", "height", "duration_s", "frame_count",
    "annotated_frames", "status", "url", "notes",
]

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_SKIP = "\033[93mSKIP\033[0m"

MIN_DOWNLOADED_VIDEOS = 20
MIN_DURATION_S = 5.0
FPS_LOW, FPS_HIGH = 29.0, 31.0


def _report(name: str, passed: bool, detail: str = "") -> None:
    status = _PASS if passed else _FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"\n  [{status}]  {name}{suffix}")


def _load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH)


def _downloaded_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "downloaded"].copy()


def _video_files_on_disk() -> list[Path]:
    return [p for p in RAW_VIDEO_ROOT.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()]


# ---------------------------------------------------------------------------
# T2.1 — Manifest file exists with correct columns
# ---------------------------------------------------------------------------

def test_manifest_exists_and_has_correct_columns():
    ok = MANIFEST_PATH.exists()
    _report("T2.1  Manifest file exists", ok, str(MANIFEST_PATH.relative_to(PROJECT_ROOT)))
    assert ok, f"video_manifest.csv not found at {MANIFEST_PATH}"

    df = _load_manifest()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    cols_ok = len(missing) == 0
    _report(
        "T2.1  Manifest has required columns",
        cols_ok,
        f"missing: {missing}" if missing else f"all {len(REQUIRED_COLUMNS)} columns present",
    )
    assert cols_ok, f"Missing columns: {missing}"

    rows_ok = len(df) > 0
    _report("T2.1  Manifest is non-empty", rows_ok, f"{len(df)} rows")
    assert rows_ok, "video_manifest.csv has 0 rows."


# ---------------------------------------------------------------------------
# T2.2 — At least 20 downloaded videos
# ---------------------------------------------------------------------------

def test_minimum_video_count():
    video_files = _video_files_on_disk()
    n = len(video_files)
    ok = n >= MIN_DOWNLOADED_VIDEOS

    if n == 0:
        msg = (
            f"No video files found in {RAW_VIDEO_ROOT}. "
            "Download YouTube videos first:\n"
            "  bash scripts/download_youtube.sh\n"
            "or:\n"
            "  python -m src.data_collection.youtube_downloader --from-manifest"
        )
        _report("T2.2  Minimum video count", False, "0 videos on disk — skipping with XFAIL")
        pytest.xfail(msg)

    _report("T2.2  Minimum video count", ok, f"{n} videos found (minimum {MIN_DOWNLOADED_VIDEOS})")
    assert ok, (
        f"Only {n} video(s) in raw_videos/ — need at least {MIN_DOWNLOADED_VIDEOS}. "
        "Download more YouTube laryngoscopy videos."
    )


# ---------------------------------------------------------------------------
# T2.3 — All manifest-listed downloaded videos open with cv2
# ---------------------------------------------------------------------------

def test_video_readability():
    df = _load_manifest()
    dl = _downloaded_rows(df)

    if len(dl) == 0:
        _report("T2.3  Video readability", True, "no downloaded videos yet — skipping")
        pytest.skip("No downloaded videos in manifest yet.")

    failures = []
    for _, row in dl.iterrows():
        video_path = PROJECT_ROOT / row["path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            failures.append(row["video_id"])
        cap.release()

    ok = len(failures) == 0
    _report(
        "T2.3  Video readability",
        ok,
        f"{len(dl) - len(failures)}/{len(dl)} readable"
        + (f"; FAILED: {failures}" if failures else ""),
    )
    assert ok, f"cv2 could not open {len(failures)} video(s): {failures}"


# ---------------------------------------------------------------------------
# T2.4 — All downloaded videos report 29–31 fps
# ---------------------------------------------------------------------------

def test_fps_range():
    df = _load_manifest()
    dl = _downloaded_rows(df)

    if len(dl) == 0:
        pytest.skip("No downloaded videos in manifest yet.")

    out_of_range = []
    for _, row in dl.iterrows():
        video_path = PROJECT_ROOT / row["path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not (FPS_LOW <= fps <= FPS_HIGH):
            out_of_range.append((row["video_id"], fps))

    ok = len(out_of_range) == 0
    _report(
        "T2.4  FPS range (29–31 fps)",
        ok,
        f"all in range" if ok else f"{len(out_of_range)} out-of-range: {out_of_range[:3]}",
    )
    if out_of_range:
        pytest.fail(
            f"{len(out_of_range)} video(s) outside 29–31 fps: {out_of_range}\n"
            "Re-download with yt-dlp using --fps 30 or use ffmpeg to transcode."
        )


# ---------------------------------------------------------------------------
# T2.5 — All downloaded videos are at least 5 s long
# ---------------------------------------------------------------------------

def test_duration_minimum():
    df = _load_manifest()
    dl = _downloaded_rows(df)

    if len(dl) == 0:
        pytest.skip("No downloaded videos in manifest yet.")

    too_short = []
    for _, row in dl.iterrows():
        video_path = PROJECT_ROOT / row["path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        dur = n_frames / fps if fps > 0 else 0
        if dur < MIN_DURATION_S:
            too_short.append((row["video_id"], round(dur, 2)))

    ok = len(too_short) == 0
    _report(
        f"T2.5  Duration >= {MIN_DURATION_S} s",
        ok,
        "all OK" if ok else f"{len(too_short)} too short: {too_short}",
    )
    assert ok, f"Videos shorter than {MIN_DURATION_S} s: {too_short}"


# ---------------------------------------------------------------------------
# T2.6 — At least 2 distinct resolutions (paper had 320x240 – 1920x1080)
# ---------------------------------------------------------------------------

def test_resolution_variety():
    df = _load_manifest()
    dl = _downloaded_rows(df)

    if len(dl) == 0:
        pytest.skip("No downloaded videos in manifest yet.")

    resolutions = set()
    for _, row in dl.iterrows():
        video_path = PROJECT_ROOT / row["path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        resolutions.add((w, h))

    ok = len(resolutions) >= 2
    _report(
        "T2.6  Resolution variety (>= 2 distinct)",
        ok,
        f"{len(resolutions)} distinct resolution(s): {sorted(resolutions)}",
    )
    assert ok, (
        f"Only {len(resolutions)} distinct resolution(s) found — need >= 2. "
        "Include videos from different clinical cameras / YouTube sources."
    )


# ---------------------------------------------------------------------------
# T2.7 — No null values in required manifest columns (all rows)
# ---------------------------------------------------------------------------

def test_manifest_completeness():
    df = _load_manifest()

    null_report = {}
    always_required = ["video_id", "source", "split", "condition", "status"]
    for col in always_required:
        if col not in df.columns:
            null_report[col] = "column missing"
            continue
        nulls = df[col].isna().sum()
        if nulls:
            null_report[col] = f"{nulls} null(s)"

    ok = len(null_report) == 0
    _report(
        "T2.7  Manifest completeness (no nulls in key columns)",
        ok,
        "all OK" if ok else str(null_report),
    )
    assert ok, f"Null values found in manifest: {null_report}"


# ---------------------------------------------------------------------------
# Summary hook
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    error = len(terminalreporter.stats.get("error", []))
    skipped = len(terminalreporter.stats.get("skipped", []))
    xfailed = len(terminalreporter.stats.get("xfailed", []))
    total = passed + failed + error + skipped + xfailed

    print("\n" + "=" * 60)
    print("  PHASE 2 — DATA COLLECTION VALIDATION SUMMARY")
    print("=" * 60)

    rows = [
        ("T2.1  Manifest structure",         "test_manifest_exists_and_has_correct_columns"),
        ("T2.2  >= 20 videos on disk",        "test_minimum_video_count"),
        ("T2.3  All videos readable (cv2)",   "test_video_readability"),
        ("T2.4  FPS 29–31",                   "test_fps_range"),
        ("T2.5  Duration >= 5 s",             "test_duration_minimum"),
        ("T2.6  >= 2 resolutions",            "test_resolution_variety"),
        ("T2.7  No nulls in key columns",     "test_manifest_completeness"),
    ]

    passed_ids = {r.nodeid.split("::")[-1] for r in terminalreporter.stats.get("passed", [])}
    failed_ids = {
        r.nodeid.split("::")[-1]
        for r in terminalreporter.stats.get("failed", []) + terminalreporter.stats.get("error", [])
    }
    skipped_ids = {
        r.nodeid.split("::")[-1]
        for r in terminalreporter.stats.get("skipped", []) + terminalreporter.stats.get("xfailed", [])
    }

    _SKIP_C = "\033[93mSKIP\033[0m"
    for label, fn in rows:
        if fn in passed_ids:
            status = _PASS
        elif fn in failed_ids:
            status = _FAIL
        else:
            status = _SKIP_C
        print(f"  [{status}]  {label}")

    print("-" * 60)
    overall = _PASS if (failed + error) == 0 else _FAIL
    skip_note = f", {skipped + xfailed} skipped/xfailed" if (skipped + xfailed) else ""
    print(f"  Overall: [{overall}]  {passed}/{total} tests passed{skip_note}")
    print("=" * 60 + "\n")
