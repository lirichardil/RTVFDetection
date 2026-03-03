#!/usr/bin/env bash
# =============================================================================
# scripts/download_youtube.sh
# Download YouTube laryngoscopy videos for the RTVF Detection dataset.
#
# The paper (Koivu et al., 2026) used 10 YouTube videos (mean 183 s each)
# showing flexible white-light in-office laryngoscopy under fair use.
#
# Usage:
#   bash scripts/download_youtube.sh                  # guided interactive mode
#   bash scripts/download_youtube.sh <URL> [<URL>...]  # download specific URLs
#
# Prerequisites:
#   conda activate rtvf-detection
#   pip install yt-dlp   (if not already installed)
#
# What to look for on YouTube:
#   Search terms used by the paper authors:
#     "flexible laryngoscopy normal vocal folds demonstration"
#     "in-office nasolaryngoscopy tutorial ENT"
#     "videolaryngoscopy vocal fold abduction adduction phonation"
#     "laryngoscopy examination vocal fold normal movement"
#     "flexible nasolaryngoscopy vocal cord normal appearance"
#
#   Selection criteria (matching Table 1, Koivu et al.):
#     ✓ White-light flexible transnasal laryngoscopy (NOT stroboscopy, NOT rigid)
#     ✓ Glottis (vocal folds) clearly visible throughout most of the video
#     ✓ Duration 60–360 s preferred (paper mean 183 s)
#     ✓ Tasks: vowel phonation (/i/), breathing, sniffing
#     ✓ Any resolution 320×240 – 1280×720 accepted
#     ✗ Exclude: stroboscopy, high-speed video, rigid endoscopy, micro-laryngoscopy
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/data/raw_videos/youtube"
MANIFEST="$PROJECT_ROOT/data/video_manifest.csv"

# yt-dlp format: best quality ≤ 720p, prefer mp4
FORMAT="bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best"

echo "============================================================"
echo "  RTVF Detection — YouTube Video Downloader"
echo "  Output dir : $OUTPUT_DIR"
echo "  Manifest   : $MANIFEST"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Check yt-dlp
# ---------------------------------------------------------------------------
if ! command -v yt-dlp &>/dev/null; then
    echo "  [ERROR] yt-dlp is not installed."
    echo "          Install it with:  pip install yt-dlp"
    exit 1
fi
echo "  yt-dlp version: $(yt-dlp --version)"
echo ""

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Determine URLs
# ---------------------------------------------------------------------------
URLS=("$@")

if [ ${#URLS[@]} -eq 0 ]; then
    echo "  No URLs provided.  Running in guided mode."
    echo ""
    echo "  You need 10 YouTube videos showing flexible in-office laryngoscopy."
    echo "  Recommended search terms:"
    echo "    1. flexible laryngoscopy normal vocal folds demonstration"
    echo "    2. in-office nasolaryngoscopy tutorial ENT"
    echo "    3. videolaryngoscopy vocal fold abduction adduction"
    echo "    4. laryngoscopy examination vocal fold normal movement"
    echo "    5. flexible nasolaryngoscopy vocal cord normal"
    echo "    6. laryngoscopy vocal fold paralysis comparison"
    echo "    7. transnasal laryngoscopy procedure demonstration"
    echo "    8. endoscopy larynx vocal cord phonation sniffing"
    echo "    9. fiberoptic laryngoscopy tutorial ENT clinic"
    echo "   10. laryngoscopy breathing phonation sniffing tasks"
    echo ""
    echo "  Once you have URLs, run:"
    echo "    bash scripts/download_youtube.sh <URL1> <URL2> ..."
    echo ""
    echo "  Or paste them into the manifest (url column) and run:"
    echo "    python -m src.data_collection.youtube_downloader --from-manifest"
    exit 0
fi

# ---------------------------------------------------------------------------
# Determine starting video_id index
# ---------------------------------------------------------------------------
NEXT_IDX=1
if [ -f "$MANIFEST" ]; then
    # Count existing YT_ rows to find next available slot
    EXISTING=$(grep -c "^YT_" "$MANIFEST" 2>/dev/null || echo "0")
    NEXT_IDX=$((EXISTING + 1))
fi

# ---------------------------------------------------------------------------
# Download loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
IDX=$NEXT_IDX

for URL in "${URLS[@]}"; do
    VIDEO_ID=$(printf "YT_%03d" "$IDX")
    OUT_TEMPLATE="$OUTPUT_DIR/${VIDEO_ID}.%(ext)s"

    echo "  Downloading $VIDEO_ID: $URL"

    if yt-dlp \
        --format "$FORMAT" \
        --output "$OUT_TEMPLATE" \
        --merge-output-format mp4 \
        --no-playlist \
        --quiet \
        --no-warnings \
        "$URL"; then

        echo "  [PASS] $VIDEO_ID downloaded."
        PASS=$((PASS + 1))
        IDX=$((IDX + 1))

        # Update the manifest if the row exists
        if command -v python &>/dev/null; then
            python - "$VIDEO_ID" "$URL" "$MANIFEST" <<'PYEOF'
import sys, csv, os, pathlib
vid_id, url, manifest = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.exists(manifest):
    sys.exit(0)
rows = []
updated = False
with open(manifest, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row["video_id"] == vid_id:
            row["url"] = url
            row["status"] = "downloaded"
            updated = True
        rows.append(row)
if updated:
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Manifest updated for {vid_id}.")
PYEOF
        fi
    else
        echo "  [FAIL] $VIDEO_ID — yt-dlp returned non-zero for $URL"
        FAIL=$((FAIL + 1))
        IDX=$((IDX + 1))
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Downloads complete: $PASS passed, $FAIL failed."
echo ""
echo "  Next steps:"
echo "    1. Run video_validator.py to update manifest with actual metadata:"
echo "         python -m src.data_collection.video_validator"
echo "    2. Run assign_splits.py to finalize train/val/test labels:"
echo "         python -m src.data_collection.assign_splits"
echo "    3. Run Phase 2 tests:"
echo "         pytest data/tests/test_phase2_data.py -v -s"
echo "============================================================"

[ "$FAIL" -gt 0 ] && exit 1 || exit 0
