#!/bin/bash
# Full pipeline — middle-frame extraction, all videos.
set -e
cd "$(dirname "$0")"

echo "================================================"
echo "  HB METER FINAL — Full Pipeline"
echo "================================================"

python 01_eda.py               && echo "✓ Stage 1  — EDA"
python 02_segment.py           && echo "✓ Stage 2  — LED segment detection"
python 03_extract.py           && echo "✓ Stage 3  — Feature extraction"
python 03c_bin_analysis.py     && echo "✓ Stage 3c — Bin discriminability"
python 04_train.py             && echo "✓ Stage 4  — Training"
python 05_evaluate.py          && echo "✓ Stage 5  — Evaluation"

echo ""
echo "Done. Results in outputs/results/"
