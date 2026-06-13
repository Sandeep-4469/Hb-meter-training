#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "================================================"
echo "  METRIC LEARNING PIPELINE"
echo "================================================"
python extract.py      && echo "✓ Step 1 — Feature extraction"
python learn.py        && echo "✓ Step 2 — LMNN metric learning"
python train.py        && echo "✓ Step 3 — Train & evaluate"
echo ""
echo "Results in metric_learning/results/"
