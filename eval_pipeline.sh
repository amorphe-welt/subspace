#!/bin/bash

################################################################################
# Full Evaluation Pipeline Execution Script
# 
# This script runs the complete evaluation pipeline from start to finish.
# Modify the variables below to suit your data and computational resources.
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data
INPUT_DATA="data/AmDi.epsilon.jsonl"
INPUT_DATA_UNLABELED ="data/unlabeled.jsonl"

# Model configuration
MODEL_NAME="pierluigic/xl-lexeme"
POOLING="mean"
BATCH_SIZE=32
LAYERS="14 16 18 20 21 22 23 24"

# PCA settings (set USE_PCA=false to disable)
USE_PCA=false
N_COMPONENTS=100

# Pseudo-labeling threshold (0-1, higher = stricter)
CONFIDENCE_THRESHOLD=0.5

# Output directory
OUTPUT_DIR="xl-lexeme"

# Evaluations directory
EVAL_DIR="xl-lexeme/evaluations"

# Random seed for reproducibility
SEED=42

# Device (cuda or cpu)
DEVICE="cuda"

# ============================================================================
# SETUP
# ============================================================================

echo "========================================"
echo "Evaluation Pipeline"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Input: $INPUT_DATA"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# STAGE 1
# ============================================================================

echo ""
echo "=========================================================" 
echo "Stage 1: Filtering Evaluation: Sense Pseudo-Label Quality"
echo "========================================================="
echo ""

python E1_evaluate_filtering.py \
    --repr-dir "$OUTPUT_DIR/embeddings/gold" \
    --subspace-dir "$OUTPUT_DIR/subspaces" \
    --output "$EVAL_DIR" \
    --thresholds 0.3 0.4 0.5

echo "✓ Stage 1 complete"

# ============================================================================
# STAGE 2
# ============================================================================

echo ""
echo "========================================"
echo "Stage 2: Subspace Reliability Evaluation"
echo "========================================"
echo ""

python E2_evaluate_subspace_reliability.py \
    --subspace-dir "$OUTPUT_DIR/subspaces" \
    --repr-dir "$OUTPUT_DIR/embeddings/gold" \
    --output "$EVAL_DIR" \
    --n-resamples 5

echo "✓ Stage 2 complete"

# ============================================================================
# STAGE 3
# ============================================================================

echo ""
echo "=========================================================" 
echo "Stage 3: Subspace Quality Analysis (Normalized)"
echo "========================================================="
echo ""

python E3_subspace_quality_analysis.py \
    --reliability-dir "$EVAL_DIR" \
    --pseudo-jsonl "OUTPUT_DIR/pseudo_labeled.jsonl" \
    --outdir "$EVAL_DIR"

echo "✓ Stage 3 complete"


# ============================================================================
# STAGE 4
# ============================================================================

echo ""
echo "=========================================="
echo "Stage 4: Pseudo-Label Filtering Evaluation"
echo "=========================================="
echo ""

python E4_pseudo_label_eval.py \
    --pseudo-jsonl "OUTPUT_DIR/pseudo_labeled.jsonl" \
    --outdir "$EVAL_DIR"

echo "✓ Stage 4 complete"

# ============================================================================
# STAGE 5
# ============================================================================

echo ""
echo "=========================================================="
echo "Stage 5: Align Subspace Quality with Pseudo-Label Behavior"
echo "=========================================================="
echo ""

python E5_analyse_quality_vs_pseudo.py \
    --subspace-json "$EVAL_DIR/subspace_quality.json" \
    --pseudo-json "$EVAL_DIR/pseudo_label_metrics.json" \
    --outdir "$EVAL_DIR"
    
echo "✓ Stage 5 complete"


# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "====================================================================="
echo "Evaluation Pipeline Complete!"
echo "====================================================================="
echo ""
