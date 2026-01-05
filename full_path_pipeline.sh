#!/bin/bash

################################################################################
# Full Pipeline Execution Script
# 
# This script runs the complete sense-subspace LSCD pipeline from start to finish.
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

# Random seed for reproducibility
SEED=42

# Device (cuda or cpu)
DEVICE="cuda"

# ============================================================================
# SETUP
# ============================================================================

echo "========================================"
echo "Sense-Subspace LSCD Pipeline"
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
# STAGE 1: EXTRACT REPRESENTATIONS
# ============================================================================

echo ""
echo "=========================================================" 
echo "Stage 1: Extracting representations for labeled dataset"
echo "========================================================="
echo ""

python 1_extract_representations.py \
    --input "$INPUT_DATA" \
    --output "$OUTPUT_DIR/embeddings/gold" \
    --model "$MODEL_NAME" \
    --pooling "$POOLING" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"
    --layers "$LAYERS"

echo "✓ Stage 1 complete"

# ============================================================================
# STAGE 2: SELECT OPTIMAL LAYER
# ============================================================================

echo ""
echo "========================================"
echo "Stage 2: Selecting optimal layer"
echo "========================================"
echo ""

python 2_select_layer.py \
    --repr-dir "$OUTPUT_DIR/embeddings/gold" \
    --output "$OUTPUT_DIR/layer_selection"

echo "✓ Stage 2 complete"

# ============================================================================
# STAGE 3: EXTRACT REPRESENTATIONS FROM UNLABELED
# ============================================================================

echo ""
echo "=========================================================" 
echo "Stage 3: Extracting representations for unlabeled dataset"
echo "========================================================="
echo ""

python 1_extract_representations.py \
    --input "$INPUT_DATA_UNLABELED" \
    --output "$OUTPUT_DIR/embeddings/unlabeled" \
    --model "$MODEL_NAME" \
    --pooling "$POOLING" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"
    --layer-map "$OUTPUT/layer_selection/layer_selection.json"

echo "✓ Stage 3 complete"


# ============================================================================
# STAGE 4: TRAIN SENSE VECTORS
# ============================================================================

echo ""
echo "========================================"
echo "Stage 4: Training sense vectors"
echo "========================================"
echo ""

if [ "$USE_PCA" = true ]; then
    PCA_FLAG="--pca"
    PCA_COMPONENTS="--n-components $N_COMPONENTS"
else
    PCA_FLAG=""
    PCA_COMPONENTS=""
fi

python 3_train_sense_vectors.py \
    --repr-dir "$OUTPUT_DIR/embeddings/gold" \
    --layer-selection "$OUTPUT_DIR/layer_selection/layer_selection.json" \
    --output "$OUTPUT_DIR/sense_vectors" \
    $PCA_FLAG \
    $PCA_COMPONENTS \
    --seed "$SEED"

echo "✓ Stage 4 complete"

# ============================================================================
# STAGE 5: BUILD SENSE SUBSPACES
# ============================================================================

echo ""
echo "========================================"
echo "Stage 5: Building sense subspaces"
echo "========================================"
echo ""

python 4_build_sense_subspace.py \
    --vectors-dir "$OUTPUT_DIR/sense_vectors" \
    --output "$OUTPUT_DIR/subspaces"

echo "✓ Stage 5 complete"

# ============================================================================
# STAGE 6: PSEUDO-LABEL UNLABELED DATA
# ============================================================================

echo ""
echo "========================================="
echo "Stage 6: Pseudo-labeling unlabeled data"
echo "========================================="
echo ""

python 6_pseudo_label_unlabeled.py \
    --labeled-repr-dir "$OUTPUT_DIR/embeddings/gold" \
    --unlabeled-repr-dir "$OUTPUT_DIR/embeddings/unlabeled" \
    --subspace-dir "$OUTPUT_DIR/subspaces" \
    --output "$OUTPUT_DIR/pseudo_labeled.jsonl" \
    --threshold "$CONFIDENCE_THRESHOLD"

echo "✓ Stage 6 complete"

# ============================================================================
# STAGE 7: PROJECT ALL DATA BY TIME
# ============================================================================

echo ""
echo "==========================================================="
echo "Stage 7: Projecting labeled + pseudo labeled data by time"
echo "==========================================================="
echo ""

python 5_project_labeled_by_time.py \
    --repr-dir "$OUTPUT_DIR/embeddings/gold" \
    --subspace-dir "$OUTPUT_DIR/subspaces" \
    --output "$OUTPUT_DIR/projections" \
    --pseudo-dir "$OUTPUT_DIR/embeddings/pseudo_labeled" \
    --include-pseudo

echo "✓ Stage 7 complete"

# ============================================================================
# STAGE 8: INJECT PSEUDO LABELS
# ============================================================================

echo ""
echo "========================================"
echo "Stage 8: Injecting pseudo labels"
echo "========================================"
echo ""

python 7_inject_pseudo_labels.py \
    --input-dir "$OUTPUT_DIR/embeddings/unlabeled" \
    --output-dir "$OUTPUT_DIR/embeddings/pseudo_labeled" \
    --pseudo-jsonl "$OUTPUT_DIR/pseudo_labeled.jsonl"

echo "✓ Stage 8 complete"

# ============================================================================
# STAGE 9: ANALYZE SEMANTIC CHANGE
# ============================================================================

echo ""
echo "========================================"
echo "Stage 9: Analyzing semantic change"
echo "========================================"
echo ""

python 8_analyze_semantic_change.py \
    --projections-dir "$OUTPUT_DIR/projections" \
    --output "$OUTPUT_DIR/analysis" \
    --plot

echo "✓ Stage 9 complete"

# ============================================================================
# STAGE 10: PLOT 3D VISUALIZATION
# ============================================================================

echo ""
echo "================================================================"
echo "Stage 10: Plot 3D Visualization of Sense Trajectories Over Time "
echo "================================================================"
echo ""

python 9_plot_semantic_subspace.py \
        --projections-dir "$OUTPUT_DIR/projections" \
        --output "$OUTPUT_DIR/trajectory_plots" 
        
echo "✓ Stage 10 complete"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "====================================================================="
echo "Pipeline Complete!"
echo "====================================================================="
echo ""
echo "Output files saved to: $OUTPUT_DIR"
echo ""
echo "Key outputs:"
echo "  - Representations: $OUTPUT_DIR/embeddings/"
echo "  - Layer selection: $OUTPUT_DIR/layer_selection/layer_selection.json"
echo "  - Sense vectors: $OUTPUT_DIR/sense_vectors/"
echo "  - Subspaces: $OUTPUT_DIR/subspaces/"
echo "  - Projections: $OUTPUT_DIR/projections/"
echo "  - Pseudo-labels: $OUTPUT_DIR/pseudo_labeled.jsonl"
echo "  - Analysis: $OUTPUT_DIR/analysis/"
echo ""
echo "======================================================================"
