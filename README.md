# Sense-Subspace Lexical Semantic Change Detection Pipeline

This repository contains a complete pipeline for **Lexical Semantic Change Detection (LSCD)** using sense-based subspaces. It leverages transformer-based embeddings (e.g., XLM-R) to extract word representations, builds robust sense subspaces, and tracks semantic trajectories across chronological timespans.

## 🚀 Overview

The pipeline implements a multi-stage approach to analyze how word senses evolve over time:
1. **Representation Extraction:** Contextualized embeddings from specific transformer layers.
2. **Subspace Modeling:** Construction of sense-specific subspaces using PCA/Orthonormal bases.
3. **Pseudo-Labeling:** Expanding the labeled dataset by projecting unlabeled data into the built subspaces.
4. **Change Analysis:** Computing Jensen-Shannon Divergence (JSD) and 3D Trajectory plotting.


---

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/sense-subspace-lscd.git](https://github.com/your-username/sense-subspace-lscd.git)
   cd sense-subspace-lscd

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```

*Required: `torch`, `transformers`, `scikit-learn`, `umap-learn`, `scipy`, `matplotlib`, `seaborn`, `numpy`.*

---

## 📂 Data Preparation

The pipeline expects data in `.jsonl` format. Each line should contain:

* `lexeme`: The target word.
* `context`: The sentence containing the word.
* `label` (for gold data): The annotated sense.
* `timespan`: The chronological period (e.g., "1850-1900").

Place your files in a `data/` directory:

* `data/AmDi.epsilon.jsonl` (Labeled/Gold data)
* `data/unlabeled.jsonl` (Unlabeled corpus data)

---

## 🏃 Running the Pipeline

You can execute the entire workflow with a single command using the provided bash script:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh

```

### Configuration

You can edit the variables at the top of `run_pipeline.sh` to customize the run:

* `MODEL_NAME`: The HuggingFace model path (default: `pierluigic/xl-lexeme`).
* `LAYERS`: The transformer layers to evaluate for optimal representation.
* `CONFIDENCE_THRESHOLD`: Stricteness of the pseudo-labeling (0.0 to 1.0).
* `DEVICE`: `cuda` for GPU acceleration or `cpu`.

---

## 📊 Pipeline Stages

| Stage | Script | Description |
| --- | --- | --- |
| **1** | `1_extract_representations.py` | Extracts embeddings for gold/labeled data. |
| **2** | `2_select_layer.py` | Identifies the layer with the highest sense-discriminative power. |
| **3** | `1_extract_representations.py` | Extracts embeddings for unlabeled data using the optimal layer. |
| **4** | `3_train_sense_vectors.py` | Calculates mean sense vectors and applies PCA. |
| **5** | `4_build_sense_subspace.py` | Constructs orthonormal bases for each sense subspace. |
| **6** | `6_pseudo_label_unlabeled.py` | Assigns sense labels to unlabeled data based on subspace distance. |
| **7** | `5_project_labeled_by_time.py` | Groups all labeled data into chronological projections. |
| **8** | `7_inject_pseudo_labels.py` | Updates the representation files with the new pseudo-labels. |
| **9** | `8_analyze_semantic_change.py` | Calculates JSD and identifies emergent signals. |
| **10** | `9_plot_semantic_subspace.py` | Generates 3D visualizations of sense trajectories. |

---

## 📈 Outputs

All results are saved to the directory specified by `$OUTPUT_DIR` (default: `xl-lexeme/`):

* **/embeddings/**: Raw and processed `.pkl` files of contextual embeddings.
* **/layer_selection/**: JSON report on the best-performing layer.
* **/subspaces/**: Mathematical definitions of the sense-specific subspaces.
* **/analysis/**: Semantic change metrics (JSD) and statistical tables.
* **/trajectory_plots/**: 3D PNG/PDF plots showing sense evolution over time.




























A modular, research-grade pipeline for detecting lexical semantic change using sense subspaces derived from pretrained language model representations.

## 📋 Overview

This pipeline implements a **sense-aware approach to LSCD** that:
- Learns **fixed sense subspaces** for each lexeme from labeled data
- Projects usages across all timespans into these shared subspaces
- Tracks **sense distributions over time**
- Performs **automatic sense labeling** (WSD/WSI) on unlabeled data
- Detects and quantifies **semantic change**

### Key Features

✅ **Scientifically rigorous**: Avoids circularity, validates stability, ensures reproducibility  
✅ **Modular design**: 7 independent CLI scripts for maximum flexibility  
✅ **Publication-ready**: Comprehensive metrics, visualizations, and documentation  
✅ **Extensible**: Easy to adapt to different models, languages, or datasets

---

## 🗂️ Pipeline Structure

```
lscd_pipeline/
├── 1_extract_representations.py    # Extract hidden states from pretrained model
├── 2_select_layer.py                # Select optimal layer per lexeme
├── 3_train_sense_vectors.py        # Train pairwise sense contrast vectors
├── 4_build_sense_subspace.py       # Build fixed sense subspaces
├── 5_project_labeled_by_time.py    # Project labeled data by timespan
├── 6_pseudo_label_unlabeled.py     # Pseudo-label unlabeled data (WSD/WSI)
├── 7_analyze_semantic_change.py    # Analyze and visualize semantic change
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── examples/
    ├── run_full_pipeline.sh         # Example end-to-end script
    └── sample_data.jsonl            # Sample data format
```

---

## 📦 Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~10GB disk space for model and outputs

### Setup

```bash
# Clone or download the pipeline
cd lscd_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=1.0.0
scipy>=1.7.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

---

## 📊 Data Format

### Labeled Data (Input JSONL)

Each line must contain:
```json
{
  "sentence": "Das Blatt des Baumes ist grün.",
  "lexeme": "blatt",
  "synset": "s25245",
  "timespan": "1950s"
}
```

- `sentence`: Context containing the lexeme
- `lexeme`: Target word (case-insensitive matching)
- `synset`: Sense identifier (e.g., WordNet synset ID)
- `timespan`: Temporal period (e.g., decade, year)

### Unlabeled Data (Input JSONL)

```json
{
  "sentence": "Er las das Blatt vollständig.",
  "lexeme": "blatt",
  "timespan": "1980s"
}
```

**Note**: Set `synset` to `null` or omit it for unlabeled data.

---

## 🚀 Usage

### Full Pipeline Example

```bash
#!/bin/bash
# Run the complete pipeline

# 1. Extract representations
python 1_extract_representations.py \
    --input data/labeled.jsonl \
    --output outputs/representations \
    --model bert-base-german-cased \
    --pooling first \
    --batch-size 32

# 2. Select optimal layer
python 2_select_layer.py \
    --repr-dir outputs/representations \
    --output outputs/layer_selection

# 3. Train sense vectors
python 3_train_sense_vectors.py \
    --repr-dir outputs/representations \
    --layer-selection outputs/layer_selection/layer_selection.json \
    --output outputs/sense_vectors \
    --pca \
    --n-components 100

# 4. Build subspaces
python 4_build_sense_subspace.py \
    --vectors-dir outputs/sense_vectors \
    --output outputs/subspaces

# 5. Project labeled data by time
python 5_project_labeled_by_time.py \
    --repr-dir outputs/representations \
    --subspace-dir outputs/subspaces \
    --output outputs/projections

# 6. Pseudo-label unlabeled data
python 6_pseudo_label_unlabeled.py \
    --repr-dir outputs/representations \
    --subspace-dir outputs/subspaces \
    --output outputs/pseudo_labeled \
    --threshold 0.5

# 7. Analyze semantic change
python 7_analyze_semantic_change.py \
    --projections-dir outputs/projections \
    --output outputs/analysis \
    --plot
```

### Individual Script Usage

#### Script 1: Extract Representations

```bash
python 1_extract_representations.py \
    --input data/labeled.jsonl \
    --output outputs/representations \
    --model bert-base-german-cased \
    --layers 8 9 10 11 12 \
    --pooling first \
    --batch-size 32 \
    --device cuda
```

**Arguments:**
- `--input`: Path to input JSONL file
- `--output`: Output directory
- `--model`: HuggingFace model name
- `--layers`: Specific layers to extract (optional, default: all)
- `--pooling`: Subword pooling strategy (`first`, `mean`, `last`)
- `--batch-size`: Batch size for processing
- `--device`: Device (`cuda` or `cpu`)

**Output:** `{lexeme}_representations.pkl` files

---

#### Script 2: Select Layer

```bash
python 2_select_layer.py \
    --repr-dir outputs/representations \
    --output outputs/layer_selection
```

**Output:** `layer_selection.json` with optimal layer per lexeme

---

#### Script 3: Train Sense Vectors

```bash
python 3_train_sense_vectors.py \
    --repr-dir outputs/representations \
    --layer-selection outputs/layer_selection/layer_selection.json \
    --output outputs/sense_vectors \
    --pca \
    --n-components 100 \
    --seed 42
```

**Arguments:**
- `--pca`: Apply PCA dimensionality reduction
- `--n-components`: Number of PCA components
- `--seed`: Random seed for reproducibility

**Output:** `{lexeme}_sense_vectors.pkl` files

---

#### Script 4: Build Subspace

```bash
python 4_build_sense_subspace.py \
    --vectors-dir outputs/sense_vectors \
    --output outputs/subspaces
```

**Output:** `{lexeme}_subspace.pkl` files with orthonormal basis

---

#### Script 5: Project by Time

```bash
python 5_project_labeled_by_time.py \
    --repr-dir outputs/representations \
    --subspace-dir outputs/subspaces \
    --output outputs/projections
```

**Output:** `{lexeme}_projections_by_time.pkl` with timespan-grouped coordinates

---

#### Script 6: Pseudo-Label

```bash
python 6_pseudo_label_unlabeled.py \
    --repr-dir outputs/representations \
    --subspace-dir outputs/subspaces \
    --output outputs/pseudo_labeled \
    --threshold 0.5
```

**Arguments:**
- `--threshold`: Confidence threshold for assignment (0-1)

**Output:** `{lexeme}_pseudo_labeled.jsonl` files

---

#### Script 7: Analyze Change

```bash
python 7_analyze_semantic_change.py \
    --projections-dir outputs/projections \
    --output outputs/analysis \
    --plot
```

**Output:**
- `semantic_change_metrics.json`: Quantitative metrics
- `{lexeme}_semantic_change.png`: Visualization plots
- `analysis_summary.txt`: Human-readable summary

---

## 📐 Methodology

### Core Principles

1. **Fixed Sense Subspaces**
   - One subspace per lexeme (not per timespan)
   - Learned only from labeled data
   - Orthonormalized via Gram-Schmidt

2. **Pairwise Sense Contrast**
   - For senses S₁, S₂: contrast vector = centroid(S₂) - centroid(S₁)
   - Captures direction of maximal separation
   - All pairs contribute to subspace

3. **Layer Selection**
   - Per-lexeme optimization
   - Combines linear probe accuracy + silhouette score
   - Finds layer with best sense separability

4. **No Circularity**
   - Subspaces trained only on gold labels
   - Unlabeled data never influences subspace definition
   - Pseudo-labeling is post-hoc

### Metrics

- **Jensen-Shannon Divergence (JSD)**: Distributional distance between timespans
- **Sense Dominance**: Proportion of each sense over time
- **Emergence Signals**: Sudden increases in sense proportions
- **Stability**: Split-half correlation of sense centroids

---

## 📈 Output Artifacts

### Per-Lexeme Files

```
outputs/
├── representations/
│   ├── {lexeme}_representations.pkl   # Hidden states per layer
│   └── extraction_config.json
├── layer_selection/
│   └── layer_selection.json            # Selected layers + scores
├── sense_vectors/
│   ├── {lexeme}_sense_vectors.pkl     # Orthonormal contrast vectors
│   └── training_summary.json
├── subspaces/
│   ├── {lexeme}_subspace.pkl          # Subspace basis + metadata
│   └── subspace_summary.json
├── projections/
│   ├── {lexeme}_projections_by_time.pkl  # Coordinates by timespan
│   └── projection_summary.json
├── pseudo_labeled/
│   ├── {lexeme}_pseudo_labeled.jsonl  # Sense assignments
│   └── pseudo_labeling_summary.json
└── analysis/
    ├── semantic_change_metrics.json   # All metrics
    ├── {lexeme}_semantic_change.png   # Plots
    └── analysis_summary.txt
```

### Summary Files

- **extraction_config.json**: Model, layers, pooling settings
- **layer_selection.json**: Selected layers with separability scores
- **training_summary.json**: Number of vectors, stability per lexeme
- **subspace_summary.json**: Subspace dimensions, orthonormality errors
- **projection_summary.json**: Timespan coverage statistics
- **pseudo_labeling_summary.json**: Assignment acceptance rates
- **semantic_change_metrics.json**: JSD matrices, emergence signals
- **analysis_summary.txt**: Human-readable summary report

---

## 🧪 Scientific Considerations

### Assumptions

1. **Labeled data is representative**: Sense proportions in labeled data reflect true distributions
2. **Senses are stable**: Sense definitions don't change over time (only their distributions)
3. **Subspace captures sense structure**: Linear separability of senses in embedding space

### Limitations

- **Requires sufficient labeled data**: ~10+ examples per sense per lexeme minimum
- **Fixed granularity**: Cannot detect sense splitting/merging within existing categories
- **Linear assumption**: Subspace is linear; non-linear structure not captured

### Validation

- **Split-half stability**: Verifies sense centroids are consistent across data splits
- **Orthonormality checks**: Ensures basis vectors are properly orthogonal
- **Cross-validation**: Layer selection uses CV to avoid overfitting

---

## 🔬 Example Research Questions

This pipeline can answer:

1. **Which senses emerged/declined over time?**
   → Analyze sense dominance curves

2. **When did the largest semantic shifts occur?**
   → Examine JSD matrix for peak changes

3. **How stable are sense representations?**
   → Check stability metrics in training output

4. **Can we automatically label historical usages?**
   → Use pseudo-labeling with confidence filtering

5. **Which lexemes changed most/least?**
   → Compare mean JSD across lexemes

---

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{sense_subspace_lscd,
  title={Sense-Subspace Lexical Semantic Change Detection Pipeline},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/lscd-pipeline}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **Non-linear subspaces**: Kernel methods, manifold learning
- **Dynamic subspaces**: Allow sense evolution over time
- **Hierarchical senses**: Handle sense taxonomies
- **Multilingual support**: Cross-lingual subspaces

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🐛 Troubleshooting

### Common Issues

**"Could not locate lexeme in sentence"**
- Check that `lexeme` matches exactly (case-insensitive)
- Verify tokenizer handles your language correctly

**"Insufficient labeled data"**
- Need minimum ~10 examples per sense
- Consider combining similar senses

**"High orthonormality error"**
- Usually harmless if < 0.01
- May indicate numerical instability with very similar senses

**Memory errors**
- Reduce `--batch-size`
- Use `--pca` with smaller `--n-components`
- Process fewer layers

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Happy semantic change detecting! 🎯**
