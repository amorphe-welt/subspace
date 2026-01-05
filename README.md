# Sense-Subspace Lexical Semantic Change Detection Pipeline

This repository contains a complete pipeline for **Lexical Semantic Change Detection (LSCD)** using sense-based subspaces. It leverages transformer-based embeddings to extract word representations, builds robust sense subspaces, and tracks semantic trajectories across chronological timespans.

## 🚀 Overview

The pipeline implements a multi-stage approach to analyze how word senses evolve over time:
1. **Representation Extraction:** Contextualized embeddings from specific transformer layers.
2. **Subspace Modeling:** Construction of sense-specific subspaces.
3. **Pseudo-Labeling:** Expanding the labeled dataset by projecting unlabeled data into the built subspaces.
4. **Change Analysis:** Computing Jensen-Shannon Divergence (JSD) and 3D Trajectory plotting.


---

## 🛠 Installation

2. **Install dependencies:**
```
pip install -r requirements.txt

```

*Required: `torch`, `transformers`, `scikit-learn`, `umap-learn`, `scipy`, `matplotlib`, `seaborn`, `numpy`.*

---

## 📂 Data Preparation

The pipeline expects data in `.jsonl` format.
Place your files in a `data/` directory:

* `data/AmDi.epsilon.jsonl` (Labeled/Gold data)
* `data/unlabeled.jsonl` (Unlabeled corpus data)

---

## 🏃 Running the Pipeline

You can execute the entire workflow with a single command using the provided bash script:

```
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
| **4** | `3_train_sense_vectors.py` | Traines sense vectors. |
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
* **/subspaces/**: Sense-specific subspaces.
* **/analysis/**: Semantic change metrics (JSD) and plots.
* **/trajectory_plots/**: 3D PNG/PDF plots showing sense evolution over time.
