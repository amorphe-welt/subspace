"""
Align Subspace Quality with Pseudo-Label Behavior
=================================================

Tests H1-H4 correlations:
- H1: Quality ↔ Coverage
- H2: Quality ↔ Mean Confidence
- H3: Quality ↔ Sense Entropy
- H4: Quality ↔ Temporal Instability

Usage:
    python analyze_quality_vs_pseudo.py \
        --subspace-json outputs/subspace_quality.json \
        --pseudo-json outputs/pseudo_label_metrics.json \
        --outdir outputs/quality_vs_pseudo
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_entropy(pseudo_info):
    """Entropy of pseudo-label distribution per lexeme"""
    counts = pseudo_info.get("SenseCounts", None)
    if counts is None or len(counts) == 0:
        return np.nan
    probs = np.array(list(counts.values()))
    probs = probs / probs.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))

def main(subspace_json, pseudo_json, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    
    #quality_data = load_json(subspace_json)
    quality_raw = load_json(subspace_json)

    # Normalize quality data to dict[lexeme -> info]
    if isinstance(quality_raw, list):
        quality_data = {item["lexeme"]: item for item in quality_raw}
    elif isinstance(quality_raw, dict):
        quality_data = quality_raw
    else:
        raise ValueError("Unsupported format for subspace quality JSON")
    
    
    
    #pseudo_data = load_json(pseudo_json)
    pseudo_raw = load_json(pseudo_json)

    if isinstance(pseudo_raw, list):
        pseudo_data = {item["Lexeme"]: item for item in pseudo_raw}
    elif isinstance(pseudo_raw, dict):
        pseudo_data = pseudo_raw
    else:
        raise ValueError("Unsupported format for pseudo-label JSON")
    
    print(f"Loaded {len(quality_data)} lexemes with quality")
    print(f"Loaded {len(pseudo_data)} lexemes with pseudo-label stats")
    print("Overlap:", len(set(quality_data) & set(pseudo_data)))
    
    rows = []
    for lex, qinfo in quality_data.items():
        if lex not in pseudo_data:
            continue
        pinfo = pseudo_data[lex]
        # entropy = compute_entropy(pinfo)
        entropy = pinfo.get("SenseEntropy", np.nan)
        instability = qinfo.get("StabilityErr", np.nan)
        rows.append({
            "lexeme": lex,
            "quality": qinfo.get("QualityScore", np.nan),
            "coverage": pinfo.get("Coverage", np.nan),
            "mean_conf": pinfo.get("MeanConfidence", np.nan),
            "entropy": entropy,
            "instability": instability
        })
    
    # Convert to arrays
    qualities = np.array([r["quality"] for r in rows])
    coverages = np.array([r["coverage"] for r in rows])
    mean_confs = np.array([r["mean_conf"] for r in rows])
    entropies = np.array([r["entropy"] for r in rows])
    instabilities = np.array([r["instability"] for r in rows])
    
    def describe(x, name):
        print(
            name,
            "min", np.min(x),
            "max", np.max(x),
            "std", np.std(x)
        )

    describe(qualities, "Quality")
    describe(coverages, "Coverage")
    describe(mean_confs, "MeanConf")
    describe(entropies, "Entropy")
    describe(instabilities, "Instability")
    
    # Correlation helper
    def corr(x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan, np.nan
        return spearmanr(x[mask], y[mask])
    
    correlations = {
        "H1: Quality vs Coverage": corr(qualities, coverages),
        "H2: Quality vs MeanConfidence": corr(qualities, mean_confs),
        "H3: Quality vs SenseEntropy": corr(qualities, entropies),
        "H4: Quality vs TemporalInstability": corr(qualities, instabilities)
    }
    
    print("Spearman Correlations (rho, p-value):")
    for h, (rho, p) in correlations.items():
        print(f"{h}: rho={rho:.3f}, p={p:.3f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    axes = axes.flatten()
    
    axes[0].scatter(qualities, coverages)
    axes[0].set_xlabel("Subspace Quality")
    axes[0].set_ylabel("Pseudo-Label Coverage")
    axes[0].set_title("H1: Quality vs Coverage")
    
    axes[1].scatter(qualities, mean_confs)
    axes[1].set_xlabel("Subspace Quality")
    axes[1].set_ylabel("Mean Confidence")
    axes[1].set_title("H2: Quality vs Mean Confidence")
    
    axes[2].scatter(qualities, entropies)
    axes[2].set_xlabel("Subspace Quality")
    axes[2].set_ylabel("Sense Entropy")
    axes[2].set_title("H3: Quality vs Sense Entropy")
    
    axes[3].scatter(qualities, instabilities)
    axes[3].set_xlabel("Subspace Quality")
    axes[3].set_ylabel("Temporal Instability")
    axes[3].set_title("H4: Quality vs Temporal Instability")
    
    plt.tight_layout()
    out_path = outdir / "quality_vs_pseudo.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subspace-json", type=Path, required=True)
    parser.add_argument("--pseudo-json", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    main(args.subspace_json, args.pseudo_json, args.outdir)
