"""
Subspace Quality Analysis (Normalized)
======================================

Compute a quality measure for each subspace based on:
- Stability (StabilityErr, inverted)
- Projection confidence (MeanConf)
- Pseudo-label coverage (accepted / total)

Outputs:
- JSON ranking of lexemes by quality
- Scatter plot: StabilityErr vs MeanConf, bubble = pseudo-label coverage
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def load_subspace_reliability(path: Path):
    """Load subspace reliability metrics (JSON dict)"""
    with open(path / "subspace_reliability.json", "r") as f:
        reliability = json.load(f)
    return reliability  # dict keyed by lexeme

def load_pseudo_labels(jsonl_path: Path):
    """Compute pseudo-label coverage per lexeme"""
    coverage = defaultdict(lambda: {"pseudo": 0, "rejected": 0})
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            lex = item["lexeme"]
            src = item["source"]
            if src == "pseudo":
                coverage[lex]["pseudo"] += 1
            elif src == "rejected":
                coverage[lex]["rejected"] += 1
    # Compute coverage fraction
    for lex in coverage:
        total = coverage[lex]["pseudo"] + coverage[lex]["rejected"]
        coverage[lex]["PseudoCoverage"] = coverage[lex]["pseudo"] / total if total > 0 else 0.0
    return coverage

def compute_quality(reliability, pseudo_results, weights=(0.33,0.33,0.34)):
    """
    weights = (stab_weight, conf_weight, pseudo_weight)
    """
    # Gather raw metrics for normalization
    stab_errs = np.array([v['resample_stability_error'] for v in reliability.values()])
    mean_confs = np.array([v['mean_confidence'] for v in reliability.values()])
    pseudo_covs = np.array([pseudo_results.get(lex, {}).get("PseudoCoverage", 0.0) for lex in reliability.keys()])

    # Normalize
    stab_norm = (stab_errs - stab_errs.min()) / (stab_errs.max() - stab_errs.min() + 1e-10)
    conf_norm = (mean_confs - mean_confs.min()) / (mean_confs.max() - mean_confs.min() + 1e-10)
    pseudo_norm = (pseudo_covs - pseudo_covs.min()) / (pseudo_covs.max() - pseudo_covs.min() + 1e-10)

    quality_list = []
    for i, lex in enumerate(reliability.keys()):
        stab_score = 1.0 - stab_norm[i]          # lower stability error => higher score
        conf_score = conf_norm[i]
        pseudo_score = pseudo_norm[i]

        qscore = weights[0]*stab_score + weights[1]*conf_score + weights[2]*pseudo_score

        quality_list.append({
            "lexeme": lex,
            "StabilityErr": reliability[lex]['resample_stability_error'],
            "MeanConf": reliability[lex]['mean_confidence'],
            "PseudoCoverage": pseudo_results.get(lex, {}).get("PseudoCoverage", 0.0),
            "QualityScore": qscore
        })

    # Sort descending
    quality_list.sort(key=lambda x: x["QualityScore"], reverse=True)
    return quality_list

def save_quality_json(results, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "subspace_quality.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ordered quality JSON to {outdir / 'subspace_quality.json'}")

def plot_quality(results, outdir: Path):
    lexemes = [r["lexeme"] for r in results]
    stabilities = [r["StabilityErr"] for r in results]
    confs = [r["MeanConf"] for r in results]
    coverages = [r["PseudoCoverage"] for r in results]
    qualities = [r["QualityScore"] for r in results]

    # 1. Shrink figure size (Width, Height)
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # 2. Tighten the axis margins (percentage of data range)
    ax.margins(0.08)
    
    # 3. Slightly smaller bubbles for smaller scale
    sc = ax.scatter(stabilities, confs, 
                    s=np.array(coverages) * 300 +10, 
                    c=qualities, 
                    cmap="viridis", 
                    alpha=0.8,
                    edgecolors='black', 
                    linewidth=0.5)
    
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Subspace Quality", fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_xlabel("Stability Error", fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Projection Confidence", fontsize=10, fontweight='bold')
    ax.set_title("Subspace Quality Analysis", fontsize=11, fontweight='bold')
    
    # 4. Optimized labels with small background pad for readability
    for i, lex in enumerate(lexemes):
        ax.text(stabilities[i], confs[i] + 0.015, lex, 
                fontsize=8, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5))

    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / "subspace_quality.pdf", dpi=400, bbox_inches='tight')
    plt.close()


def plot_quality_(results, outdir: Path):
    lexemes = [r["lexeme"] for r in results]
    stabilities = [r["StabilityErr"] for r in results]
    confs = [r["MeanConf"] for r in results]
    coverages = [r["PseudoCoverage"] for r in results]
    qualities = [r["QualityScore"] for r in results]

    plt.figure(figsize=(10,7))
    sc = plt.scatter(stabilities, confs, s=np.array(coverages)*500, c=qualities, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="Subspace Quality")
    plt.xlabel("Stability Error")
    plt.ylabel("Mean Projection Confidence")
    plt.title("Subspace Quality Analysis (Bubble = Pseudo-Label Coverage)")
    for i, lex in enumerate(lexemes):
        plt.text(stabilities[i]+0.002, confs[i]+0.002, lex, fontsize=8, alpha=0.7)
    plt.tight_layout()
    plt.savefig(outdir / "subspace_quality.png", dpi=300)
    plt.close()
    print(f"Saved quality scatter plot to {outdir / 'subspace_quality.png'}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Subspace Quality Analysis")
    parser.add_argument("--reliability-dir", type=Path, required=True, help="Dir with subspace_reliability.json")
    parser.add_argument("--pseudo-jsonl", type=Path, required=True, help="Pseudo-label JSONL")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    reliability = load_subspace_reliability(args.reliability_dir)
    pseudo_results = load_pseudo_labels(args.pseudo_jsonl)
    results = compute_quality(reliability, pseudo_results)
    save_quality_json(results, args.outdir)
    plot_quality(results, args.outdir)

if __name__ == "__main__":
    main()
