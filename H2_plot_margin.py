"""
Semantic Subspace Quality: Confidence vs. Margin Analysis
=========================================================
This script visualizes the distribution of pseudo-labeled instances within
the lexeme-specific subspaces generated via XL-Lexeme.

METRICS:
    - Confidence: The L2-norm distance to the nearest sense centroid, 
      inverted and normalized. High confidence indicates a sample is 
      prototypical for a sense.
    - Margin: The distance gap between the top-1 and top-2 predicted 
      senses (d2 - d1). High margin indicates low ambiguity.

PURPOSE:
    - Validates the 0.5 confidence threshold for sample rejection.
    - Diagnoses semantic overlap (low-margin clusters).
    - Identifies 'rejected' samples (e.g., Schimmel with < 0.5 confidence).

USAGE:
    python plot_margin.py path/to/results.jsonl --output-dir ./plots
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def plot_confidence_margin(jsonl_path, output_dir):
    # 1. Load data
    data = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found.")
        return
    
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)


    # 2. Setup Figure - Academic Style
    plt.figure(figsize=(7, 5))
    sns.set_style("ticks")

    # 3. Create Scatter Plot
    # We use a muted palette and small markers to manage high-density data
    sc = sns.scatterplot(
        data=df, 
        x="margin", 
        y="confidence", 
        hue="lexeme",
        alpha=0.5, 
        s=1,
        edgecolor=None,
        palette="turbo"
    )
    
    # Set Log Scale for X-axis
    # plt.xscale('log')
    # plt.yscale('log')

    # 4. Threshold & Labels
    # Dotted line at 0.5 highlights the rejection boundary
    plt.axhline(y=0.5, color='black', linestyle=':', linewidth=1.2, label='Threshold ($0.5$)')
    
    #plt.ylim(0, 1.05)
    plt.ylim(0.25,0.75)
    plt.xlim(0.0,0.75)
    # plt.xlim(0, df["margin"].max() * 1.05)
    plt.xlabel("Margin ($d_2 - d_1$)", fontsize=12, fontweight='bold')
    plt.ylabel("Projection Confidence", fontsize=12, fontweight='bold')
    plt.title("Confidence vs. Margin", fontsize=14, fontweight='bold')
    # Legend placement
    plt.legend(title="Lexeme", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=8)
    plt.legend().set_visible(False)

    sns.despine() 
    plt.tight_layout()
    
    # 5. Save and Close
    save_file = output_path / "confidence_margin_scatter.png"
    plt.savefig(save_file, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated quality plot: {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pseudo-label quality via Confidence vs. Margin scatter plots.")
    parser.add_argument("jsonl", type=str, help="Path to input JSONL containing 'confidence', 'margin', and 'lexeme'")
    parser.add_argument("--output-dir", "-o", type=str, default=".", help="Directory to save the resulting .pdf")
    
    args = parser.parse_args()
    plot_confidence_margin(args.jsonl, args.output_dir)
