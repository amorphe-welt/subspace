"""
Script 8: Analyze Semantic Change
==================================

Analyzes semantic change by comparing sense distributions across timespans.
Computes distributional distances and generates visualizations.

Usage:
    python 8_analyze_semantic_change.py \
        --projections-dir outputs/projections \
        --output outputs/analysis \
        --plot

Input:
    {lexeme}_projections_by_time.pkl from script 5
    
Output:
    - semantic_change_metrics.json with quantitative measures
    - Visualization plots for each lexeme
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pickle
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def compute_stepwise_significance(projections_data, timespans):
    """Computes p-values for transitions between adjacent timespans."""
    ts_data = projections_data['timespan_data']
    sig_results = {}

    for i in range(len(timespans) - 1):
        t1, t2 = timespans[i], timespans[i+1]
        
        # Pull raw counts (e.g., {"sense_1": 45, "sense_2": 12})
        counts1 = ts_data[t1].get('sense_counts', {})
        counts2 = ts_data[t2].get('sense_counts', {})
        
        all_senses = sorted(set(list(counts1.keys()) + list(counts2.keys())))
        all_senses = [s for s in all_senses if s is not None]

        # Table: Rows = Timespans, Cols = Senses
        table = np.array([
            [counts1.get(s, 0) for s in all_senses],
            [counts2.get(s, 0) for s in all_senses]
        ])

        # Filter out senses with 0 total occurrences in these two spans
        table = table[:, table.sum(axis=0) > 0]

        if table.shape[1] < 2: # Cannot test if only 1 sense exists
            p = 1.0
        else:
            _, p, _, _ = chi2_contingency(table)
            
        #sig_results[f"{t1}->{t2}"] = {"p_val": float(p), "significant": p < 0.05}
        sig_results[f"{t1}->{t2}"] = {
            "p_val": float(p), 
            # FIX: explicitly cast to bool() to avoid numpy.bool_
            "significant": bool(p < 0.05) 
        }
    return sig_results

def compute_centroid_drift(projections_data, timespans):
    """Calculates cosine distance between the mean embeddings of adjacent timespans."""
    ts_data = projections_data['timespan_data']
    drift_results = {}

    for i in range(len(timespans) - 1):
        t1, t2 = timespans[i], timespans[i+1]
        
        # mean_embedding should be the average of all vectors in that span
        vec1 = ts_data[t1].get('mean_embedding')
        vec2 = ts_data[t2].get('mean_embedding')

        if vec1 is not None and vec2 is not None:
            # Cosine Distance = 1 - Cosine Similarity
            dist = 1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            drift_results[f"{t1}->{t2}"] = float(dist)
            
    return drift_results
    
def compute_jsd_matrix(
    sense_distributions: Dict[str, Dict[str, float]],
    timespans: List[str]
) -> np.ndarray:
    """Compute Jensen-Shannon divergence matrix between timespan distributions"""
    n = len(timespans)
    jsd_matrix = np.zeros((n, n))
    
    # Collect all valid senses
    all_senses = sorted(set(
        sense
        for dist in sense_distributions.values()
        for sense in dist.keys()
        if sense is not None
    ))
    
    for i, t1 in enumerate(timespans):
        for j, t2 in enumerate(timespans):
            dist1 = sense_distributions[t1]
            dist2 = sense_distributions[t2]
            
            # Align distributions
            p = np.array([dist1.get(s, 0) for s in all_senses])
            q = np.array([dist2.get(s, 0) for s in all_senses])
            
            # Avoid log(0)
            p = p + 1e-10
            q = q + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            
            jsd_matrix[i, j] = jensenshannon(p, q)
    
    return jsd_matrix


def detect_sense_emergence(
    projections_data: Dict,
    threshold: float = 0.2
) -> Dict:
    """
    Detect potential new sense emergence by analyzing distribution changes.
    Ignores None senses.
    """
    timespan_data = projections_data['timespan_data']
    timespans = sorted(timespan_data.keys())
    emergence_signals = {}
    
    for i, timespan in enumerate(timespans):
        if i == 0:
            continue
        
        prev_timespan = timespans[i-1]
        curr_dist = timespan_data[timespan]['sense_distribution']
        prev_dist = timespan_data[prev_timespan]['sense_distribution']
        
        for sense in curr_dist:
            if sense is None:
                continue  # Skip None
            change = curr_dist.get(sense, 0) - prev_dist.get(sense, 0)
            if change > threshold:
                if timespan not in emergence_signals:
                    emergence_signals[timespan] = []
                emergence_signals[timespan].append({
                    'sense': sense,
                    'change': float(change),
                    'prev_proportion': float(prev_dist.get(sense, 0)),
                    'curr_proportion': float(curr_dist.get(sense, 0))
                })
    
    return emergence_signals


def analyze_lexeme(
    projections_file: Path,
    output_dir: Path,
    create_plots: bool,
    exclude
) -> Dict:
    """Analyze semantic change for a single lexeme, ignoring None senses"""
    
    with open(projections_file, 'rb') as f:
        data = pickle.load(f)
    
    lexeme = data['lexeme']
    logging.info(f"Analyzing semantic change for: {lexeme}")
    
    
    # Filter out the specific unwanted strings
    #exclude = {"1995-2014", "2006-2023"}
    exclude_set = set(exclude)

    timespan_data = data['timespan_data']
    timespans = sorted([
        ts for ts in data["timespan_data"].keys() 
        if ts not in exclude_set
    ])
    #timespans = sorted(timespan_data.keys())
    
    logging.info(f"  Timespans: {timespans}")
    
    # Extract sense distributions, ignoring None senses
    sense_distributions = {}
    for timespan in timespans:
        dist = {s: p for s, p in timespan_data[timespan]['sense_distribution'].items() if s is not None}
        sense_distributions[timespan] = dist
    
    # Compute JSD matrix
    jsd_matrix = compute_jsd_matrix(sense_distributions, timespans)
    
    # --- NEW: Call new metrics ---
    # These functions look at transitions (T1 -> T2, T2 -> T3)
    stepwise_significance = compute_stepwise_significance(data, timespans)
    centroid_drifts = compute_centroid_drift(data, timespans)
    
    # Detect sense emergence (ignore None)
    emergence_raw = detect_sense_emergence(data)
    emergence = {}
    for ts, signals in emergence_raw.items():
        filtered = [s for s in signals if s['sense'] is not None]
        if filtered:
            emergence[ts] = filtered
    
    # Compute change metrics
    metrics = {
        'lexeme': lexeme,
        'n_timespans': len(timespans),
        'timespans': timespans,
        'sense_distributions': sense_distributions,
        'jsd_matrix': jsd_matrix.tolist(),
        'stepwise_significance': stepwise_significance, 
        'centroid_drifts': centroid_drifts,            
        'max_jsd': float(jsd_matrix.max()),
        'mean_jsd': float(jsd_matrix[np.triu_indices_from(jsd_matrix, k=1)].mean()),
        'emergence_signals': emergence
    }
    
    # Identify most/least stable adjacent pairs
    adjacent_jsds = [jsd_matrix[i, i+1] for i in range(len(timespans)-1)]
    if adjacent_jsds:
        max_change_idx = np.argmax(adjacent_jsds)
        metrics['largest_change'] = {
            'from': timespans[max_change_idx],
            'to': timespans[max_change_idx + 1],
            'jsd': float(adjacent_jsds[max_change_idx])
        }
    
    logging.info(f"  Mean JSD: {metrics['mean_jsd']:.4f}")
    logging.info(f"  Max JSD: {metrics['max_jsd']:.4f}")
    if emergence:
        logging.info(f"  Emergence signals: {sum(len(v) for v in emergence.values())}")
    
    # Create visualizations
    if create_plots:
        create_visualizations(lexeme, timespans, timespan_data, sense_distributions, 
                              jsd_matrix, output_dir, stepwise_sig=stepwise_significance, drifts=centroid_drifts)
    
    return metrics

def analyze_all_semantic_change(
    projections_dir: Path,
    output_dir: Path,
    create_plots: bool,
    exclude
):
    """Analyze semantic change for all lexemes, ignoring None senses"""
    
    output_dir.mkdir(exist_ok=True, parents=True)
    files = list(projections_dir.glob("*_projections_by_time.pkl"))
    logging.info(f"Found {len(files)} lexemes to analyze")
    
    all_metrics = []
    for f in files:
        metrics = analyze_lexeme(f, output_dir, create_plots, exclude)
        all_metrics.append(metrics)
        logging.info("")
    
    # Save metrics
    metrics_path = output_dir / "semantic_change_metrics.json"
    with open(metrics_path, 'w') as out_f:
        json.dump(all_metrics, out_f, indent=2)
    logging.info(f"Saved metrics to {metrics_path}")


def create_visualizations(
    lexeme: str,
    timespans: List[str],
    timespan_data,
    sense_distributions: Dict[str, Dict[str, float]],
    jsd_matrix: np.ndarray,
    output_dir: Path,
    emergence_signals: Dict = None,
    stepwise_sig=None, drifts=None
):
    """Create publication-ready visualizations including semantic change over time"""

    # --- Panel 1: Sense dominance over time (stacked line plot) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Collect all non-None senses
    all_senses = sorted(set(
        sense
        for dist in sense_distributions.values()
        for sense in dist.keys()
        if sense is not None
    ))
    colors = sns.color_palette("husl", len(all_senses))
    
    # 1a: Line plot per sense
    ax1 = axes[0]
    for sense, color in zip(all_senses, colors):
        proportions = [sense_distributions[t].get(sense, 0) for t in timespans]
        ax1.plot(timespans, proportions, marker='o', label=sense, color=color, linewidth=2, markersize=6)
    ax1.set_xlabel("", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Sense Proportion", fontsize=12, fontweight='bold')
    ax1.set_title(f"Sense Dominance Over Time: {lexeme}", fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])
    ax1.set_xticks(list(timespans))
    ax1.set_xticklabels(timespans, ha='right')
    if len(timespans) > 5:
        ax1.tick_params(axis='x', rotation=45)

    # --- Panel 2: JSD heatmap ---
    ax2 = axes[1]
    sns.heatmap(
        jsd_matrix,
        xticklabels=timespans,
        yticklabels=timespans,
        cmap='YlOrRd',
        ax=ax2,
        cbar_kws={'label': 'Jensen-Shannon Divergence'},
        annot=True if len(timespans) <= 8 else False,
        fmt='.3f',
        square=True
    )
    ax2.set_title(f"Distributional Distance: {lexeme}", fontsize=14, fontweight='bold')

    # --- Panel 3: Semantic change over time (adjacent JSD) ---
    #ax3 = axes[2]
    #adjacent_jsds = [jsd_matrix[i, i+1] for i in range(len(timespans)-1)]
    #ax3.plot(timespans[1:], adjacent_jsds, marker='o', color='darkblue', linewidth=2, markersize=6)
    #ax3.set_xlabel("Timespan", fontsize=12, fontweight='bold')
    #ax3.set_ylabel("Semantic Change (JSD)", fontsize=12, fontweight='bold')
    #ax3.set_title(f"Semantic Change Over Time: {lexeme}", fontsize=14, fontweight='bold')
    #ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Optional: highlight emergence signals
    #if emergence_signals:
    #    for ts, signals in emergence_signals.items():
    #        if ts in timespans:
    #            idx = timespans.index(ts)
    #            if idx > 0:  # avoid first timespan
    #                ax3.axvline(timespans[idx], color='red', linestyle='--', alpha=0.5)
    
    # Panel 3: Multidimensional Change Over Time
    ax3 = axes[2]
    adj_timespans = timespans[1:] # T2, T3, T4...
    adjacent_jsds = [jsd_matrix[i, i+1] for i in range(len(timespans)-1)]
    
    # Plot JSD Line
    ax3.plot(adj_timespans, adjacent_jsds, marker='o', label='JSD (Sense Shift)', color='blue', linewidth=2)
    
    # Plot Centroid Drift if available
    if drifts:
        drift_values = [drifts[f"{timespans[i]}->{timespans[i+1]}"] for i in range(len(timespans)-1)]
        ax3.plot(adj_timespans, drift_values, marker='s', label='Centroid Drift (Context Shift)', color='green', linestyle='--', linewidth=2)

    # Highlight Significant Changes (Chi2)
    if stepwise_sig:
        for i, ts_pair in enumerate(adj_timespans):
            key = f"{timespans[i]}->{timespans[i+1]}"
            if stepwise_sig[key]['significant']:
                ax3.annotate('$\diamond$', (ts_pair, adjacent_jsds[i]), color='red', fontsize=20, ha='center')

    ax3.set_xticks(list(adj_timespans))
    ax3.set_xticklabels(adj_timespans, ha='right')
    if len(timespans) > 5:
        ax3.tick_params(axis='x', rotation=45)
    ax3.set_title("Combined Change Metrics",  fontsize=14, fontweight='bold')
    ax3.legend()
    
    
    plt.tight_layout()
    plot_path = output_dir / f"{lexeme}_semantic_change.pdf"
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    plt.close()
    logging.info(f"  ✓ Saved semantic change plot to {plot_path}")

    # --- Stacked area plot (existing) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sense_data = {sense: [] for sense in all_senses}
    
    # Collect sample counts per timespan
    sample_counts = [timespan_data[ts].get('n_samples', 0) for ts in timespans]
    
    for timespan in timespans:
        for sense in all_senses:
            sense_data[sense].append(sense_distributions[timespan].get(sense, 0))
    
    
    plt.figure(figsize=(6, 5)) 
    ax = plt.gca()

    # Plotting the stackplot
    ax.stackplot(range(len(timespans)), 
                 *[sense_data[sense] for sense in all_senses],
                 labels=all_senses,
                 colors=colors,
                 alpha=0.8)
                 
    # Formatting ticks
    ax.set_xticks(range(len(timespans)))
    ax.set_xticklabels(timespans, rotation=45, ha='right')
    
    # Labels and Title
    ax.set_xlabel('Timespan', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sense Proportion', fontsize=12, fontweight='bold')
    ax.set_title(f'Sense Distribution: {lexeme}', fontsize=14, fontweight='bold')
    
    # --- FIXED: Legend moved inside the plot area ---
    # loc='best' allows matplotlib to try and find an empty corner, 
    # but 'upper right' is usually safe for decline-style plots.
    ax.legend(loc='lower right', frameon=True, shadow=False, fontsize=9, framealpha=0.9)
    
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(timespans) - 1]) # dynamic x-limit based on data
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # tight_layout is essential when using rotation on labels
    plt.tight_layout()
    
    stacked_path = output_dir / f"{lexeme}_stacked_distribution.pdf"
    plt.savefig(stacked_path, dpi=400, bbox_inches='tight')
    plt.close()        
    
    logging.info(f"  ✓ Saved stacked plot to {stacked_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze semantic change")
    parser.add_argument("--projections-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--exclude", nargs='+', default=[], help="List of timespans to ignore")
    args = parser.parse_args()
    
    setup_logging()
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    analyze_all_semantic_change(args.projections_dir, args.output, args.plot, args.exclude)


if __name__ == "__main__":
    main()

