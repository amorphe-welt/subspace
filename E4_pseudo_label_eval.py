"""
Pseudo-Label Filtering Evaluation
=================================

Evaluates pseudo-label behavior per lexeme without assuming gold labels.

Metrics:
- Coverage (accepted / total unlabeled)
- Confidence statistics (mean, std)
- Sense distribution entropy
- Temporal stability of sense distributions
- Plot: Coverage vs Mean Confidence

Usage:
    python pseudo_label_eval.py \
        --pseudo-jsonl outputs/all_samples.jsonl \
        --outdir outputs/pseudo_eval
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_pseudo_jsonl(jsonl_path: Path):
    """
    Expected fields per item:
    - lexeme
    - source: gold | pseudo | rejected
    - synset (for pseudo)
    - confidence (for pseudo)
    - timespan
    """
    lex_data = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            lex_data[item["lexeme"]].append(item)
    return lex_data


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(lex_data):
    results = {}

    for lexeme, items in lex_data.items():
        pseudo = [x for x in items if x["source"] == "pseudo"]
        rejected = [x for x in items if x["source"] == "rejected"]

        n_pseudo = len(pseudo)
        n_rejected = len(rejected)
        total = n_pseudo + n_rejected

        coverage = n_pseudo / total if total > 0 else 0.0

        # Confidence statistics
        confidences = [x["confidence"] for x in pseudo]
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        std_conf = float(np.std(confidences)) if confidences else 0.0

        # Sense distribution entropy
        senses = [x["synset"] for x in pseudo if x.get("synset") is not None]
        sense_counts = Counter(senses)
        probs = np.array(list(sense_counts.values()), dtype=float)
        probs /= probs.sum() if probs.sum() > 0 else 1.0
        sense_entropy = float(entropy(probs)) if len(probs) > 1 else 0.0

        # Temporal stability (Jensen–Shannon over adjacent timespans)
        by_time = defaultdict(list)
        for x in pseudo:
            if x.get("timespan") is not None:
                by_time[x["timespan"]].append(x["synset"])

        sorted_times = sorted(by_time.keys())
        jsd_scores = []

        for t1, t2 in zip(sorted_times[:-1], sorted_times[1:]):
            c1 = Counter(by_time[t1])
            c2 = Counter(by_time[t2])

            all_senses = set(c1) | set(c2)
            p = np.array([c1[s] for s in all_senses], dtype=float)
            q = np.array([c2[s] for s in all_senses], dtype=float)

            if p.sum() == 0 or q.sum() == 0:
                continue

            p /= p.sum()
            q /= q.sum()

            m = 0.5 * (p + q)
            jsd = 0.5 * (entropy(p, m) + entropy(q, m))
            jsd_scores.append(jsd)

        temporal_instability = float(np.mean(jsd_scores)) if jsd_scores else 0.0

        results[lexeme] = {
            "Lexeme": lexeme,
            "PseudoCount": n_pseudo,
            "RejectedCount": n_rejected,
            "Coverage": coverage,
            "MeanConfidence": mean_conf,
            "StdConfidence": std_conf,
            "SenseEntropy": sense_entropy,
            "TemporalInstability": temporal_instability
        }

    return results


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_json(results, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "pseudo_label_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {out_path}")


def plot_coverage_vs_confidence(results, outdir: Path):
    lexemes, coverages, confidences = [], [], []

    for lex, r in results.items():
        lexemes.append(lex)
        coverages.append(r["Coverage"])
        confidences.append(r["MeanConfidence"])

    plt.figure(figsize=(10, 7))
    plt.scatter(coverages, confidences, s=80, alpha=0.7)
    plt.xlabel("Pseudo-Label Coverage")
    plt.ylabel("Mean Confidence")
    plt.title("Pseudo-Label Filtering Behavior per Lexeme")

    for i, lex in enumerate(lexemes):
        plt.text(
            coverages[i] + 0.005,
            confidences[i] + 0.005,
            lex,
            fontsize=8,
            alpha=0.7
        )

    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / "coverage_vs_confidence.png", dpi=300)
    plt.close()
    print(f"Saved plot to {outdir / 'coverage_vs_confidence.png'}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pseudo-Label Filtering Evaluation")
    parser.add_argument("--pseudo-jsonl", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    lex_data = load_pseudo_jsonl(args.pseudo_jsonl)
    results = compute_metrics(lex_data)
    save_json(results, args.outdir)
    plot_coverage_vs_confidence(results, args.outdir)


if __name__ == "__main__":
    main()
