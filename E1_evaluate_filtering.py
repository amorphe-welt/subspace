"""
Filtering Evaluation: Sense Pseudo-Label Quality
================================================

Evaluates the quality of pseudo-labels produced by sense subspaces.
This is NOT WSD evaluation.

Measures:
- Precision (purity) of accepted pseudo-labels
- Coverage (% of data accepted)
- Per-sense filtering quality

Usage:
    python evaluate_filtering.py \
        --repr-dir outputs/representations \
        --subspace-dir outputs/subspaces \
        --output outputs/filtering_eval \
        --thresholds 0.3 0.4 0.5
"""

import argparse
import json
import logging
from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


# ---------- Geometry ----------

def project(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project vector into sense subspace"""
    return basis @ vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


# ---------- Evaluation ----------

def evaluate_lexeme_filtering(
    repr_file: Path,
    subspace_file: Path,
    thresholds: list[float]
):
    with open(repr_file, "rb") as f:
        repr_data = pickle.load(f)

    with open(subspace_file, "rb") as f:
        subspace = pickle.load(f)

    lexeme = repr_data["lexeme"]
    layer = subspace["layer"]

    logging.info(f"Filtering evaluation: {lexeme}")

    hidden = repr_data["hidden_states"][layer]
    metadata = repr_data["metadata"]

    # Only gold-labeled samples
    indices = [i for i, m in enumerate(metadata) if m["synset"] is not None]
    X = hidden[indices]
    y = [metadata[i]["synset"] for i in indices]

    if subspace["pca_model"] is not None:
        X = subspace["pca_model"].transform(X)

    # Compute sense centroids (gold)
    sense_centroids = defaultdict(list)
    for vec, sense in zip(X, y):
        sense_centroids[sense].append(vec)

    sense_centroids = {
        s: np.mean(vs, axis=0)
        for s, vs in sense_centroids.items()
    }

    # Project centroids
    centroids_proj = {
        s: project(c, subspace["basis"])
        for s, c in sense_centroids.items()
    }

    results = {}

    for threshold in thresholds:
        accepted = []
        correct = 0

        per_sense = defaultdict(lambda: {"accepted": 0, "correct": 0})

        for vec, gold in zip(X, y):
            v_proj = project(vec, subspace["basis"])

            scores = {
                s: cosine(v_proj, c_proj)
                for s, c_proj in centroids_proj.items()
            }

            pred = max(scores, key=scores.get)
            conf = scores[pred]

            if conf >= threshold:
                accepted.append((pred, gold))
                per_sense[pred]["accepted"] += 1
                if pred == gold:
                    correct += 1
                    per_sense[pred]["correct"] += 1

        n_total = len(X)
        n_acc = len(accepted)

        precision = correct / n_acc if n_acc else 0.0
        coverage = n_acc / n_total if n_total else 0.0

        results[threshold] = {
            "precision": precision,
            "coverage": coverage,
            "accepted": n_acc,
            "correct": correct,
            "per_sense": {
                s: {
                    "accepted": v["accepted"],
                    "precision": (
                        v["correct"] / v["accepted"]
                        if v["accepted"] else 0.0
                    )
                }
                for s, v in per_sense.items()
            }
        }

        logging.info(
            f"  t={threshold:.2f} | "
            f"coverage={coverage:.1%} | "
            f"precision={precision:.3f} | "
            f"accepted={n_acc}"
        )

    return {
        "lexeme": lexeme,
        "n_samples": len(X),
        "subspace_dim": subspace["dimension"],
        "results": results
    }


# ---------- Runner ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr-dir", type=Path, required=True)
    parser.add_argument("--subspace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5])

    args = parser.parse_args()
    setup_logging()

    args.output.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for subspace_file in args.subspace_dir.glob("*_subspace.pkl"):
        lexeme = subspace_file.stem.replace("_subspace", "")
        repr_file = args.repr_dir / f"{lexeme}_representations.pkl"

        if not repr_file.exists():
            continue

        res = evaluate_lexeme_filtering(
            repr_file,
            subspace_file,
            args.thresholds
        )

        all_results[lexeme] = res

    with open(args.output / "filtering_eval.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logging.info("Filtering evaluation complete.")


if __name__ == "__main__":
    main()
