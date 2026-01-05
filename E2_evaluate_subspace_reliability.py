"""
Script 5: Subspace Reliability Evaluation
=========================================

Evaluates the reliability of sense subspaces:
  - Orthonormality of basis
  - Projection stability across resamples
  - Confidence distributions for labeled and control vectors

Usage:
    python evaluate_subspace_reliability.py \
        --subspace-dir outputs/subspaces \
        --repr-dir outputs/representations \
        --output outputs/reliability \
        --n-resamples 5
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pickle
from collections import defaultdict

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_orthonormality(basis: np.ndarray):
    gram = basis @ basis.T
    error = np.linalg.norm(gram - np.eye(len(basis)))
    return error

def projection_confidence(vec: np.ndarray, centroids_proj: dict):
    # inverse distance confidence
    distances = {sense: float(np.linalg.norm(vec - c)) for sense, c in centroids_proj.items()}
    nearest_sense = min(distances, key=distances.get)
    all_dists = np.array(list(distances.values()))
    inv_dists = 1.0 / (all_dists + 1e-10)
    confidences = inv_dists / inv_dists.sum()
    return nearest_sense, float(confidences[list(distances.keys()).index(nearest_sense)])

def evaluate_subspace(subspace_file: Path, repr_file: Path, n_resamples=5):
    with open(subspace_file, 'rb') as f:
        subspace = pickle.load(f)
    with open(repr_file, 'rb') as f:
        repr_data = pickle.load(f)
    
    lexeme = subspace['lexeme']
    layer = subspace['layer']
    basis = subspace['basis']
    
    ortho_error = check_orthonormality(basis)
    
    hidden = repr_data['hidden_states'][layer]
    meta = repr_data['metadata']
    
    labeled_idx = [i for i, m in enumerate(meta) if m['synset'] is not None]
    X = hidden[labeled_idx]
    y = [meta[i]['synset'] for i in labeled_idx]
    
    # Compute centroids
    centroids = defaultdict(list)
    for xi, yi in zip(X, y):
        centroids[yi].append(xi)
    centroids = {s: np.mean(vs, axis=0) for s, vs in centroids.items()}
    centroids_proj = {s: basis @ c for s, c in centroids.items()}
    
    # Confidence analysis
    confidences = []
    for xi in X:
        proj = basis @ xi
        _, conf = projection_confidence(proj, centroids_proj)
        confidences.append(conf)
    
    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))
    
    # Resampling stability (subspace similarity across subsamples)
    if n_resamples > 1:
        resample_errors = []
        n_samples = X.shape[0]
        for _ in range(n_resamples):
            idxs = np.random.choice(n_samples, n_samples//2, replace=False)
            subX = X[idxs]
            # PCA or Gram-based check could go here; simplified as mean distance
            # between projections
            proj_mean = basis @ subX.mean(axis=0)
            resample_errors.append(np.linalg.norm(proj_mean))
        stability_error = float(np.std(resample_errors))
    else:
        stability_error = None
    
    return {
        'lexeme': lexeme,
        'orthonormality_error': ortho_error,
        'mean_confidence': mean_conf,
        'std_confidence': std_conf,
        'resample_stability_error': stability_error
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate subspace reliability")
    parser.add_argument("--subspace-dir", type=Path, required=True)
    parser.add_argument("--repr-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=5)
    
    args = parser.parse_args()
    setup_logging()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    subspace_files = list(args.subspace_dir.glob("*_subspace.pkl"))
    
    results = []
    for sf in subspace_files:
        lexeme = sf.stem.replace("_subspace","")
        repr_file = args.repr_dir / f"{lexeme}_representations.pkl"
        if not repr_file.exists():
            logging.warning(f"No representations for {lexeme}, skipping")
            continue
        res = evaluate_subspace(sf, repr_file, args.n_resamples)
        results.append(res)
        logging.info(f"{lexeme}: Ortho={res['orthonormality_error']:.4f}, "
                     f"MeanConf={res['mean_confidence']:.3f}, "
                     f"StabilityErr={res['resample_stability_error']}")
    
    # Convert results to dict keyed by lexeme
    results_dict = {res['lexeme']: res for res in results}

    # Save as JSON
    import json
    with open(args.output / "subspace_reliability.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    logging.info(f"Saved subspace reliability results to {args.output / 'subspace_reliability.json'}")

if __name__ == "__main__":
    main()
