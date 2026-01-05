"""
Script 6: Pseudo-Label Unlabeled Data (Combined JSONL)
======================================================

Pseudo-labels unlabeled data using subspaces derived from gold/labeled data.
Outputs a single JSONL with gold + pseudo + rejected instances.
"""

import json
import argparse
import logging
from pathlib import Path
import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def project_to_subspace(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return basis @ vec

def assign_sense_with_confidence_(proj, centroids_proj, threshold):
    distances = {sense: float(np.linalg.norm(proj - c)) for sense, c in centroids_proj.items()}
    nearest_sense = min(distances, key=distances.get)
    all_dists = np.array(list(distances.values()))
    inv_dists = 1.0 / (all_dists + 1e-10)
    confidences = inv_dists / inv_dists.sum()
    nearest_conf = float(confidences[list(distances.keys()).index(nearest_sense)])
    if nearest_conf >= threshold:
        assigned = nearest_sense
        labeled = True
    else:
        # always assign
        #assigned = None
        assigned = nearest_sense
        labeled = False
    return labeled, assigned, nearest_conf, distances
    
def assign_sense_with_confidence(proj, centroids_proj, threshold):
    # compute distances to each centroid
    distances = {sense: float(np.linalg.norm(proj - c)) for sense, c in centroids_proj.items()}

    # find nearest sense
    nearest_sense = min(distances, key=distances.get)

    # convert distances to inverse-normalized "confidence"
    all_dists = np.array(list(distances.values()))
    inv_dists = 1.0 / (all_dists + 1e-10)
    confidences = inv_dists / inv_dists.sum()

    nearest_index = list(distances.keys()).index(nearest_sense)
    nearest_conf = float(confidences[nearest_index])

    # compute margin: difference to 2nd best
    sorted_conf = np.sort(confidences)[::-1]  # descending
    if len(sorted_conf) > 1:
        margin = float(sorted_conf[0] - sorted_conf[1])
    else:
        margin = float(sorted_conf[0])  # only one sense

    # apply threshold
    if nearest_conf >= threshold:
        labeled = True
    else:
        labeled = False

    return labeled, nearest_sense, nearest_conf, distances, margin


def pseudo_label_lexeme(labeled_file, unlabeled_file, subspace_file, threshold):
    with open(labeled_file, 'rb') as f:
        labeled = pickle.load(f)
    with open(unlabeled_file, 'rb') as f:
        unlabeled = pickle.load(f)
    with open(subspace_file, 'rb') as f:
        subspace = pickle.load(f)

    lexeme = labeled['lexeme']
    layer = subspace['layer']

    labeled_hidden = labeled['hidden_states'][layer]
    labeled_meta = labeled['metadata']
    unlabeled_hidden = unlabeled['hidden_states'][layer]
    unlabeled_meta = unlabeled['metadata']

    # Apply PCA if available
    if subspace.get('pca_model') is not None:
        labeled_hidden = subspace['pca_model'].transform(labeled_hidden)
        unlabeled_hidden = subspace['pca_model'].transform(unlabeled_hidden)

    # Compute centroids from gold/labeled
    centroids = {}
    for i, m in enumerate(labeled_meta):
        if m['synset'] is not None:
            centroids.setdefault(m['synset'], []).append(labeled_hidden[i])
    for sense in centroids:
        centroids[sense] = np.mean(centroids[sense], axis=0)
    centroids_proj = {s: project_to_subspace(c, subspace['basis']) for s, c in centroids.items()}

    # Combine gold + pseudo-labeled
    all_items = []
    stats = {'gold': 0, 'pseudo': 0, 'rejected': 0}
    # Gold
    #print(labeled_meta[0])
    
    for i, m in enumerate(labeled_meta):
        all_items.append({
            'id': m['id'],
            'token': m.get('token'),
            'sentence': m['sentence'],
            'lexeme': lexeme,
            'synset': m['synset'],
            'timespan': m.get('timespan'),
            'span': m.get('span'),
            'source': 'gold',
            'confidence': 1.0,
            'margin': 1.0
        })
        stats['gold'] += 1
    # Pseudo-label
    for i, m in enumerate(unlabeled_meta):
        vec = unlabeled_hidden[i]
        proj = project_to_subspace(vec, subspace['basis'])
        labeled, assigned, conf, distances, margin = assign_sense_with_confidence(proj, centroids_proj, threshold)
        if labeled is True:
            source = 'pseudo'
            stats['pseudo'] += 1
        else:
            source = 'rejected'
            stats['rejected'] += 1
        all_items.append({
            'id': m['id'],
            'token': m['token'],
            'sentence': m['sentence'],
            'lexeme': lexeme,
            'synset': assigned,
            'timespan': m.get('timespan'),
            'span': m.get('span'),
            'source': source,
            'confidence': conf,
            'margin': margin
        })
    logging.info(f"{lexeme}: Gold={stats['gold']} Pseudo={stats['pseudo']} Rejected={stats['rejected']}")
    return all_items

def pseudo_label_all(labeled_dir, unlabeled_dir, subspace_dir, output_file, threshold):
    labeled_files = {f.stem.replace('_representations',''): f for f in labeled_dir.glob('*.pkl')}
    unlabeled_files = {f.stem.replace('_representations',''): f for f in unlabeled_dir.glob('*.pkl')}
    subspace_files = {f.stem.replace('_subspace',''): f for f in subspace_dir.glob('*.pkl')}

    all_items = []

    for lexeme, subspace_file in subspace_files.items():
        if lexeme not in labeled_files:
            logging.info(f"Skipping {lexeme} (no labeled data)")
            continue
        if lexeme not in unlabeled_files:
            logging.info(f"Skipping {lexeme} (no unlabeled data)")
            continue
        labeled_file = labeled_files[lexeme]
        unlabeled_file = unlabeled_files[lexeme]
        items = pseudo_label_lexeme(labeled_file, unlabeled_file, subspace_file, threshold)
        all_items.extend(items)

    # Save combined JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info(f"Saved combined pseudo-labeled data to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Pseudo-label unlabeled data (combined JSONL)")
    parser.add_argument("--labeled-repr-dir", type=Path, required=True)
    parser.add_argument("--unlabeled-repr-dir", type=Path, required=True)
    parser.add_argument("--subspace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    setup_logging()
    pseudo_label_all(args.labeled_repr_dir, args.unlabeled_repr_dir, args.subspace_dir, args.output, args.threshold)

if __name__ == "__main__":
    main()
