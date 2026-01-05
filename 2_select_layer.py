"""
Script 2: Select Optimal Layer
===============================

Selects the best layer for each lexeme based on sense separability.
Uses linear probe accuracy and silhouette scores to evaluate layers.

Usage:
    python 2_select_layer.py \
        --repr-dir outputs/representations \
        --output outputs/layer_selection

Input:
    {lexeme}_representations.pkl from script 1
    
Output:
    layer_selection.json with selected layer per lexeme
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def evaluate_layer_separability(
    X: np.ndarray,
    y: List[str],
    layer_id: int
) -> Dict[str, float]:
    """
    Evaluate how well senses are separated in this layer.
    
    Returns:
        Dictionary with probe_accuracy, silhouette, and combined score
    """
    unique_senses = list(set(y))
    
    if len(unique_senses) < 2:
        return {
            'probe_accuracy': 0.0,
            'silhouette': 0.0,
            'combined': 0.0,
            'n_senses': len(unique_senses)
        }
    
    # Linear probe with cross-validation
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    
    # Use cross-validation if enough samples
    if len(X) >= 10:
        cv_scores = cross_val_score(clf, X, y, cv=min(5, len(X) // 2))
        probe_acc = cv_scores.mean()
    else:
        clf.fit(X, y)
        probe_acc = clf.score(X, y)
    
    # Silhouette score
    if len(unique_senses) <= len(X) - 1:
        try:
            sil_score = silhouette_score(X, y)
        except:
            sil_score = 0.0
    else:
        sil_score = 0.0
    
    # Combined score (weighted average)
    combined = 0.6 * probe_acc + 0.4 * max(0, sil_score)
    
    return {
        'probe_accuracy': float(probe_acc),
        'silhouette': float(sil_score),
        'combined': float(combined),
        'n_senses': len(unique_senses),
        'n_samples': len(X)
    }


def select_layer_for_lexeme(
    repr_file: Path
) -> Dict:
    """Select best layer for a single lexeme"""
    
    with open(repr_file, 'rb') as f:
        data = pickle.load(f)
    
    lexeme = data['lexeme']
    logging.info(f"Evaluating layers for: {lexeme}")
    
    # Filter for labeled data only
    labeled_indices = [
        i for i, m in enumerate(data['metadata'])
        if m['synset'] is not None
    ]
    
    if len(labeled_indices) < 10:
        logging.warning(f"Insufficient labeled data for {lexeme}: {len(labeled_indices)} samples")
        return None
    
    # Get labels
    labels = [data['metadata'][i]['synset'] for i in labeled_indices]
    unique_senses = list(set(labels))
    
    if len(unique_senses) < 2:
        logging.warning(f"Only one sense for {lexeme}")
        return None
    
    logging.info(f"  {len(labeled_indices)} labeled samples, {len(unique_senses)} senses")
    
    # Evaluate each layer
    layer_scores = {}
    
    for layer, hidden_states in data['hidden_states'].items():
        X = hidden_states[labeled_indices]
        
        scores = evaluate_layer_separability(X, labels, layer)
        layer_scores[int(layer)] = scores
        
        logging.info(
            f"  Layer {layer}: probe={scores['probe_accuracy']:.3f}, "
            f"silhouette={scores['silhouette']:.3f}, "
            f"combined={scores['combined']:.3f}"
        )
    
    # Select best layer
    best_layer = max(
        layer_scores.keys(),
        key=lambda l: layer_scores[l]['combined']
    )
    
    logging.info(f"  ✓ Selected layer {best_layer} for {lexeme}")
    
    return {
        'lexeme': lexeme,
        'selected_layer': best_layer,
        'layer_scores': layer_scores,
        'n_labeled_samples': len(labeled_indices),
        'n_senses': len(unique_senses),
        'sense_labels': unique_senses
    }


def select_layers(repr_dir: Path, output_dir: Path):
    """Select optimal layer for all lexemes"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    repr_files = list(repr_dir.glob("*_representations.pkl"))
    logging.info(f"Found {len(repr_files)} lexemes to process")
    
    for repr_file in repr_files:
        result = select_layer_for_lexeme(repr_file)
        
        if result is not None:
            results[result['lexeme']] = result
    
    # Save results
    output_path = output_dir / "layer_selection.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nSaved layer selection to {output_path}")
    logging.info(f"Successfully processed {len(results)} lexemes")
    
    # Print summary
    print("\n" + "="*60)
    print("LAYER SELECTION SUMMARY")
    print("="*60)
    for lexeme, info in sorted(results.items()):
        print(f"{lexeme:20} -> Layer {info['selected_layer']:2d} "
              f"(score: {info['layer_scores'][info['selected_layer']]['combined']:.3f})")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Select optimal layer for each lexeme based on sense separability"
    )
    parser.add_argument(
        "--repr-dir",
        type=Path,
        required=True,
        help="Directory containing representations from script 1"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for layer selection results"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    select_layers(args.repr_dir, args.output)


if __name__ == "__main__":
    main()
