"""
Script 3: Train Sense Vectors
==============================

Trains pairwise sense contrast vectors from labeled data.
Optionally applies PCA before training, then orthonormalizes vectors.

Usage:
    python 3_train_sense_vectors.py \
        --repr-dir outputs/representations \
        --layer-selection outputs/layer_selection/layer_selection.json \
        --output outputs/sense_vectors \
        --pca \
        --n-components 100

Input:
    - Representations from script 1
    - Layer selection from script 2
    
Output:
    {lexeme}_sense_vectors.pkl with orthonormalized contrast vectors
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pickle
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Orthonormalize vectors using Gram-Schmidt process.
    
    Args:
        vectors: Array of shape (n_vectors, dim)
        
    Returns:
        Orthonormalized vectors
    """
    ortho = []
    for v in vectors:
        # Subtract projections onto previous vectors
        for u in ortho:
            v = v - np.dot(v, u) * u
        
        # Normalize
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            ortho.append(v / norm)
        else:
            logging.warning("Near-zero vector encountered during orthonormalization")
    
    return np.array(ortho)


def compute_stability_metrics(
    X: np.ndarray,
    y: List[str],
    sense_vectors: np.ndarray,
    n_splits: int = 5
) -> Dict:
    """
    Compute split-half stability for sense representations.
    
    Tests whether sense centroids are stable across random data splits.
    """
    unique_senses = sorted(set(y))
    
    # Collect stability scores across splits
    all_stabilities = []
    
    for split in range(n_splits):
        # Random split
        n = len(X)
        perm = np.random.permutation(n)
        half = n // 2
        
        indices1 = perm[:half]
        indices2 = perm[half:]
        
        X1 = X[indices1]
        y1 = [y[i] for i in indices1]
        
        X2 = X[indices2]
        y2 = [y[i] for i in indices2]
        
        # Compute centroids for each sense in each split
        split_stabilities = []
        for sense in unique_senses:
            idx1 = [i for i, s in enumerate(y1) if s == sense]
            idx2 = [i for i, s in enumerate(y2) if s == sense]
            
            if len(idx1) > 0 and len(idx2) > 0:
                c1 = X1[idx1].mean(axis=0)
                c2 = X2[idx2].mean(axis=0)
                
                # Cosine similarity
                similarity = 1 - cosine(c1, c2)
                split_stabilities.append(similarity)
        
        if split_stabilities:
            all_stabilities.extend(split_stabilities)
    
    return {
        'mean_stability': float(np.mean(all_stabilities)) if all_stabilities else 0.0,
        'std_stability': float(np.std(all_stabilities)) if all_stabilities else 0.0,
        'min_stability': float(np.min(all_stabilities)) if all_stabilities else 0.0,
        'n_senses': len(unique_senses)
    }


def train_sense_vectors_for_lexeme(
    repr_file: Path,
    selected_layer: int,
    apply_pca: bool,
    n_components: int
) -> Dict:
    """Train sense vectors for a single lexeme"""
    
    with open(repr_file, 'rb') as f:
        data = pickle.load(f)
    
    lexeme = data['lexeme']
    logging.info(f"Training sense vectors for: {lexeme} (layer {selected_layer})")
    
    # Get representations from selected layer
    hidden_states = data['hidden_states'][selected_layer]
    
    # Filter labeled data
    labeled_indices = [
        i for i, m in enumerate(data['metadata'])
        if m['synset'] is not None
    ]
    
    X = hidden_states[labeled_indices]
    y = [data['metadata'][i]['synset'] for i in labeled_indices]
    
    unique_senses = sorted(set(y))
    logging.info(f"  {len(X)} samples, {len(unique_senses)} senses: {unique_senses}")
    
    if len(unique_senses) < 2:
        logging.warning(f"Need at least 2 senses, skipping {lexeme}")
        return None
    
    # Optional PCA dimensionality reduction
    pca_model = None
    if apply_pca and X.shape[1] > n_components:
        logging.info(f"  Applying PCA: {X.shape[1]} -> {n_components} dimensions")
        pca_model = PCA(n_components=n_components, random_state=42)
        X_transformed = pca_model.fit_transform(X)
        explained_var = pca_model.explained_variance_ratio_.sum()
        logging.info(f"  Explained variance: {explained_var:.3f}")
    else:
        X_transformed = X
    
    # Train pairwise sense contrast vectors
    sense_vectors = []
    sense_pairs = []
    
    logging.info("  Computing pairwise contrast vectors:")
    for i, sense1 in enumerate(unique_senses):
        for sense2 in unique_senses[i+1:]:
            # Get examples for each sense
            idx1 = [j for j, s in enumerate(y) if s == sense1]
            idx2 = [j for j, s in enumerate(y) if s == sense2]
            
            X1 = X_transformed[idx1]
            X2 = X_transformed[idx2]
            
            # Compute centroids
            c1 = X1.mean(axis=0)
            c2 = X2.mean(axis=0)
            
            # Contrast vector (direction from sense1 to sense2)
            contrast = c2 - c1
            contrast_norm = np.linalg.norm(contrast)
            
            if contrast_norm > 1e-10:
                contrast = contrast / contrast_norm
                sense_vectors.append(contrast)
                sense_pairs.append((sense1, sense2))
                
                logging.info(f"    {sense1} <-> {sense2}: magnitude={contrast_norm:.3f}")
    
    if len(sense_vectors) == 0:
        logging.warning(f"No valid contrast vectors for {lexeme}")
        return None
    
    # Orthonormalize using Gram-Schmidt
    sense_vectors = np.array(sense_vectors)
    logging.info(f"  Orthonormalizing {len(sense_vectors)} vectors...")
    sense_vectors_ortho = gram_schmidt(sense_vectors)
    
    logging.info(f"  Result: {len(sense_vectors_ortho)} orthonormal vectors")
    
    # Compute stability metrics
    logging.info("  Computing stability metrics...")
    stability = compute_stability_metrics(X_transformed, y, sense_vectors_ortho)
    logging.info(f"  Stability: {stability['mean_stability']:.3f} ± {stability['std_stability']:.3f}")
    
    return {
        'lexeme': lexeme,
        'layer': selected_layer,
        'sense_vectors': sense_vectors_ortho,
        'sense_pairs': sense_pairs,
        'unique_senses': unique_senses,
        'pca_model': pca_model,
        'stability_metrics': stability,
        'n_labeled_samples': len(X)
    }


def train_sense_vectors(
    repr_dir: Path,
    layer_selection_path: Path,
    output_dir: Path,
    apply_pca: bool,
    n_components: int
):
    """Train sense vectors for all lexemes"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load layer selection
    with open(layer_selection_path, 'r') as f:
        layer_selection = json.load(f)
    
    logging.info(f"Loaded layer selection for {len(layer_selection)} lexemes")
    
    results = {}
    
    for lexeme, info in layer_selection.items():
        repr_file = repr_dir / f"{lexeme}_representations.pkl"
        
        if not repr_file.exists():
            logging.warning(f"Representation file not found for {lexeme}")
            continue
        
        selected_layer = info['selected_layer']
        
        result = train_sense_vectors_for_lexeme(
            repr_file,
            selected_layer,
            apply_pca,
            n_components
        )
        
        if result is not None:
            # Save individual file
            output_path = output_dir / f"{lexeme}_sense_vectors.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            
            logging.info(f"  ✓ Saved to {output_path}\n")
            results[lexeme] = {
                'n_vectors': len(result['sense_vectors']),
                'stability': result['stability_metrics']['mean_stability']
            }
    
    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nTraining complete! Processed {len(results)} lexemes")
    logging.info(f"Summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SENSE VECTOR TRAINING SUMMARY")
    print("="*70)
    for lexeme, info in sorted(results.items()):
        print(f"{lexeme:20} -> {info['n_vectors']:2d} vectors, "
              f"stability: {info['stability']:.3f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Train pairwise sense contrast vectors from labeled data"
    )
    parser.add_argument(
        "--repr-dir",
        type=Path,
        required=True,
        help="Directory containing representations"
    )
    parser.add_argument(
        "--layer-selection",
        type=Path,
        required=True,
        help="Path to layer_selection.json from script 2"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for sense vectors"
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Apply PCA before training vectors"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=100,
        help="Number of PCA components (if --pca enabled)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    # Set random seed
    np.random.seed(args.seed)
    
    train_sense_vectors(
        args.repr_dir,
        args.layer_selection,
        args.output,
        args.pca,
        args.n_components
    )


if __name__ == "__main__":
    main()
