"""
Script 4: Build Sense Subspace
===============================

Builds fixed sense subspaces from orthonormalized sense vectors.
The subspace is defined by the span of the sense contrast vectors.

Usage:
    python 4_build_sense_subspace.py \
        --vectors-dir outputs/sense_vectors \
        --output outputs/subspaces

Input:
    {lexeme}_sense_vectors.pkl from script 3
    
Output:
    {lexeme}_subspace.pkl with subspace basis and metadata
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import pickle


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def build_subspace_for_lexeme(vectors_file: Path) -> Dict:
    """Build subspace for a single lexeme"""
    
    with open(vectors_file, 'rb') as f:
        data = pickle.load(f)
    
    lexeme = data['lexeme']
    sense_vectors = data['sense_vectors']
    
    logging.info(f"Building subspace for: {lexeme}")
    logging.info(f"  Basis dimension: {sense_vectors.shape[0]}")
    logging.info(f"  Embedding dimension: {sense_vectors.shape[1]}")
    
    # The subspace is simply the span of the orthonormal sense vectors
    # Each row is a basis vector
    subspace = {
        'lexeme': lexeme,
        'layer': data['layer'],
        'basis': sense_vectors,  # Shape: (n_vectors, embedding_dim)
        'dimension': sense_vectors.shape[0],
        'embedding_dim': sense_vectors.shape[1],
        'unique_senses': data['unique_senses'],
        'sense_pairs': data['sense_pairs'],
        'pca_model': data.get('pca_model'),
        'stability_metrics': data.get('stability_metrics', {}),
        'n_labeled_samples': data.get('n_labeled_samples', 0)
    }
    
    # Verify orthonormality
    gram_matrix = sense_vectors @ sense_vectors.T
    identity = np.eye(len(sense_vectors))
    orthonormality_error = np.linalg.norm(gram_matrix - identity)
    
    logging.info(f"  Orthonormality error: {orthonormality_error:.6f}")
    
    if orthonormality_error > 1e-3:
        logging.warning(f"  High orthonormality error for {lexeme}!")
    
    subspace['orthonormality_error'] = float(orthonormality_error)
    
    return subspace


def build_subspaces(vectors_dir: Path, output_dir: Path):
    """Build subspaces for all lexemes"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vector_files = list(vectors_dir.glob("*_sense_vectors.pkl"))
    logging.info(f"Found {len(vector_files)} lexemes to process")
    
    results = {}
    
    for vec_file in vector_files:
        subspace = build_subspace_for_lexeme(vec_file)
        
        # Save subspace
        output_path = output_dir / f"{subspace['lexeme']}_subspace.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(subspace, f)
        
        logging.info(f"  ✓ Saved to {output_path}\n")
        
        results[subspace['lexeme']] = {
            'dimension': subspace['dimension'],
            'embedding_dim': subspace['embedding_dim'],
            'n_senses': len(subspace['unique_senses']),
            'orthonormality_error': subspace['orthonormality_error'],
            'stability': subspace['stability_metrics'].get('mean_stability', 0.0)
        }
    
    # Save summary
    summary_path = output_dir / "subspace_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nSubspace construction complete!")
    logging.info(f"Processed {len(results)} lexemes")
    logging.info(f"Summary saved to {summary_path}")
    
    # Print summary table
    print("\n" + "="*90)
    print("SENSE SUBSPACE SUMMARY")
    print("="*90)
    print(f"{'Lexeme':<20} {'Subspace Dim':<15} {'# Senses':<12} {'Orthonorm Error':<18} {'Stability':<12}")
    print("-"*90)
    
    for lexeme, info in sorted(results.items()):
        print(f"{lexeme:<20} {info['dimension']:<15} {info['n_senses']:<12} "
              f"{info['orthonormality_error']:<18.6f} {info['stability']:<12.3f}")
    
    print("="*90)
    
    # Print statistics
    dims = [info['dimension'] for info in results.values()]
    errors = [info['orthonormality_error'] for info in results.values()]
    stabilities = [info['stability'] for info in results.values()]
    
    print("\nStatistics:")
    print(f"  Average subspace dimension: {np.mean(dims):.1f} ± {np.std(dims):.1f}")
    print(f"  Average orthonormality error: {np.mean(errors):.6f}")
    print(f"  Average stability: {np.mean(stabilities):.3f} ± {np.std(stabilities):.3f}")
    print("="*90)


def main():
    parser = argparse.ArgumentParser(
        description="Build fixed sense subspaces from sense vectors"
    )
    parser.add_argument(
        "--vectors-dir",
        type=Path,
        required=True,
        help="Directory containing sense vectors from script 3"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for subspaces"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    build_subspaces(args.vectors_dir, args.output)


if __name__ == "__main__":
    main()
