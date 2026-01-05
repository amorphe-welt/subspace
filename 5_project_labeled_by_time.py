"""
Script 5: Project Labeled Data by Timespan (with optional pseudo-labeled data)
=============================================================================

Projects labeled usages into sense subspace, grouped by timespan.
Optionally includes automatically labeled (pseudo-labeled) data.

Usage:
    python 5_project_labeled_by_time.py \
        --repr-dir outputs/representations \
        --subspace-dir outputs/subspaces \
        --output outputs/projections \
        [--pseudo-dir outputs/pseudo_representations --include-pseudo]

Input:
    - Representations from script 1 (gold)
    - Optional pseudo-labeled representations
    - Subspaces from script 4

Output:
    {lexeme}_projections_by_time.pkl with coordinates grouped by timespan
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pickle
from collections import defaultdict


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def project_to_subspace(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project vector onto subspace spanned by basis vectors (orthonormal)"""
    return basis @ vec


def compute_distribution_statistics(projections: np.ndarray) -> Dict:
    """Compute statistics for projected coordinates"""
    return {
        'mean': projections.mean(axis=0).tolist(),
        'std': projections.std(axis=0).tolist(),
        'min': projections.min(axis=0).tolist(),
        'max': projections.max(axis=0).tolist(),
        'n_samples': len(projections)
    }


def load_representations(file_path: Path) -> Dict:
    """Load a pickled representation file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def project_lexeme_by_time(
    repr_files: List[Path],
    subspace_file: Path
) -> Dict:
    """Project labeled data (gold + optional pseudo) for a single lexeme, grouped by timespan"""
    
    # Load subspace
    with open(subspace_file, 'rb') as f:
        subspace = pickle.load(f)
    
    lexeme = subspace['lexeme']
    logging.info(f"Projecting {lexeme} by timespan")

    # Initialize timespan data
    timespan_data = defaultdict(lambda: {
        'representations': [],
        'senses': [],
        'projections': [],
        'sentences': []
    })

    n_total = 0
    n_labeled = 0

    for repr_file in repr_files:
        repr_data = load_representations(repr_file)
        layer = subspace['layer']
        hidden_states = repr_data['hidden_states'][layer]

        # Apply PCA if used
        if subspace['pca_model'] is not None:
            hidden_states = subspace['pca_model'].transform(hidden_states)


        for i, meta in enumerate(repr_data['metadata']):
            n_total += 1

            synset = meta['synset']
            if synset in [None, "None"]:
                continue
            timespan = meta['timespan']

            # Skip if no synset or no timespan
            if synset is None or timespan is None:
                if timespan is None:
                    logging.warning(f"  Sample {i} has no timespan, skipping")
                continue

            n_labeled += 1

            vec = hidden_states[i]
            projection = project_to_subspace(vec, subspace['basis'])

            timespan_data[timespan]['representations'].append(vec)
            timespan_data[timespan]['senses'].append(synset)  # only non-None here
            timespan_data[timespan]['projections'].append(projection)
            timespan_data[timespan]['sentences'].append(meta['sentence'])

    logging.info(f"  Total samples: {n_total}, Labeled: {n_labeled}")
    logging.info(f"  Found {len(timespan_data)} timespans")

    # Convert lists to arrays and compute statistics
    for timespan in timespan_data:
        data = timespan_data[timespan]
        
        # 1. Store the Mean Embedding (Centroid) for Drift Analysis
        # We use the high-dimensional representations here
        repr_array = np.array(data['representations'])
        data['mean_embedding'] = repr_array.mean(axis=0)

        # 2. Store Raw Sense Counts for Chi-Squared Significance
        sense_counts = {}
        for s in data['senses']:
            sense_counts[s] = sense_counts.get(s, 0) + 1
        data['sense_counts'] = sense_counts
        
        # 3. Standard processing (Projections and Distribution)
        data['projections'] = np.array(data['projections'])
        data['statistics'] = compute_distribution_statistics(data['projections'])
        
        # Only include valid senses (filter out None)
        valid_senses = [s for s in data['senses'] if s is not None]

        # Count occurrences
        sense_counts = {}
        for s in valid_senses:
            sense_counts[s] = sense_counts.get(s, 0) + 1

        total = sum(sense_counts.values())
        if total > 0:
            data['sense_distribution'] = {s: c / total for s, c in sense_counts.items()}
        else:
            data['sense_distribution'] = {}

    return {
        'lexeme': lexeme,
        'subspace_dimension': subspace['dimension'],
        'layer': layer,
        'timespan_data': dict(timespan_data),
        'all_senses': subspace['unique_senses'],
        'n_labeled_projected': n_labeled
    }

def project_all_by_time(
    repr_dir: Path,
    subspace_dir: Path,
    output_dir: Path,
    pseudo_dir: Path = None,
    include_pseudo: bool = False
):
    """Project all lexemes by timespan, optionally including pseudo-labeled data"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    subspace_files = list(subspace_dir.glob("*_subspace.pkl"))
    logging.info(f"Found {len(subspace_files)} lexemes to process")

    results = {}

    for subspace_file in subspace_files:
        lexeme = subspace_file.stem.replace("_subspace", "")
        repr_files = [repr_dir / f"{lexeme}_representations.pkl"]

        # Include pseudo-labeled file if requested
        if include_pseudo and pseudo_dir is not None:
            pseudo_file = pseudo_dir / f"{lexeme}_representations.pkl"
            if pseudo_file.exists():
                repr_files.append(pseudo_file)
            else:
                logging.warning(f"Pseudo-labeled file not found for {lexeme}, skipping pseudo data")

        # Skip if no representation files exist
        repr_files = [f for f in repr_files if f.exists()]
        if not repr_files:
            logging.warning(f"No representation files found for {lexeme}, skipping")
            continue

        projection_data = project_lexeme_by_time(repr_files, subspace_file)

        output_path = output_dir / f"{lexeme}_projections_by_time.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(projection_data, f)

        logging.info(f"  ✓ Saved to {output_path}\n")

        results[lexeme] = {
            'n_timespans': len(projection_data['timespan_data']),
            'n_samples': projection_data['n_labeled_projected'],
            'timespans': sorted(projection_data['timespan_data'].keys())
        }

    summary_path = output_dir / "projection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"\nProjection complete!")
    logging.info(f"Processed {len(results)} lexemes")
    logging.info(f"Summary saved to {summary_path}")

    # Print summary
    print("\n" + "="*80)
    print("PROJECTION SUMMARY")
    print("="*80)
    print(f"{'Lexeme':<20} {'# Timespans':<15} {'# Samples':<15} {'Timespans':<30}")
    print("-"*80)
    for lexeme, info in sorted(results.items()):
        timespans_str = ", ".join(info['timespans'][:3])
        if len(info['timespans']) > 3:
            timespans_str += "..."
        print(f"{lexeme:<20} {info['n_timespans']:<15} {info['n_samples']:<15} {timespans_str:<30}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Project labeled usages into sense subspace by timespan (optional pseudo-labeled data)"
    )
    parser.add_argument(
        "--repr-dir",
        type=Path,
        required=True,
        help="Directory containing gold representations"
    )
    parser.add_argument(
        "--subspace-dir",
        type=Path,
        required=True,
        help="Directory containing subspaces from script 4"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for projections"
    )
    parser.add_argument(
        "--pseudo-dir",
        type=Path,
        help="Directory containing pseudo-labeled representations"
    )
    parser.add_argument(
        "--include-pseudo",
        action="store_true",
        help="Include pseudo-labeled data when projecting"
    )

    args = parser.parse_args()
    setup_logging()

    project_all_by_time(args.repr_dir, args.subspace_dir, args.output,
                        pseudo_dir=args.pseudo_dir,
                        include_pseudo=args.include_pseudo)


if __name__ == "__main__":
    main()
