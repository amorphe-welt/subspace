"""
Inject Pseudo-Labels into Representations
=========================================

Takes unlabeled BERT representations and injects pseudo-labels from auto_labeled.jsonl
so that projection scripts can use them.

Usage:
    python inject_pseudo_labels.py \
        --input-dir outputs/embeddings/unlabeled \
        --pseudo-jsonl outputs/auto_labeled.jsonl \
        --output-dir outputs/embeddings/pseudo_labeled
"""

import argparse
from pathlib import Path
import json
import pickle
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def load_pseudo_labels(pseudo_jsonl: Path):
    pseudo_map = {}
    with open(pseudo_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            uid = item["id"]
            synset = item.get("synset")
            if synset is not None:
                pseudo_map[uid] = synset
    logging.info(f"Loaded {len(pseudo_map)} pseudo-labels from {pseudo_jsonl}")
    return pseudo_map

def inject_labels(input_dir: Path, output_dir: Path, pseudo_map: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    rep_files = list(input_dir.glob("*_representations.pkl"))
    logging.info(f"Found {len(rep_files)} representation files in {input_dir}")

    for rep_file in rep_files:
        with open(rep_file, "rb") as f:
            reps = pickle.load(f)

        n_injected = 0
        for meta in reps["metadata"]:
            uid = meta["id"]
            if uid in pseudo_map:
                meta["synset"] = pseudo_map[uid]
                n_injected += 1

        out_path = output_dir / rep_file.name
        with open(out_path, "wb") as f:
            pickle.dump(reps, f)
        logging.info(f"{rep_file.name}: injected {n_injected} pseudo-labels -> {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Inject pseudo-labels into representations")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder with unlabeled representations")
    parser.add_argument("--pseudo-jsonl", type=Path, required=True, help="JSONL with pseudo-labeled data")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder to save updated representations")
    args = parser.parse_args()

    setup_logging()
    pseudo_map = load_pseudo_labels(args.pseudo_jsonl)
    inject_labels(args.input_dir, args.output_dir, pseudo_map)

if __name__ == "__main__":
    main()
