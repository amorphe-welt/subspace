"""
Export Projections to JSONL
===========================

Reads all *_projections_by_time.pkl files in a folder and exports
all sample data (including projections) to a single JSONL file.

Usage:
    python export_projections_jsonl.py \
        --pkl-dir outputs/projections \
        --output outputs/all_samples.jsonl
"""

import argparse
import logging
from pathlib import Path
import pickle
import json
import numpy as np

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def export_projections_to_jsonl(pkl_dir: Path, output_file: Path):
    files = list(pkl_dir.glob("*_projections_by_time.pkl"))
    logging.info(f"Found {len(files)} projection files in {pkl_dir}")

    all_items = []

    for f in files:
        with open(f, "rb") as pf:
            data = pickle.load(pf)
        lexeme = data.get("lexeme", f.stem)
        timespan_data = data.get("timespan_data", {})

        for ts, ts_data in timespan_data.items():
            projections = ts_data.get("projections", [])
            senses = ts_data.get("senses", [])
            sentences = ts_data.get("sentences", [])
            ids = ts_data.get("ids", list(range(len(senses))))  # fallback if no ids

            for i in range(len(senses)):
                item = {
                    "id": ids[i] if i < len(ids) else i,
                    "lexeme": lexeme,
                    "sentence": sentences[i] if i < len(sentences) else None,
                    "synset": senses[i],
                    "timespan": ts,
                    "projection": projections[i].tolist() if i < len(projections) else None
                }
                all_items.append(item)

    # Write JSONL
    with open(output_file, "w", encoding="utf-8") as out_f:
        for item in all_items:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"Exported {len(all_items)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Export projection PKLs to JSONL")
    parser.add_argument("--pkl-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_logging()
    export_projections_to_jsonl(args.pkl_dir, args.output)


if __name__ == "__main__":
    main()
