"""
Script 1: Extract Representations (Updated)
===========================================

Supports gold/labeled data (multiple layers) and unlabeled data (best layer per lexeme).

Usage:

# Gold/labeled: extract multiple layers
python 1_extract_representations.py \
    --input labeled.jsonl \
    --output outputs/representations/labeled \
    --model bert-base-german-cased \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12

# Unlabeled: extract only best layer per lexeme (from Script 2)
python 1_extract_representations.py \
    --input unlabeled.jsonl \
    --output outputs/representations/unlabeled \
    --model bert-base-german-cased \
    --layer-map best_layers.json
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from tqdm import tqdm

@dataclass
class Usage:
    id: str
    sentence: str
    lexeme: str
    token: Optional[str] = None  # Added token field
    span: Optional[List[int]] = None
    synset: Optional[str] = None
    timespan: Optional[str] = None
    corpus: Optional[str] = None

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_token_indices_from_whitespace_span(sentence: str, search_token: Optional[str], span: Optional[List[int]] = None) -> List[int]:
    tokens = sentence.split()
    
    # Case 1: Span is provided (highest priority)
    if span is not None:
        if span[1] > len(tokens):
            raise ValueError(f"Span {span} out of range for sentence with {len(tokens)} tokens")
        return list(range(span[0], span[1]))
    
    # Case 2: Span is missing, search for the 'token' string in the split sentence
    if search_token:
        try:
            # Look for exact match in whitespace-split list
            idx = tokens.index(search_token)
            return [idx]
        except ValueError:
            # Fallback: Find the first token that contains the search_token string
            for i, t in enumerate(tokens):
                if search_token in t:
                    return [i]
            raise ValueError(f"Token '{search_token}' not found in sentence: {sentence}")
    
    raise ValueError("Both 'span' and 'token' are missing; cannot locate target.")

def pool_subword_representations(hidden: torch.Tensor, pooling: str) -> np.ndarray:
    if pooling == "first":
        return hidden[0].cpu().numpy()
    elif pooling == "mean":
        return hidden.mean(dim=0).cpu().numpy()
    elif pooling == "last":
        return hidden[-1].cpu().numpy()
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")

def extract_representations(
    input_jsonl: Path,
    output_dir: Path,
    model_name: str,
    layers: Optional[List[int]],
    layer_map: Optional[Dict[str,int]],
    pooling: str,
    batch_size: int,
    device: str
):
    logging.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    usages: List[Usage] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            usages.append(
                Usage(
                    id=d["id"],
                    sentence=d["sentence"],
                    lexeme=d["lexeme"],
                    token=d.get("token"), # Load token from JSONL
                    span=d.get("span"),
                    synset=d.get("synset"),
                    timespan=d.get("timespan"),
                    corpus=d.get("corpus")
                )
            )

    lexeme_usages = {}
    for u in usages:
        lexeme_usages.setdefault(u.lexeme, []).append(u)

    output_dir.mkdir(parents=True, exist_ok=True)
    lexeme_layers_used = {}

    for lexeme, lex_usages in lexeme_usages.items():
        if layer_map is not None:
            if lexeme not in layer_map:
                logging.warning(f"No layer specified for {lexeme} in layer_map, skipping")
                continue
            layers_to_extract = [layer_map[lexeme]]
        else:
            if layers is None:
                raise ValueError("Must specify --layers for gold/labeled data")
            layers_to_extract = layers

        lexeme_layers_used[lexeme] = layers_to_extract
        logging.info(f"Processing '{lexeme}' ({len(lex_usages)} usages) with layers {layers_to_extract}")

        representations = {
            "lexeme": lexeme,
            "hidden_states": {l: [] for l in layers_to_extract},
            "metadata": []
        }

        with torch.no_grad():
            for i in tqdm(range(0, len(lex_usages), batch_size), desc=f"Extracting {lexeme}"):
                batch = lex_usages[i:i + batch_size]
                sentences = [u.sentence for u in batch]

                encoded = tokenizer(
                    sentences,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)

                outputs = model(**encoded)

                for j, usage in enumerate(batch):
                    try:
                        # Use usage.token for the search if span is missing
                        target_indices = find_token_indices_from_whitespace_span(
                            usage.sentence, 
                            usage.token, 
                            usage.span
                        )
                    except ValueError as e:
                        logging.warning(f"ID {usage.id}: {str(e)}")
                        continue

                    for layer in layers_to_extract:
                        # Map whitespace token indices to subword indices
                        # Note: This logic assumes 1:1 mapping or simple pooling. 
                        # To keep your existing logic intact, we slice the hidden states:
                        hidden = outputs.hidden_states[layer][j, target_indices, :]
                        pooled = pool_subword_representations(hidden, pooling)
                        representations["hidden_states"][layer].append(pooled)

                    representations["metadata"].append({
                        "id": usage.id,
                        "sentence": usage.sentence,
                        "lexeme": usage.lexeme,
                        "token": usage.token,
                        "span": usage.span,
                        "synset": usage.synset,
                        "timespan": usage.timespan,
                        "corpus": usage.corpus
                    })

        for layer in layers_to_extract:
            representations["hidden_states"][layer] = np.array(representations["hidden_states"][layer])

        if len(representations["metadata"]) > 0:
            out_path = output_dir / f"{lexeme}_representations.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(representations, f)
            logging.info(f"Saved {out_path}")

    # Save config
    config = {
        "model_name": model_name,
        "pooling": pooling,
        "batch_size": batch_size,
        "device": device,
        "lexeme_layers": lexeme_layers_used
    }
    with open(output_dir / "extraction_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Extract hidden representations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="bert-base-german-cased")
    parser.add_argument("--layers", type=int, nargs='+', help="List of layers")
    parser.add_argument("--layer-map", type=Path, help="JSON file with lexeme->best layer")
    parser.add_argument("--pooling", type=str, default="first", choices=["first","mean","last"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    setup_logging()

    layer_map = None
    if args.layer_map is not None:
        with open(args.layer_map, "r", encoding="utf-8") as f:
            layer_map = {lex: info["selected_layer"] for lex, info in json.load(f).items()}

    extract_representations(
        input_jsonl=args.input,
        output_dir=args.output,
        model_name=args.model,
        layers=args.layers,
        layer_map=layer_map,
        pooling=args.pooling,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()
