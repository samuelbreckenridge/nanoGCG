"""Utility to preprocess allowable strings for constrained GCG.

This script:
1. Loads a set of allowable strings from a text or JSON file
2. Tokenizes all strings
3. Computes mean-pooled embeddings for efficient similarity search
4. Saves tokenized strings, embeddings, and FAISS index to disk

Usage (text format):
    python -m nanogcg.preprocess_allowable_strings \
        --strings_file allowable_strings.txt \
        --output_dir ./preprocessed \
        --model_name mistralai/Mistral-7B-Instruct-v0.2

Usage (JSON format):
    python -m nanogcg.preprocess_allowable_strings \
        --strings_file allowable_strings.json \
        --input_format json \
        --output_dir ./preprocessed \
        --model_name mistralai/Mistral-7B-Instruct-v0.2

For JSON format, the file should contain a list where each item is converted
to a formatted JSON string representation.
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_strings_from_file(file_path: str, input_format: str = "text") -> List[str]:
    """Load strings from a file.

    Args:
        file_path: Path to the input file
        input_format: Format of the input file ("text" or "json")
            - "text": One string per line
            - "json": JSON list of strings

    Returns:
        List of strings
    """
    logger.info(f"Loading strings from {file_path} (format: {input_format})")

    if input_format == "text":
        with open(file_path, 'r') as f:
            strings = [line.strip() for line in f if line.strip()]
    elif input_format == "json":
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list, got {type(data)}")
            strings = [json.dumps(item, indent=2) for item in data]
    else:
        raise ValueError(f"Unknown input format: {input_format}. Use 'text' or 'json'.")

    logger.info(f"Loaded {len(strings)} strings")
    return strings


def tokenize_strings(strings: List[str], tokenizer) -> List[List[int]]:
    """Tokenize all strings and return token IDs."""
    logger.info("Tokenizing strings...")
    tokenized = []

    for string in tqdm(strings, desc="Tokenizing"):
        tokens = tokenizer(
            string,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0].tolist()
        tokenized.append(tokens)

    return tokenized


def compute_embeddings(
    tokenized: List[List[int]],
    model,
    embedding_layer,
    device: str = "cuda",
    batch_size: int = 32
) -> np.ndarray:
    """Compute mean-pooled embeddings for all tokenized strings.

    Args:
        tokenized: List of token ID lists
        model: The transformer model
        embedding_layer: The model's embedding layer
        device: Device to use for computation
        batch_size: Batch size for processing

    Returns:
        Array of shape (n_strings, embedding_dim) with mean-pooled embeddings
    """
    logger.info("Computing embeddings...")

    # Get embedding dimension from the model
    embed_dim = embedding_layer.weight.shape[1]
    n_strings = len(tokenized)

    # Pre-allocate the output array to avoid memory issues with large lists
    all_embeddings = np.zeros((n_strings, embed_dim), dtype=np.float32)

    # Convert to tensors and pad
    max_len = max(len(ids) for ids in tokenized)

    for i in tqdm(range(0, len(tokenized), batch_size), desc="Computing embeddings"):
        batch = tokenized[i:i + batch_size]

        # Pad sequences in batch
        padded_batch = []
        lengths = []
        for ids in batch:
            padded = ids + [0] * (max_len - len(ids))  # Pad with 0
            padded_batch.append(padded)
            lengths.append(len(ids))

        # Convert to tensor
        batch_tensor = torch.tensor(padded_batch, dtype=torch.long).to(device)

        with torch.no_grad():
            # Get embeddings from embedding layer
            embeds = embedding_layer(batch_tensor)  # (batch_size, max_len, embed_dim)

            # Mean pool over the actual (non-padded) tokens
            for j, length in enumerate(lengths):
                mean_embed = embeds[j, :length, :].mean(dim=0)  # (embed_dim,)
                # Write directly to pre-allocated array
                all_embeddings[i + j] = mean_embed.cpu().numpy()

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Preprocess allowable strings for constrained GCG")
    parser.add_argument(
        "--strings_file",
        type=str,
        required=True,
        help="Path to file with allowable strings"
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Format of input file: 'text' (one string per line) or 'json' (JSON list). "
             "For JSON, each item is converted to a formatted JSON string."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save preprocessed data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Model name or path to use for tokenization and embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding computation"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device
    )
    embedding_layer = model.get_input_embeddings()

    # Load strings
    strings = load_strings_from_file(args.strings_file, input_format=args.input_format)

    # Tokenize
    tokenized = tokenize_strings(strings, tokenizer)

    # Compute embeddings
    embeddings = compute_embeddings(
        tokenized,
        model,
        embedding_layer,
        device=args.device,
        batch_size=args.batch_size
    )

    # Save results
    logger.info(f"Saving preprocessed data to {output_dir}")

    # Save original strings
    logger.info("Saving strings.txt...")
    strings_path = output_dir / "strings.txt"
    with open(strings_path, 'w') as f:
        for s in strings:
            f.write(s + '\n')
    logger.info(f"Saved {len(strings)} strings to {strings_path}")

    # Save tokenized strings (as list of lists)
    logger.info("Saving tokenized.npy...")
    tokenized_path = output_dir / "tokenized.npy"
    np.save(tokenized_path, np.array(tokenized, dtype=object), allow_pickle=True)
    logger.info(f"Saved tokenized strings to {tokenized_path}")

    # Save embeddings
    logger.info("Saving embeddings.npy...")
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path}")

    # Build and save FAISS index
    try:
        import faiss
        logger.info("Building FAISS index...")

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-8)

        embed_dim = embeddings.shape[1]
        n_strings = len(strings)

        # Use IVF index for all dataset sizes
        # Adjust number of clusters based on dataset size
        if n_strings < 1000:
            nlist = max(1, n_strings // 100)  # Very small datasets
        elif n_strings < 10000:
            nlist = max(10, n_strings // 100)  # Small datasets
        elif n_strings < 100000:
            nlist = max(50, n_strings // 1000)  # Medium datasets
        else:
            nlist = 100  # Large datasets (1M strings)

        logger.info(f"Building IVF index with {nlist} clusters for {n_strings} strings")

        quantizer = faiss.IndexFlatIP(embed_dim)
        index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings_normalized)
        index.add(embeddings_normalized)

        # Set number of clusters to probe during search
        index.nprobe = min(10, max(1, nlist // 10))

        # Save FAISS index
        faiss_path = output_dir / "faiss.index"
        faiss.write_index(index, str(faiss_path))
        logger.info(f"FAISS IVF index saved to {faiss_path} (nprobe={index.nprobe})")

    except ImportError:
        logger.warning("FAISS not available. Skipping index creation. Install with: pip install faiss-cpu")

    # Save metadata
    metadata_path = output_dir / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"model_name: {args.model_name}\n")
        f.write(f"num_strings: {len(strings)}\n")
        f.write(f"embedding_dim: {embeddings.shape[1]}\n")
        f.write(f"strings_file: {args.strings_file}\n")

    logger.info("Preprocessing complete!")
    logger.info(f"  Strings: {len(strings)}")
    logger.info(f"  Embedding shape: {embeddings.shape}")
    logger.info(f"  Files saved to: {output_dir}")


if __name__ == "__main__":
    main()