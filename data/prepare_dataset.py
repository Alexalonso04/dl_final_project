import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Any

# Initialize tokenizer globally
enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc._special_tokens['<|endoftext|>']

def tokenize(doc: Dict[str, Any]) -> np.ndarray:
    """
    Tokenize a single document.
    
    Args:
        doc: Dictionary containing the document text under 'text' key
        
    Returns:
        numpy array of uint16 tokens
    """
    tokens = [EOT_TOKEN]  # Start with end of text token
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_shard(filename: str, tokens_np: np.ndarray) -> None:
    """Write a shard of tokens to disk"""
    np.save(filename, tokens_np)

def prepare_fineweb_dataset(output_dir: str = "data/edu_fineweb10B", 
                          remote_name: str = "sample-10BT",
                          shard_size: int = int(1e8)) -> Path:
    """
    Downloads and preprocesses the FineWeb-Edu dataset.
    
    Args:
        output_dir: Directory to save the processed shards
        remote_name: Name of the remote dataset to download
        shard_size: Number of tokens per shard (default: 100M)
        
    Returns:
        Path to the output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    logging.info(f"Downloading FineWeb-Edu dataset ({remote_name})...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    
    # Process documents with multiprocessing
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                # Add tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Write current shard
                split = "val" if shard_index == 0 else "train"
                filename = output_dir / f"edufineweb_{split}_{shard_index:06d}"
                remainder = shard_size - token_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_shard(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                
                # Start new shard with remaining tokens
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
        
        # Write final shard if needed
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = output_dir / f"edufineweb_{split}_{shard_index:06d}"
            write_shard(filename, all_tokens_np[:token_count])
            
    logging.info(f"Dataset processing complete. Shards saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_fineweb_dataset()