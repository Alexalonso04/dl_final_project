# Based on https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py
import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import Iterator, Tuple, Literal, Optional
from dataclasses import dataclass

@dataclass
class DataConfig:
    # Data source configuration
    data_mode: Literal['cached', 'full', 'local'] = 'cached'  # Which dataset version to use
    cache_dir: str = "data/cache"  # For cached version
    full_data_dir: str = "data/edu_fineweb10B"  # For full version
    local_data_path: Optional[str] = None  # For local version
    
    # Dataset parameters
    num_train_chunks: int = 103  # Only used for cached version
    use_cached: bool = False  # Deprecated, use data_mode instead

class DatasetManager:
    """Manages dataset downloading and preparation"""
    
    def __init__(
        self,
        config: DataConfig,
        vocab_size: int = 50257
    ):
        self.config = config
        self.vocab_size = vocab_size
        
    def ensure_data_available(self) -> Path:
        """Ensure dataset is available and return path to data directory"""
        if self.config.data_mode == 'local' and self.config.local_data_path:
            path = Path(self.config.local_data_path)
            assert path.exists(), f"Local data path {path} does not exist"
            return path
            
        elif self.config.data_mode == 'cached':
            return self._ensure_cached_data()
            
        elif self.config.data_mode == 'full':
            return self._ensure_full_data()
            
        else:
            raise ValueError(f"Invalid data_mode: {self.config.data_mode}")
    
    def _ensure_cached_data(self) -> Path:
        """Download and prepare cached version of dataset"""
        cache_dir = Path(self.config.cache_dir)
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
            
        # Download validation chunk
        self._download_chunk("fineweb_val_000000.bin")
        
        # Download training chunks
        for i in range(1, self.config.num_train_chunks + 1):
            self._download_chunk(f"fineweb_train_{i:06d}.bin")
            
        return cache_dir
    
    def _ensure_full_data(self) -> Path:
        """Ensure full version of dataset is available"""
        data_dir = Path(self.config.full_data_dir)
        if not data_dir.exists() or not any(data_dir.glob("edufineweb_*.npy")) and self.config.data_mode != 'cached':
            logging.info("Full dataset not found. Please run prepare_dataset.py first")
            raise FileNotFoundError(
                f"Full dataset not found in {data_dir}. "
                "Run prepare_dataset.py to download and process the dataset."
            )
        return data_dir
    
    def _download_chunk(self, fname: str) -> Path:
        """Download a single chunk from HuggingFace"""
        local_path = Path(self.config.cache_dir) / fname
        if not local_path.exists():
            logging.info(f"Downloading {fname}")
            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=self.config.cache_dir
            )
        return local_path

class TokenDataLoader:
    """Data loader implementation matching nanoGPT exactly"""
    def __init__(
        self,
        data_path: Path,
        split: Literal['train', 'val'],
        batch_size: int,
        sequence_length: int,
        process_rank: int = 0,
        num_processes: int = 1,
        vocab_size: int = 50257
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.vocab_size = vocab_size
        
        # Find all data files
        if data_path.name == "cache":
            pattern = "fineweb_val_*.bin" if split == 'val' else "fineweb_train_*.bin"
        else:
            pattern = "edufineweb_val_*.npy" if split == 'val' else "edufineweb_train_*.npy"
            
        self.shards = sorted(data_path.glob(pattern))
        assert len(self.shards) > 0, f"No shards found in {data_path} for split {split}"
        
        # Reset position
        self.reset()
    
    def _load_shard(self, path: Path) -> np.ndarray:
        """Load a data shard, ensuring int32 type for compatibility"""
        if path.suffix == '.bin':
            with open(path, "rb") as f:
                header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
                assert header[0] == 20240520, "magic number mismatch"
                assert header[1] == 1, "unsupported version"
                ntok = header[2]
                data = np.frombuffer(f.read(), dtype=np.uint16)
                assert len(data) == ntok
        else:
            data = np.load(path)
            data = data.astype(np.int32)
            data = torch.tensor(data, dtype=torch.long)

        return data
    
    def reset(self):
        self.current_shard = 0
        self.tokens = self._load_shard(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.sequence_length * self.process_rank
        
    
    def advance(self):
        """Advance to next shard, ensuring proper process rank offset"""
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.current_position = self.process_rank * self.batch_size * self.sequence_length
        self.tokens = self._load_shard(self.shards[self.current_shard])
    
    def __iter__(self):
        return self
        
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch, exactly matching nanoGPT's implementation"""
        batch_size = self.batch_size * self.sequence_length * self.num_processes
        buf = self.tokens[self.current_position:self.current_position + self.batch_size * self.sequence_length + 1]
        x = torch.tensor(buf[:-1], dtype=torch.long).view(self.batch_size, -1)
        y = torch.tensor(buf[1:], dtype=torch.long).view(self.batch_size, -1)
        
        self.current_position += batch_size
        if self.current_position + batch_size >= len(self.tokens):
            self.advance()
            
        return x, y

def create_dataloader(
    data_config: DataConfig,
    split: Literal['train', 'val'],
    batch_size: int,
    sequence_length: int,
    process_rank: int = 0,
    num_processes: int = 1,
    vocab_size: int = 50257
) -> TokenDataLoader:
    manager = DatasetManager(data_config, vocab_size)
    data_path = manager.ensure_data_available()
    return TokenDataLoader(
        data_path=data_path,
        split=split,
        batch_size=batch_size,
        sequence_length=sequence_length,
        process_rank=process_rank,
        num_processes=num_processes,
        vocab_size=vocab_size
    )