from dataclasses import dataclass
from typing import Optional, Literal
import yaml

@dataclass
class DebugConfig:
    enabled: bool = False
    log_level: str = "INFO"
    log_tensors: bool = False
    cuda_launch_blocking: bool = False

@dataclass
class DataConfig:
    # Data source configuration
    data_mode: Literal['cached', 'full', 'local'] = 'cached'  # Which dataset version to use
    cache_dir: str = "data/cache"  # For cached version
    full_data_dir: str = "data/edu_fineweb10B"  # For full version
    local_data_path: Optional[str] = None  # For local version
    
    # Dataset parameters
    num_train_chunks: int = 103  # Only used for cached version

@dataclass
class TrainingConfig:
    # Batch sizes
    total_batch_size: int = 524288  # ~0.5M tokens
    micro_batch_size: int = 64
    sequence_length: int = 1024
    
    # Learning rate settings
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 192
    cooldown_steps: int = 192

    max_steps: int = 19073
    
    # Optimization settings
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    
    # Training loop settings
    validate_every: int = 250
    save_every: int = 5000
    eval_samples: int = 20
    
    # Hardware settings
    device: str = "cuda"
    mixed_precision: bool = True

@dataclass 
class ModelConfig:
    model_type: Literal['base', 'differential', 'selective']
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocabulary size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

    use_rope: bool = True

    
    # Special attention parameters
    diff_lambda_init: Optional[float] = None  # For differential attention
    sel_topk: Optional[int] = None  # For selective attention

@dataclass
class OptimizerConfig:
    type: Literal['adamw', 'muon'] = 'adamw'  # Default to AdamW
    # Muon specific settings
    momentum: float = 0.95
    nesterov: bool = True
    backend: Literal['svd', 'newtonschulz5'] = 'newtonschulz5'
    backend_steps: int = 5
    muon_warmup_steps: int = 0

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    debug: DebugConfig
    optimizer: OptimizerConfig 
    save_dir: str = "checkpoints"
    log_dir: str = "logs"

    @staticmethod
    def load_config(config_path: str) -> 'Config':
        """Load config from YAML file"""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])
        data_config = DataConfig(**config_dict.get("data", {}))
        debug_config = DebugConfig(**config_dict.get("debug", {}))
        optimizer_config = OptimizerConfig(**config_dict.get("optimizer", {})) 

        return Config(
            model=model_config,
            training=training_config,
            data=data_config,
            debug=debug_config,
            optimizer=optimizer_config, 
            save_dir=config_dict.get("save_dir", "checkpoints"),
            log_dir=config_dict.get("log_dir", "logs")
        )