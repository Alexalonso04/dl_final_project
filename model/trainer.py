import os
import time
import math
import torch
import inspect
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
from typing import Optional, Dict, Any
import logging
from pathlib import Path
from datetime import datetime

from configs.config import Config
from model.transformer import GPT
from data.data import create_dataloader

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        
        self.step = 0
        self.setup_distributed()
        self.setup_logging()
        self.setup_model()
        self.setup_dataloaders()
        self.setup_optimizer()
        self.debug = config.debug
        
        self.best_val_loss = float('inf')
    
    def setup_distributed(self):
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.is_master_process = self.ddp_rank == 0
        else:
            self.ddp_rank = 0
            self.world_size = 1
            self.device = self.config.training.device
            self.is_master_process = True
    
    def setup_logging(self):
        if self.is_master_process:
            # Create log directory if it doesn't exist
            os.makedirs(self.config.log_dir, exist_ok=True)
            
            # Generate timestamp for log filename
            current_time = datetime.now().strftime("%m_%d_%y_%H%M")
            log_filename = f"{current_time}_train.log"
            
            # Set up logging configuration
            log_level = getattr(logging, self.config.debug.log_level.upper())
            logging.basicConfig(
                format='%(asctime)s - %(levelname)s - %(message)s',
                level=log_level,
                handlers=[
                    logging.FileHandler(os.path.join(self.config.log_dir, log_filename)),
                    logging.StreamHandler()
                ]
            )
            
            # Log system information
            logging.info("=== System Information ===")
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logging.info(f"Number of GPUs: {gpu_count}")
                for i in range(gpu_count):
                    gpu_props = torch.cuda.get_device_properties(i)
                    logging.info(f"GPU {i}: {gpu_props.name}")
                    logging.info(f"  - Total memory: {gpu_props.total_memory / 1024**3:.2f} GB")
                    logging.info(f"  - Compute capability: {gpu_props.major}.{gpu_props.minor}")
            else:
                logging.info("No GPU available, running on CPU")
                
            # Log PyTorch version
            logging.info(f"PyTorch version: {torch.__version__}")
            
            # Log configuration
            logging.info("\n=== Configuration ===")

            logging.info("\nModel Configuration:")
            for key, value in vars(self.config.model).items():
                logging.info(f"  {key}: {value}")
                
            logging.info("\nTraining Configuration:")
            for key, value in vars(self.config.training).items():
                logging.info(f"  {key}: {value}")
                
            logging.info("\nData Configuration:")
            for key, value in vars(self.config.data).items():
                logging.info(f"  {key}: {value}")
                
            logging.info("\nDebug Configuration:")
            for key, value in vars(self.config.debug).items():
                logging.info(f"  {key}: {value}")
                
            # Handle debug mode configuration
            if self.config.debug.enabled:
                if self.config.debug.cuda_launch_blocking:
                    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                logging.info("\nDebug mode enabled")
                
            # Log memory info before training
            if torch.cuda.is_available():
                logging.info("\nInitial GPU Memory Usage:")
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**2
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**2
                    logging.info(f"GPU {i}:")
                    logging.info(f"  - Allocated: {mem_allocated:.2f} MB")
                    logging.info(f"  - Reserved: {mem_reserved:.2f} MB")
                    
            logging.info("\n=== Training Start ===")
    
    def setup_model(self):
        model = GPT(self.config)  

        # Set float32 matmul precision to high
        torch.set_float32_matmul_precision('high')

        model.to(self.device)

        if self.ddp:
            model = DDP(model, device_ids=[self.ddp_local_rank])

        self.model = model
        self.raw_model = model.module if self.ddp else model
    
    def setup_dataloaders(self):
        self.train_loader = create_dataloader(
            self.config.data,
            'train',
            self.config.training.micro_batch_size,
            self.config.training.sequence_length,
            self.ddp_rank,
            self.world_size,
            vocab_size=self.config.model.vocab_size
        )

        self.val_loader = create_dataloader(
            self.config.data,
            'val',
            self.config.training.micro_batch_size,
            self.config.training.sequence_length,
            self.ddp_rank,
            self.world_size,
            vocab_size=self.config.model.vocab_size
        )
    
    def setup_optimizer(self):
        # Configure base optimizers with their target learning rates
        optimizers = self.raw_model.configure_optimizers(
            self.config.training.weight_decay,
            self.config.training.max_lr,
            device_type="cuda" if self.device.startswith("cuda") else "cpu"
        )

        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # Create schedulers for each optimizer
        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt, lambda step: self.get_lr())
            for opt in optimizers
        ]

        self.optimizers = optimizers
    
        self.grad_accum_steps = (
            self.config.training.total_batch_size // 
            (self.config.training.micro_batch_size * 
             self.config.training.sequence_length * 
             self.world_size)
        )
    
    def get_lr(self) -> float:
        """Get learning rate multiplier based on step - follows warmup->constant->warmdown pattern"""
        assert self.step <= self.config.training.max_steps
        warmup_steps = self.config.training.warmup_steps
        warmdown_steps = 1300
        max_steps = self.config.training.max_steps

        # TODO: Update this to latest version of modded-nanoGPT. Someone found performance improvements since last pull. 

        # 1) linear warmup
        if self.step < warmup_steps: 
            return .0036 # return (it+1) / args.warmup_iters

        # 2) constant lr until warmdown
        elif self.step < (max_steps - warmdown_steps):
            return 1.0

        # 3) linear warmdown
        else:
            decay_ratio = (max_steps - self.step) / warmdown_steps
            return decay_ratio


    def evaluate(self) -> float:
        """Run evaluation and return validation loss"""
        self.model.eval()
        
        with torch.no_grad():
            total_loss = 0.0
            for _ in range(self.config.training.eval_samples):
                batch = next(iter(self.val_loader))
                x, y = [t.to(self.device) for t in batch]
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                
                total_loss += loss.item()
                
        avg_loss = total_loss / self.config.training.eval_samples
        if self.ddp:
            dist.all_reduce(torch.tensor(avg_loss, device=self.device), op=dist.ReduceOp.AVG)
            
        return avg_loss
    
    def save_checkpoint(self, loss: float):
        """Save model checkpoint with support for multiple optimizers"""
        if not self.is_master_process:
            return

        checkpoint = {
            'model': self.raw_model.state_dict(),
            'optimizer_states': [opt.state_dict() for opt in self.optimizers],
            'config': self.config,
            'step': self.step,
            'val_loss': loss
        }

        path = Path(self.config.save_dir)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = path / f'checkpoint_{self.step:06d}.pt'
        best_path = path / 'checkpoint_best.pt'

        torch.save(checkpoint, checkpoint_path)
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint with support for multiple optimizers"""
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.raw_model.load_state_dict(checkpoint['model'])

        # Handle both single-optimizer checkpoints and new muon ones
        if 'optimizer_states' in checkpoint:
            assert len(checkpoint['optimizer_states']) == len(self.optimizers), (
                f"Checkpoint has {len(checkpoint['optimizer_states'])} optimizers but "
                f"model was configured with {len(self.optimizers)} optimizers"
            )
            for opt, state in zip(self.optimizers, checkpoint['optimizer_states']):
                opt.load_state_dict(state)
        elif 'optimizer' in checkpoint:
            # Old format with single optimizer
            logging.warning("Loading legacy checkpoint format with single optimizer")
            if len(self.optimizers) == 1:
                self.optimizers[0].load_state_dict(checkpoint['optimizer'])
            else:
                logging.warning("Model configured with multiple optimizers but "
                              "loading from single-optimizer checkpoint. "
                              "Optimizer states will be fresh initialized.")

        self.step = checkpoint['step']
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    def train_step(self):
        # Zero gradients
        for opt in self.optimizers:
            opt.zero_grad()

        loss_accum = 0.0

        for micro_step in range(self.grad_accum_steps):
            x, y = self.train_loader.__next__()
            x, y = x.to(self.device), y.to(self.device)

            if self.ddp:
                self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)

            with torch.amp.autocast('cuda', enabled=self.config.training.mixed_precision, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)

            loss = loss / self.grad_accum_steps
            loss.backward()

            loss_accum += loss.detach()

        if self.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)

        # Step optimizers and their schedulers
        for opt, scheduler in zip(self.optimizers, self.schedulers):
            opt.step()
            scheduler.step()

        return loss_accum.item()
    
    def train(self):
        """Main training loop with support for multiple optimizers"""
        torch.manual_seed(1337 + self.ddp_rank)

        if self.is_master_process:
            logging.info(f"Starting training, total steps: {self.config.training.max_steps}")
            if len(self.optimizers) > 1:
                logging.info("Using multiple optimizers:")
                for i, opt in enumerate(self.optimizers):
                    logging.info(f"Optimizer {i}: {opt.__class__.__name__}")
                    for group in opt.param_groups:
                        params_shape = [tuple(p.shape) for p in group['params']]
                        logging.info(f"  Parameter shapes: {params_shape}")

        while self.step < self.config.training.max_steps:
            t0 = time.time()

            loss = self.train_step()

            if self.step % self.config.training.validate_every == 0:
                val_loss = self.evaluate()
                if self.is_master_process:
                    lrs = self.get_lr()
                    if isinstance(lrs, list):
                        lr_str = "lrs=[" + ",".join(f"{lr:.2e}" for lr in lrs) + "]"
                    else:
                        lr_str = f"lr={lrs:.2e}"
                    logging.info(
                        f"Step {self.step}: train_loss={loss:.4f}, "
                        f"val_loss={val_loss:.4f}, {lr_str}"
                    )

            if self.step % self.config.training.save_every == 0:
                self.save_checkpoint(val_loss)

            dt = time.time() - t0
            if self.is_master_process:
                lrs = self.get_lr()
                if isinstance(lrs, list):
                    lr_str = "lrs=[" + ",".join(f"{lr:.2e}" for lr in lrs) + "]"
                else:
                    lr_str = f"lr={lrs:.2e}"
                logging.info(f"Step {self.step}: train_loss={loss:.4f}, "
                           f"time={dt*1000:.2f}ms/step, {lr_str}")

            self.step += 1

        # Final evaluation and save
        val_loss = self.evaluate()
        self.save_checkpoint(val_loss)

        if self.ddp:
            destroy_process_group()

def train(config_path: str):
    """Main training function"""
    config = Config.load_config(config_path)
    trainer = Trainer(config)
    trainer.train()