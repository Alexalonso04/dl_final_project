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
        self.start_time = datetime.now().strftime("%m_%d_%y_%H%M")
        self.model_type = self.config.model.model_type
        self.optim_type = self.config.optimizer.type
        self.use_rope =  'rope_' if self.config.model.use_rope else ''
        self.path_prefix = f'{self.model_type}_{self.optim_type}_{self.use_rope}{self.start_time}'
        
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

        print(f'Distributed setup complete - DDP: {self.ddp}, '
                     f'Rank: {self.ddp_rank}, World Size: {self.world_size}, '
                     f'Device: {self.device}, Is Master: {self.is_master_process}')
    
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
            log_filename = f"{self.path_prefix}_train.log"
            
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

            logging.info("\Optimizer Configuration:")
            for key, value in vars(self.config.optimizer).items():
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

        # Ensure self.optimizers is always a list
        if isinstance(optimizers, torch.optim.Optimizer):
            self.optimizers = [optimizers]
        else:
            # It's already a list
            self.optimizers = optimizers

        if self.config.optimizer.type == 'muon':
            # Create schedulers for each optimizer
            self.schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, self.get_lr_muon) for opt in optimizers]

        self.grad_accum_steps = (
            self.config.training.total_batch_size //
            (self.config.training.micro_batch_size *
             self.config.training.sequence_length *
             self.world_size)
        )
    
    def get_lr_muon(self, step) -> float:
        """Get learning rate multiplier based on step - follows warmup->constant->warmdown pattern"""
        assert self.step <= self.config.training.max_steps
        warmup_steps = self.config.training.warmup_steps
        max_steps = self.config.training.max_steps
        cooldown_steps = self.config.training.cooldown_steps

        # 1) linear warmup
        if self.step < warmup_steps: 
            return (self.step+1) / warmup_steps

        # 2) constant lr until warmdown
        elif self.step < (max_steps - cooldown_steps):
            return 1.0

        # 3) linear cooldown
        else:
            decay_ratio = (max_steps - self.step) / cooldown_steps
            return decay_ratio

    def get_lr_adamW(self) -> float:
        """Get learning rate multiplier based on step"""
        warmup_steps = self.config.training.warmup_steps
        max_lr = self.config.training.max_lr
        min_lr = self.config.training.min_lr
        max_steps = self.config.training.max_steps

        # 1) linear warmup
        if self.step < warmup_steps: 
            return max_lr * (self.step + 1) / warmup_steps

        # 2) constant lr until warmdown
        elif self.step > max_steps:
            return min_lr

        decay_ratio = (self.step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)


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
        
        if self.config.optimizer.type == 'muon':
            optimizer_state = [opt.state_dict() for opt in self.optimizers]
        else:
            optimizer_state = self.optimizers[0].state_dict()

        checkpoint = {
            'model': self.raw_model.state_dict(),
            'optimizer_states': optimizer_state,
            'config': self.config,
            'step': self.step,
            'val_loss': loss
        }

        path = Path(self.config.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        

        checkpoint_path = path / f'{self.path_prefix}_checkpoint_{self.step:06d}.pt'
        best_path = path / f'{self.path_prefix}_checkpoint_best.pt'

        torch.save(checkpoint, checkpoint_path)
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint with support for multiple optimizers"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load model state
        self.raw_model.load_state_dict(checkpoint['model'])
        self.config = checkpoint['config']

        if self.config.optimizer.type == 'adamw':
            self.optimizers[0].load_state_dict(checkpoint['optimizer_states'])
        else:
            for opt, state in zip(self.optimizers, checkpoint['optimizer_states']):
                opt.load_state_dict(state)

        self.step = checkpoint['step']
        self.val_loss = checkpoint.get('val_loss', float('inf'))
    
    def train_step(self):
        # Zero gradients
        if self.config.optimizer.type == 'muon':
            for opt in self.optimizers:
                opt.zero_grad()
        else:
            self.optimizers[0].zero_grad()

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

        if self.config.optimizer.type == 'muon':
            # Step optimizers and their schedulers
            for opt, scheduler in zip(self.optimizers, self.schedulers):
                opt.step()
                scheduler.step()
        else:
            lr = self.get_lr_adamW()
            for param_group in self.optimizers[0].param_groups:
                param_group['lr'] = lr
            self.optimizers[0].step()

        return loss_accum.item()
    
    def train(self):
        """Main training loop with support for multiple optimizers"""
        torch.manual_seed(1337 + self.ddp_rank)

        if self.is_master_process:
            logging.info(f"Starting training, total steps: {self.config.training.max_steps}")
            if self.config.optimizer.type == 'muon':
                logging.debug("Using multiple optimizers:")
                for i, opt in enumerate(self.optimizers):
                    logging.debug(f"Optimizer {i}: {opt.__class__.__name__}")
                    for group in opt.param_groups:
                        params_shape = [tuple(p.shape) for p in group['params']]
                        logging.debug(f"  Parameter shapes: {params_shape}")

        while self.step < self.config.training.max_steps:
            t0 = time.time()

            loss = self.train_step()

            if self.step % self.config.training.validate_every == 0:
                self.val_loss = self.evaluate()
                if self.is_master_process:
                    
                    logging.info(
                        f"Step {self.step}: train_loss={loss:.4f}, "
                        f"val_loss={self.val_loss:.4f}"
                    )

            if self.step % self.config.training.save_every == 0:
                self.save_checkpoint(self.val_loss)

            dt = time.time() - t0
            if self.is_master_process:
                
                logging.info(f"Step {self.step}: train_loss={loss:.4f}, "
                           f"time={dt*1000:.2f}ms/step")

            self.step += 1

        # Final evaluation and save
        self.val_loss = self.evaluate()
        self.save_checkpoint(self.val_loss)

        if self.ddp:
            destroy_process_group()