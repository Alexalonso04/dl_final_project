import argparse
from pathlib import Path
from model.trainer import Trainer
from configs.config import Config

def main():
    parser = argparse.ArgumentParser(description='Train a transformer model')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = Config.load_config(args.config)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        trainer.load_checkpoint(str(checkpoint_path))
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()