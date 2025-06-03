"""
Main training script for facial keypoints detection
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import FacialKeypointsDataset
from data.preprocessing import DataPreprocessor
from models.cnn_model import create_model
from training.trainer import FacialKeypointsTrainer, create_optimizer, create_scheduler
from utils.visualization import plot_training_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train facial keypoints detection model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV file containing training data')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test split ratio')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='resnet18',
                        choices=['basic_cnn', 'deep_cnn', 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b2'],
                        help='Type of model to train')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights for backbone models')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate for regularization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau'],
                        help='Learning rate scheduler type')
    
    # Loss function arguments
    parser.add_argument('--loss_function', type=str, default='mse',
                        choices=['mse', 'l1', 'smooth_l1'],
                        help='Loss function to use')
    
    # Data processing arguments
    parser.add_argument('--image_size', type=int, nargs=2, default=[96, 96],
                        help='Input image size (height width)')
    parser.add_argument('--handle_missing', type=str, default='drop',
                        choices=['drop', 'interpolate', 'zero'],
                        help='How to handle missing keypoints')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Checkpoint arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save_frequency', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # ClearML arguments
    parser.add_argument('--clearml_config', type=str, default=None,
                        help='Path to ClearML configuration file')
    parser.add_argument('--project_name', type=str, default='facial_keypoints_detection',
                        help='ClearML project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='ClearML experiment name')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Set deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg):
    """Get the appropriate device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_loss_function(loss_type):
    """Create loss function."""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")


def load_clearml_config(config_path):
    """Load ClearML configuration from YAML file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('clearml', {})
    return None


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load ClearML config
    clearml_config = load_clearml_config(args.clearml_config)
    if clearml_config:
        clearml_config['project_name'] = args.project_name
        if args.experiment_name:
            clearml_config['task_name'] = args.experiment_name
        else:
            clearml_config['task_name'] = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("="*60)
    print("FACIAL KEYPOINTS DETECTION TRAINING")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("="*60)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(image_size=tuple(args.image_size))
    
    # Split data
    print("Splitting data...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    train_df, val_df, test_df = preprocessor.split_data(
        args.data_path,
        val_size=args.val_split,
        test_size=args.test_split,
        random_state=args.seed
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = FacialKeypointsDataset(
        csv_file=None,  # We'll pass the dataframe directly
        augmentation=preprocessor.get_train_transforms(),
        handle_missing=args.handle_missing,
        image_size=tuple(args.image_size)
    )
    train_dataset.data = train_df.reset_index(drop=True)
    
    val_dataset = FacialKeypointsDataset(
        csv_file=None,
        augmentation=preprocessor.get_val_transforms(),
        handle_missing=args.handle_missing,
        image_size=tuple(args.image_size)
    )
    val_dataset.data = val_df.reset_index(drop=True)
    
    test_dataset = FacialKeypointsDataset(
        csv_file=None,
        augmentation=preprocessor.get_val_transforms(),
        handle_missing=args.handle_missing,
        image_size=tuple(args.image_size)
    )
    test_dataset.data = test_df.reset_index(drop=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type=args.model_type,
        num_keypoints=30,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate
    )
    
    print(f"Model created: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = create_loss_function(args.loss_function)
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate
    )
    
    # Create scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='step',
            step_size=30,
            gamma=0.1
        )
    elif args.scheduler == 'cosine':
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='cosine',
            T_max=args.epochs
        )
    elif args.scheduler == 'plateau':
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='plateau',
            mode='min',
            patience=10,
            factor=0.5
        )
    
    # Create trainer
    trainer = FacialKeypointsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        clearml_config=clearml_config,
        save_dir=args.save_dir
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        num_epochs=args.epochs,
        resume_from=args.resume_from,
        save_frequency=args.save_frequency
    )
    
    # Plot and save training history
    print("Saving training plots...")
    trainer.plot_training_history(
        save_path=os.path.join(args.save_dir, 'training_history.png')
    )
    
    # Evaluate on test set
    if len(test_dataset) > 0:
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate_model(test_loader)
        
        print("\nTest Results:")
        print(f"Test Loss: {test_metrics['test_loss']:.6f}")
        print(f"MSE: {test_metrics['mse']:.6f}")
        print(f"MAE: {test_metrics['mae']:.6f}")
        
        # Save test results
        import json
        with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f} at epoch {trainer.best_epoch + 1}")
    print(f"Model checkpoints saved in: {args.save_dir}")


if __name__ == "__main__":
    main()