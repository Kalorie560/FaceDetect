#!/usr/bin/env python3
"""
Test script to verify that config loading works correctly
"""

import sys
import os
sys.path.append('src')

from training.train import parse_args, load_config, update_args_from_config

def test_config_loading():
    """Test if config loading works correctly."""
    
    print("Testing config loading functionality...")
    
    # Create a mock args object by parsing with --data_path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/training_config.yaml')
    parser.add_argument('--data_path', default='test.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_type', default='resnet18')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--image_size', type=int, nargs=2, default=[96, 96])
    parser.add_argument('--handle_missing', default='drop')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--loss_function', default='mse')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--scheduler', default='step')
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--save_frequency', type=int, default=5)
    parser.add_argument('--resume_from', default=None)
    parser.add_argument('--project_name', default='facial_keypoints_detection')
    parser.add_argument('--experiment_name', default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args(['--data_path', 'test.csv'])
    
    print(f"Before loading config: epochs = {args.epochs}")
    
    # Load config
    config = load_config(args.config)
    if config:
        print(f"Config loaded successfully from: {args.config}")
        print(f"Config epochs value: {config.get('training', {}).get('epochs', 'NOT FOUND')}")
        
        # Update args from config
        args = update_args_from_config(args, config)
        print(f"After loading config: epochs = {args.epochs}")
        
        if args.epochs == 50:
            print("✅ SUCCESS: Config loading works! Epochs correctly set to 50 from config file.")
        else:
            print(f"❌ FAILED: Expected epochs=50, got epochs={args.epochs}")
    else:
        print(f"❌ FAILED: Could not load config from {args.config}")

if __name__ == "__main__":
    test_config_loading()