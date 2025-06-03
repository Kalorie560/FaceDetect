#!/usr/bin/env python3
"""
Submission file generation script for Kaggle Facial Keypoints Detection competition.

This script generates submission files in the required format using trained models.
"""

import os
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.cnn_model import create_model
from src.utils.inference import create_submission_file, SubmissionGenerator, load_predictor


def main():
    parser = argparse.ArgumentParser(
        description='Generate submission file for Kaggle Facial Keypoints Detection competition'
    )
    
    # Required arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to saved model weights (.pth file)'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default='test.csv',
        help='Path to test.csv file (default: test.csv)'
    )
    parser.add_argument(
        '--submission_format',
        type=str,
        default='submissionFileFormat.csv',
        help='Path to submissionFileFormat.csv file (default: submissionFileFormat.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Output submission file path (default: submission.csv)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model_type',
        type=str,
        default='resnet18',
        choices=['basic_cnn', 'deep_cnn', 'resnet18', 'resnet34', 'resnet50', 
                 'efficientnet_b0', 'efficientnet_b2'],
        help='Model architecture type (default: resnet18)'
    )
    parser.add_argument(
        '--num_keypoints',
        type=int,
        default=30,
        help='Number of output keypoints (default: 30)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: cuda)'
    )
    
    # Additional model parameters
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Use pretrained backbone (for ResNet/EfficientNet models)'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        help='Dropout rate (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.test_csv):
        print(f"Error: Test CSV file not found: {args.test_csv}")
        print("Please download test.csv from the Kaggle competition page.")
        sys.exit(1)
    
    if not os.path.exists(args.submission_format):
        print(f"Error: Submission format file not found: {args.submission_format}")
        print("Please download submissionFileFormat.csv from the Kaggle competition page.")
        sys.exit(1)
    
    print("=" * 60)
    print("Kaggle Facial Keypoints Detection - Submission Generator")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Submission format: {args.submission_format}")
    print(f"Output file: {args.output}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    try:
        # Prepare model arguments
        model_kwargs = {
            'num_keypoints': args.num_keypoints,
            'dropout_rate': args.dropout_rate
        }
        
        # Add pretrained parameter for ResNet/EfficientNet models
        if args.model_type.startswith(('resnet', 'efficientnet')):
            model_kwargs['pretrained'] = args.pretrained
        
        # Generate submission file
        output_path = create_submission_file(
            model_path=args.model_path,
            model_class=lambda **kwargs: create_model(args.model_type, **kwargs),
            test_csv_file=args.test_csv,
            submission_format_file=args.submission_format,
            output_file=args.output,
            device=args.device,
            batch_size=args.batch_size,
            **model_kwargs
        )
        
        print("=" * 60)
        print(f"✓ Submission file generated successfully: {output_path}")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Validate the submission file format")
        print("2. Check for any missing predictions")
        print("3. Upload to Kaggle competition")
        print("\nGood luck! 🚀")
        
    except Exception as e:
        print(f"\n❌ Error generating submission file: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()