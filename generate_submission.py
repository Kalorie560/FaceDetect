#!/usr/bin/env python3
"""
Simple submission file generation script for Kaggle Facial Keypoints Detection competition.

This script automatically detects trained models and generates submission files
without requiring command line arguments.
"""

import os
import sys
import glob
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.cnn_model import create_model
from src.utils.inference import SubmissionGenerator, KeypointsPredictor


def find_model_files():
    """Find available model files in the project."""
    model_patterns = [
        "*.pth",
        "models/*.pth",
        "**/*.pth"
    ]
    
    model_files = []
    for pattern in model_patterns:
        model_files.extend(glob.glob(pattern, recursive=True))
    
    return model_files


def detect_model_type(model_path):
    """Automatically detect model type from filename."""
    filename = os.path.basename(model_path).lower()
    
    # Model type detection based on filename
    if 'resnet18' in filename or 'resnet_18' in filename:
        return 'resnet18'
    elif 'resnet34' in filename or 'resnet_34' in filename:
        return 'resnet34'
    elif 'resnet50' in filename or 'resnet_50' in filename:
        return 'resnet50'
    elif 'efficientnet_b0' in filename or 'efficient_b0' in filename:
        return 'efficientnet_b0'
    elif 'efficientnet_b2' in filename or 'efficient_b2' in filename:
        return 'efficientnet_b2'
    elif 'deep_cnn' in filename or 'deepcnn' in filename:
        return 'deep_cnn'
    elif 'basic_cnn' in filename or 'basiccnn' in filename:
        return 'basic_cnn'
    else:
        # Default to resnet18 if cannot detect
        print(f"Warning: Could not detect model type from filename '{filename}'. Using resnet18 as default.")
        return 'resnet18'


def generate_default_submission_format():
    """Generate default submission format without requiring submissionFileFormat.csv."""
    
    # Standard keypoint names in the required order
    keypoint_names = [
        'left_eye_center_x', 'left_eye_center_y',
        'right_eye_center_x', 'right_eye_center_y',
        'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
        'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
        'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
        'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
        'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
        'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
        'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        'nose_tip_x', 'nose_tip_y',
        'mouth_left_corner_x', 'mouth_left_corner_y',
        'mouth_right_corner_x', 'mouth_right_corner_y',
        'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
    ]
    
    # Check if test.csv exists to determine number of test images
    if os.path.exists('test.csv'):
        test_df = pd.read_csv('test.csv')
        num_test_images = len(test_df)
    else:
        # Default to 1783 images (standard competition size)
        num_test_images = 1783
        print("Warning: test.csv not found. Using default 1783 test images.")
    
    # Generate submission format
    submission_rows = []
    row_id = 1
    
    for image_id in range(1, num_test_images + 1):
        for feature_name in keypoint_names:
            submission_rows.append({
                'RowId': row_id,
                'ImageId': image_id,
                'FeatureName': feature_name,
                'Location': '?'  # Placeholder that will be replaced with predictions
            })
            row_id += 1
    
    return pd.DataFrame(submission_rows)


def main():
    """Main function to generate submission file."""
    print("=" * 60)
    print("Kaggle Facial Keypoints Detection - Simple Submission Generator")
    print("=" * 60)
    
    # Check for required files
    if not os.path.exists('test.csv'):
        print("❌ Error: test.csv not found!")
        print("Please download test.csv from the Kaggle competition page.")
        sys.exit(1)
    
    # Find available model files
    model_files = find_model_files()
    
    if not model_files:
        print("❌ Error: No model files (.pth) found!")
        print("Please train a model first or place a trained model in the project directory.")
        sys.exit(1)
    
    # Use the first available model (or most recent)
    model_path = sorted(model_files, key=os.path.getmtime)[-1]  # Most recent model
    model_type = detect_model_type(model_path)
    
    print(f"📁 Found model: {model_path}")
    print(f"🤖 Detected model type: {model_type}")
    print(f"📊 Test data: test.csv")
    print(f"📄 Output: submission.csv")
    print("=" * 60)
    
    try:
        # Generate submission format
        print("🔧 Generating submission format...")
        submission_format_df = generate_default_submission_format()
        
        # Save temporary submission format file
        temp_format_file = 'temp_submission_format.csv'
        submission_format_df.to_csv(temp_format_file, index=False)
        
        # Prepare model arguments
        model_kwargs = {
            'num_keypoints': 30,
            'dropout_rate': 0.5
        }
        
        # Add pretrained parameter for ResNet/EfficientNet models
        if model_type.startswith(('resnet', 'efficientnet')):
            model_kwargs['pretrained'] = False  # Use False since we're loading trained weights
        
        # Create model and predictor
        print(f"🚀 Loading {model_type} model...")
        model = create_model(model_type, **model_kwargs)
        predictor = KeypointsPredictor(
            model=model,
            model_path=model_path,
            device='cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
        )
        
        # Create submission generator
        submission_generator = SubmissionGenerator(predictor)
        
        # Generate submission file
        print("🔮 Generating predictions...")
        output_path = submission_generator.generate_submission_file(
            test_csv_file='test.csv',
            submission_format_file=temp_format_file,
            output_file='submission.csv',
            batch_size=32
        )
        
        # Clean up temporary file
        if os.path.exists(temp_format_file):
            os.remove(temp_format_file)
        
        print("=" * 60)
        print(f"✅ Submission file generated successfully: {output_path}")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Check the submission.csv file")
        print("2. Upload to Kaggle competition")
        print("\n🚀 Good luck!")
        
    except Exception as e:
        print(f"\n❌ Error generating submission file: {e}")
        # Clean up temporary file if it exists
        if os.path.exists('temp_submission_format.csv'):
            os.remove('temp_submission_format.csv')
        sys.exit(1)


if __name__ == '__main__':
    main()