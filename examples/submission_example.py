#!/usr/bin/env python3
"""
Example script demonstrating how to use the submission file generator.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir / 'src'))

from src.models.cnn_model import BasicCNN, ResNetKeypointDetector, create_model
from src.utils.inference import (
    KeypointsPredictor, SubmissionGenerator, 
    create_submission_file, load_predictor
)


def example_1_basic_usage():
    """
    Example 1: Basic usage with a simple function call
    """
    print("Example 1: Basic submission file generation")
    print("-" * 50)
    
    # Using the convenience function
    try:
        output_path = create_submission_file(
            model_path='path/to/your/model.pth',
            model_class=lambda **kwargs: create_model('resnet18', **kwargs),
            test_csv_file='test.csv',
            submission_format_file='submissionFileFormat.csv',
            output_file='submission_basic.csv',
            device='cpu',  # Use CPU for this example
            batch_size=16
        )
        print(f"Submission file created: {output_path}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("This is expected if you don't have the actual files.")


def example_2_advanced_usage():
    """
    Example 2: Advanced usage with custom configuration
    """
    print("\nExample 2: Advanced submission generation with custom model")
    print("-" * 50)
    
    try:
        # Create model manually
        model = ResNetKeypointDetector(
            num_keypoints=30,
            backbone='resnet18',
            pretrained=False,
            dropout_rate=0.3
        )
        
        # Create predictor
        predictor = KeypointsPredictor(
            model=model,
            model_path='path/to/your/model.pth',  # This would fail, but shows the pattern
            device='cpu',
            image_size=(96, 96)
        )
        
        # Create submission generator
        submission_generator = SubmissionGenerator(predictor)
        
        # Generate submission file
        output_path = submission_generator.generate_submission_file(
            test_csv_file='test.csv',
            submission_format_file='submissionFileFormat.csv',
            output_file='submission_advanced.csv',
            batch_size=32
        )
        
        # Validate the submission
        validation_results = submission_generator.validate_submission(output_path)
        print("Validation results:", validation_results)
        
    except Exception as e:
        print(f"Error (expected): {e}")
        print("This shows the advanced usage pattern.")


def example_3_model_types():
    """
    Example 3: Using different model types
    """
    print("\nExample 3: Different model architectures")
    print("-" * 50)
    
    model_configs = [
        ('basic_cnn', {'num_keypoints': 30, 'dropout_rate': 0.5}),
        ('resnet18', {'num_keypoints': 30, 'pretrained': True}),
        ('efficientnet_b0', {'num_keypoints': 30, 'pretrained': True}),
    ]
    
    for model_type, kwargs in model_configs:
        print(f"\nModel type: {model_type}")
        try:
            # This would create the submission file for each model type
            # output_path = create_submission_file(
            #     model_path=f'models/{model_type}_best.pth',
            #     model_class=lambda **k: create_model(model_type, **k),
            #     test_csv_file='test.csv',
            #     submission_format_file='submissionFileFormat.csv',
            #     output_file=f'submission_{model_type}.csv',
            #     **kwargs
            # )
            print(f"  Configuration: {kwargs}")
            print(f"  Would create: submission_{model_type}.csv")
        except Exception as e:
            print(f"  Error: {e}")


def show_expected_format():
    """
    Show the expected format for submission files
    """
    print("\nExpected submission format:")
    print("-" * 50)
    print("""
The submission file should have the following structure:

RowId,ImageId,FeatureName,Location
1,1,left_eye_center_x,37.5
2,1,left_eye_center_y,32.4
3,1,right_eye_center_x,59.6
4,1,right_eye_center_y,32.8
...

Key points:
- RowId: Sequential ID for each row
- ImageId: ID of the test image (1-1783)
- FeatureName: Name of the facial keypoint (e.g., 'left_eye_center_x')
- Location: Predicted coordinate value

The submissionFileFormat.csv file tells you exactly which 
predictions are required for each image.
""")


def show_usage_instructions():
    """
    Show usage instructions
    """
    print("\nUsage Instructions:")
    print("=" * 60)
    print("""
1. Download competition data:
   - test.csv (test images)
   - submissionFileFormat.csv (required predictions format)

2. Train your model and save the weights:
   - Use the training scripts in src/training/
   - Save model as .pth file

3. Generate submission file:
   
   Method A - Command line script:
   ```bash
   python generate_submission.py \\
       --model_path models/resnet18_best.pth \\
       --model_type resnet18 \\
       --test_csv test.csv \\
       --submission_format submissionFileFormat.csv \\
       --output submission.csv
   ```
   
   Method B - Python script:
   ```python
   from src.utils.inference import create_submission_file
   from src.models.cnn_model import create_model
   
   create_submission_file(
       model_path='models/resnet18_best.pth',
       model_class=lambda **kwargs: create_model('resnet18', **kwargs),
       test_csv_file='test.csv',
       submission_format_file='submissionFileFormat.csv',
       output_file='submission.csv'
   )
   ```

4. Validate and submit:
   - Check the generated CSV file
   - Upload to Kaggle competition
""")


if __name__ == '__main__':
    print("Facial Keypoints Detection - Submission Examples")
    print("=" * 60)
    
    show_usage_instructions()
    show_expected_format()
    
    # Run examples (these will show errors due to missing files, but demonstrate usage)
    example_1_basic_usage()
    example_2_advanced_usage()
    example_3_model_types()
    
    print("\n" + "=" * 60)
    print("Examples completed! Check the usage patterns above.")
    print("=" * 60)