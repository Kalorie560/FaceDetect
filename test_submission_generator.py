#!/usr/bin/env python3
"""
Test script for the submission file generator.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from src.models.cnn_model import BasicCNN
    from src.utils.inference import SubmissionGenerator, KeypointsPredictor
    print("✓ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_mock_data():
    """Create mock data for testing"""
    print("Creating mock test data...")
    
    # Create mock test CSV data
    test_data = []
    for i in range(3):  # 3 test images
        # Create 96x96 random image data
        image_pixels = np.random.randint(0, 255, 96*96)
        image_string = ' '.join(map(str, image_pixels))
        test_data.append({'Image': image_string})
    
    test_df = pd.DataFrame(test_data)
    
    # Create mock submission format
    submission_data = []
    row_id = 1
    for image_id in range(1, 4):  # 3 images
        for feature_name in SubmissionGenerator.KEYPOINT_NAMES[:6]:  # First 6 features only
            submission_data.append({
                'RowId': row_id,
                'ImageId': image_id,
                'FeatureName': feature_name,
                'Location': np.nan  # Will be filled with predictions
            })
            row_id += 1
    
    submission_format_df = pd.DataFrame(submission_data)
    
    return test_df, submission_format_df


def test_submission_generator():
    """Test the submission generator functionality"""
    print("Testing submission generator...")
    
    try:
        # Create a simple model for testing
        model = BasicCNN(num_keypoints=30, dropout_rate=0.5)
        
        # Create predictor (without loading weights for this test)
        predictor = KeypointsPredictor(
            model=model,
            model_path=None,  # Don't load weights for test
            device='cpu',
            image_size=(96, 96)
        )
        
        # Create submission generator
        submission_generator = SubmissionGenerator(predictor)
        
        print("✓ Successfully created SubmissionGenerator")
        
        # Test with mock data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Create mock data files
            test_df, submission_format_df = create_mock_data()
            
            test_csv_path = temp_dir / 'test.csv'
            submission_format_path = temp_dir / 'submissionFileFormat.csv'
            output_path = temp_dir / 'submission.csv'
            
            # Save mock data
            test_df.to_csv(test_csv_path, index=False)
            submission_format_df.to_csv(submission_format_path, index=False)
            
            print(f"✓ Created mock data files in {temp_dir}")
            
            # Test loading submission format
            loaded_format = SubmissionGenerator.load_submission_format(str(submission_format_path))
            print(f"✓ Loaded submission format: {len(loaded_format)} rows")
            
            # Test generating submission file
            try:
                output_file = submission_generator.generate_submission_file(
                    test_csv_file=str(test_csv_path),
                    submission_format_file=str(submission_format_path),
                    output_file=str(output_path),
                    batch_size=2
                )
                print(f"✓ Generated submission file: {output_file}")
                
                # Validate the submission
                validation_results = submission_generator.validate_submission(output_file)
                print("✓ Validation results:")
                for key, value in validation_results.items():
                    print(f"    {key}: {value}")
                
                # Check the output file
                result_df = pd.read_csv(output_file)
                print(f"✓ Output file has {len(result_df)} rows")
                print(f"✓ Columns: {list(result_df.columns)}")
                
                # Show sample predictions
                print("\nSample predictions:")
                print(result_df.head(10).to_string(index=False))
                
            except Exception as e:
                print(f"❌ Error generating submission: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        return False


def test_keypoint_names():
    """Test that keypoint names are correct"""
    print("\nTesting keypoint names...")
    
    expected_names = [
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
    
    if SubmissionGenerator.KEYPOINT_NAMES == expected_names:
        print(f"✓ Keypoint names are correct ({len(expected_names)} total)")
        return True
    else:
        print("❌ Keypoint names mismatch!")
        print("Expected:", expected_names[:5], "...")
        print("Got:", SubmissionGenerator.KEYPOINT_NAMES[:5], "...")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Submission File Generator")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test keypoint names
    all_tests_passed &= test_keypoint_names()
    
    # Test submission generator
    all_tests_passed &= test_submission_generator()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ All tests passed!")
        print("The submission generator is ready to use.")
    else:
        print("❌ Some tests failed!")
        print("Please check the errors above.")
    print("=" * 60)
    
    return all_tests_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)