#!/usr/bin/env python3
"""
Kaggle Facial Keypoints Detection submission generator.

This script generates a Submission.csv file that follows the exact Kaggle competition format:
- Reads IdLookupTable.csv to get the required RowId and FeatureName mapping
- Generates predictions for each RowId based on ImageId and FeatureName
- Outputs a 2-column CSV file: RowId, Location

For now, uses dummy prediction logic (center coordinates at 48.0) as placeholder.
This can be replaced with actual model predictions later.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

def create_sample_idlookuptable():
    """
    Create a sample IdLookupTable.csv for testing purposes.
    This demonstrates the expected format and can be used when the actual file is not available.
    """
    print("Creating sample IdLookupTable.csv for demonstration...")
    
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
    
    # Create sample data for 10 test images
    sample_rows = []
    row_id = 1
    
    for image_id in range(1, 11):  # Sample 10 images
        for feature_name in keypoint_names:
            sample_rows.append({
                'RowId': row_id,
                'ImageId': image_id,
                'FeatureName': feature_name
            })
            row_id += 1
    
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv('IdLookupTable.csv', index=False)
    print(f"✅ Created sample IdLookupTable.csv with {len(sample_df)} rows")
    return sample_df

def dummy_prediction_logic(image_id, feature_name):
    """
    Dummy prediction logic that generates placeholder coordinates.
    
    Args:
        image_id: The ID of the image being predicted
        feature_name: The name of the facial keypoint feature
    
    Returns:
        float: Predicted coordinate value
    """
    # Use center coordinates (48.0) for all predictions as placeholder
    # This follows the user's suggestion for dummy values
    
    # Add some variation based on feature type for more realistic dummy data
    if '_x' in feature_name:
        # X coordinates: vary around center (48.0) ± 10
        base_x = 48.0
        variation = np.sin(hash(feature_name) % 100) * 10
        return round(base_x + variation, 1)
    elif '_y' in feature_name:
        # Y coordinates: vary around center (48.0) ± 10
        base_y = 48.0
        variation = np.cos(hash(feature_name) % 100) * 10
        return round(base_y + variation, 1)
    else:
        # Default to center
        return 48.0

def generate_submission_from_idlookuptable(idlookuptable_path='IdLookupTable.csv', output_path='Submission.csv'):
    """
    Generate Kaggle submission file based on IdLookupTable.csv format.
    
    Args:
        idlookuptable_path: Path to the IdLookupTable.csv file
        output_path: Path for the output Submission.csv file
    
    Returns:
        str: Path to the generated submission file
    """
    print(f"Reading IdLookupTable from: {idlookuptable_path}")
    
    # Read the IdLookupTable
    try:
        lookup_df = pd.read_csv(idlookuptable_path)
        print(f"✅ Loaded IdLookupTable with {len(lookup_df)} rows")
    except FileNotFoundError:
        print(f"❌ IdLookupTable.csv not found. Creating sample file...")
        lookup_df = create_sample_idlookuptable()
    
    # Validate required columns
    required_columns = ['RowId', 'ImageId', 'FeatureName']
    if not all(col in lookup_df.columns for col in required_columns):
        raise ValueError(f"IdLookupTable must contain columns: {required_columns}")
    
    print("🔮 Generating predictions using dummy logic...")
    
    # Generate predictions for each row
    submission_rows = []
    for _, row in lookup_df.iterrows():
        row_id = row['RowId']
        image_id = row['ImageId']
        feature_name = row['FeatureName']
        
        # Apply dummy prediction logic
        predicted_location = dummy_prediction_logic(image_id, feature_name)
        
        submission_rows.append({
            'RowId': row_id,
            'Location': predicted_location
        })
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_rows)
    
    # Ensure RowId order is preserved (sort by RowId just in case)
    submission_df = submission_df.sort_values('RowId')
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"✅ Submission file generated: {output_path}")
    print(f"📊 Total predictions: {len(submission_df)}")
    print(f"🎯 Format: {list(submission_df.columns)}")
    
    # Show sample of the output
    print("\n📋 Sample of generated submission:")
    print(submission_df.head(10).to_string(index=False))
    
    return output_path

def main():
    """Main function to generate Kaggle submission file."""
    print("=" * 60)
    print("Kaggle Facial Keypoints Detection - Submission Generator")
    print("Following IdLookupTable.csv format exactly")
    print("=" * 60)
    
    try:
        # Generate submission file
        output_file = generate_submission_from_idlookuptable()
        
        print("\n" + "=" * 60)
        print(f"✅ SUCCESS: Kaggle submission file generated!")
        print(f"📁 File: {output_file}")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Replace dummy prediction logic with actual model predictions")
        print("2. Verify the submission format matches Kaggle requirements")
        print("3. Upload Submission.csv to Kaggle competition")
        print("\n🚀 Ready for submission!")
        
    except Exception as e:
        print(f"\n❌ Error generating submission file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()