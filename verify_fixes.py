#!/usr/bin/env python3
"""
Quick verification script for the test fixes
"""
import sys
import os
import tempfile
import numpy as np
import pandas as pd
sys.path.append('src')

from models.cnn_model import BasicCNN, DeepCNN, create_model
from data.dataset import FacialKeypointsDataset
import torch

def test_pretrained_parameter():
    """Test that models accept pretrained parameter"""
    print("Testing pretrained parameter handling...")
    
    # Test BasicCNN with pretrained parameter
    try:
        model = create_model('basic_cnn', num_keypoints=30, pretrained=False)
        print("✅ BasicCNN accepts pretrained parameter")
    except Exception as e:
        print(f"❌ BasicCNN failed: {e}")
        return False
    
    # Test DeepCNN with pretrained parameter  
    try:
        model = create_model('deep_cnn', num_keypoints=30, pretrained=False)
        print("✅ DeepCNN accepts pretrained parameter")
    except Exception as e:
        print(f"❌ DeepCNN failed: {e}")
        return False
    
    return True

def test_parameter_count():
    """Test that DeepCNN has more parameters than BasicCNN"""
    print("\nTesting parameter counts...")
    
    basic_model = BasicCNN(num_keypoints=30)
    deep_model = DeepCNN(num_keypoints=30)
    
    basic_params = sum(p.numel() for p in basic_model.parameters())
    deep_params = sum(p.numel() for p in deep_model.parameters())
    
    print(f"BasicCNN parameters: {basic_params:,}")
    print(f"DeepCNN parameters: {deep_params:,}")
    
    if deep_params > basic_params:
        print("✅ DeepCNN has more parameters than BasicCNN")
        return True
    else:
        print("❌ DeepCNN has fewer parameters than BasicCNN")
        return False

def test_dataset_creation():
    """Test dataset creation with reduced missing values"""
    print("\nTesting dataset creation...")
    
    # Create mock data similar to the test
    num_samples = 10
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
    
    data = {}
    data['Image'] = [' '.join([str(np.random.randint(0, 256)) for _ in range(96*96)]) 
                    for _ in range(num_samples)]
    
    for col in keypoint_names:
        # Add some random keypoints with reduced missing values
        values = np.random.uniform(0, 96, num_samples)
        # Only 15% chance of having missing values, and only 1 NaN per column
        if np.random.random() < 0.15:
            values[np.random.choice(num_samples, 1, replace=False)] = np.nan
        data[col] = values
    
    df = pd.DataFrame(data)
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        dataset = FacialKeypointsDataset(
            csv_file=temp_file,
            handle_missing='drop'
        )
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Original data length: {len(df)}")
        print(f"Complete rows: {len(df.dropna())}")
        
        if len(dataset) > 0:
            print("✅ Dataset creation test PASSED")
            return True
        else:
            print("❌ Dataset creation test FAILED - empty dataset")
            return False
            
    except Exception as e:
        print(f"❌ Dataset creation test FAILED - {e}")
        return False
    finally:
        os.unlink(temp_file)

def test_forward_pass():
    """Test forward pass works for both models"""
    print("\nTesting forward pass...")
    
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 96, 96)
    
    # Test BasicCNN
    try:
        basic_model = BasicCNN(num_keypoints=30)
        output = basic_model(input_tensor)
        assert output.shape == (batch_size, 30)
        print("✅ BasicCNN forward pass works")
    except Exception as e:
        print(f"❌ BasicCNN forward pass failed: {e}")
        return False
    
    # Test DeepCNN
    try:
        deep_model = DeepCNN(num_keypoints=30)  
        output = deep_model(input_tensor)
        assert output.shape == (batch_size, 30)
        print("✅ DeepCNN forward pass works")
    except Exception as e:
        print(f"❌ DeepCNN forward pass failed: {e}")
        return False
    
    return True

def main():
    print("🔧 Verifying test fixes...")
    print("="*50)
    
    all_passed = True
    
    all_passed &= test_pretrained_parameter()
    all_passed &= test_parameter_count()
    all_passed &= test_dataset_creation()
    all_passed &= test_forward_pass()
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All fixes verified successfully!")
    else:
        print("❌ Some fixes failed verification")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)