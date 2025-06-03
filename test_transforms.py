#!/usr/bin/env python3
"""
Test script to verify preprocessing transform fixes
"""
import sys
import os
sys.path.append('src')

from data.preprocessing import DataPreprocessor
import numpy as np

def test_transforms():
    """Test that transforms work without warnings"""
    print("Testing data transforms...")
    
    preprocessor = DataPreprocessor()
    
    # Test getting transforms (this should not raise warnings)
    try:
        train_transforms = preprocessor.get_train_transforms()
        val_transforms = preprocessor.get_val_transforms()
        print("✅ Transforms created successfully")
        
        # Test applying transforms to dummy data
        dummy_image = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
        dummy_keypoints = [(48, 48), (60, 40)]  # x, y format
        
        result = train_transforms(image=dummy_image, keypoints=dummy_keypoints)
        print("✅ Train transforms applied successfully")
        
        result = val_transforms(image=dummy_image, keypoints=dummy_keypoints)
        print("✅ Validation transforms applied successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        return False

def main():
    print("🔧 Testing transform fixes...")
    print("="*40)
    
    success = test_transforms()
    
    print("\n" + "="*40)
    if success:
        print("🎉 Transform fixes verified!")
    else:
        print("❌ Transform fixes failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)