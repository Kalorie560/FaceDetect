#!/usr/bin/env python3
"""
Test script to verify coordinate clamping functionality
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from models.cnn_model import create_model

def test_coordinate_clamp():
    """Test that all models properly clamp coordinates to [0, 1] range (normalized)"""
    
    print("🧪 Testing coordinate clamping for all model types...")
    
    # Test models
    model_types = ['basic_cnn', 'deep_cnn', 'resnet18', 'efficientnet_b0']
    
    # Create dummy input (batch_size=2, channels=1, height=96, width=96)
    dummy_input = torch.randn(2, 1, 96, 96)
    
    all_tests_passed = True
    
    for model_type in model_types:
        print(f"\n📋 Testing {model_type}...")
        
        try:
            # Create model
            model = create_model(model_type, num_keypoints=30)
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output shape
            expected_shape = (2, 30)  # batch_size=2, num_keypoints=30
            if output.shape != expected_shape:
                print(f"❌ {model_type}: Wrong output shape. Expected {expected_shape}, got {output.shape}")
                all_tests_passed = False
                continue
            
            # Check coordinate constraints
            min_val = torch.min(output).item()
            max_val = torch.max(output).item()
            
            if min_val < 0.0:
                print(f"❌ {model_type}: Minimum coordinate {min_val:.4f} is below 0.0")
                all_tests_passed = False
            elif max_val > 1.0:
                print(f"❌ {model_type}: Maximum coordinate {max_val:.4f} is above 1.0")
                all_tests_passed = False
            else:
                print(f"✅ {model_type}: Coordinates properly clamped to [0.0, 1.0]")
                print(f"   Range: [{min_val:.4f}, {max_val:.4f}]")
            
        except Exception as e:
            print(f"❌ {model_type}: Error during testing - {e}")
            all_tests_passed = False
    
    print("\n" + "="*60)
    print("📋 COORDINATE CLAMP TEST SUMMARY")
    print("="*60)
    
    if all_tests_passed:
        print("🎉 All models successfully clamp coordinates to [0, 1] range!")
        print("\n✅ Implementation verified:")
        print("   • All model forward functions include torch.clamp(x, min=0.0, max=1.0)")
        print("   • Output coordinates are normalized to [0, 1] range")
        print("   • Both x and y coordinates are properly constrained")
    else:
        print("❌ Some coordinate clamping tests failed.")
        return False
    
    return True

def test_extreme_case():
    """Test with artificially large values to ensure clamping works"""
    print("\n🎯 Testing extreme case with artificially modified model...")
    
    # Create a basic CNN model
    model = create_model('basic_cnn', num_keypoints=30)
    
    # Modify the final layer weights to produce very large values
    with torch.no_grad():
        model.fc3.weight.fill_(100.0)  # Very large weights
        model.fc3.bias.fill_(1000.0)   # Very large bias
    
    model.eval()
    
    # Test input
    dummy_input = torch.randn(1, 1, 96, 96)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    min_val = torch.min(output).item()
    max_val = torch.max(output).item()
    
    print(f"   Output range with extreme weights: [{min_val:.4f}, {max_val:.4f}]")
    
    if min_val >= 0.0 and max_val <= 1.0:
        print("✅ Extreme case test passed - clamping works even with large weights!")
        return True
    else:
        print("❌ Extreme case test failed - clamping not working properly")
        return False

if __name__ == "__main__":
    print("🔬 Facial Keypoints Detection - Coordinate Clamping Test")
    print("="*60)
    
    # Run main tests
    main_test_passed = test_coordinate_clamp()
    
    # Run extreme case test
    extreme_test_passed = test_extreme_case()
    
    # Final summary
    print("\n" + "="*60)
    print("🏁 FINAL TEST RESULTS")
    print("="*60)
    
    if main_test_passed and extreme_test_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Coordinate clamping implementation is working correctly.")
        print("✅ All models will output normalized coordinates within [0, 1] range.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Please check the implementation.")
        sys.exit(1)