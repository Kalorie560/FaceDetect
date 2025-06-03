#!/usr/bin/env python3
"""
Quick verification script for the test fixes
"""
import sys
import os
sys.path.append('src')

from models.cnn_model import BasicCNN, DeepCNN, create_model
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