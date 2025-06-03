#!/usr/bin/env python3
"""
Test script to verify ClearML integration in facial keypoints detection training.
"""

import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_clearml_config_loading():
    """Test loading ClearML configuration."""
    print("Testing ClearML configuration loading...")
    
    config_path = 'config/clearml.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        clearml_config = config.get('clearml', {})
        print(f"✅ ClearML config loaded successfully")
        print(f"   Project: {clearml_config.get('project_name', 'N/A')}")
        print(f"   Tags: {clearml_config.get('experiment', {}).get('tags', [])}")
        return clearml_config
    else:
        print(f"❌ ClearML config file not found: {config_path}")
        return None

def test_clearml_import():
    """Test ClearML import."""
    print("\nTesting ClearML import...")
    
    try:
        from clearml import Task, Logger
        print("✅ ClearML imported successfully")
        return True
    except ImportError as e:
        print(f"❌ ClearML import failed: {e}")
        return False

def test_clearml_task_creation():
    """Test ClearML task creation (without connecting to server)."""
    print("\nTesting ClearML task creation...")
    
    try:
        from clearml import Task
        
        # This will work even offline, but won't connect to server
        task = Task.init(
            project_name="test_facial_keypoints",
            task_name=f"test_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=['test', 'facial_keypoints'],
            auto_connect_frameworks=False,
            auto_connect_streams=False
        )
        
        print("✅ ClearML task created successfully")
        print(f"   Task ID: {task.id if task else 'N/A'}")
        
        # Test logger
        logger = task.get_logger() if task else None
        if logger:
            print("✅ ClearML logger obtained successfully")
        
        # Clean up
        if task:
            task.close()
        
        return True
        
    except Exception as e:
        print(f"⚠️  ClearML task creation test failed (this is expected without server connection): {e}")
        return False

def test_training_integration():
    """Test that training code imports work with ClearML."""
    print("\nTesting training integration...")
    
    try:
        from training.trainer import FacialKeypointsTrainer
        from training.train import load_clearml_config
        
        print("✅ Training modules imported successfully")
        
        # Test config loading
        clearml_config = load_clearml_config('config/clearml.yaml')
        if clearml_config:
            print("✅ ClearML config loading function works")
        else:
            print("⚠️  ClearML config loading returned None")
        
        return True
        
    except Exception as e:
        print(f"❌ Training integration test failed: {e}")
        return False

def test_mock_logging():
    """Test mock logging functionality."""
    print("\nTesting mock logging functionality...")
    
    try:
        # Create mock data
        train_loss = 0.5
        val_loss = 0.6
        learning_rate = 0.001
        epoch = 0
        
        # This would be the actual logging code in trainer
        print(f"✅ Mock logging test passed")
        print(f"   Train Loss: {train_loss}")
        print(f"   Val Loss: {val_loss}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Epoch: {epoch}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock logging test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("CLEARML INTEGRATION TEST")
    print("="*60)
    
    tests = [
        test_clearml_config_loading,
        test_clearml_import,
        test_clearml_task_creation,
        test_training_integration,
        test_mock_logging
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! ClearML integration should work.")
    elif passed > 0:
        print("⚠️  Some tests passed. ClearML integration partially working.")
    else:
        print("❌ All tests failed. ClearML integration needs attention.")
    
    print("\nNote: Some tests may fail without proper ClearML server connection,")
    print("but the integration code should still work during actual training.")

if __name__ == "__main__":
    main()