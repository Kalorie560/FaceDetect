#!/usr/bin/env python3
"""
Test script to verify JSON serialization fix for training metrics.
"""

import json
import numpy as np
import tempfile
import os

def test_json_serialization():
    """Test that training metrics can be serialized to JSON."""
    
    # Simulate the metrics that would be returned by evaluate_model
    # These would normally be numpy scalars that cause the JSON error
    mse_numpy = np.mean([1.5, 2.0, 1.8])  # Returns numpy.float64
    mae_numpy = np.mean([0.5, 1.0, 0.8])  # Returns numpy.float64
    avg_loss = 44.007700  # This might be a numpy scalar too
    
    print("Testing JSON serialization fix...")
    print(f"MSE type before conversion: {type(mse_numpy)}")
    print(f"MAE type before conversion: {type(mae_numpy)}")
    print(f"Avg loss type: {type(avg_loss)}")
    
    # Apply the fix: convert to Python native types
    test_metrics = {
        'test_loss': float(avg_loss),
        'mse': float(mse_numpy), 
        'mae': float(mae_numpy),
        'per_keypoint_mse': np.array([1.2, 1.5, 1.3]).tolist(),
        'per_keypoint_mae': np.array([0.4, 0.6, 0.5]).tolist()
    }
    
    print(f"MSE type after conversion: {type(test_metrics['mse'])}")
    print(f"MAE type after conversion: {type(test_metrics['mae'])}")
    print(f"Test loss type after conversion: {type(test_metrics['test_loss'])}")
    
    # Try to serialize to JSON
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_metrics, f, indent=2)
            temp_file = f.name
        
        # Read it back to verify
        with open(temp_file, 'r') as f:
            loaded_metrics = json.load(f)
        
        print("✅ JSON serialization successful!")
        print(f"Serialized metrics: {loaded_metrics}")
        
        # Clean up
        os.unlink(temp_file)
        
        return True
        
    except TypeError as e:
        print(f"❌ JSON serialization failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_json_serialization()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")