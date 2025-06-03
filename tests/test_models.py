"""
Unit tests for model architectures
"""

import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cnn_model import (
    BasicCNN, 
    ResNetKeypointDetector, 
    EfficientNetKeypointDetector,
    DeepCNN,
    create_model,
    get_model_info
)


class TestModels(unittest.TestCase):
    """Test cases for model architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.image_size = (1, 96, 96)  # (C, H, W)
        self.num_keypoints = 30
        self.input_tensor = torch.randn(self.batch_size, *self.image_size)
    
    def test_basic_cnn(self):
        """Test BasicCNN model."""
        model = BasicCNN(num_keypoints=self.num_keypoints)
        
        # Test forward pass
        output = model(self.input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_keypoints))
        
        # Check that gradients can flow
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_deep_cnn(self):
        """Test DeepCNN model."""
        model = DeepCNN(num_keypoints=self.num_keypoints)
        
        # Test forward pass
        output = model(self.input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_keypoints))
        
        # Check that model has more parameters than BasicCNN
        basic_model = BasicCNN(num_keypoints=self.num_keypoints)
        deep_params = sum(p.numel() for p in model.parameters())
        basic_params = sum(p.numel() for p in basic_model.parameters())
        self.assertGreater(deep_params, basic_params)
    
    def test_resnet_detector(self):
        """Test ResNet-based detector."""
        for backbone in ['resnet18', 'resnet34']:
            with self.subTest(backbone=backbone):
                model = ResNetKeypointDetector(
                    num_keypoints=self.num_keypoints,
                    backbone=backbone,
                    pretrained=False  # Don't download weights in tests
                )
                
                # Test forward pass
                output = model(self.input_tensor)
                
                # Check output shape
                self.assertEqual(output.shape, (self.batch_size, self.num_keypoints))
    
    def test_efficientnet_detector(self):
        """Test EfficientNet-based detector."""
        for backbone in ['efficientnet_b0']:  # Test one backbone to avoid long test times
            with self.subTest(backbone=backbone):
                model = EfficientNetKeypointDetector(
                    num_keypoints=self.num_keypoints,
                    backbone=backbone,
                    pretrained=False  # Don't download weights in tests
                )
                
                # Test forward pass
                output = model(self.input_tensor)
                
                # Check output shape
                self.assertEqual(output.shape, (self.batch_size, self.num_keypoints))
    
    def test_create_model_factory(self):
        """Test model factory function."""
        model_types = ['basic_cnn', 'deep_cnn', 'resnet18']
        
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model = create_model(
                    model_type=model_type,
                    num_keypoints=self.num_keypoints,
                    pretrained=False
                )
                
                # Test forward pass
                output = model(self.input_tensor)
                
                # Check output shape
                self.assertEqual(output.shape, (self.batch_size, self.num_keypoints))
    
    def test_model_info(self):
        """Test model info function."""
        model = BasicCNN(num_keypoints=self.num_keypoints)
        info = get_model_info(model)
        
        # Check that info contains expected keys
        expected_keys = ['total_parameters', 'trainable_parameters', 'model_class', 'model_size_mb']
        for key in expected_keys:
            self.assertIn(key, info)
        
        # Check that values are reasonable
        self.assertGreater(info['total_parameters'], 0)
        self.assertGreater(info['trainable_parameters'], 0)
        self.assertEqual(info['model_class'], 'BasicCNN')
        self.assertGreater(info['model_size_mb'], 0)
    
    def test_different_input_sizes(self):
        """Test models with different input sizes."""
        # Test with different batch sizes
        for batch_size in [1, 2, 8]:
            with self.subTest(batch_size=batch_size):
                input_tensor = torch.randn(batch_size, 1, 96, 96)
                model = BasicCNN(num_keypoints=self.num_keypoints)
                output = model(input_tensor)
                self.assertEqual(output.shape, (batch_size, self.num_keypoints))
    
    def test_model_eval_mode(self):
        """Test that models work in evaluation mode."""
        model = BasicCNN(num_keypoints=self.num_keypoints)
        model.eval()
        
        with torch.no_grad():
            output = model(self.input_tensor)
            self.assertEqual(output.shape, (self.batch_size, self.num_keypoints))
    
    def test_model_device_compatibility(self):
        """Test model device compatibility."""
        model = BasicCNN(num_keypoints=self.num_keypoints)
        
        # Test CPU
        model_cpu = model.to('cpu')
        input_cpu = self.input_tensor.to('cpu')
        output_cpu = model_cpu(input_cpu)
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            input_cuda = self.input_tensor.to('cuda')
            output_cuda = model_cuda(input_cuda)
            self.assertEqual(output_cuda.device.type, 'cuda')


if __name__ == '__main__':
    unittest.main()