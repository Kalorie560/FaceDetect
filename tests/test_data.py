"""
Unit tests for data processing modules
"""

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import FacialKeypointsDataset
from data.preprocessing import DataPreprocessor


class TestDataset(unittest.TestCase):
    """Test cases for FacialKeypointsDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock data
        self.num_samples = 10
        self.num_keypoints = 30
        
        # Create mock CSV data
        data = {}
        data['Image'] = [' '.join([str(np.random.randint(0, 256)) for _ in range(96*96)]) 
                        for _ in range(self.num_samples)]
        
        # Add keypoint columns
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
        
        for col in keypoint_names:
            # Add some random keypoints with some missing values
            values = np.random.uniform(0, 96, self.num_samples)
            # Make some values NaN to simulate missing data
            if np.random.random() < 0.3:  # 30% chance of having missing values
                values[np.random.choice(self.num_samples, 2, replace=False)] = np.nan
            data[col] = values
        
        self.df = pd.DataFrame(data)
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = FacialKeypointsDataset(
            csv_file=self.temp_file.name,
            handle_missing='drop'
        )
        
        # Check that dataset was created
        self.assertIsInstance(dataset, FacialKeypointsDataset)
        self.assertGreater(len(dataset), 0)
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        dataset = FacialKeypointsDataset(
            csv_file=self.temp_file.name,
            handle_missing='drop'
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            
            # Check sample structure
            self.assertIn('image', sample)
            self.assertIn('keypoints', sample)
            
            # Check tensor types
            self.assertIsInstance(sample['image'], torch.Tensor)
            self.assertIsInstance(sample['keypoints'], torch.Tensor)
            
            # Check shapes
            self.assertEqual(sample['image'].shape, (1, 96, 96))  # (C, H, W)
            self.assertEqual(sample['keypoints'].shape, (30,))
    
    def test_dataset_missing_value_handling(self):
        """Test different missing value handling strategies."""
        strategies = ['drop', 'zero']
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                dataset = FacialKeypointsDataset(
                    csv_file=self.temp_file.name,
                    handle_missing=strategy
                )
                
                if len(dataset) > 0:
                    sample = dataset[0]
                    keypoints = sample['keypoints']
                    
                    # Check that no NaN values exist
                    self.assertFalse(torch.isnan(keypoints).any())
    
    def test_dataset_info(self):
        """Test dataset info method."""
        dataset = FacialKeypointsDataset(
            csv_file=self.temp_file.name,
            handle_missing='drop'
        )
        
        info = dataset.get_sample_info()
        
        # Check info keys
        expected_keys = ['num_samples', 'image_size', 'num_keypoints', 'missing_values', 'keypoint_names']
        for key in expected_keys:
            self.assertIn(key, info)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor(image_size=(96, 96))
    
    def test_transforms_creation(self):
        """Test that transforms are created correctly."""
        train_transforms = self.preprocessor.get_train_transforms()
        val_transforms = self.preprocessor.get_val_transforms()
        inference_transforms = self.preprocessor.get_inference_transforms()
        
        # Check that transforms are not None
        self.assertIsNotNone(train_transforms)
        self.assertIsNotNone(val_transforms)
        self.assertIsNotNone(inference_transforms)
    
    def test_image_preprocessing(self):
        """Test image preprocessing for inference."""
        # Create test image
        test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        # Preprocess image
        processed = DataPreprocessor.preprocess_image_for_inference(
            test_image, target_size=(96, 96)
        )
        
        # Check output shape and type
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (1, 1, 96, 96))  # (B, C, H, W)
    
    def test_keypoints_denormalization(self):
        """Test keypoints denormalization."""
        # Create normalized keypoints
        normalized_keypoints = torch.rand(1, 30)  # Random values in [0, 1]
        
        # Denormalize
        denormalized = DataPreprocessor.denormalize_keypoints(
            normalized_keypoints, image_size=(96, 96)
        )
        
        # Check that values are in expected range
        self.assertTrue((denormalized >= 0).all())
        self.assertTrue((denormalized <= 96).all())
        self.assertEqual(denormalized.shape, normalized_keypoints.shape)
    
    def test_split_data(self):
        """Test data splitting functionality."""
        # Create temporary CSV file with mock data
        num_samples = 100
        data = {
            'Image': [' '.join([str(np.random.randint(0, 256)) for _ in range(96*96)]) 
                     for _ in range(num_samples)]
        }
        
        # Add keypoint columns
        for i in range(30):
            data[f'keypoint_{i}'] = np.random.uniform(0, 96, num_samples)
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Split data
            train_df, val_df, test_df = DataPreprocessor.split_data(
                temp_file, val_size=0.2, test_size=0.1, random_state=42
            )
            
            # Check that splits have correct sizes
            total_samples = len(train_df) + len(val_df) + len(test_df)
            self.assertLessEqual(total_samples, num_samples)  # Some may be dropped due to missing values
            
            # Check that splits are disjoint
            train_indices = set(train_df.index)
            val_indices = set(val_df.index)
            test_indices = set(test_df.index)
            
            self.assertEqual(len(train_indices & val_indices), 0)
            self.assertEqual(len(train_indices & test_indices), 0)
            self.assertEqual(len(val_indices & test_indices), 0)
            
        finally:
            os.unlink(temp_file)
    
    def test_dataset_analysis(self):
        """Test dataset analysis functionality."""
        # Create temporary CSV file with mock data
        num_samples = 50
        data = {
            'Image': [' '.join([str(np.random.randint(0, 256)) for _ in range(96*96)]) 
                     for _ in range(num_samples)]
        }
        
        # Add keypoint columns with some missing values
        for i in range(30):
            values = np.random.uniform(0, 96, num_samples)
            # Add some NaN values
            nan_indices = np.random.choice(num_samples, 5, replace=False)
            values[nan_indices] = np.nan
            data[f'keypoint_{i}'] = values
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Analyze dataset
            stats = DataPreprocessor.analyze_dataset(temp_file)
            
            # Check that analysis contains expected keys
            expected_keys = [
                'total_samples', 'num_keypoints', 'keypoint_names',
                'missing_values_per_column', 'samples_with_all_keypoints',
                'samples_with_missing_data', 'missing_percentages', 'keypoint_statistics'
            ]
            
            for key in expected_keys:
                self.assertIn(key, stats)
            
            # Check values
            self.assertEqual(stats['total_samples'], num_samples)
            self.assertEqual(stats['num_keypoints'], 30)
            
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()