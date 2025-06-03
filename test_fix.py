#!/usr/bin/env python3
"""
Test script to verify the dataset loading fix
"""

import sys
import os
sys.path.append('src')

from data.dataset import FacialKeypointsDataset
from data.preprocessing import DataPreprocessor
import pandas as pd

def test_dataset_fix():
    print('Testing dataset loading...')
    
    # Load the test data
    df = pd.read_csv('test_training.csv')
    print(f'Data shape: {df.shape}')
    print(f'Columns: {df.columns.tolist()[:5]}...')
    
    # Create dataset and test
    dataset = FacialKeypointsDataset(csv_file=None, image_size=(96, 96))
    dataset.set_data(df.head(10))  # Test with first 10 rows
    
    print(f'Dataset length: {len(dataset)}')
    print(f'Keypoint columns: {len(dataset.keypoint_cols)}')
    
    if len(dataset.keypoint_cols) > 0:
        # Test one sample
        sample = dataset[0]
        print(f'Image shape: {sample["image"].shape}')
        print(f'Keypoints shape: {sample["keypoints"].shape}')
        print('✅ Dataset loading test passed!')
        return True
    else:
        print('❌ No keypoint columns found!')
        return False

if __name__ == "__main__":
    test_dataset_fix()