"""
Dataset class for Kaggle Facial Keypoints Detection
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Optional, Callable, Tuple, Dict, Any


class FacialKeypointsDataset(Dataset):
    """
    Dataset class for facial keypoints detection.
    
    Handles loading and preprocessing of facial images and their corresponding
    keypoint coordinates from the Kaggle facial keypoints detection dataset.
    """
    
    def __init__(
        self,
        csv_file: str,
        transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        handle_missing: str = 'drop',
        image_size: Tuple[int, int] = (96, 96)
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to the CSV file containing image data and keypoints
            transform: Transform to apply to images
            augmentation: Augmentation to apply to both images and keypoints
            handle_missing: How to handle missing keypoints ('drop', 'interpolate', 'zero')
            image_size: Target size for images (height, width)
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.augmentation = augmentation
        self.image_size = image_size
        
        # Handle missing values
        if handle_missing == 'drop':
            self.data = self.data.dropna()
        elif handle_missing == 'interpolate':
            self.data = self._interpolate_missing()
        elif handle_missing == 'zero':
            self.data = self.data.fillna(0)
        
        # Extract keypoint column names
        self.keypoint_cols = [col for col in self.data.columns if col != 'Image']
        
        # Reset index after potential dropping
        self.data = self.data.reset_index(drop=True)
        
    def _interpolate_missing(self) -> pd.DataFrame:
        """Interpolate missing keypoint values using available data."""
        data_copy = self.data.copy()
        
        # Group keypoints by pairs (x, y coordinates)
        for i in range(0, len(self.keypoint_cols), 2):
            if i + 1 < len(self.keypoint_cols):
                x_col = self.keypoint_cols[i]
                y_col = self.keypoint_cols[i + 1]
                
                # Only interpolate if we have some valid data for this keypoint pair
                mask = data_copy[x_col].notna() & data_copy[y_col].notna()
                if mask.sum() > 0:
                    data_copy[x_col] = data_copy[x_col].interpolate()
                    data_copy[y_col] = data_copy[y_col].interpolate()
        
        return data_copy
    
    def _parse_image(self, image_str: str) -> np.ndarray:
        """
        Parse image string from CSV to numpy array.
        
        Args:
            image_str: Space-separated pixel values as string
            
        Returns:
            Image as numpy array
        """
        image = np.array(image_str.split(), dtype=np.float32)
        image = image.reshape(self.image_size)
        image = image / 255.0  # Normalize to [0, 1]
        return image
    
    def _normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints to [0, 1] range based on image size.
        
        Args:
            keypoints: Array of keypoint coordinates
            
        Returns:
            Normalized keypoints
        """
        keypoints = keypoints.copy()
        # Normalize x coordinates (even indices)
        keypoints[::2] = keypoints[::2] / self.image_size[1]
        # Normalize y coordinates (odd indices)
        keypoints[1::2] = keypoints[1::2] / self.image_size[0]
        return keypoints
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing 'image' and 'keypoints' tensors
        """
        row = self.data.iloc[idx]
        
        # Parse image
        image = self._parse_image(row['Image'])
        
        # Extract keypoints
        keypoints = row[self.keypoint_cols].values.astype(np.float32)
        
        # Normalize keypoints
        keypoints = self._normalize_keypoints(keypoints)
        
        # Apply augmentations (if any)
        if self.augmentation:
            # Reshape keypoints for augmentation (albumentations format)
            keypoints_2d = keypoints.reshape(-1, 2)
            
            # Apply augmentation
            augmented = self.augmentation(
                image=image,
                keypoints=keypoints_2d
            )
            image = augmented['image']
            keypoints = np.array(augmented['keypoints']).flatten()
        
        # Convert to tensor and add channel dimension for grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
        
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        
        return {
            'image': image,
            'keypoints': keypoints
        }
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        return {
            'num_samples': len(self.data),
            'image_size': self.image_size,
            'num_keypoints': len(self.keypoint_cols),
            'missing_values': self.data.isnull().sum().sum(),
            'keypoint_names': self.keypoint_cols
        }