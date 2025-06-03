"""
Data preprocessing utilities for facial keypoints detection
"""

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional, Dict, Any
import cv2
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Data preprocessing utilities for facial keypoints detection.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (96, 96)):
        """
        Initialize the preprocessor.
        
        Args:
            image_size: Target image size (height, width)
        """
        self.image_size = image_size
    
    def get_train_transforms(self) -> A.Compose:
        """
        Get training data augmentation pipeline.
        
        Returns:
            Albumentations compose object for training transforms
        """
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.Affine(
                    translate_percent=0.1,
                    scale=0.9,
                    rotate=10,
                    p=0.5
                ),
            ], p=0.8),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ], p=0.5),
            A.Normalize(mean=[0.485], std=[0.229]),  # ImageNet stats for grayscale
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            angle_in_degrees=True
        ))
    
    def get_val_transforms(self) -> A.Compose:
        """
        Get validation data transforms.
        
        Returns:
            Albumentations compose object for validation transforms
        """
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=[0.485], std=[0.229]),
        ], keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        ))
    
    def get_inference_transforms(self) -> A.Compose:
        """
        Get inference transforms.
        
        Returns:
            Albumentations compose object for inference
        """
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=[0.485], std=[0.229]),
        ])
    
    @staticmethod
    def split_data(
        csv_file: str,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            csv_file: Path to the CSV file
            val_size: Proportion of data for validation
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = pd.read_csv(csv_file)
        
        # Remove rows with too many missing values
        missing_threshold = len([col for col in df.columns if col != 'Image']) * 0.5
        df = df.dropna(thresh=len(df.columns) - missing_threshold)
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
        
        # Second split: separate train and validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=None
        )
        
        return train_df, val_df, test_df
    
    @staticmethod
    def analyze_dataset(csv_file: str) -> Dict[str, Any]:
        """
        Analyze the dataset and provide statistics.
        
        Args:
            csv_file: Path to the CSV file
            
        Returns:
            Dictionary with dataset statistics
        """
        df = pd.read_csv(csv_file)
        keypoint_cols = [col for col in df.columns if col != 'Image']
        
        stats = {
            'total_samples': len(df),
            'num_keypoints': len(keypoint_cols),
            'keypoint_names': keypoint_cols,
            'missing_values_per_column': df.isnull().sum().to_dict(),
            'samples_with_all_keypoints': len(df.dropna()),
            'samples_with_missing_data': len(df) - len(df.dropna()),
        }
        
        # Calculate missing percentage for each keypoint
        missing_percentages = {}
        for col in keypoint_cols:
            missing_percentages[col] = (df[col].isnull().sum() / len(df)) * 100
        
        stats['missing_percentages'] = missing_percentages
        
        # Keypoint statistics (for non-missing values)
        keypoint_stats = {}
        for col in keypoint_cols:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                keypoint_stats[col] = {
                    'mean': non_null_values.mean(),
                    'std': non_null_values.std(),
                    'min': non_null_values.min(),
                    'max': non_null_values.max(),
                    'count': len(non_null_values)
                }
        
        stats['keypoint_statistics'] = keypoint_stats
        
        return stats
    
    @staticmethod
    def preprocess_image_for_inference(
        image: np.ndarray,
        target_size: Tuple[int, int] = (96, 96)
    ) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Input image as numpy array
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize image
        image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean, std = 0.485, 0.229
        image = (image - mean) / std
        
        # Add batch and channel dimensions
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        
        return image
    
    @staticmethod
    def denormalize_keypoints(
        keypoints: torch.Tensor,
        image_size: Tuple[int, int] = (96, 96)
    ) -> torch.Tensor:
        """
        Denormalize keypoints from [0, 1] range to image coordinates.
        
        Args:
            keypoints: Normalized keypoints tensor
            image_size: Original image size (height, width)
            
        Returns:
            Denormalized keypoints tensor
        """
        keypoints = keypoints.clone()
        
        # Denormalize x coordinates (even indices)
        keypoints[..., ::2] *= image_size[1]
        # Denormalize y coordinates (odd indices)
        keypoints[..., 1::2] *= image_size[0]
        
        return keypoints
    
    @staticmethod
    def create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset=None,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create data loaders for training, validation, and testing.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset (optional)
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            
        Returns:
            Dictionary of data loaders
        """
        loaders = {}
        
        loaders['train'] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        loaders['val'] = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        if test_dataset is not None:
            loaders['test'] = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        
        return loaders