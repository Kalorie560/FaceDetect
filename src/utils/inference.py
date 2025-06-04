"""
Inference utilities for facial keypoints detection
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
from typing import Tuple, Union, Optional, List, Dict
from PIL import Image
import albumentations as A
from data.preprocessing import DataPreprocessor


class KeypointsPredictor:
    """
    Class for performing inference with trained facial keypoints detection models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        image_size: Tuple[int, int] = (96, 96)
    ):
        """
        Initialize the predictor.
        
        Args:
            model: PyTorch model instance
            model_path: Path to saved model weights
            device: Device to run inference on
            image_size: Input image size for the model
        """
        self.device = device
        self.image_size = image_size
        self.model = model.to(device)
        
        # Load model weights if path provided
        if model_path:
            self.load_model(model_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize preprocessing
        self.preprocessor = DataPreprocessor(image_size=image_size)
        self.transform = self.preprocessor.get_inference_transforms()
    
    def load_model(self, model_path: str):
        """
        Load model weights from checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess input image for inference.
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        
        # Apply transformations
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Ensure we have a tensor (convert from numpy array if needed)
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor)
        
        # Add batch dimension and ensure correct shape
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add channel and batch dims
        elif len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dim
        
        return image_tensor.to(self.device)
    
    def postprocess_keypoints(
        self, 
        keypoints: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Postprocess predicted keypoints.
        
        Args:
            keypoints: Predicted keypoints tensor (1, 30) or (30,)
            original_size: Original image size for scaling back
            
        Returns:
            Processed keypoints as numpy array
        """
        # Convert to numpy and remove batch dimension if present
        if keypoints.dim() == 2:
            keypoints = keypoints.squeeze(0)
        
        keypoints = keypoints.cpu().numpy()
        
        # Denormalize keypoints to model input size
        keypoints_denorm = DataPreprocessor.denormalize_keypoints(
            torch.tensor(keypoints).unsqueeze(0),
            self.image_size
        ).squeeze(0).numpy()
        
        # Scale to original image size if provided
        if original_size is not None:
            scale_x = original_size[1] / self.image_size[1]
            scale_y = original_size[0] / self.image_size[0]
            
            # Scale x coordinates (even indices)
            keypoints_denorm[::2] *= scale_x
            # Scale y coordinates (odd indices)
            keypoints_denorm[1::2] *= scale_y
        
        return keypoints_denorm
    
    def predict(
        self, 
        image: Union[np.ndarray, Image.Image],
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Predict facial keypoints for a single image.
        
        Args:
            image: Input image
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Predicted keypoints (and confidence if requested)
        """
        original_size = None
        if isinstance(image, np.ndarray):
            original_size = image.shape[:2]
        elif isinstance(image, Image.Image):
            original_size = (image.height, image.width)
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
            
            # Calculate confidence as negative mean squared prediction
            # (this is a simple heuristic - could be improved with proper uncertainty estimation)
            if return_confidence:
                confidence = 1.0 / (1.0 + torch.mean(predictions ** 2).item())
        
        # Postprocess keypoints
        keypoints = self.postprocess_keypoints(predictions, original_size)
        
        if return_confidence:
            return keypoints, confidence
        else:
            return keypoints
    
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 8
    ) -> List[np.ndarray]:
        """
        Predict facial keypoints for a batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            
        Returns:
            List of predicted keypoints for each image
        """
        all_predictions = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_tensors = []
            original_sizes = []
            
            # Preprocess batch
            for img in batch_images:
                if isinstance(img, np.ndarray):
                    original_sizes.append(img.shape[:2])
                elif isinstance(img, Image.Image):
                    original_sizes.append((img.height, img.width))
                
                img_tensor = self.preprocess_image(img)
                batch_tensors.append(img_tensor.squeeze(0))  # Remove batch dim for stacking
            
            # Stack into batch
            batch_tensor = torch.stack(batch_tensors)
            
            # Perform inference
            with torch.no_grad():
                batch_predictions = self.model(batch_tensor)
            
            # Postprocess each prediction
            for j, pred in enumerate(batch_predictions):
                keypoints = self.postprocess_keypoints(
                    pred.unsqueeze(0), 
                    original_sizes[j]
                )
                all_predictions.append(keypoints)
        
        return all_predictions
    
    def detect_face_region(
        self,
        image: np.ndarray,
        padding: float = 0.2
    ) -> Optional[np.ndarray]:
        """
        Detect and crop face region from image using OpenCV's face detector.
        
        Args:
            image: Input image
            padding: Padding around detected face (as fraction of face size)
            
        Returns:
            Cropped face image or None if no face detected
        """
        # Load OpenCV's face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Use the largest detected face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Add padding
        padding_x = int(w * padding)
        padding_y = int(h * padding)
        
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(image.shape[1], x + w + padding_x)
        y2 = min(image.shape[0], y + h + padding_y)
        
        # Crop face region
        if len(image.shape) == 3:
            face_crop = image[y1:y2, x1:x2]
        else:
            face_crop = image[y1:y2, x1:x2]
        
        return face_crop
    
    def predict_with_face_detection(
        self,
        image: Union[np.ndarray, Image.Image],
        auto_crop: bool = True
    ) -> Optional[np.ndarray]:
        """
        Predict keypoints with automatic face detection and cropping.
        
        Args:
            image: Input image
            auto_crop: Whether to automatically crop detected face
            
        Returns:
            Predicted keypoints or None if no face detected
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if auto_crop:
            # Detect and crop face
            face_crop = self.detect_face_region(image)
            if face_crop is None:
                print("No face detected in the image")
                return None
            
            # Predict on cropped face
            keypoints = self.predict(face_crop)
        else:
            # Predict on full image
            keypoints = self.predict(image)
        
        return keypoints


def load_predictor(
    model_class: type,
    model_path: str,
    device: str = 'cuda',
    **model_kwargs
) -> KeypointsPredictor:
    """
    Factory function to load a predictor with a specific model.
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to saved model weights
        device: Device to run on
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized KeypointsPredictor
    """
    # Create model instance
    model = model_class(**model_kwargs)
    
    # Create predictor
    predictor = KeypointsPredictor(
        model=model,
        model_path=model_path,
        device=device
    )
    
    return predictor


class SubmissionGenerator:
    """
    Class for generating submission files for the Kaggle Facial Keypoints Detection competition.
    """
    
    # Standard keypoint names in the required order
    KEYPOINT_NAMES = [
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
    
    def __init__(self, predictor: KeypointsPredictor):
        """
        Initialize the submission generator.
        
        Args:
            predictor: Trained KeypointsPredictor instance
        """
        self.predictor = predictor
    
    @classmethod
    def load_submission_format(cls, submission_format_file: str) -> pd.DataFrame:
        """
        Load the submission format file to understand required predictions.
        
        Args:
            submission_format_file: Path to submissionFileFormat.csv
            
        Returns:
            DataFrame with required submission format
        """
        try:
            submission_df = pd.read_csv(submission_format_file)
            required_columns = ['RowId', 'ImageId', 'FeatureName', 'Location']
            
            # Validate format
            if not all(col in submission_df.columns for col in required_columns):
                raise ValueError(f"Submission format file must contain columns: {required_columns}")
            
            return submission_df
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Submission format file not found: {submission_format_file}. "
                "Please download submissionFileFormat.csv from the competition data page."
            )
    
    def predict_test_images(
        self,
        test_csv_file: str,
        batch_size: int = 32
    ) -> Dict[int, np.ndarray]:
        """
        Predict keypoints for all test images.
        
        Args:
            test_csv_file: Path to test.csv file
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping ImageId to predicted keypoints
        """
        try:
            test_df = pd.read_csv(test_csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Test file not found: {test_csv_file}. "
                "Please ensure test.csv is in the correct location."
            )
        
        predictions = {}
        
        # Process images in batches
        for start_idx in range(0, len(test_df), batch_size):
            end_idx = min(start_idx + batch_size, len(test_df))
            batch_df = test_df.iloc[start_idx:end_idx]
            
            # Prepare batch images
            batch_images = []
            image_ids = []
            
            for _, row in batch_df.iterrows():
                # Parse image data (space-separated pixel values)
                image_data = np.array([int(pixel) for pixel in row['Image'].split()], dtype=np.uint8)
                image = image_data.reshape(96, 96)
                
                batch_images.append(image)
                # ImageId is typically the index + 1 (1-based indexing)
                image_ids.append(row.name + 1 if 'ImageId' not in row else row['ImageId'])
            
            # Predict keypoints for batch
            batch_predictions = self.predictor.predict_batch(batch_images, batch_size=len(batch_images))
            
            # Store predictions with ImageId
            for img_id, pred in zip(image_ids, batch_predictions):
                predictions[img_id] = pred
            
            print(f"Processed batch {start_idx//batch_size + 1}/{(len(test_df) + batch_size - 1)//batch_size}")
        
        return predictions
    
    def generate_submission_file(
        self,
        test_csv_file: str,
        submission_format_file: str,
        output_file: str = 'submission.csv',
        batch_size: int = 32
    ) -> str:
        """
        Generate submission file for the competition.
        
        Args:
            test_csv_file: Path to test.csv file
            submission_format_file: Path to submissionFileFormat.csv
            output_file: Output submission file path
            batch_size: Batch size for processing
            
        Returns:
            Path to generated submission file
        """
        print("Loading submission format...")
        submission_format = self.load_submission_format(submission_format_file)
        
        print("Predicting keypoints for test images...")
        predictions = self.predict_test_images(test_csv_file, batch_size)
        
        print("Generating submission file...")
        submission_rows = []
        
        # Process each row in the submission format
        for _, row in submission_format.iterrows():
            row_id = row['RowId']
            image_id = row['ImageId']
            feature_name = row['FeatureName']
            
            # Get prediction for this image
            if image_id in predictions:
                pred_keypoints = predictions[image_id]
                
                # Find the index of this feature in our standard order
                if feature_name in self.KEYPOINT_NAMES:
                    feature_idx = self.KEYPOINT_NAMES.index(feature_name)
                    predicted_location = pred_keypoints[feature_idx]
                else:
                    # If feature not found, use NaN (will need manual handling)
                    predicted_location = np.nan
                    print(f"Warning: Unknown feature name: {feature_name}")
            else:
                # If image not found, use NaN
                predicted_location = np.nan
                print(f"Warning: No prediction found for ImageId: {image_id}")
            
            submission_rows.append({
                'RowId': row_id,
                'ImageId': image_id,
                'FeatureName': feature_name,
                'Location': predicted_location
            })
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(submission_rows)
        
        # Save to file
        submission_df.to_csv(output_file, index=False)
        
        print(f"Submission file saved to: {output_file}")
        print(f"Total predictions: {len(submission_df)}")
        print(f"Missing predictions: {submission_df['Location'].isna().sum()}")
        
        return output_file
    
    def validate_submission(self, submission_file: str) -> Dict[str, any]:
        """
        Validate the generated submission file.
        
        Args:
            submission_file: Path to submission file
            
        Returns:
            Dictionary with validation results
        """
        submission_df = pd.read_csv(submission_file)
        
        validation_results = {
            'total_rows': len(submission_df),
            'missing_values': submission_df['Location'].isna().sum(),
            'unique_images': submission_df['ImageId'].nunique(),
            'unique_features': submission_df['FeatureName'].nunique(),
            'feature_names': sorted(submission_df['FeatureName'].unique()),
            'valid_predictions': len(submission_df) - submission_df['Location'].isna().sum(),
            'completion_rate': (len(submission_df) - submission_df['Location'].isna().sum()) / len(submission_df) * 100
        }
        
        # Check for required columns
        required_columns = ['RowId', 'ImageId', 'FeatureName', 'Location']
        missing_columns = [col for col in required_columns if col not in submission_df.columns]
        validation_results['missing_columns'] = missing_columns
        
        return validation_results


def create_submission_file(
    model_path: str,
    model_class: type,
    test_csv_file: str,
    submission_format_file: str,
    output_file: str = 'submission.csv',
    device: str = 'cuda',
    batch_size: int = 32,
    **model_kwargs
) -> str:
    """
    Convenience function to create submission file from model and data files.
    
    Args:
        model_path: Path to saved model weights
        model_class: Model class to instantiate
        test_csv_file: Path to test.csv file
        submission_format_file: Path to submissionFileFormat.csv
        output_file: Output submission file path
        device: Device to run inference on
        batch_size: Batch size for processing
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        Path to generated submission file
    """
    # Load predictor
    predictor = load_predictor(
        model_class=model_class,
        model_path=model_path,
        device=device,
        **model_kwargs
    )
    
    # Create submission generator
    submission_generator = SubmissionGenerator(predictor)
    
    # Generate submission file
    output_path = submission_generator.generate_submission_file(
        test_csv_file=test_csv_file,
        submission_format_file=submission_format_file,
        output_file=output_file,
        batch_size=batch_size
    )
    
    # Validate submission
    validation_results = submission_generator.validate_submission(output_path)
    
    print("\nSubmission Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    
    return output_path