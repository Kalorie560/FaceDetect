"""
Inference utilities for facial keypoints detection
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Union, Optional, List
from PIL import Image
import albumentations as A
from ..data.preprocessing import DataPreprocessor


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