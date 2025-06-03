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
        transformed = self.transform(image=image)\n        image_tensor = transformed['image']\n        \n        # Add batch dimension and ensure correct shape\n        if len(image_tensor.shape) == 2:\n            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add channel and batch dims\n        elif len(image_tensor.shape) == 3:\n            image_tensor = image_tensor.unsqueeze(0)  # Add batch dim\n        \n        return image_tensor.to(self.device)\n    \n    def postprocess_keypoints(\n        self, \n        keypoints: torch.Tensor,\n        original_size: Optional[Tuple[int, int]] = None\n    ) -> np.ndarray:\n        \"\"\"\n        Postprocess predicted keypoints.\n        \n        Args:\n            keypoints: Predicted keypoints tensor (1, 30) or (30,)\n            original_size: Original image size for scaling back\n            \n        Returns:\n            Processed keypoints as numpy array\n        \"\"\"\n        # Convert to numpy and remove batch dimension if present\n        if keypoints.dim() == 2:\n            keypoints = keypoints.squeeze(0)\n        \n        keypoints = keypoints.cpu().numpy()\n        \n        # Denormalize keypoints to model input size\n        keypoints_denorm = DataPreprocessor.denormalize_keypoints(\n            torch.tensor(keypoints).unsqueeze(0),\n            self.image_size\n        ).squeeze(0).numpy()\n        \n        # Scale to original image size if provided\n        if original_size is not None:\n            scale_x = original_size[1] / self.image_size[1]\n            scale_y = original_size[0] / self.image_size[0]\n            \n            # Scale x coordinates (even indices)\n            keypoints_denorm[::2] *= scale_x\n            # Scale y coordinates (odd indices)\n            keypoints_denorm[1::2] *= scale_y\n        \n        return keypoints_denorm\n    \n    def predict(\n        self, \n        image: Union[np.ndarray, Image.Image],\n        return_confidence: bool = False\n    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:\n        \"\"\"\n        Predict facial keypoints for a single image.\n        \n        Args:\n            image: Input image\n            return_confidence: Whether to return prediction confidence\n            \n        Returns:\n            Predicted keypoints (and confidence if requested)\n        \"\"\"\n        original_size = None\n        if isinstance(image, np.ndarray):\n            original_size = image.shape[:2]\n        elif isinstance(image, Image.Image):\n            original_size = (image.height, image.width)\n        \n        # Preprocess image\n        image_tensor = self.preprocess_image(image)\n        \n        # Perform inference\n        with torch.no_grad():\n            predictions = self.model(image_tensor)\n            \n            # Calculate confidence as negative mean squared prediction\n            # (this is a simple heuristic - could be improved with proper uncertainty estimation)\n            if return_confidence:\n                confidence = 1.0 / (1.0 + torch.mean(predictions ** 2).item())\n        \n        # Postprocess keypoints\n        keypoints = self.postprocess_keypoints(predictions, original_size)\n        \n        if return_confidence:\n            return keypoints, confidence\n        else:\n            return keypoints\n    \n    def predict_batch(\n        self,\n        images: List[Union[np.ndarray, Image.Image]],\n        batch_size: int = 8\n    ) -> List[np.ndarray]:\n        \"\"\"\n        Predict facial keypoints for a batch of images.\n        \n        Args:\n            images: List of input images\n            batch_size: Batch size for processing\n            \n        Returns:\n            List of predicted keypoints for each image\n        \"\"\"\n        all_predictions = []\n        \n        for i in range(0, len(images), batch_size):\n            batch_images = images[i:i + batch_size]\n            batch_tensors = []\n            original_sizes = []\n            \n            # Preprocess batch\n            for img in batch_images:\n                if isinstance(img, np.ndarray):\n                    original_sizes.append(img.shape[:2])\n                elif isinstance(img, Image.Image):\n                    original_sizes.append((img.height, img.width))\n                \n                img_tensor = self.preprocess_image(img)\n                batch_tensors.append(img_tensor.squeeze(0))  # Remove batch dim for stacking\n            \n            # Stack into batch\n            batch_tensor = torch.stack(batch_tensors)\n            \n            # Perform inference\n            with torch.no_grad():\n                batch_predictions = self.model(batch_tensor)\n            \n            # Postprocess each prediction\n            for j, pred in enumerate(batch_predictions):\n                keypoints = self.postprocess_keypoints(\n                    pred.unsqueeze(0), \n                    original_sizes[j]\n                )\n                all_predictions.append(keypoints)\n        \n        return all_predictions\n    \n    def detect_face_region(\n        self,\n        image: np.ndarray,\n        padding: float = 0.2\n    ) -> Optional[np.ndarray]:\n        \"\"\"\n        Detect and crop face region from image using OpenCV's face detector.\n        \n        Args:\n            image: Input image\n            padding: Padding around detected face (as fraction of face size)\n            \n        Returns:\n            Cropped face image or None if no face detected\n        \"\"\"\n        # Load OpenCV's face detector\n        face_cascade = cv2.CascadeClassifier(\n            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'\n        )\n        \n        # Convert to grayscale if needed\n        if len(image.shape) == 3:\n            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)\n        else:\n            gray = image\n        \n        # Detect faces\n        faces = face_cascade.detectMultiScale(\n            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)\n        )\n        \n        if len(faces) == 0:\n            return None\n        \n        # Use the largest detected face\n        face = max(faces, key=lambda x: x[2] * x[3])\n        x, y, w, h = face\n        \n        # Add padding\n        padding_x = int(w * padding)\n        padding_y = int(h * padding)\n        \n        x1 = max(0, x - padding_x)\n        y1 = max(0, y - padding_y)\n        x2 = min(image.shape[1], x + w + padding_x)\n        y2 = min(image.shape[0], y + h + padding_y)\n        \n        # Crop face region\n        if len(image.shape) == 3:\n            face_crop = image[y1:y2, x1:x2]\n        else:\n            face_crop = image[y1:y2, x1:x2]\n        \n        return face_crop\n    \n    def predict_with_face_detection(\n        self,\n        image: Union[np.ndarray, Image.Image],\n        auto_crop: bool = True\n    ) -> Optional[np.ndarray]:\n        \"\"\"\n        Predict keypoints with automatic face detection and cropping.\n        \n        Args:\n            image: Input image\n            auto_crop: Whether to automatically crop detected face\n            \n        Returns:\n            Predicted keypoints or None if no face detected\n        \"\"\"\n        # Convert PIL to numpy if needed\n        if isinstance(image, Image.Image):\n            image = np.array(image)\n        \n        if auto_crop:\n            # Detect and crop face\n            face_crop = self.detect_face_region(image)\n            if face_crop is None:\n                print(\"No face detected in the image\")\n                return None\n            \n            # Predict on cropped face\n            keypoints = self.predict(face_crop)\n        else:\n            # Predict on full image\n            keypoints = self.predict(image)\n        \n        return keypoints\n\n\ndef load_predictor(\n    model_class: type,\n    model_path: str,\n    device: str = 'cuda',\n    **model_kwargs\n) -> KeypointsPredictor:\n    \"\"\"\n    Factory function to load a predictor with a specific model.\n    \n    Args:\n        model_class: Model class to instantiate\n        model_path: Path to saved model weights\n        device: Device to run on\n        **model_kwargs: Additional arguments for model initialization\n        \n    Returns:\n        Initialized KeypointsPredictor\n    \"\"\"\n    # Create model instance\n    model = model_class(**model_kwargs)\n    \n    # Create predictor\n    predictor = KeypointsPredictor(\n        model=model,\n        model_path=model_path,\n        device=device\n    )\n    \n    return predictor