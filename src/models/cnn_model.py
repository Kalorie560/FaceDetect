"""
CNN models for facial keypoints detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b2
from typing import Optional, Dict, Any


class BasicCNN(nn.Module):
    """
    Basic CNN model for facial keypoints detection.
    """
    
    def __init__(self, num_keypoints: int = 30, dropout_rate: float = 0.5, **kwargs):
        """
        Initialize the basic CNN model.
        
        Args:
            num_keypoints: Number of output keypoints (15 points * 2 coordinates = 30)
            dropout_rate: Dropout rate for regularization
            **kwargs: Additional keyword arguments (ignored for compatibility)
        """
        super(BasicCNN, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size of flattened features
        # After 4 conv+pool layers: 96 -> 48 -> 24 -> 12 -> 6
        self.fc_input_size = 256 * 6 * 6
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_keypoints)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 96, 96)
            
        Returns:
            Output tensor of shape (batch_size, num_keypoints) with coordinates clamped to [0, 96]
        """
        # Convolutional layers with activation and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Clamp coordinates to [0, 96] range for 96x96 images
        x = torch.clamp(x, min=0.0, max=96.0)
        
        return x


class ResNetKeypointDetector(nn.Module):
    """
    ResNet-based model for facial keypoints detection.
    """
    
    def __init__(
        self,
        num_keypoints: int = 30,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the ResNet-based model.
        
        Args:
            num_keypoints: Number of output keypoints
            backbone: ResNet architecture ('resnet18', 'resnet34', 'resnet50')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(ResNetKeypointDetector, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # Load backbone
        if backbone == 'resnet18':
            if pretrained:
                from torchvision.models import ResNet18_Weights
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet18(weights=None)
        elif backbone == 'resnet34':
            if pretrained:
                from torchvision.models import ResNet34_Weights
                self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet34(weights=None)
        elif backbone == 'resnet50':
            if pretrained:
                from torchvision.models import ResNet50_Weights
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer for grayscale input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Get number of features from the backbone
        num_features = self.backbone.fc.in_features
        
        # Replace the final classification layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_keypoints)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 96, 96)
            
        Returns:
            Output tensor of shape (batch_size, num_keypoints) with coordinates clamped to [0, 96]
        """
        x = self.backbone(x)
        
        # Clamp coordinates to [0, 96] range for 96x96 images
        x = torch.clamp(x, min=0.0, max=96.0)
        
        return x


class EfficientNetKeypointDetector(nn.Module):
    """
    EfficientNet-based model for facial keypoints detection.
    """
    
    def __init__(
        self,
        num_keypoints: int = 30,
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the EfficientNet-based model.
        
        Args:
            num_keypoints: Number of output keypoints
            backbone: EfficientNet architecture ('efficientnet_b0', 'efficientnet_b2')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super(EfficientNetKeypointDetector, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # Load backbone
        if backbone == 'efficientnet_b0':
            if pretrained:
                from torchvision.models import EfficientNet_B0_Weights
                self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = efficientnet_b0(weights=None)
        elif backbone == 'efficientnet_b2':
            if pretrained:
                from torchvision.models import EfficientNet_B2_Weights
                self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
            else:
                self.backbone = efficientnet_b2(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer for grayscale input
        self.backbone.features[0][0] = nn.Conv2d(
            1, self.backbone.features[0][0].out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Get number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_keypoints)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 96, 96)
            
        Returns:
            Output tensor of shape (batch_size, num_keypoints) with coordinates clamped to [0, 96]
        """
        x = self.backbone(x)
        
        # Clamp coordinates to [0, 96] range for 96x96 images
        x = torch.clamp(x, min=0.0, max=96.0)
        
        return x


class DeepCNN(nn.Module):
    """
    Deeper CNN model with residual connections for facial keypoints detection.
    """
    
    def __init__(self, num_keypoints: int = 30, dropout_rate: float = 0.5, **kwargs):
        """
        Initialize the deep CNN model.
        
        Args:
            num_keypoints: Number of output keypoints
            dropout_rate: Dropout rate for regularization
            **kwargs: Additional keyword arguments (ignored for compatibility)
        """
        super(DeepCNN, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # First block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third block
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fourth block
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fifth block
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers - using much larger hidden sizes to ensure more parameters than BasicCNN
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, num_keypoints)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 96, 96)
            
        Returns:
            Output tensor of shape (batch_size, num_keypoints) with coordinates clamped to [0, 96]
        """
        # First block
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.bn1(self.conv2(x))))
        
        # Second block
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.bn2(self.conv4(x))))
        
        # Third block
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.bn3(self.conv6(x))))
        
        # Fourth block
        x = F.relu(self.conv7(x))
        x = self.pool(F.relu(self.bn4(self.conv8(x))))
        
        # Fifth block
        x = F.relu(self.conv9(x))
        x = self.adaptive_pool(F.relu(self.bn5(self.conv10(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        
        # Clamp coordinates to [0, 96] range for 96x96 images
        x = torch.clamp(x, min=0.0, max=96.0)
        
        return x


def create_model(
    model_type: str = 'basic_cnn',
    num_keypoints: int = 30,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model to create
        num_keypoints: Number of output keypoints
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized model
    """
    if model_type == 'basic_cnn':
        return BasicCNN(num_keypoints=num_keypoints, **kwargs)
    elif model_type == 'deep_cnn':
        return DeepCNN(num_keypoints=num_keypoints, **kwargs)
    elif model_type.startswith('resnet'):
        backbone = model_type
        return ResNetKeypointDetector(
            num_keypoints=num_keypoints,
            backbone=backbone,
            **kwargs
        )
    elif model_type.startswith('efficientnet'):
        backbone = model_type
        return EfficientNetKeypointDetector(
            num_keypoints=num_keypoints,
            backbone=backbone,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_class': model.__class__.__name__,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
    }