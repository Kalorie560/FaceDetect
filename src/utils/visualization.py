"""
Visualization utilities for facial keypoints detection
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional, Union
import torch
import seaborn as sns


# Define keypoint names for the 15 facial keypoints
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

# Define colors for different facial features
FEATURE_COLORS = {
    'eyes': (0, 255, 0),        # Green
    'eyebrows': (255, 0, 0),    # Red
    'nose': (0, 0, 255),        # Blue
    'mouth': (255, 255, 0)      # Yellow
}


def plot_keypoints_on_image(
    image: np.ndarray,
    keypoints: np.ndarray,
    title: str = "Facial Keypoints",
    denormalize: bool = True,
    image_size: Tuple[int, int] = (96, 96),
    show_labels: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot keypoints on an image.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        keypoints: Keypoints array of shape (30,) or (15, 2)
        title: Title for the plot
        denormalize: Whether to denormalize keypoints from [0,1] to image coordinates
        image_size: Image size for denormalization
        show_labels: Whether to show keypoint labels
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Ensure image is 2D
    if len(image.shape) == 3:
        if image.shape[0] == 1:  # (1, H, W)
            image = image.squeeze(0)
        elif image.shape[2] == 1:  # (H, W, 1)
            image = image.squeeze(2)
    
    # Display image
    ax.imshow(image, cmap='gray')
    
    # Reshape keypoints if needed
    if keypoints.shape[0] == 30:
        keypoints = keypoints.reshape(15, 2)
    
    # Denormalize keypoints if needed
    if denormalize:
        keypoints_plot = keypoints.copy()
        keypoints_plot[:, 0] *= image_size[1]  # x coordinates
        keypoints_plot[:, 1] *= image_size[0]  # y coordinates
    else:
        keypoints_plot = keypoints
    
    # Define feature groups
    feature_groups = {
        'left_eye': [0, 2, 3],  # left_eye_center, inner_corner, outer_corner
        'right_eye': [1, 4, 5],  # right_eye_center, inner_corner, outer_corner
        'left_eyebrow': [6, 7],  # left_eyebrow_inner_end, outer_end
        'right_eyebrow': [8, 9], # right_eyebrow_inner_end, outer_end
        'nose': [10],            # nose_tip
        'mouth': [11, 12, 13, 14] # mouth corners and center lips
    }
    
    # Plot keypoints with different colors for different features
    for feature, indices in feature_groups.items():
        if 'eye' in feature:
            color = FEATURE_COLORS['eyes']
        elif 'eyebrow' in feature:
            color = FEATURE_COLORS['eyebrows']
        elif 'nose' in feature:
            color = FEATURE_COLORS['nose']
        elif 'mouth' in feature:
            color = FEATURE_COLORS['mouth']
        else:
            color = (255, 255, 255)  # White for unknown
        
        # Convert color to matplotlib format
        color_mpl = tuple(c/255.0 for c in color)
        
        for idx in indices:
            if idx < len(keypoints_plot):
                x, y = keypoints_plot[idx]
                ax.plot(x, y, 'o', color=color_mpl, markersize=8)
                
                if show_labels:
                    ax.annotate(f'{idx}', (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8, 
                              color='white', weight='bold')
    
    ax.set_title(title, fontsize=14, weight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def plot_batch_predictions(
    images: torch.Tensor,
    true_keypoints: torch.Tensor,
    predicted_keypoints: torch.Tensor,
    num_samples: int = 8,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a batch of images with true and predicted keypoints.
    
    Args:
        images: Batch of images (B, C, H, W)
        true_keypoints: True keypoints (B, 30)
        predicted_keypoints: Predicted keypoints (B, 30)
        num_samples: Number of samples to plot
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    num_samples = min(num_samples, images.shape[0])
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Get image and keypoints
        image = images[i].cpu().numpy()
        if image.shape[0] == 1:
            image = image.squeeze(0)
        
        true_kp = true_keypoints[i].cpu().numpy().reshape(15, 2)
        pred_kp = predicted_keypoints[i].cpu().numpy().reshape(15, 2)
        
        # Display image
        ax.imshow(image, cmap='gray')
        
        # Plot true keypoints in green
        ax.scatter(true_kp[:, 0] * 96, true_kp[:, 1] * 96, 
                  c='green', s=30, alpha=0.8, label='True')
        
        # Plot predicted keypoints in red
        ax.scatter(pred_kp[:, 0] * 96, pred_kp[:, 1] * 96, 
                  c='red', s=30, alpha=0.8, label='Predicted')
        
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.axis('off')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Batch predictions plot saved to {save_path}")
    
    return fig


def create_keypoint_heatmap(
    keypoints: np.ndarray,
    image_size: Tuple[int, int] = (96, 96),
    sigma: float = 2.0
) -> np.ndarray:
    """
    Create heatmap representation of keypoints.
    
    Args:
        keypoints: Keypoints array (15, 2) or (30,)
        image_size: Size of the output heatmap
        sigma: Standard deviation for Gaussian heatmap
        
    Returns:
        Heatmap array of shape (15, H, W)
    """
    if keypoints.shape[0] == 30:
        keypoints = keypoints.reshape(15, 2)
    
    heatmaps = np.zeros((15, image_size[0], image_size[1]))
    
    for i, (x, y) in enumerate(keypoints):
        # Skip if keypoint is missing (negative or out of bounds)
        if x < 0 or y < 0 or x >= image_size[1] or y >= image_size[0]:
            continue
        
        # Create Gaussian heatmap
        xx, yy = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        heatmaps[i] = heatmap
    
    return heatmaps


def plot_training_metrics(
    history: dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training metrics from history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot losses
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot learning rate
    axes[0, 1].plot(epochs, history['learning_rate'], label='Learning Rate', 
                   marker='d', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 0].plot(epochs, loss_diff, label='Val - Train Loss', 
                   marker='v', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Difference')
    axes[1, 0].set_title('Overfitting Indicator')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot smoothed losses
    window_size = max(1, len(epochs) // 10)
    if len(epochs) >= window_size:
        train_smooth = np.convolve(history['train_loss'], 
                                 np.ones(window_size)/window_size, mode='valid')
        val_smooth = np.convolve(history['val_loss'], 
                               np.ones(window_size)/window_size, mode='valid')
        epochs_smooth = epochs[window_size-1:]
        
        axes[1, 1].plot(epochs_smooth, train_smooth, label='Train (Smoothed)', linewidth=2)
        axes[1, 1].plot(epochs_smooth, val_smooth, label='Val (Smoothed)', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Smoothed Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor smoothing', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Smoothed Losses')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to {save_path}")
    
    return fig


def visualize_data_distribution(
    keypoints_data: np.ndarray,
    keypoint_names: List[str] = KEYPOINT_NAMES,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the distribution of keypoints data.
    
    Args:
        keypoints_data: Array of keypoints data (N, 30)
        keypoint_names: Names of the keypoints
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, keypoint_names)):
        data = keypoints_data[:, i]
        # Remove NaN values for plotting
        data_clean = data[~np.isnan(data)]
        
        if len(data_clean) > 0:
            ax.hist(data_clean, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name}\n(n={len(data_clean)})', fontsize=10)
            ax.set_xlabel('Coordinate Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'{name}\n(n=0)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Data distribution plot saved to {save_path}")
    
    return fig