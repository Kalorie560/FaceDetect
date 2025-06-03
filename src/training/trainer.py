"""
Training utilities and trainer class for facial keypoints detection
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task, Logger
import yaml
from datetime import datetime


class FacialKeypointsTrainer:
    """
    Trainer class for facial keypoints detection models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        clearml_config: Optional[Dict[str, Any]] = None,
        save_dir: str = './checkpoints'
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            clearml_config: ClearML configuration
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize ClearML if config provided
        self.clearml_task = None
        self.clearml_logger = None
        if clearml_config:
            self._init_clearml(clearml_config)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def _init_clearml(self, config: Dict[str, Any]):
        """Initialize ClearML task and logger."""
        try:
            self.clearml_task = Task.init(
                project_name=config.get('project_name', 'facial_keypoints_detection'),
                task_name=config.get('task_name', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                tags=config.get('tags', ['facial_keypoints', 'cnn'])
            )
            self.clearml_logger = self.clearml_task.get_logger()
            
            # Log configuration
            self.clearml_task.connect_configuration(config)
            
            print("ClearML initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize ClearML: {e}")
            self.clearml_task = None
            self.clearml_logger = None
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            keypoints = batch['keypoints'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, keypoints)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{avg_loss:.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to ClearML
            if self.clearml_logger and batch_idx % 10 == 0:
                self.clearml_logger.report_scalar(
                    'train',
                    'batch_loss',
                    value=loss.item(),
                    iteration=epoch * num_batches + batch_idx
                )
        
        return total_loss / num_batches
    
    def validate_epoch(self, epoch: int) -> float:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            for batch in pbar:
                images = batch['image'].to(self.device)
                keypoints = batch['keypoints'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, keypoints)
                
                total_loss += loss.item()
                avg_loss = total_loss / len(self.val_loader)
                
                # Update progress bar
                pbar.set_postfix({'Val Loss': f'{avg_loss:.6f}'})
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {val_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Starting epoch number
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
        
        return start_epoch
    
    def train(
        self,
        num_epochs: int,
        resume_from: Optional[str] = None,
        save_frequency: int = 5
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
            save_frequency: How often to save checkpoints
            
        Returns:
            Training history
        """
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
        
        print(f"Starting training from epoch {start_epoch + 1}/{num_epochs}")
        
        for epoch in range(start_epoch, num_epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate epoch
            val_loss = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            
            # Log to ClearML
            if self.clearml_logger:
                self.clearml_logger.report_scalar('epoch', 'train_loss', value=train_loss, iteration=epoch)
                self.clearml_logger.report_scalar('epoch', 'val_loss', value=val_loss, iteration=epoch)
                self.clearml_logger.report_scalar('epoch', 'learning_rate', 
                                                 value=self.optimizer.param_groups[0]['lr'], iteration=epoch)
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            if is_best:
                print(f"  *** New best model! ***")
            print()
        
        print(f"Training completed! Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch + 1}")
        
        return self.history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.history['train_loss']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot learning rate
        axes[1].plot(epochs, self.history['learning_rate'], label='Learning Rate', marker='d', color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                images = batch['image'].to(self.device)
                keypoints = batch['keypoints'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, keypoints)
                
                total_loss += loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(keypoints.cpu().numpy())
        
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        mse = np.mean((all_predictions - all_targets) ** 2)
        mae = np.mean(np.abs(all_predictions - all_targets))
        
        # Calculate per-keypoint metrics
        per_keypoint_mse = np.mean((all_predictions - all_targets) ** 2, axis=0)
        per_keypoint_mae = np.mean(np.abs(all_predictions - all_targets), axis=0)
        
        metrics = {
            'test_loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'per_keypoint_mse': per_keypoint_mse.tolist(),
            'per_keypoint_mae': per_keypoint_mae.tolist()
        }
        
        return metrics


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    learning_rate: float = 0.001,
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer for training.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'step',
    **kwargs
) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")