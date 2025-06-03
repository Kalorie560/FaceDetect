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
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])\n            \n            # Check if this is the best model\n            is_best = val_loss < self.best_val_loss\n            if is_best:\n                self.best_val_loss = val_loss\n                self.best_epoch = epoch\n            \n            # Log to ClearML\n            if self.clearml_logger:\n                self.clearml_logger.report_scalar('epoch', 'train_loss', value=train_loss, iteration=epoch)\n                self.clearml_logger.report_scalar('epoch', 'val_loss', value=val_loss, iteration=epoch)\n                self.clearml_logger.report_scalar('epoch', 'learning_rate', \n                                                 value=self.optimizer.param_groups[0]['lr'], iteration=epoch)\n            \n            # Save checkpoint\n            if (epoch + 1) % save_frequency == 0 or is_best:\n                self.save_checkpoint(epoch, val_loss, is_best)\n            \n            # Print epoch summary\n            print(f\"Epoch {epoch+1}/{num_epochs}:\")\n            print(f\"  Train Loss: {train_loss:.6f}\")\n            print(f\"  Val Loss: {val_loss:.6f}\")\n            print(f\"  LR: {self.optimizer.param_groups[0]['lr']:.2e}\")\n            if is_best:\n                print(f\"  *** New best model! ***\")\n            print()\n        \n        print(f\"Training completed! Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch + 1}\")\n        \n        return self.history\n    \n    def plot_training_history(self, save_path: Optional[str] = None):\n        \"\"\"\n        Plot training history.\n        \n        Args:\n            save_path: Path to save the plot\n        \"\"\"\n        if not self.history['train_loss']:\n            print(\"No training history to plot\")\n            return\n        \n        fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n        \n        # Plot losses\n        epochs = range(1, len(self.history['train_loss']) + 1)\n        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')\n        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')\n        axes[0].set_xlabel('Epoch')\n        axes[0].set_ylabel('Loss')\n        axes[0].set_title('Training and Validation Loss')\n        axes[0].legend()\n        axes[0].grid(True)\n        \n        # Plot learning rate\n        axes[1].plot(epochs, self.history['learning_rate'], label='Learning Rate', marker='d', color='orange')\n        axes[1].set_xlabel('Epoch')\n        axes[1].set_ylabel('Learning Rate')\n        axes[1].set_title('Learning Rate Schedule')\n        axes[1].set_yscale('log')\n        axes[1].legend()\n        axes[1].grid(True)\n        \n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n            print(f\"Training history plot saved to {save_path}\")\n        \n        plt.show()\n    \n    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:\n        \"\"\"\n        Evaluate model on test set.\n        \n        Args:\n            test_loader: Test data loader\n            \n        Returns:\n            Evaluation metrics\n        \"\"\"\n        self.model.eval()\n        total_loss = 0.0\n        all_predictions = []\n        all_targets = []\n        \n        with torch.no_grad():\n            for batch in tqdm(test_loader, desc='Evaluating'):\n                images = batch['image'].to(self.device)\n                keypoints = batch['keypoints'].to(self.device)\n                \n                outputs = self.model(images)\n                loss = self.criterion(outputs, keypoints)\n                \n                total_loss += loss.item()\n                all_predictions.append(outputs.cpu().numpy())\n                all_targets.append(keypoints.cpu().numpy())\n        \n        # Concatenate all predictions and targets\n        all_predictions = np.concatenate(all_predictions, axis=0)\n        all_targets = np.concatenate(all_targets, axis=0)\n        \n        # Calculate metrics\n        avg_loss = total_loss / len(test_loader)\n        mse = np.mean((all_predictions - all_targets) ** 2)\n        mae = np.mean(np.abs(all_predictions - all_targets))\n        \n        # Calculate per-keypoint metrics\n        per_keypoint_mse = np.mean((all_predictions - all_targets) ** 2, axis=0)\n        per_keypoint_mae = np.mean(np.abs(all_predictions - all_targets), axis=0)\n        \n        metrics = {\n            'test_loss': avg_loss,\n            'mse': mse,\n            'mae': mae,\n            'per_keypoint_mse': per_keypoint_mse.tolist(),\n            'per_keypoint_mae': per_keypoint_mae.tolist()\n        }\n        \n        return metrics\n\n\ndef create_optimizer(\n    model: nn.Module,\n    optimizer_type: str = 'adam',\n    learning_rate: float = 0.001,\n    **kwargs\n) -> optim.Optimizer:\n    \"\"\"\n    Create optimizer for training.\n    \n    Args:\n        model: PyTorch model\n        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')\n        learning_rate: Learning rate\n        **kwargs: Additional optimizer arguments\n        \n    Returns:\n        Optimizer instance\n    \"\"\"\n    if optimizer_type.lower() == 'adam':\n        return optim.Adam(model.parameters(), lr=learning_rate, **kwargs)\n    elif optimizer_type.lower() == 'sgd':\n        return optim.SGD(model.parameters(), lr=learning_rate, **kwargs)\n    elif optimizer_type.lower() == 'adamw':\n        return optim.AdamW(model.parameters(), lr=learning_rate, **kwargs)\n    else:\n        raise ValueError(f\"Unsupported optimizer type: {optimizer_type}\")\n\n\ndef create_scheduler(\n    optimizer: optim.Optimizer,\n    scheduler_type: str = 'step',\n    **kwargs\n) -> optim.lr_scheduler._LRScheduler:\n    \"\"\"\n    Create learning rate scheduler.\n    \n    Args:\n        optimizer: Optimizer instance\n        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau')\n        **kwargs: Additional scheduler arguments\n        \n    Returns:\n        Scheduler instance\n    \"\"\"\n    if scheduler_type.lower() == 'step':\n        return optim.lr_scheduler.StepLR(optimizer, **kwargs)\n    elif scheduler_type.lower() == 'cosine':\n        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)\n    elif scheduler_type.lower() == 'plateau':\n        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)\n    else:\n        raise ValueError(f\"Unsupported scheduler type: {scheduler_type}\")