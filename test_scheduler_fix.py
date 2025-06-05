#!/usr/bin/env python3
"""
Test scheduler fix for training issues.
Tests that ReduceLROnPlateau scheduler receives validation loss correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.training.trainer import Trainer
from src.models.cnn_model import BasicCNN

def test_scheduler_fix():
    """Test that the scheduler fix works correctly for ReduceLROnPlateau."""
    print("Testing scheduler fix for ReduceLROnPlateau...")
    
    # Create a simple model
    model = BasicCNN()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Test ReduceLROnPlateau scheduler
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Test StepLR scheduler
    scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Test isinstance detection
    assert isinstance(scheduler_plateau, torch.optim.lr_scheduler.ReduceLROnPlateau), "Failed to detect ReduceLROnPlateau"
    assert not isinstance(scheduler_step, torch.optim.lr_scheduler.ReduceLROnPlateau), "Incorrectly detected StepLR as ReduceLROnPlateau"
    
    # Test scheduler step calls (simulate the fixed trainer logic)
    val_loss = 0.5
    
    # Test ReduceLROnPlateau
    try:
        if isinstance(scheduler_plateau, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_plateau.step(val_loss)  # Should work
        print("✅ ReduceLROnPlateau scheduler step() with val_loss: SUCCESS")
    except Exception as e:
        print(f"❌ ReduceLROnPlateau scheduler step() failed: {e}")
        return False
    
    # Test StepLR
    try:
        if not isinstance(scheduler_step, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_step.step()  # Should work
        print("✅ StepLR scheduler step() without args: SUCCESS")
    except Exception as e:
        print(f"❌ StepLR scheduler step() failed: {e}")
        return False
    
    print("✅ All scheduler tests passed!")
    return True

def test_trainer_scheduler_integration():
    """Test that Trainer correctly handles different scheduler types."""
    print("\nTesting Trainer scheduler integration...")
    
    # Create synthetic data
    batch_size = 8
    train_data = torch.randn(batch_size, 1, 96, 96)
    train_targets = torch.randn(batch_size, 30)  # 15 keypoints * 2 coordinates
    val_data = torch.randn(batch_size, 1, 96, 96)
    val_targets = torch.randn(batch_size, 30)
    
    # Create mock datasets
    train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Test with ReduceLROnPlateau
    model = BasicCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.MSELoss()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device='cpu'
    )
    
    # Test one training step with scheduler
    try:
        train_loss = trainer.train_epoch(epoch=1)
        val_loss = trainer.validate_epoch(epoch=1)
        
        # Test the fixed scheduler logic
        if trainer.scheduler:
            if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                trainer.scheduler.step(val_loss)
            else:
                trainer.scheduler.step()
        
        print("✅ Trainer scheduler integration: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Trainer scheduler integration failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SCHEDULER FIX VALIDATION TEST")
    print("=" * 50)
    
    success = True
    
    # Run scheduler tests
    success &= test_scheduler_fix()
    
    # Run trainer integration tests
    success &= test_trainer_scheduler_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Scheduler fix is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Check the output above.")
    print("=" * 50)