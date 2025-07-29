from typing import Optional, List, Literal, Union
import torch
import numpy as np
import sys
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

def train_per_epoch(
    train_loader: DataLoader, 
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn: torch.nn.Module,
    device: str = "cpu",
    max_norm_grad: Optional[float] = None,
):
    """Train model for one epoch."""
    model.train()
    model.to(device)
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (past_state, past_control, future_control_prompt) in enumerate(train_loader):
        # Skip batch if batch size is too small
        if past_state.size(0) <= 1:
            continue
            
        # Move data to device
        past_state = past_state.to(device)
        past_control = past_control.to(device)
        target = future_control_prompt.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(past_state, past_control)
        loss = loss_fn(output, target)
        
        # Check for invalid loss
        if not torch.isfinite(loss):
            print(f"Warning: Invalid loss at batch {batch_idx}")
            continue
            
        # Backward pass
        loss.backward()
        
        # Gradient clipping if specified
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)
            
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    # Step scheduler if provided
    if scheduler:
        scheduler.step()
    
    # Return average loss
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

def valid_per_epoch(
    valid_loader: DataLoader, 
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str = "cpu",
):
    """Validate model for one epoch."""
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (past_state, past_control, future_control_prompt) in enumerate(valid_loader):
            # Skip batch if batch size is too small
            if past_state.size(0) <= 1:
                continue
                
            # Move data to device
            past_state = past_state.to(device)
            past_control = past_control.to(device)
            target = future_control_prompt.to(device)
            
            # Forward pass
            output = model(past_state, past_control)
            loss = loss_fn(output, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    # Return average loss
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

def valid_per_epoch_multi_step(
    valid_loader: DataLoader, 
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str = "cpu",
):
    """Multi-step validation (simplified to regular validation for now)."""
    # For now, just use regular validation
    # You can implement multi-step prediction logic here later
    return valid_per_epoch(valid_loader, model, optimizer, loss_fn, device)

def simple_evaluate(
    test_loader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: str = "cpu"
):
    """Simple evaluation function that returns basic metrics."""
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    num_samples = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (past_state, past_control, future_control_prompt) in enumerate(test_loader):
            if past_state.size(0) <= 1:
                continue
                
            # Move data to device
            past_state = past_state.to(device)
            past_control = past_control.to(device)
            target = future_control_prompt.to(device)
            
            # Forward pass
            output = model(past_state, past_control)
            loss = loss_fn(output, target)
            
            # Calculate MSE
            mse = torch.mean((output - target) ** 2)
            
            total_loss += loss.item()
            total_mse += mse.item()
            num_batches += 1
            
            # Store predictions and targets for R2 calculation
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
    
    # Calculate averages
    avg_loss = total_loss / max(num_batches, 1)
    avg_mse = total_mse / max(num_batches, 1)
    rmse = np.sqrt(avg_mse)
    
    # Calculate R2 score
    if all_predictions and all_targets:
        pred_concat = torch.cat(all_predictions, dim=0).numpy()
        target_concat = torch.cat(all_targets, dim=0).numpy()
        
        # Simple R2 calculation
        ss_res = np.sum((target_concat - pred_concat) ** 2)
        ss_tot = np.sum((target_concat - np.mean(target_concat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero
        
        # MAE calculation
        mae = np.mean(np.abs(target_concat - pred_concat))
    else:
        r2 = 0.0
        mae = 0.0
    
    return avg_loss, avg_mse, rmse, mae, r2

def train(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn: torch.nn.Module,
    device: str = "cpu",
    num_epoch: int = 64,
    verbose: Optional[int] = 8,
    save_best: str = "./weights/best.pt",
    save_last: str = "./weights/last.pt",
    max_norm_grad: Optional[float] = None,
    test_for_check_per_epoch: Optional[DataLoader] = None
):
    """
    Simplified training function that maintains all original parameters.
    """
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_best), exist_ok=True)
    os.makedirs(os.path.dirname(save_last), exist_ok=True)
    
    # Lists for storing metrics
    train_loss_list = []
    valid_loss_list = []
    test_epochs = []
    test_loss_list = []
    test_mse_list = []
    test_rmse_list = []
    test_mae_list = []
    test_r2_list = []
    
    best_epoch = 0
    best_loss = float('inf')
    
    print(f"Starting training for {num_epoch} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Main training loop
    for epoch in tqdm(range(num_epoch), desc="Training"):
        
        # Training phase
        train_loss = train_per_epoch(
            train_loader, model, optimizer, scheduler, loss_fn, device, max_norm_grad
        )
        
        # Validation phase
        valid_loss = valid_per_epoch(
            valid_loader, model, optimizer, loss_fn, device
        )
        
        # Store losses
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        # Verbose output and testing
        if verbose and (epoch % verbose == 0):
            print(f"Epoch {epoch+1:03d}/{num_epoch} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Valid Loss: {valid_loss:.6f}")
            
            # Test evaluation if provided
            if test_for_check_per_epoch:
                try:
                    test_loss, mse, rmse, mae, r2 = simple_evaluate(
                        test_for_check_per_epoch, model, loss_fn, device
                    )
                    
                    test_epochs.append(epoch + 1)
                    test_loss_list.append(test_loss)
                    test_mse_list.append(mse)
                    test_rmse_list.append(rmse)
                    test_mae_list.append(mae)
                    test_r2_list.append(r2)
                    
                    print(f"        Test | Loss: {test_loss:.6f} | "
                          f"MSE: {mse:.6f} | R²: {r2:.4f}")
                          
                except Exception as e:
                    print(f"        Test evaluation failed: {e}")
        
        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_best)
        
        # Save last model
        torch.save(model.state_dict(), save_last)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_loss:.6f} at epoch {best_epoch+1}")
    
    # Return training history
    history = {
        'train_loss': train_loss_list,
        'valid_loss': valid_loss_list,
        'test_epochs': test_epochs,
        'test_loss': test_loss_list,
        'test_mse': test_mse_list,
        'test_rmse': test_rmse_list,
        'test_mae': test_mae_list,
        'test_r2': test_r2_list
    }
    
    return history
