from typing import Optional, Any
import torch
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import os

# Ensure the path is correct for your 'evaluate' module
sys.path.append(os.path.expanduser('~/transformer_model'))
from evaluate import evaluate

def train_per_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scheduler: Optional[_LRScheduler],
    max_norm_grad: float
):
    """Trains the model for one epoch."""
    model.train()
    model.to(device)
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Training", leave=False)
    for data_0D, data_ctrl, target in pbar:
        if data_0D.size(0) <= 1: continue

        data_0D, data_ctrl, target = data_0D.to(device), data_ctrl.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True) # More efficient
        output = model(data_0D, data_ctrl)
        loss = loss_fn(output, target)

        if not torch.isfinite(loss):
            print("WARNING: Loss is not finite. Skipping batch.")
            continue
        
        loss.backward()

        # Gradient Clipping: Prevents exploding gradients, crucial for Transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)
        
        optimizer.step()
        
        # Scheduler is stepped after each batch for fine-grained LR control
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
    return total_loss / len(train_loader)

def valid_per_epoch(
    model: torch.nn.Module,
    valid_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str
):
    """Validates the model for one epoch."""
    model.eval()
    model.to(device)
    total_loss = 0.0

    with torch.no_grad():
        for data_0D, data_ctrl, target in valid_loader:
            if data_0D.size(0) <= 1: continue
            data_0D, data_ctrl, target = data_0D.to(device), data_ctrl.to(device), target.to(device)
            output = model(data_0D, data_ctrl)
            loss = loss_fn(output, target)
            total_loss += loss.item()

    return total_loss / len(valid_loader)

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    num_epoch: int,
    save_best: str,
    save_last: str,
    # --- MODIFICATION: Key hyperparameters for effective training ---
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int = 0,
    max_norm_grad: float = 1.0,
    verbose: int = 1,
    test_for_check_per_epoch: Optional[DataLoader] = None
):
    """
    Orchestrates a robust training lifecycle with best practices for generalization.
    """
    history = {k: [] for k in ['train_loss', 'valid_loss', 'test_epochs', 'test_loss', 'test_r2']}
    best_loss = float('inf')

    # --- MODIFICATION: Optimizer setup with Weight Decay ---
    # AdamW is preferred as it decouples weight decay from the learning rate update,
    # which is a more effective form of regularization.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- MODIFICATION: Advanced Scheduler with Warmup ---
    # This setup is crucial for Transformers. It starts with a low LR, "warms up" to the
    # target LR, and then decays it. This stabilizes training.
    main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch * len(train_loader) - warmup_steps, eta_min=1e-6)
    
    scheduler: _LRScheduler
    if warmup_steps > 0:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    else:
        scheduler = main_scheduler

    # --- Main Training Loop ---
    for epoch in tqdm(range(num_epoch), desc="Overall Training Progress"):
        train_loss = train_per_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scheduler=scheduler,
            max_norm_grad=max_norm_grad
        )
        
        valid_loss = valid_per_epoch(
            model, valid_loader, loss_fn, device
        )

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        if verbose and (epoch + 1) % verbose == 0:
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

            if test_for_check_per_epoch:
                # Note: 'evaluate' should not take an optimizer. This is a corrected call.
                test_loss, _, _, _, r2 = evaluate(
                    test_loader=test_for_check_per_epoch, model=model,
                    loss_fn=loss_fn, device=device, is_print=False
                )
                history['test_epochs'].append(epoch + 1)
                history['test_loss'].append(test_loss)
                history['test_r2'].append(r2)
                print(f"   -> Test Metrics | Loss: {test_loss:.4f}, R2 Score: {r2:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), save_best)
            tqdm.write(f"   -> New best model saved at epoch {epoch+1} with validation loss: {best_loss:.4f}")

        torch.save(model.state_dict(), save_last)
        
    print(f"\nTraining process finished. Best validation loss: {best_loss:.4f}")
    return history
