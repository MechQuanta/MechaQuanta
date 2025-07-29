import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =================================================================================================
# IMPROVED DATA PREPARATION
# =================================================================================================

class ImprovedReactorDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_state, past_control, _, future_control_prompt = self.samples[idx]
        
        # Data augmentation during training
        if self.augment and torch.rand(1) < 0.3:  # 30% chance
            # Add small noise
            noise_scale = 0.01
            past_state = past_state + torch.randn_like(past_state) * noise_scale
            past_control = past_control + torch.randn_like(past_control) * noise_scale
        
        return past_state, past_control, future_control_prompt

def create_improved_sliding_windows(
    data: np.ndarray, 
    input_seq_len: int = 20, 
    output_pred_len: int = 4,
    input_cols: List[int] = [2, 3],  # Last 2 columns as input features
    stride: int = 1
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create sliding windows from data matrix
    data: (1000, 4) numpy array
    input_cols: which columns to use as input features
    """
    samples = []
    num_rows = data.shape[0]
    
    for i in range(0, num_rows - input_seq_len - output_pred_len + 1, stride):
        # Input sequence: rows i to i+input_seq_len-1, using specified columns
        input_data = data[i:i+input_seq_len, input_cols]
        past_state = torch.tensor(input_data, dtype=torch.float32)
        
        # For compatibility with existing model, create minimal control input
        past_control = torch.zeros(input_seq_len, 1, dtype=torch.float32)
        
        # Target: next output_pred_len rows from the same columns
        target_data = data[i+input_seq_len:i+input_seq_len+output_pred_len, input_cols]
        future_control_prompt = torch.tensor(target_data, dtype=torch.float32)
        
        # Placeholder for future_state_target (not used in this setup)
        future_state_target = torch.ones(output_pred_len, len(input_cols), dtype=torch.float32)
        
        samples.append((past_state, past_control, future_state_target, future_control_prompt))
    
    return samples

# =================================================================================================
# IMPROVED LOSS FUNCTIONS
# =================================================================================================

class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred, target):
        loss = (pred - target) ** 2
        if self.weights is not None:
            loss = loss * self.weights
        return torch.mean(loss)

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 
                          0.5 * diff ** 2 / self.beta,
                          diff - 0.5 * self.beta)
        return torch.mean(loss)

# =================================================================================================
# EARLY STOPPING CLASS
# =================================================================================================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

# =================================================================================================
# IMPROVED TRAINING FUNCTIONS
# =================================================================================================

def train_epoch_improved(
    train_loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
    max_norm_grad: float = 1.0,
    accumulation_steps: int = 1
):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (past_state, past_control, future_control_prompt) in enumerate(train_loader):
        if past_state.size(0) <= 1:
            continue
            
        past_state = past_state.to(device)
        past_control = past_control.to(device)
        target = future_control_prompt.to(device)
        
        # Forward pass
        output = model(past_state, past_control)
        loss = loss_fn(output, target) / accumulation_steps
        
        if not torch.isfinite(loss):
            continue
            
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if max_norm_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def validate_epoch_improved(
    valid_loader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: str
):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for past_state, past_control, future_control_prompt in valid_loader:
            if past_state.size(0) <= 1:
                continue
                
            past_state = past_state.to(device)
            past_control = past_control.to(device)
            target = future_control_prompt.to(device)
            
            output = model(past_state, past_control)
            loss = loss_fn(output, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)

# =================================================================================================
# MAIN TRAINING FUNCTION
# =================================================================================================

def train_improved_model(
    data_matrix: np.ndarray,  # (1000, 4) data matrix
    model_params: Dict,
    training_params: Dict,
    input_cols: List[int] = [2, 3],  # Last 2 columns
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Improved training function that addresses overfitting
    """
    print(f"Using device: {device}")
    print(f"Data shape: {data_matrix.shape}")
    
    # =================================================================================================
    # DATA PREPROCESSING
    # =================================================================================================
    
    # Normalize data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_matrix)
    
    # Create sliding windows
    input_seq_len = training_params.get('input_seq_len', 20)
    output_pred_len = training_params.get('output_pred_len', 4)
    stride = training_params.get('stride', 1)
    
    samples = create_improved_sliding_windows(
        normalized_data, 
        input_seq_len=input_seq_len,
        output_pred_len=output_pred_len,
        input_cols=input_cols,
        stride=stride
    )
    
    print(f"Total samples created: {len(samples)}")
    
    # Split data
    n_samples = len(samples)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Create datasets and dataloaders
    train_dataset = ImprovedReactorDataset(train_samples, augment=True)
    val_dataset = ImprovedReactorDataset(val_samples, augment=False)
    test_dataset = ImprovedReactorDataset(test_samples, augment=False)
    
    batch_size = training_params.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # =================================================================================================
    # MODEL INITIALIZATION
    # =================================================================================================
    
    # Import your Transformer class here
    from trnsfrm2 import Transformer  # Replace with actual import
    
    # Update model parameters
    model_params.update({
        'input_0D_dim': len(input_cols),
        'input_0D_seq_len': input_seq_len,
        'output_0D_pred_len': output_pred_len,
        'output_0D_dim': len(input_cols)
    })
    
    model = Transformer(**model_params).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # =================================================================================================
    # OPTIMIZER AND SCHEDULER
    # =================================================================================================
    
    lr = training_params.get('learning_rate', 1e-4)
    weight_decay = training_params.get('weight_decay', 1e-5)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=training_params.get('scheduler_T0', 10),
        T_mult=2,
        eta_min=lr * 0.01
    )
    
    # Loss function
    loss_type = training_params.get('loss_type', 'mse')
    if loss_type == 'smooth_l1':
        loss_fn = SmoothL1Loss(beta=training_params.get('smooth_l1_beta', 1.0))
    elif loss_type == 'weighted_mse':
        weights = training_params.get('loss_weights', None)
        loss_fn = WeightedMSELoss(weights)
    else:
        loss_fn = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=training_params.get('patience', 20),
        min_delta=training_params.get('min_delta', 1e-6)
    )
    
    # =================================================================================================
    # TRAINING LOOP
    # =================================================================================================
    
    num_epochs = training_params.get('num_epochs', 100)
    max_norm_grad = training_params.get('max_norm_grad', 1.0)
    accumulation_steps = training_params.get('accumulation_steps', 1)
    
    # Storage for metrics
    train_losses = []
    val_losses = []
    learning_rates = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Create directories
    save_dir = training_params.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch_improved(
            train_loader, model, optimizer, loss_fn, device, 
            max_norm_grad, accumulation_steps
        )
        
        # Validation
        val_loss = validate_epoch_improved(val_loader, model, loss_fn, device)
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(current_lr)
        
        # Print progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'scaler': scaler
            }, os.path.join(save_dir, 'best_model.pt'))
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    
    # =================================================================================================
    # FINAL EVALUATION
    # =================================================================================================
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss = validate_epoch_improved(test_loader, model, loss_fn, device)
    print(f"Final test loss: {test_loss:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'scaler': scaler
    }

# =================================================================================================
# ADDITIONAL UTILITIES
# =================================================================================================

def analyze_model_predictions(model, test_loader, device, scaler=None, num_samples=5):
    """Analyze model predictions vs actual values"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for i, (past_state, past_control, target) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            past_state = past_state.to(device)
            past_control = past_control.to(device)
            
            pred = model(past_state, past_control)
            
            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Denormalize if scaler provided
    if scaler is not None:
        # Assuming we want to denormalize the last 2 columns
        pred_full = np.zeros((predictions.shape[0], predictions.shape[1], 4))
        target_full = np.zeros((targets.shape[0], targets.shape[1], 4))
        
        pred_full[:, :, -2:] = predictions
        target_full[:, :, -2:] = targets
        
        pred_denorm = scaler.inverse_transform(pred_full.reshape(-1, 4))
        target_denorm = scaler.inverse_transform(target_full.reshape(-1, 4))
        
        predictions = pred_denorm[:, -2:].reshape(predictions.shape)
        targets = target_denorm[:, -2:].reshape(targets.shape)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for i in range(min(4, predictions.shape[0])):
        ax = axes[i]
        for j in range(predictions.shape[2]):
            ax.plot(predictions[i, :, j], 'r--', label=f'Pred Feature {j}', alpha=0.8)
            ax.plot(targets[i, :, j], 'b-', label=f'True Feature {j}', alpha=0.8)
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return predictions, targets

def find_optimal_hyperparameters(data_matrix, input_cols=[2, 3]):
    """Simple grid search for hyperparameters"""
    
    hyperparams_grid = {
        'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
        'batch_size': [16, 32, 64],
        'dropout': [0.05, 0.1, 0.15, 0.2],
        'n_layers': [2, 3, 4],
        'feature_dim': [64, 128, 256]
    }
    
    best_val_loss = float('inf')
    best_params = {}
    
    print("Starting hyperparameter search...")
    
    # Simplified search - test a few combinations
    test_combinations = [
        {'learning_rate': 1e-4, 'batch_size': 32, 'dropout': 0.1, 'n_layers': 3, 'feature_dim': 128},
        {'learning_rate': 5e-5, 'batch_size': 64, 'dropout': 0.15, 'n_layers': 4, 'feature_dim': 256},
        {'learning_rate': 1e-4, 'batch_size': 16, 'dropout': 0.05, 'n_layers': 2, 'feature_dim': 64}
    ]
    
    for i, params in enumerate(test_combinations):
        print(f"\nTesting combination {i+1}/{len(test_combinations)}: {params}")
        
        model_params = {
            'n_layers': params['n_layers'],
            'n_heads': 8,
            'dim_feedforward': params['feature_dim'] * 4,
            'dropout': params['dropout'],
            'feature_dim': params['feature_dim'],
            'use_cnn': False,
            'RIN': True,
            'noise_std': 0.01
        }
        
        training_params = {
            'input_seq_len': 20,
            'output_pred_len': 4,
            'batch_size': params['batch_size'],
            'learning_rate': params['learning_rate'],
            'weight_decay': 1e-5,
            'num_epochs': 30,  # Reduced for quick search
            'patience': 10,
            'max_norm_grad': 1.0,
            'save_dir': f'./temp_search_{i}'
        }
        
        try:
            results = train_improved_model(
                data_matrix=data_matrix,
                model_params=model_params,
                training_params=training_params,
                input_cols=input_cols,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )
            
            if results['best_val_loss'] < best_val_loss:
                best_val_loss = results['best_val_loss']
                best_params = params.copy()
                
        except Exception as e:
            print(f"Error with combination {i+1}: {e}")
            continue
    
    print(f"\nBest hyperparameters found:")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best parameters: {best_params}")
    
    return best_params, best_val_loss

# =================================================================================================
# EXAMPLE USAGE
# =================================================================================================

if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load your actual data here
    # data_matrix = pd.read_csv('your_data.csv').values  # Replace with your data loading
    
    # For demo, create synthetic data with some patterns
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    data_matrix = np.column_stack([
        t,  # Column 0: time
        np.sin(t) + 0.1 * np.random.randn(1000),  # Column 1: signal with noise
        np.cos(t * 0.5) + 0.05 * np.random.randn(1000),  # Column 2: slower oscillation
        np.sin(t * 2) * np.exp(-t * 0.1) + 0.1 * np.random.randn(1000)  # Column 3: damped oscillation
    ])
    
    print("Data matrix shape:", data_matrix.shape)
    print("Data matrix statistics:")
    print(f"Mean: {np.mean(data_matrix, axis=0)}")
    print(f"Std: {np.std(data_matrix, axis=0)}")
    
    # Option 1: Use default parameters
    print("\n" + "="*60)
    print("TRAINING WITH DEFAULT PARAMETERS")
    print("="*60)
    
    model_params = {
        'n_layers': 3,
        'n_heads': 8,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'feature_dim': 128,
        'use_cnn': False,
        'RIN': True,
        'noise_std': 0.005  # Reduced noise for better convergence
    }
    
    training_params = {
        'input_seq_len': 20,
        'output_pred_len': 4,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'patience': 25,
        'max_norm_grad': 1.0,
        'accumulation_steps': 1,
        'stride': 1,
        'loss_type': 'mse',
        'scheduler_T0': 15,
        'save_dir': './checkpoints'
    }
    
    # Train the model
    results = train_improved_model(
        data_matrix=data_matrix,
        model_params=model_params,
        training_params=training_params,
        input_cols=[2, 3],  # Use columns 2 and 3 as features
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"\nTraining Results:")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Final test loss: {results['test_loss']:.6f}")
    print(f"Train/Val loss ratio: {results['train_losses'][-1]/results['val_losses'][-1]:.3f}")
    
    # Option 2: Hyperparameter search (uncomment to
