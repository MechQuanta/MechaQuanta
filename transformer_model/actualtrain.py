import torch
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any

# Import your existing modules
sys.path.append(os.path.expanduser('~/transformer_model'))
from mData import create_data_loaders, estimate_epochs, load_excel_data, ReactorDataset
from traintrnsfrmmodi import train  # Your existing train function
from trnsfrm import Transformer  # Your existing Transformer model


def get_range_of_output(data: pd.DataFrame, columns: list) -> Dict[str, list]:
    """
    Calculate range information for output normalization.
    Based on your Transformer model, it expects range_info as Dict[str, list] where 
    each list contains [min, max] values.
    
    Args:
        data: DataFrame containing the data
        columns: list of column names to calculate ranges for
        
    Returns:
        Dictionary with min/max values for each column in the format your Transformer expects
    """
    range_info = {}
    for col in columns:
        if col in data.columns:
            range_info[col] = [float(data[col].min()), float(data[col].max())]
        else:
            print(f"Warning: Column '{col}' not found in data")
            range_info[col] = [0.0, 1.0]
    
    return range_info


def create_enhanced_data_loaders(
    excel_file_path: str,
    args: Dict[str, Any],
    seq_len: int,
    pred_len: int,
    cols_0D: list,
    cols_control: list
):
    """
    Create enhanced data loaders with additional dataset properties.
    
    Args:
        excel_file_path: path to Excel file
        args: training arguments dictionary
        seq_len: sequence length for input
        pred_len: prediction length for output
        cols_0D: column names for 0D data
        cols_control: column names for control data
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, range_info)
    """
    
    # Load the raw data first to calculate range info
    df = load_excel_data(excel_file_path)
    
    # Create range info for the specified columns
    # Map your column indices to actual column names
    column_mapping = {
        0: 'step',
        1: 'days', 
        2: 'k_eff',
        3: 'control_rod_position'
    }
    
    # Create a mock DataFrame with the expected column structure for range calculation
    ts_data = pd.DataFrame()
    for i, col_name in enumerate(cols_0D):
        if i < len(df.columns)+1:
            ts_data[col_name] = df.iloc[:, i + 1]  # Skip step column, start from days
    
    range_info = get_range_of_output(ts_data, cols_0D)
    
    # Create the data loaders using your custom dataset
    train_loader, valid_loader, test_loader = create_data_loaders(
        file_path=excel_file_path,
        batch_size=args['batch_size'],
        sequence_length=seq_len,
        prediction_length=pred_len,
        train_split=0.7,
        valid_split=0.15,
        test_split=0.15,
        shuffle_train=True,
        random_seed=42
    )
    
    # Modify the data loaders to include additional DataLoader parameters
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=args['batch_size'],
        num_workers=args.get('num_workers', 0),
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_loader.dataset,
        batch_size=args['batch_size'],
        num_workers=args.get('num_workers', 0),
        shuffle=False,  # Don't shuffle validation data
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_loader.dataset,
        batch_size=args['batch_size'],
        num_workers=args.get('num_workers', 0),
        shuffle=False,  # Don't shuffle test data
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, valid_loader, test_loader, range_info


def main_training_pipeline(
    excel_file_path: str,
    args: Dict[str, Any],
    tag: str = "reactor_experiment"
):
    """
    Main training pipeline that integrates Excel data loading with your existing training code.
    
    Args:
        excel_file_path: path to your Excel file
        args: dictionary containing training arguments
        tag: experiment tag for saving models and logs
    """
    
    # Define your column specifications
    cols_0D = ['days', 'k_eff']  # Input features (columns 1, 2 from Excel)
    cols_control = ['control_signal']  # Control features (dummy for now)
    
    # Define sequence parameters
    seq_len = 21  # Input sequence length (rows 0-20)
    pred_len = 4  # Prediction sequence length (rows 21-24)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders with enhanced functionality
    print("Creating data loaders...")
    train_loader, valid_loader, test_loader, range_info = create_enhanced_data_loaders(
        excel_file_path=excel_file_path,
        args=args,
        seq_len=seq_len,
        pred_len=pred_len,
        cols_0D=cols_0D,
        cols_control=cols_control
    )
    
    print(f"Range info calculated: {range_info}")
    
    # Estimate recommended number of epochs
    df = load_excel_data(excel_file_path)
    recommended_epochs = estimate_epochs(
        total_sequences=len(df) - seq_len - pred_len + 1,
        batch_size=args['batch_size'],
        train_split=0.7,
        target_iterations=1000
    )
    
    # Use recommended epochs if not specified
    if 'num_epoch' not in args or args['num_epoch'] is None:
        args['num_epoch'] = recommended_epochs
        print(f"Using recommended epochs: {recommended_epochs}")
    
    # Initialize your transformer model with the provided arguments
    model = Transformer(
        n_layers=args.get('n_layers', 4), 
        n_heads=args.get('n_heads', 8), 
        dim_feedforward=args.get('dim_feedforward', 512), 
        dropout=args.get('dropout', 0.1),        
        RIN=args.get('RIN', False),
        input_0D_dim=len(cols_0D),
        input_0D_seq_len=seq_len,
        input_ctrl_dim=len(cols_control),
        input_ctrl_seq_len=seq_len + pred_len,
        output_0D_pred_len=pred_len,
        output_0D_dim=1,  # Predicting control rod position (1 output)
        feature_dim=args.get('feature_dim', 64),
        range_info=range_info,
        noise_mean=args.get('noise_mean', 0.0),
        noise_std=args.get('noise_std', 0.1),
        kernel_size=args.get('kernel_size', 3)
    )

    # Print model summary
    try:
        model.summary()
    except AttributeError:
        print("Model summary not available")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args['step_size'], 
        gamma=args['gamma']
    )
    
    # Setup save directories
    save_best_dir = os.path.join(args['root_dir'], f"{tag}_best.pt")
    save_last_dir = os.path.join(args['root_dir'], f"{tag}_last.pt")
    
    # Create directories if they don't exist
    os.makedirs(args['root_dir'], exist_ok=True)
    
    # Initialize loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    # Load existing model if available (optional)
    if os.path.exists(save_last_dir):
        try:
            model.load_state_dict(torch.load(save_last_dir, map_location=device))
            print(f"Loaded existing model from {save_last_dir}")
        except Exception as e:
            print(f"Could not load existing model: {e}")
    
    print("=" * 50)
    print("Training Process")
    print("=" * 50)
    print(f"Process: {tag}")
    print(f"Device: {device}")
    print(f"Epochs: {args['num_epoch']}")
    print(f"Batch size: {args['batch_size']}")
    print(f"Learning rate: {args['lr']}")
    print(f"Sequence length: {seq_len}")
    print(f"Prediction length: {pred_len}")
    
    # Start training using your existing train function
    history = train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        num_epoch=args['num_epoch'],
        verbose=args.get('verbose', 8),
        save_best=save_best_dir,
        save_last=save_last_dir,
        max_norm_grad=args.get('max_norm_grad', None),
        test_for_check_per_epoch=test_loader
    )
    
    print("=" * 50)
    print("Training Completed!")
    print("=" * 50)
    
    return history, model, (train_loader, valid_loader, test_loader)


# Example usage
if __name__ == "__main__":
    # Define your training arguments
    training_args = {
        'batch_size': 32,
        'num_workers': 0,  # Set to 0 if you have issues with multiprocessing
        'lr': 0.001,
        'step_size': 20,
        'gamma': 0.8,
        'num_epoch': 50,  # Will be overridden by recommended if None
        'verbose': 8,
        'max_norm_grad': 1.0,
        'root_dir': './checkpoints',
        
        # Transformer model hyperparameters
        'n_layers': 4,
        'n_heads': 8,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'RIN': False,
        'feature_dim': 64,
        'noise_mean': 0.0,
        'noise_std': 0.1,
        'kernel_size': 3
    }
    
    # Path to your Excel file
    excel_file_path = "expanded_keff_1000_steps.xlsx"  # Replace with your actual file path
    
    # Run the training pipeline
    try:
        history, trained_model, data_loaders = main_training_pipeline(
            excel_file_path=excel_file_path,
            args=training_args,
            tag="reactor_transformer_v1"
        )
        
        print("Training completed successfully!")
        
        # You can now use the trained model and history for further analysis
        train_loader, valid_loader, test_loader = data_loaders
        
        # Example: Plot training curves
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['valid_loss'], label='Valid Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            plt.subplot(1, 2, 2)
            if history['test_r2']:
                plt.plot(history['test_epochs'], history['test_r2'], label='Test R²')
                plt.xlabel('Epoch')
                plt.ylabel('R² Score')
                plt.legend()
                plt.title('Test R² Score')
            
            plt.tight_layout()
            plt.savefig('training_results.png')
            print("Training plots saved as 'training_results.png'")
        except ImportError:
            print("Matplotlib not available for plotting")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
