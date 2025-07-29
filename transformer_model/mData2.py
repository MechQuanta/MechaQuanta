import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import os

class ReactorDataset(Dataset):
    """
    Custom Dataset class for reactor data with sliding window approach.
    
    Creates sequences where:
    - Input: rows i to i+sequence_length-1 (columns: days, k_eff, control_rod_position)
    - Target: rows i+sequence_length to i+sequence_length+prediction_length-1 (column: control_rod_position)
    """
    
    def __init__(
        self, 
        data: np.ndarray, 
        sequence_length: int = 21, 
        prediction_length: int = 4,
        input_columns: list = [1, 2],  # days, k_eff, control_rod_position (0-indexed)
        target_column: int = 3             # control_rod_position (0-indexed)
    ):
        """
        Args:
            data: numpy array with shape (n_samples, n_features)
            sequence_length: length of input sequence (default: 21 for rows 0-20)
            prediction_length: length of prediction sequence (default: 4 for rows 21-24)
            input_columns: which columns to use as input features
            target_column: which column to use as target
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_columns = input_columns
        self.target_column = target_column
        
        # Calculate valid starting indices
        self.max_start_idx = len(data) - sequence_length - prediction_length + 1
        
        if self.max_start_idx <= 0:
            raise ValueError(f"Data too short. Need at least {sequence_length + prediction_length} rows, got {len(data)}")
    
    def __len__(self):
        return self.max_start_idx
    
    def __getitem__(self, idx):
        # Input sequence: rows idx to idx+sequence_length-1
        input_start = idx
        input_end = idx + self.sequence_length
        
        # Target sequence: rows idx+sequence_length to idx+sequence_length+prediction_length-1
        target_start = idx + self.sequence_length
        target_end = idx + self.sequence_length + self.prediction_length
        
        # Extract input features (days, k_eff, control_rod_position)
        data_0D = self.data[input_start:input_end, self.input_columns]
        
        # Create a minimal control input for compatibility (can be dummy)
        data_ctrl = np.zeros((self.sequence_length, 1))  # Dummy control data
        
        # Extract target (control rod position)
        target = self.data[target_start:target_end, self.target_column:self.target_column+1]
        
        # Convert to tensors
        data_0D = torch.FloatTensor(data_0D)
        data_ctrl = torch.FloatTensor(data_ctrl)
        target = torch.FloatTensor(target)
        
        return data_0D, data_ctrl, target

def load_excel_data(file_path: str) -> pd.DataFrame:
    """
    Load Excel file and validate the expected structure.
    
    Args:
        file_path: path to Excel file
        
    Returns:
        pandas DataFrame with the loaded data
        
    Expected columns:
        0: step
        1: days  
        2: k_eff
        3: control_rod_position
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    # Try reading Excel file
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")
    
    # Validate structure
    if len(df.columns) < 4:
        raise ValueError(f"Expected at least 4 columns, got {len(df.columns)}")
    
    if len(df) < 25:  # Minimum for sequence_length=21 + prediction_length=4
        raise ValueError(f"Expected at least 25 rows, got {len(df)}")
    
    # Rename columns for clarity (optional)
    expected_columns = ['step', 'days', 'k_eff', 'control_rod_position']
    df.columns = expected_columns[:len(df.columns)]
    
    print(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Data range - Days: {df['days'].min():.2f} to {df['days'].max():.2f}")
    print(f"Data range - K_eff: {df['k_eff'].min():.4f} to {df['k_eff'].max():.4f}")
    print(f"Data range - Control Rod: {df['control_rod_position'].min():.2f} to {df['control_rod_position'].max():.2f}")
    
    return df

def create_data_loaders(
    file_path: str,
    batch_size: int = 32,
    sequence_length: int = 21,
    prediction_length: int = 4,
    train_split: float = 0.7,
    valid_split: float = 0.15,
    test_split: float = 0.15,
    shuffle_train: bool = True,
    random_seed: Optional[int] = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from Excel file.
    
    Args:
        file_path: path to Excel file
        batch_size: batch size for data loaders
        sequence_length: length of input sequences (default: 21)
        prediction_length: length of prediction sequences (default: 4)
        train_split: fraction of data for training
        valid_split: fraction of data for validation  
        test_split: fraction of data for testing
        shuffle_train: whether to shuffle training data
        random_seed: random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    
    # Validate splits
    if abs((train_split + valid_split + test_split) - 1.0) > 1e-6:
        raise ValueError("Train, valid, and test splits must sum to 1.0")
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Load data
    df = load_excel_data(file_path)
    data = df.values  # Convert to numpy array
    
    # Create dataset to get the total number of valid sequences
    full_dataset = ReactorDataset(
        data=data,
        sequence_length=sequence_length,
        prediction_length=prediction_length
    )
    
    total_sequences = len(full_dataset)
    print(f"Total valid sequences: {total_sequences}")
    
    # Calculate split indices
    train_size = int(total_sequences * train_split)
    valid_size = int(total_sequences * valid_split)
    test_size = total_sequences - train_size - valid_size
    
    print(f"Split sizes - Train: {train_size}, Valid: {valid_size}, Test: {test_size}")
    
    # Create indices for splitting
    indices = np.arange(total_sequences)
    if shuffle_train:
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True  # Ensures consistent batch sizes
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    
    print(f"Created data loaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(valid_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, valid_loader, test_loader

def estimate_epochs(
    total_sequences: int,
    batch_size: int = 32,
    train_split: float = 0.7,
    target_iterations: int = 1000
) -> int:
    """
    Estimate reasonable number of epochs based on data size.
    
    Args:
        total_sequences: total number of sequences in dataset
        batch_size: batch size
        train_split: fraction of data used for training
        target_iterations: target number of training iterations
        
    Returns:
        Recommended number of epochs
    """
    train_sequences = int(total_sequences * train_split)
    batches_per_epoch = train_sequences // batch_size
    
    if batches_per_epoch == 0:
        recommended_epochs = 1
    else:
        recommended_epochs = max(1, target_iterations // batches_per_epoch)
    
    print(f"Epoch estimation:")
    print(f"  Training sequences: {train_sequences}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Recommended epochs: {recommended_epochs}")
    
    return recommended_epochs

# Example usage function
def main():
    """
    Example usage of the data loading functions.
    """
    # Example parameters
    excel_file = "expanded_keff_1000_steps.xlsx"  # Replace with your file path
    batch_size = 32
    sequence_length = 21
    prediction_length = 4
    
    try:
        # Create data loaders
        train_loader, valid_loader, test_loader = create_data_loaders(
            file_path=excel_file,
            batch_size=batch_size,
            sequence_length=sequence_length,
            prediction_length=prediction_length
        )
        
        # Estimate epochs
        df = load_excel_data(excel_file)
        dataset = ReactorDataset(df.values, sequence_length, prediction_length)
        recommended_epochs = estimate_epochs(len(dataset), batch_size)
        
        # Test a batch
        print("\nTesting data loader output:")
        for batch_idx, (data_0D, data_ctrl, target) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  data_0D shape: {data_0D.shape}")  # Should be [batch_size, sequence_length, n_input_features]
            print(f"  data_ctrl shape: {data_ctrl.shape}")  # Should be [batch_size, sequence_length, n_ctrl_features]
            print(f"  target shape: {target.shape}")  # Should be [batch_size, prediction_length, 1]
            
            if batch_idx == 0:  # Only show first batch
                break
        
        return train_loader, valid_loader, test_loader, recommended_epochs
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None

if __name__ == "__main__":
    main()
