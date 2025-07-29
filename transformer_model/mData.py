import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
import os
from sklearn.preprocessing import StandardScaler
import pickle

class ReactorDataset(Dataset):
    """
    Custom Dataset class for reactor data with sliding window approach.
    
    Creates sequences exactly like create_fission_sliding_windows:
    - past_state: rows i to i+input_seq_len-1 (state columns: days, k_eff)
    - past_control: rows i to i+input_seq_len-1 (control columns: control_rod_position)
    - future_control_prompt: rows i+input_seq_len to i+input_seq_len+output_pred_len-1 (control_rod_position)
    """
    
    def __init__(
        self, 
        data: np.ndarray, 
        input_seq_len: int = 20, 
        output_pred_len: int = 4,
        state_columns: list = [1, 2],      # days, k_eff (0-indexed)
        control_columns: list = [3],       # control_rod_position (0-indexed)
        scalers: Optional[Dict] = None,
        fit_scalers: bool = False
    ):
        """
        Args:
            data: numpy array with shape (n_samples, n_features)
            input_seq_len: length of input sequence (default: 20)
            output_pred_len: length of prediction sequence (default: 4)
            state_columns: which columns to use as state features
            control_columns: which columns to use as control features
            scalers: dictionary of fitted scalers for normalization
            fit_scalers: whether to fit new scalers (only for training data)
        """
        self.input_seq_len = input_seq_len
        self.output_pred_len = output_pred_len
        self.state_columns = state_columns
        self.control_columns = control_columns
        
        # Initialize or use provided scalers
        if scalers is None:
            self.scalers = {
                'state': StandardScaler(),
                'control': StandardScaler()
            }
        else:
            self.scalers = scalers
        
        # Normalize data
        self.data = self._normalize_data(data, fit_scalers)
        
        # Calculate valid starting indices
        # For sliding window: need input_seq_len + output_pred_len total rows
        self.num_samples = len(self.data) - input_seq_len - output_pred_len + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"Data too short. Need at least {input_seq_len + output_pred_len} rows, got {len(data)}")
    
    def _normalize_data(self, data: np.ndarray, fit_scalers: bool) -> np.ndarray:
        """Normalize state and control data using StandardScaler."""
        normalized_data = np.copy(data)
        
        # Normalize state features (days, k_eff)
        state_data = data[:, self.state_columns]
        if fit_scalers:
            normalized_state = self.scalers['state'].fit_transform(state_data)
        else:
            normalized_state = self.scalers['state'].transform(state_data)
        normalized_data[:, self.state_columns] = normalized_state
        
        # Normalize control features (control_rod_position)
        control_data = data[:, self.control_columns]
        if fit_scalers:
            normalized_control = self.scalers['control'].fit_transform(control_data)
        else:
            normalized_control = self.scalers['control'].transform(control_data)
        normalized_data[:, self.control_columns] = normalized_control
        
        return normalized_data
    
    def get_scalers(self) -> Dict:
        """Return the fitted scalers for use in validation/test sets."""
        return self.scalers
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Create sliding window sample exactly like create_fission_sliding_windows:
        
        For iteration i:
        - past_state: rows [i : i+input_seq_len] with state_columns (shape: [input_seq_len, len(state_columns)])
        - past_control: rows [i : i+input_seq_len] with control_columns (shape: [input_seq_len, len(control_columns)])
        - future_control_prompt: rows [i+input_seq_len : i+input_seq_len+output_pred_len] with control_columns (shape: [output_pred_len, len(control_columns)])
        """
        # Input sequence indices
        start_input = idx
        end_input = idx + self.input_seq_len
        
        # Target sequence indices
        start_target = end_input
        end_target = end_input + self.output_pred_len
        
        # Extract past state (days, k_eff) for input sequence
        past_state = self.data[start_input:end_input, self.state_columns]
        
        # Extract past control (control_rod_position) for input sequence
        past_control = self.data[start_input:end_input, self.control_columns]
        
        # Extract future control prompt (control_rod_position) for target sequence
        future_control_prompt = self.data[start_target:end_target, self.control_columns]
        
        # Convert to tensors (matching your reference code dtype)
        past_state = torch.tensor(past_state, dtype=torch.float32)
        past_control = torch.tensor(past_control, dtype=torch.float32)
        future_control_prompt = torch.tensor(future_control_prompt, dtype=torch.float32)
        
        return past_state, past_control, future_control_prompt

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
    
    if len(df) < 25:  # Minimum for sequence_length=20 + prediction_length=4
        raise ValueError(f"Expected at least 25 rows, got {len(df)}")
    
    # Rename columns for clarity (optional)
    expected_columns = ['step', 'days', 'k_eff', 'control_rod_position']
    df.columns = expected_columns[:len(df.columns)]
    
    # Check for missing values
    if df.isnull().any().any():
        print("Warning: Found missing values in data. Consider handling them.")
        print(df.isnull().sum())
    
    print(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Data range - Days: {df['days'].min():.2f} to {df['days'].max():.2f}")
    print(f"Data range - K_eff: {df['k_eff'].min():.4f} to {df['k_eff'].max():.4f}")
    print(f"Data range - Control Rod: {df['control_rod_position'].min():.2f} to {df['control_rod_position'].max():.2f}")
    
    return df

def create_data_loaders(
    file_path: str,
    batch_size: int = 32,
    input_seq_len: int = 20,
    output_pred_len: int = 4,
    train_split: float = 0.7,
    valid_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: Optional[int] = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders from Excel file.
    Uses sequential splitting to maintain temporal order.
    
    Args:
        file_path: path to Excel file
        batch_size: batch size for data loaders
        input_seq_len: length of input sequences (default: 20)
        output_pred_len: length of prediction sequences (default: 4)
        train_split: fraction of data for training
        valid_split: fraction of data for validation  
        test_split: fraction of data for testing
        random_seed: random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, scalers)
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
    
    # Calculate split points based on raw data length (not sequences)
    total_rows = len(data)
    train_end = int(total_rows * train_split)
    valid_end = int(total_rows * (train_split + valid_split))
    
    print(f"Data splitting:")
    print(f"  Total rows: {total_rows}")
    print(f"  Train rows: 0 to {train_end-1}")
    print(f"  Valid rows: {train_end} to {valid_end-1}")
    print(f"  Test rows: {valid_end} to {total_rows-1}")
    
    # Split data sequentially (maintaining temporal order)
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    
    # Create training dataset and fit scalers
    train_dataset = ReactorDataset(
        data=train_data,
        input_seq_len=input_seq_len,
        output_pred_len=output_pred_len,
        fit_scalers=True  # Fit scalers on training data
    )
    
    # Get scalers from training dataset
    scalers = train_dataset.get_scalers()
    
    # Create validation and test datasets using the same scalers
    valid_dataset = ReactorDataset(
        data=valid_data,
        input_seq_len=input_seq_len,
        output_pred_len=output_pred_len,
        scalers=scalers,
        fit_scalers=False  # Use existing scalers
    )
    
    test_dataset = ReactorDataset(
        data=test_data,
        input_seq_len=input_seq_len,
        output_pred_len=output_pred_len,
        scalers=scalers,
        fit_scalers=False  # Use existing scalers
    )
    
    print(f"Dataset sizes:")
    print(f"  Train sequences: {len(train_dataset)}")
    print(f"  Valid sequences: {len(valid_dataset)}")
    print(f"  Test sequences: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffle within training set is OK
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
    
    return train_loader, valid_loader, test_loader, scalers

def save_scalers(scalers: Dict, file_path: str):
    """Save scalers to file for later use."""
    with open(file_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to {file_path}")

def load_scalers(file_path: str) -> Dict:
    """Load scalers from file."""
    with open(file_path, 'rb') as f:
        scalers = pickle.load(f)
    print(f"Scalers loaded from {file_path}")
    return scalers

def denormalize_predictions(predictions: np.ndarray, scaler) -> np.ndarray:
    """Denormalize predictions back to original scale."""
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

def estimate_epochs(
    total_sequences: int,
    batch_size: int = 32,
    target_iterations: int = 1000
) -> int:
    """
    Estimate reasonable number of epochs based on data size.
    
    Args:
        total_sequences: total number of training sequences
        batch_size: batch size
        target_iterations: target number of training iterations
        
    Returns:
        Recommended number of epochs
    """
    batches_per_epoch = max(1, total_sequences // batch_size)
    recommended_epochs = max(1, target_iterations // batches_per_epoch)
    
    print(f"Epoch estimation:")
    print(f"  Training sequences: {total_sequences}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Recommended epochs: {recommended_epochs}")
    
    return recommended_epochs

def validate_data_loader(data_loader: DataLoader, name: str = "DataLoader"):
    """Validate that data loader produces expected output shapes."""
    print(f"\nValidating {name}:")
    
    try:
        for batch_idx, (past_state, past_control, future_control_prompt) in enumerate(data_loader):
            print(f"  Batch {batch_idx + 1}:")
            print(f"    past_state shape: {past_state.shape}")        # [batch_size, input_seq_len, 2]
            print(f"    past_control shape: {past_control.shape}")    # [batch_size, input_seq_len, 1]
            print(f"    future_control_prompt shape: {future_control_prompt.shape}")  # [batch_size, output_pred_len, 1]
            
            # Check for NaN values
            if torch.isnan(past_state).any():
                print("    WARNING: NaN values found in past_state")
            if torch.isnan(past_control).any():
                print("    WARNING: NaN values found in past_control")
            if torch.isnan(future_control_prompt).any():
                print("    WARNING: NaN values found in future_control_prompt")
            
            # Show data ranges
            print(f"    past_state range: [{past_state.min():.4f}, {past_state.max():.4f}]")
            print(f"    past_control range: [{past_control.min():.4f}, {past_control.max():.4f}]")
            print(f"    future_control_prompt range: [{future_control_prompt.min():.4f}, {future_control_prompt.max():.4f}]")
            
            if batch_idx == 0:  # Only show first batch
                break
                
    except Exception as e:
        print(f"    ERROR validating {name}: {e}")

# Example usage function
def main():
    """
    Example usage of the fixed data loading functions.
    """
    # Example parameters
    excel_file = "expanded_keff_1000_steps.xlsx"  # Replace with your file path
    batch_size = 32
    input_seq_len = 20
    output_pred_len = 4
    
    try:
        # Create data loaders
        train_loader, valid_loader, test_loader, scalers = create_data_loaders(
            file_path=excel_file,
            batch_size=batch_size,
            input_seq_len=input_seq_len,
            output_pred_len=output_pred_len
        )
        
        # Save scalers for later use
        save_scalers(scalers, "reactor_scalers.pkl")
        
        # Estimate epochs
        recommended_epochs = estimate_epochs(len(train_loader.dataset), batch_size)
        
        # Validate all data loaders
        validate_data_loader(train_loader, "Training DataLoader")
        validate_data_loader(valid_loader, "Validation DataLoader")
        validate_data_loader(test_loader, "Test DataLoader")
        
        print(f"\nData loader output format:")
        print(f"  past_state: [batch_size={batch_size}, input_seq_len={input_seq_len}, state_features=2]")
        print(f"  past_control: [batch_size={batch_size}, input_seq_len={input_seq_len}, control_features=1]")
        print(f"  future_control_prompt: [batch_size={batch_size}, output_pred_len={output_pred_len}, control_features=1]")
        
        print(f"\nSliding window example:")
        print(f"  Iteration 0: input rows 0-19 (state + control), target rows 20-23 (control)")
        print(f"  Iteration 1: input rows 1-20 (state + control), target rows 21-24 (control)")
        print(f"  Iteration 2: input rows 2-21 (state + control), target rows 22-25 (control)")
        print(f"  ... and so on")
        
        return train_loader, valid_loader, test_loader, scalers, recommended_epochs
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None

if __name__ == "__main__":
    main()
