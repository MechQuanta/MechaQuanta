import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# ==============================================================================
# 1. THE CORE LOGIC: CREATING SLIDING WINDOWS
# ==============================================================================

def create_sliding_windows(
    data_0D: torch.Tensor,
    data_ctrl: torch.Tensor,
    input_seq_len: int,
    output_pred_len: int
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Takes long, continuous time-series data and chops it into smaller,
    overlapping windows suitable for a sequence-to-sequence model.

    This function is the heart of the data preprocessing pipeline.

    Args:
        data_0D (torch.Tensor): The time-series of the plasma state.
                                Shape: [Total_Timesteps, Num_0D_Features]
        data_ctrl (torch.Tensor): The time-series of the control actions.
                                  Shape: [Total_Timesteps, Num_Ctrl_Features]
        input_seq_len (int): The length of the historical window for the model's input.
        output_pred_len (int): The length of the future window for the model to predict.

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        A list of samples. Each sample is a tuple containing:
        (input_window_0D, input_window_ctrl, target_window).
    """
    samples = []
    num_timesteps = data_0D.size(0)

    # We can only create windows if the total time series is long enough
    if num_timesteps < input_seq_len + output_pred_len:
        print(f"Warning: Time series is too short ({num_timesteps} steps) to create any windows.")
        return []

    # Iterate through the time series, creating one sample at each possible starting point.
    # The loop stops when there's no longer enough data left to form a full target window.
    for i in range(num_timesteps - input_seq_len - output_pred_len + 1):
        
        # --- Define the slice for the historical input window ---
        start_idx_input = i
        end_idx_input = i + input_seq_len
        
        # --- Define the slice for the future target window ---
        start_idx_target = end_idx_input
        end_idx_target = end_idx_input + output_pred_len

        # --- Extract the windows from the data tensors ---
        
        # Input Window 1: Historical plasma state
        input_window_0D = data_0D[start_idx_input:end_idx_input]
        
        # Input Window 2: Historical control actions taken
        input_window_ctrl = data_ctrl[start_idx_input:end_idx_input]
        
        # Target Window: The actual future plasma state that occurred. This is the "answer sheet".
        target_window = data_0D[start_idx_target:end_idx_target]

        # Append the complete sample (a tuple of three tensors) to our list
        samples.append((input_window_0D, input_window_ctrl, target_window))
        
    return samples

# ==============================================================================
# 2. PYTORCH DATASET WRAPPER
# ==============================================================================

class PlasmaDataset(Dataset):
    """
    A standard PyTorch Dataset class.
    It takes the list of windowed samples and makes it compatible with
    PyTorch's DataLoader, which handles batching and shuffling.
    """
    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        """Returns the total number of samples (windows) in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fetches one specific sample from the dataset.
        The DataLoader calls this method to create a batch.
        """
        # The sample is already a tuple of (input_0D, input_ctrl, target), so we just return it.
        return self.samples[idx]


# ==============================================================================
# 3. DEMONSTRATION OF HOW TO USE THE CODE
# ==============================================================================

if __name__ == "__main__":
    
    # --- Configuration for our example ---
    # Model configuration
    INPUT_SEQ_LEN = 20    # Model looks at 20 past time steps
    OUTPUT_PRED_LEN = 4   # Model predicts 4 future time steps

    # Data configuration
    NUM_0D_FEATURES = 12    # e.g., q95, betan, etc.
    NUM_CTRL_FEATURES = 14  # e.g., NBI powers, coil currents
    TOTAL_TIMESTEPS = 1000  # Length of our raw experimental data

    print("--- 1. Generating Raw Synthetic Data ---")
    # This simulates one long, continuous plasma experiment.
    # Shape: [Total_Timesteps, Num_Features]
    raw_data_0D = torch.randn(TOTAL_TIMESTEPS, NUM_0D_FEATURES)
    raw_data_ctrl = torch.randn(TOTAL_TIMESTEPS, NUM_CTRL_FEATURES)
    print(f"Shape of raw_data_0D: {raw_data_0D.shape}")
    print(f"Shape of raw_data_ctrl: {raw_data_ctrl.shape}")
    print("-" * 30)


    print("--- 2. Creating Sliding Windows ---")
    # This is the core data extraction step.
    all_samples = create_sliding_windows(
        data_0D=raw_data_0D,
        data_ctrl=raw_data_ctrl,
        input_seq_len=INPUT_SEQ_LEN,
        output_pred_len=OUTPUT_PRED_LEN
    )
    print(f"Total number of samples created: {len(all_samples)}")
    print("-" * 30)

    
    print("--- 3. Verifying the Shape of One Sample ---")
    # Let's inspect the very first sample to make sure it's correct.
    first_sample = all_samples[0]
    input_0D, input_ctrl, target = first_sample
    
    print(f"Shape of one input_0D sample:  {input_0D.shape}")
    print("   -> Expected: [INPUT_SEQ_LEN, NUM_0D_FEATURES]")
    print(f"   -> Actual:   [{INPUT_SEQ_LEN}, {NUM_0D_FEATURES}]")
    
    print(f"\nShape of one input_ctrl sample: {input_ctrl.shape}")
    print("   -> Expected: [INPUT_SEQ_LEN, NUM_CTRL_FEATURES]")
    print(f"   -> Actual:   [{INPUT_SEQ_LEN}, {NUM_CTRL_FEATURES}]")

    print(f"\nShape of one target sample:    {target.shape}")
    print("   -> Expected: [OUTPUT_PRED_LEN, NUM_0D_FEATURES]")
    print(f"   -> Actual:   [{OUTPUT_PRED_LEN}, {NUM_0D_FEATURES}]")
    print("-" * 30)


    print("--- 4. Using the DataLoader to Create a Batch ---")
    # We wrap our list of samples in the Dataset and then the DataLoader
    plasma_dataset = PlasmaDataset(all_samples)
    data_loader = DataLoader(plasma_dataset, batch_size=32, shuffle=True)

    # The DataLoader automatically stacks the individual samples into a batch.
    # Let's get the first batch.
    first_batch = next(iter(data_loader))
    batch_input_0D, batch_input_ctrl, batch_target = first_batch
    
    print("The DataLoader takes individual samples and adds a 'Batch' dimension.")
    print(f"Shape of one batch_input_0D:  {batch_input_0D.shape}")
    print("   -> Expected: [BATCH_SIZE, INPUT_SEQ_LEN, NUM_0D_FEATURES]")

    print(f"\nShape of one batch_input_ctrl: {batch_input_ctrl.shape}")
    print("   -> Expected: [BATCH_SIZE, INPUT_SEQ_LEN, NUM_CTRL_FEATURES]")

    print(f"\nShape of one batch_target:    {batch_target.shape}")
    print("   -> Expected: [BATCH_SIZE, OUTPUT_PRED_LEN, NUM_0D_FEATURES]")
    print("-" * 30)
