import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# ==============================================================================
# 1. RAW DATA ENTRY
# The raw data remains the same.
# ==============================================================================

data = {
    'Time': [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336, 343, 350, 357, 364, 371, 378, 385, 392, 399, 406, 413, 420, 427, 434, 441, 448, 455, 462, 469, 476, 483, 490, 497, 504, 511, 518, 525, 532, 539, 546, 553, 560, 567, 574, 581, 588, 595, 602, 609, 616, 623, 630, 637, 644, 651, 658, 665, 672, 679, 686, 693, 700],
    'k_eff': [1.092701838, 1.06437497, 1.065012769, 1.065876591, 1.061800388, 1.059393499, 1.060614603, 1.063856496, 1.064774233, 1.064784642, 1.062011291, 1.058414622, 1.056781631, 1.056440099, 1.054649913, 1.054347157, 1.053944907, 1.055532894, 1.053386437, 1.055860147, 1.056721943, 1.052291958, 1.054961126, 1.047525306, 1.050114415, 1.051093718, 1.051564438, 1.051993156, 1.051348951, 1.045463353, 1.051382467, 1.045560739, 1.049884343, 1.058042499, 1.047672517, 1.047410899, 1.040462134, 1.043748403, 1.046928182, 1.044807698, 1.047513216, 1.046018435, 1.044084828, 1.049869271, 1.044060654, 1.040598198, 1.042895967, 1.047227289, 1.040904451, 1.036620829, 1.007865863, 1.013749508, 1.009513742, 1.009092075, 1.002139306, 1.001377206, 1.003260468, 1.003304676, 1.003607456, 1.00990017, 1.001703215, 1.005317874, 1.003662281, 1.002612317, 0.995874254, 1.002976495, 1.001561763, 1.007400907, 0.999787012, 1.002976495, 1.001561763, 1.007400907, 0.999787012, 0.992451728, 0.992451728, 0.993248797, 0.995308213, 0.996167148, 0.995550332, 0.994285127, 0.993313284, 0.992451728, 0.995308213, 0.992619516, 0.994285127, 0.993248797, 0.996167148, 0.993313284, 0.996167148, 0.993313284, 0.993248797, 0.994285127, 0.996167148, 0.994285127, 0.996167148, 0.996167148, 0.993313284,0.994285127, 0.996167148, 0.996167148, 0.993313284],
    'CR_Pos': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1]
}
print("--- A simple check to verify lengths before creating the DataFrame ---")
print(f"Length of 'Time':   {len(data['Time'])}")
print(f"Length of 'k_eff':  {len(data['k_eff'])}")
print(f"Length of 'CR_Pos': {len(data['CR_Pos'])}")

# Now create the DataFrame
df = pd.DataFrame(data)

print("\nDataFrame created successfully!")
print(df.tail()) # Print the last few rows to see the end


# ==============================================================================
# 2. THE CORE DATA PREPARATION LOGIC
# These functions remain unchanged.
# ==============================================================================

def create_fission_sliding_windows(
    full_data: pd.DataFrame, input_seq_len: int, output_pred_len: int,
    state_cols: List[str], control_cols: List[str]
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    samples = []
    num_timesteps = len(full_data)
    for i in range(num_timesteps - input_seq_len - output_pred_len + 1):
        start_input = i
        end_input = i + input_seq_len
        start_target = end_input
        end_target = end_input + output_pred_len
        
        past_state = torch.tensor(full_data[state_cols].iloc[start_input:end_input].values, dtype=torch.float32)
        past_control = torch.tensor(full_data[control_cols].iloc[start_input:end_input].values, dtype=torch.float32)
        future_state_target = torch.ones(output_pred_len, len(state_cols), dtype=torch.float32)
        future_control_prompt = torch.tensor(full_data[control_cols].iloc[start_target:end_target].values, dtype=torch.float32)
        
        samples.append((past_state, past_control, future_state_target, future_control_prompt))
    return samples

class ReactorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]

# ==============================================================================
# 3. DEMONSTRATION OF DATA SPLITTING AND LOADER CREATION
# ==============================================================================

if __name__ == "__main__":
    
    # --- Configuration for our Fission Problem ---
    INPUT_SEQ_LEN = 20
    OUTPUT_PRED_LEN = 4
    BATCH_SIZE = 16
    NUM_EPOCHS = 30 # The number of epochs for the training process
    
    STATE_COLUMNS = ['k_eff']
    CONTROL_COLUMNS = ['CR_Pos']
    
    print("--- 1. Creating All Possible Samples from Raw Data ---")
    all_samples = create_fission_sliding_windows(
        full_data=df,
        input_seq_len=INPUT_SEQ_LEN,
        output_pred_len=OUTPUT_PRED_LEN,
        state_cols=STATE_COLUMNS,
        control_cols=CONTROL_COLUMNS
    )
    print(f"Total number of samples created: {len(all_samples)}")
    print("-" * 50)

    # --- 2. Splitting Samples into Train, Validation, and Test Sets ---
    # A standard split is 70% for training, 15% for validation, 15% for testing.
    # It's crucial to NOT shuffle before splitting to keep the time order,
    # especially for time-series data.
    
    # Calculate split indices
    total_samples_count = len(all_samples)
    train_split_idx = int(total_samples_count * 0.70)
    val_split_idx = int(total_samples_count * 0.85) # 70% + 15%

    # Split the list of samples
    train_samples = all_samples[:train_split_idx]
    val_samples = all_samples[train_split_idx:val_split_idx]
    test_samples = all_samples[val_split_idx:]
    
    print("--- 2.1 Data Split Results ---")
    print(f"Training samples:   {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Testing samples:    {len(test_samples)}")
    print("-" * 50)
    
    # --- 3. Creating a Separate DataLoader for Each Set ---
    # Training loader should shuffle to introduce randomness between epochs
    train_dataset = ReactorDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Validation and Test loaders should NOT shuffle to ensure consistent evaluation
    val_dataset = ReactorDataset(val_samples)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    test_dataset = ReactorDataset(test_samples)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("--- 3.1 DataLoader Verification ---")
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader:   {len(val_loader)}")
    print(f"Number of batches in test_loader:  {len(test_loader)}")
    print("\nData is now ready for a full training lifecycle.")
    print("The model would be trained for the specified number of epochs (e.g., 30)")
    print("using `train_loader` and validated against `val_loader` at each epoch.")
    print("After training is complete, the final best model would be tested once on `test_loader`.")
    print("-" * 50)
