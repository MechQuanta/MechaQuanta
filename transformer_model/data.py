import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# ==============================================================================
# 1. RAW DATA ENTRY
# Manually transcribed from the image you provided.
# ==============================================================================

# In a real scenario, this would be loaded from a CSV file.
import pandas as pd

# The corrected data dictionary
data = {
    'Time': [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336, 343, 350, 357, 364, 371, 378, 385, 392, 399, 406, 413, 420, 427, 434, 441, 448, 455, 462, 469, 476, 483, 490, 497, 504, 511, 518, 525, 532, 539, 546, 553, 560, 567, 574, 581, 588, 595, 602, 609, 616, 623, 630, 637, 644, 651, 658, 665, 672, 679, 686, 693, 700],
    'k_eff': [1.092701838, 1.06437497, 1.065012769, 1.065876591, 1.061800388, 1.059393499, 1.060614603, 1.063856496, 1.064774233, 1.064784642, 1.062011291, 1.058414622, 1.056781631, 1.056440099, 1.054649913, 1.054347157, 1.053944907, 1.055532894, 1.053386437, 1.055860147, 1.056721943, 1.052291958, 1.054961126, 1.047525306, 1.050114415, 1.051093718, 1.051564438, 1.051993156, 1.051348951, 1.045463353, 1.051382467, 1.045560739, 1.049884343, 1.058042499, 1.047672517, 1.047410899, 1.040462134, 1.043748403, 1.046928182, 1.044807698, 1.047513216, 1.046018435, 1.044084828, 1.049869271, 1.044060654, 1.040598198, 1.042895967, 1.047227289, 1.040904451, 1.036620829, 1.007865863, 1.013749508, 1.009513742, 1.009092075, 1.002139306, 1.001377206, 1.003260468, 1.003304676, 1.003607456, 1.00990017, 1.001703215, 1.005317874, 1.003662281, 1.002612317, 0.995874254, 1.002976495, 1.001561763, 1.007400907, 0.999787012, 1.002976495, 1.001561763, 1.007400907, 0.999787012, 0.992451728, 0.992451728, 0.993248797, 0.995308213, 0.996167148, 0.995550332, 0.994285127, 0.993313284, 0.992451728, 0.995308213, 0.992619516, 0.994285127, 0.993248797, 0.996167148, 0.993313284, 0.996167148, 0.993313284, 0.993248797, 0.994285127, 0.996167148, 0.994285127, 0.996167148, 0.996167148, 0.993313284,0.995874254, 1.002976495, 1.001561763, 1.007400907],
    'CR_Pos': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1] # <-- FINAL 19.1 ADDED HERE
}

# This will now work without error
print("--- A simple check to verify lengths before creating the DataFrame ---")
print(f"Length of 'Time':   {len(data['Time'])}")
print(f"Length of 'k_eff':  {len(data['k_eff'])}")
print(f"Length of 'CR_Pos': {len(data['CR_Pos'])}")

# Now create the DataFrame
df = pd.DataFrame(data)

print("\nDataFrame created successfully!")
print(df.tail()) # Print the last few rows to see the end


# ==============================================================================
# 2. THE CORE LOGIC: CREATING SLIDING WINDOWS
# This function is now adapted for the fission control problem.
# ==============================================================================

def create_fission_sliding_windows(
    full_data: pd.DataFrame,
    input_seq_len: int,
    output_pred_len: int,
    state_cols: List[str],
    control_cols: List[str]
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Creates the four tensors required by the FissionReactorTransformer for each sample.
    (past_state, past_control, future_state_target, future_control_prompt)
    """
    samples = []
    num_timesteps = len(full_data)

    for i in range(num_timesteps - input_seq_len - output_pred_len + 1):
        
        # --- Define window slices ---
        start_input = i
        end_input = i + input_seq_len
        start_target = end_input
        end_target = end_input + output_pred_len

        # --- Extract windows from DataFrame and convert to tensors ---
        
        # TENSOR 1: Historical Reactor State
        past_state = torch.tensor(full_data[state_cols].iloc[start_input:end_input].values, dtype=torch.float32)

        # TENSOR 2: Historical Control Rod Positions
        past_control = torch.tensor(full_data[control_cols].iloc[start_input:end_input].values, dtype=torch.float32)
        
        # TENSOR 3: Target Future State (THE GOAL)
        # For a reactor, the goal is often to keep k_eff at 1.0 (criticality).
        # We will create this tensor manually as our "plan" for the future.
        # It needs to have shape [L, F1]
        future_state_target = torch.ones(output_pred_len, len(state_cols), dtype=torch.float32)

        # TENSOR 4: The ground-truth future control action, used for "teacher forcing".
        # This is what the model's output will be compared against in the loss function.
        future_control_prompt = torch.tensor(full_data[control_cols].iloc[start_target:end_target].values, dtype=torch.float32)

        samples.append((past_state, past_control, future_state_target, future_control_prompt))
        
    return samples

# ==============================================================================
# 3. PYTORCH DATASET WRAPPER
# ==============================================================================

class ReactorDataset(Dataset):
    """
    Wraps the list of samples for use with a PyTorch DataLoader.
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]

# ==============================================================================
# 4. DEMONSTRATION OF HOW TO USE THE CODE
# ==============================================================================

if __name__ == "__main__":
    
    # --- Configuration for our Fission Problem ---
    INPUT_SEQ_LEN = 20      # Model looks at 20 past time steps
    OUTPUT_PRED_LEN = 4     # Model must predict 4 future control rod positions
    BATCH_SIZE = 32
    
    # Define which columns are state and which are control
    STATE_COLUMNS = ['k_eff']
    CONTROL_COLUMNS = ['CR_Pos']
    
    print("--- 1. Raw Data ---")
    print("The first 5 rows of the raw reactor data:")
    print(df.head())
    print(f"\nTotal number of time steps in raw data: {len(df)}")
    print("-" * 50)

    print("--- 2. Creating Sliding Windows ---")
    all_samples = create_fission_sliding_windows(
        full_data=df,
        input_seq_len=INPUT_SEQ_LEN,
        output_pred_len=OUTPUT_PRED_LEN,
        state_cols=STATE_COLUMNS,
        control_cols=CONTROL_COLUMNS
    )
    print(f"Total number of training samples created: {len(all_samples)}")
    print("-" * 50)

    print("--- 3. Verifying the Shape of One Sample ---")
    # Let's inspect the very first sample to ensure all four tensors are correct.
    first_sample = all_samples[0]
    p_state, p_ctrl, f_state, f_ctrl = first_sample
    
    print(f"Shape of past_state:         {p_state.shape}")
    print(f"   -> Expected: [{INPUT_SEQ_LEN}, Num_State_Features]")
    
    print(f"\nShape of past_control:       {p_ctrl.shape}")
    print(f"   -> Expected: [{INPUT_SEQ_LEN}, Num_Control_Features]")

    print(f"\nShape of future_state_target: {f_state.shape}")
    print(f"   -> Expected: [{OUTPUT_PRED_LEN}, Num_State_Features]")
    
    print(f"\nShape of future_control_prompt: {f_ctrl.shape}")
    print(f"   -> Expected: [{OUTPUT_PRED_LEN}, Num_Control_Features]")
    print("-" * 50)

    print("--- 4. Using the DataLoader to Create a Batch ---")
    # This prepares the data to be fed into the model during training.
    reactor_dataset = ReactorDataset(all_samples)
    data_loader = DataLoader(reactor_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get the first batch to see its final shape
    first_batch = next(iter(data_loader))
    batch_past_state, batch_past_control, batch_future_state, batch_future_control = first_batch
    
    print("The DataLoader takes individual samples and adds a 'Batch' dimension.")
    print(f"Final shape of batch_past_state:         {batch_past_state.shape}")
    print(f"Final shape of batch_past_control:       {batch_past_control.shape}")
    print(f"Final shape of batch_future_state_target: {batch_future_state.shape}")
    print(f"Final shape of batch_future_control_prompt: {batch_future_control.shape}")
    print("\nData is now perfectly formatted to be passed to the FissionReactorTransformer.forward method.")
    print("-" * 50)
