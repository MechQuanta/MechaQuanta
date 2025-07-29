import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict

# ======================================================================================
# 1. HELPER MODULES (Standard implementations to make the code runnable)
# ======================================================================================

class PositionalEncoding(nn.Module):
    """Standard Positional Encoding from the PyTorch tutorials."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class NoiseLayer(nn.Module):
    """Adds Gaussian noise to the input tensor."""
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x

class GELU(nn.Module):
    """Gaussian Error Linear Unit activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

class CNNencoder(nn.Module):
    """A simple 1D CNN block used for embedding."""
    def __init__(self, in_channels, out_channels, kernel_size, padding, is_last_layer):
        super().__init__()
        # This is a simplified version. The original code likely had more layers.
        # This version just demonstrates the principle.
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv_block(x)

# ======================================================================================
# 2. THE MAIN TRANSFORMER MODEL ARCHITECTURE
# ======================================================================================

class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        RIN: bool = False,
        input_0D_dim: int = 12,
        input_0D_seq_len: int = 20,
        input_ctrl_dim: int = 14,
        input_ctrl_seq_len: int = 24,
        output_0D_pred_len: int = 4,
        output_0D_dim: int = 12,
        feature_dim: int = 128,
        noise_mean: float = 0,
        noise_std: float = 0.01,
        kernel_size: int = 3,
        ):

        super(Transformer, self).__init__()

        # --- Store configuration ---
        self.input_0D_dim = input_0D_dim
        self.input_ctrl_dim = input_ctrl_dim
        self.feature_dim = feature_dim
        self.RIN = RIN
        self.input_seq_len = min(input_0D_seq_len, input_ctrl_seq_len)

        # --- Initialize Layers ---
        self.noise = NoiseLayer(mean=noise_mean, std=noise_std)
        padding = (kernel_size - 1) // 2

        # 1. Convolutional Encoders for feature extraction
        self.encoder_input_0D = CNNencoder(input_0D_dim, feature_dim // 2, kernel_size, padding, False)
        self.encoder_input_ctrl = CNNencoder(input_ctrl_dim, feature_dim // 2, kernel_size, padding, True)

        # 2. Transformer Encoder for contextualization
        self.pos = PositionalEncoding(d_model=feature_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation=GELU(),
            batch_first=False # IMPORTANT: PyTorch's default is [Seq, Batch, Dim]
        )
        self.trans_enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.src_mask = None # Will be created in forward pass

        # 3. Fully Connected (Linear) Decoder for forecasting
        self.lc_seq = nn.Linear(self.input_seq_len, output_0D_pred_len)
        self.lc_feat = nn.Linear(feature_dim, output_0D_dim)

        # 4. Reversible Instance Normalization Parameters
        if self.RIN:
            self.affine_weight_0D = nn.Parameter(torch.ones(1, 1, input_0D_dim))
            self.affine_bias_0D = nn.Parameter(torch.zeros(1, 1, input_0D_dim))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.lc_feat.bias.data.zero_()
        self.lc_feat.weight.data.uniform_(-initrange, initrange)
        self.lc_seq.bias.data.zero_()
        self.lc_seq.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, size: int):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x_0D: torch.Tensor, x_ctrl: torch.Tensor):
        # Truncate to the shortest sequence length
        x_0D = x_0D[:, :self.input_seq_len, :]
        x_ctrl = x_ctrl[:, :self.input_seq_len, :]

        # Optional: Add noise to the input for robust performance
        x_0D = self.noise(x_0D)

        # Reversible Instance Normalization (RIN)
        if self.RIN:
            means_0D = x_0D.mean(1, keepdim=True).detach()
            stdev_0D = torch.sqrt(torch.var(x_0D, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_0D_normalized = (x_0D - means_0D) / stdev_0D
            x_0D = x_0D_normalized * self.affine_weight_0D + self.affine_bias_0D

        # --- Stage 1: Feature Extraction ---
        # Note the permute: Conv1d expects [Batch, Channels, SeqLen]
        x_0D_emb = self.encoder_input_0D(x_0D.permute(0, 2, 1)).permute(0, 2, 1)
        x_ctrl_emb = self.encoder_input_ctrl(x_ctrl.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([x_0D_emb, x_ctrl_emb], dim=2) # Shape: [B, S, feature_dim]

        # --- Stage 2: Contextualization ---
        # Permute to [SeqLen, Batch, Dim] for the Transformer Encoder
        x = x.permute(1, 0, 2)
        
        # Generate the causal mask. This is used to make the model autoregressive.
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            self.src_mask = self._generate_square_subsequent_mask(len(x)).to(x.device)

        x = self.pos(x)
        x = self.trans_enc(x, self.src_mask)
        
        # Permute back to [Batch, SeqLen, Dim]
        x = x.permute(1, 0, 2)

        # --- Stage 3: Forecasting ---
        # Apply the sequence reducer: permute to apply Linear layer to the time axis
        x = x.permute(0, 2, 1) # Shape: [B, Dim, SeqLen]
        x = self.lc_seq(x)     # Shape: [B, Dim, PredLen]
        x = torch.nn.functional.relu(x)
        
        # Apply the feature reducer: permute back to apply Linear layer to the feature axis
        x = x.permute(0, 2, 1) # Shape: [B, PredLen, Dim]
        x = self.lc_feat(x)    # Shape: [B, PredLen, OutputDim]

        # De-Normalization if RIN was used
        if self.RIN:
            x = x - self.affine_bias_0D
            x = x / (self.affine_weight_0D + 1e-6)
            x = x * stdev_0D
            x = x + means_0D

        # Clamping and stability
        x = torch.clamp(x, min=-10.0, max=10.0)
        x = torch.nan_to_num(x, nan=0)

        return x

# ======================================================================================
# 3. DATA HANDLING (Dataset Class and Sliding Window)
# ======================================================================================

def create_sliding_windows(data_0D, data_ctrl, input_seq_len, output_pred_len):
    """
    Takes a long time series and creates smaller input/target windows.
    This is the core data preparation step.
    """
    samples = []
    num_timesteps = len(data_0D)

    for i in range(num_timesteps - input_seq_len - output_pred_len + 1):
        # The input window is the historical data
        input_window_0D = data_0D[i : i + input_seq_len]
        input_window_ctrl = data_ctrl[i : i + input_seq_len]
        
        # The target window is the future data we want to predict
        target_window = data_0D[i + input_seq_len : i + input_seq_len + output_pred_len]

        samples.append((input_window_0D, input_window_ctrl, target_window))
        
    return samples

class PlasmaDataset(Dataset):
    """Custom PyTorch Dataset for the plasma data."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Fetches one sample (input_0D, input_ctrl, target)
        return self.samples[idx]

# ======================================================================================
# 4. TRAINING AND VALIDATION LOOPS
# ======================================================================================

def train_per_epoch(train_loader, model, optimizer, loss_fn, device, max_norm_grad=1.0):
    model.train() # Set model to training mode
    total_loss = 0

    for batch_idx, (data_0D, data_ctrl, target) in enumerate(train_loader):
        # Move data to the selected device (GPU or CPU)
        data_0D = data_0D.to(device)
        data_ctrl = data_ctrl.to(device)
        target = target.to(device)
        
        if data_0D.size(0) <= 1: # Safety check for BatchNorm
            continue

        # --- Forward pass ---
        optimizer.zero_grad() # Reset gradients from previous batch
        output = model(data_0D, data_ctrl)
        
        # --- Loss calculation ---
        loss = loss_fn(output, target)
        
        if not torch.isfinite(loss):
            print("Warning: Loss is NaN or Inf, skipping batch.")
            continue # Skip this batch
        
        # --- Backward pass and optimization ---
        loss.backward() # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad) # Prevent exploding gradients
        optimizer.step() # Update model weights

        total_loss += loss.item()

    return total_loss / (batch_idx + 1)

def validate_per_epoch(val_loader, model, loss_fn, device):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad(): # Disable gradient calculation for efficiency
        for batch_idx, (data_0D, data_ctrl, target) in enumerate(val_loader):
            data_0D = data_0D.to(device)
            data_ctrl = data_ctrl.to(device)
            target = target.to(device)

            output = model(data_0D, data_ctrl)
            loss = loss_fn(output, target)
            total_loss += loss.item()

    return total_loss / (batch_idx + 1)

# ======================================================================================
# 5. MAIN EXECUTION BLOCK
# ======================================================================================

if __name__ == "__main__":
    
    # --- Hyperparameters and Configuration ---
    # Data params
    INPUT_0D_DIM = 12
    INPUT_CTRL_DIM = 14
    INPUT_SEQ_LEN = 20
    OUTPUT_PRED_LEN = 4

    # Model params
    FEATURE_DIM = 128
    N_HEADS = 8
    N_LAYERS = 3
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    USE_RIN = True # Set to True to test Reversible Instance Norm

    # Training params
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    
    # --- Setup Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Generate Synthetic Dummy Data ---
    # This simulates one long plasma experiment with 1000 time steps
    print("Generating synthetic data...")
    total_timesteps = 1000
    dummy_data_0D = torch.randn(total_timesteps, INPUT_0D_DIM)
    dummy_data_ctrl = torch.randn(total_timesteps, INPUT_CTRL_DIM)

    # --- Create Sliding Windows for Training and Validation ---
    print("Creating sliding windows...")
    all_samples = create_sliding_windows(dummy_data_0D, dummy_data_ctrl, INPUT_SEQ_LEN, OUTPUT_PRED_LEN)
    
    # Split into training and validation sets
    train_size = int(0.8 * len(all_samples))
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:]
    print(f"Total samples: {len(all_samples)}, Train: {len(train_samples)}, Validation: {len(val_samples)}")

    # --- Create Datasets and DataLoaders ---
    train_dataset = PlasmaDataset(train_samples)
    val_dataset = PlasmaDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Initialize Model, Loss, and Optimizer ---
    print("Initializing model...")
    model = Transformer(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        RIN=USE_RIN,
        input_0D_dim=INPUT_0D_DIM,
        input_0D_seq_len=INPUT_SEQ_LEN,
        input_ctrl_dim=INPUT_CTRL_DIM,
        input_ctrl_seq_len=INPUT_SEQ_LEN, # Use same length for simplicity
        output_0D_pred_len=OUTPUT_PRED_LEN,
        output_0D_dim=INPUT_0D_DIM, # Predict the same state variables
        feature_dim=FEATURE_DIM,
    ).to(device)

    loss_fn = nn.MSELoss() # Mean Squared Error is a good choice for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- The Main Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_per_epoch(train_loader, model, optimizer, loss_fn, device)
        val_loss = validate_per_epoch(val_loader, model, loss_fn, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')
            print(f"   -> New best model saved with validation loss: {best_val_loss:.6f}")
    
    print("\nTraining finished!")
