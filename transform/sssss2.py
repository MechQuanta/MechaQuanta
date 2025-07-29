import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import time

# --- Model Classes ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    def __init__(self, feature_size=3, d_model=60, num_layers=2, nhead=6, dropout=0.1):
        super(Transformer, self).__init__()
        self.model_type = "Transformer"
        
        # Input projection to match d_model
        self.input_projection = nn.Linear(feature_size, d_model)
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Output projection back to feature size
        self.decoder = nn.Linear(d_model, feature_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: [seq_len, batch_size, feature_size]
        
        # Project input features to d_model
        src = self.input_projection(src)  # [seq_len, batch_size, d_model]
        
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)  # [seq_len, batch_size, feature_size]
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

# --- Data Processing Functions ---
def create_inout_sequences(input_data, tw, output_window):
    """
    Create input-output sequences for training
    input_data: numpy array of shape (n_samples, n_features)
    tw: time window (input sequence length)
    output_window: prediction horizon
    """
    inout_seq = []
    L = len(input_data)
    
    for i in range(L - tw - output_window + 1):
        train_seq = input_data[i:i+tw]  # Input sequence
        train_label = input_data[i+tw:i+tw+output_window]  # Target sequence
        inout_seq.append((train_seq, train_label))
    
    return inout_seq

def get_data(data, split, input_window, output_window):
    """
    Prepare training and test data
    data: pandas DataFrame or numpy array (1000, 3)
    """
    if isinstance(data, pd.DataFrame):
        series = data.values
    else:
        series = data
    
    split_idx = round(split * len(series))
    train_data = series[:split_idx]
    test_data = series[split_idx:]

    # Apply cumulative sum and scaling
    #train_data = train_data.cumsum(axis=0)
    train_data = 2 * train_data  # Data augmentation
    
    #test_data = test_data.cumsum(axis=0)

    # Create sequences
    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    test_sequence = create_inout_sequences(test_data, input_window, output_window)

    return train_sequence, test_sequence

def get_batch(source, i, batch_size):
    """
    Get batch data for training
    source: list of (input, target) tuples
    """
    seq_len = min(batch_size, len(source) - i)
    batch_data = source[i:i + seq_len]
    
    # Extract inputs and targets
    inputs = torch.stack([torch.FloatTensor(item[0]) for item in batch_data])
    targets = torch.stack([torch.FloatTensor(item[1]) for item in batch_data])
    
    # Reshape for transformer: [seq_len, batch_size, features]
    inputs = inputs.transpose(0, 1)  # [input_window, batch_size, 3]
    targets = targets.transpose(0, 1)  # [output_window, batch_size, 3]
    
    return inputs, targets

# --- Training Functions ---
import matplotlib.pyplot as plt

def train_epoch(model, train_data, optimizer, criterion, scheduler, batch_size, device, epoch, train_losses):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - batch_size, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Use only the last output for prediction
        loss = criterion(output[-1:], targets)  # Compare last output with target
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        train_losses.append(loss.item())  # Add loss for each step
        
        log_interval = max(1, len(train_data) // batch_size // 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print("| epoch {:3d} | {:5d}/{:5d} batches | "
                  "lr {:02.10f} | {:5.2f} ms | "
                  "loss {:5.7f}".format(
                    epoch, batch, len(train_data) // batch_size, 
                    scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source, criterion, device):
    eval_model.eval()
    total_loss = 0.0
    eval_batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(data_source) - eval_batch_size, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            data, targets = data.to(device), targets.to(device)
            output = eval_model(data)
            loss = criterion(output[-1:], targets)
            total_loss += loss.item()
    
    return total_loss / (len(data_source) // eval_batch_size)

def predict_sequence(model, input_sequence, device, steps=1):
    """
    Make predictions on new data
    input_sequence: numpy array of shape (sequence_length, 3)
    """
    model.eval()
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        seq = torch.FloatTensor(input_sequence).unsqueeze(1).to(device)  # [seq_len, 1, 3]
        
        predictions = []
        for _ in range(steps):
            output = model(seq)
            pred = output[-1:]  # Last output
            predictions.append(pred.cpu().numpy())
            
            # Use prediction as input for next step (for multi-step prediction)
            seq = torch.cat([seq[1:], pred], dim=0)
    
    return np.array(predictions).squeeze()

# --- Main Training Script ---
def main():
    # Parameters
    input_window = 10
    output_window = 4
    batch_size = 32 # Reduced for 1000 samples
    lr = 0.0001  # Slightly higher learning rate
    epochs = 100
    split_ratio = 0.7  # 70% train, 30% test
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create or load your 1000x3 data matrix
    # Replace this with your actual data loading
    close_logreturn = pd.DataFrame(np.random.randn(1000, 3), columns=['feature_1', 'feature_2', 'feature_3'])
    
    print(f"Data shape: {close_logreturn.shape}")
    
    # Prepare data
    train_data, val_data = get_data(close_logreturn, split_ratio, input_window, output_window)
    
    print(f"Training sequences: {len(train_data)}")
    print(f"Validation sequences: {len(val_data)}")
    
    # Create model
    model = Transformer(
        feature_size=3,      # Your 3 features
        d_model=60,          # Must be divisible by nhead
        num_layers=2,
        nhead=6,             # 60/6 = 10, good ratio
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    # Initialize loss tracking
    train_losses = []
    val_losses = []
    
    # Training loop
    print("Starting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Training
        train_epoch(model, train_data, optimizer, criterion, scheduler, batch_size, device, epoch, train_losses)
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            val_loss = evaluate(model, val_data, criterion, device)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_transformer_1000x3.pth')
            
            print("-" * 80)
            print("| end of epoch {:3d} | time: {:5.2f}s | valid loss: {:5.7f} | best: {:5.7f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, best_val_loss))
            print("-" * 80)
        else:
            print("-" * 80)
            print("| end of epoch {:3d} | time: {:5.2f}s".format(epoch, (time.time() - epoch_start_time)))
            print("-" * 80)

        scheduler.step()
    
    print(f"Training completed! Best validation loss: {best_val_loss:.7f}")
    
    # Simple loss plotting
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    val_epochs = list(range(10, epochs + 1, 10))  # Every 10 epochs
    plt.plot(val_epochs, val_losses)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Example prediction
    print("\nMaking sample prediction...")
    sample_input = close_logreturn.iloc[-input_window:].values  # Last 10 rows
    prediction = predict_sequence(model, sample_input, device)
    print(f"Input shape: {sample_input.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction: {prediction}")
    
    return model

if __name__ == "__main__":
    model = main()
