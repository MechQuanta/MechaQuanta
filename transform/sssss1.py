import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import time

# --- Model and Helper Classes ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # ADDED: A check to ensure d_model is even
        if d_model % 2 != 0:
            raise ValueError(f"d_model (feature_size) must be even, but got {d_model}")
            
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [sequence_length, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    # CHANGED: init signature to decouple input features from model dimension
    def __init__(self, input_features, d_model=256, num_layers=1, nhead=8, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # ADDED: Input embedding layer to project input_features to d_model
        self.input_embedding = nn.Linear(input_features, d_model)
        
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # The decoder projects from d_model back to the desired output feature count
        self.decoder = nn.Linear(d_model, input_features)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.input_embedding.weight.data.uniform_(-initrange, initrange)


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src shape: [batch_size, sequence_length, input_features]
        
        # CHANGED: Pass through embedding layer first
        src = self.input_embedding(src) # Now shape: [batch_size, seq_len, d_model]
        
        # Transformer expects: [sequence_length, batch_size, d_model]
        src = src.permute(1, 0, 2)
        
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask.to(device)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        
        # Permute back to [batch_size, sequence_length, features] for loss calculation
        output = output.permute(1, 0, 2)
        return output

# --- Data Processing Functions (Unchanged) ---
def create_inout_sequences(input_data, tw, output_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - output_window + 1):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw : i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def get_data(data, split, input_window, output_window):
    series = data.values
    split_idx = round(split * len(series))
    train_data = series[:split_idx]
    test_data = series[split_idx:]
    train_data = 2 * train_data
    train_seq = create_inout_sequences(train_data, input_window, output_window)
    test_seq = create_inout_sequences(test_data, input_window, output_window)
    return train_seq, test_seq

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - i)
    batch_data = source[i:i + seq_len]
    inputs = torch.stack([torch.FloatTensor(item[0]) for item in batch_data])
    targets = torch.stack([torch.FloatTensor(item[1]) for item in batch_data])
    return inputs, targets

# --- Training and Evaluation Functions ---
def train_epoch(model, train_data, optimizer, criterion, scheduler, batch_size, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    loss = []
    
    for batch_idx, i in enumerate(range(0, len(train_data), batch_size)):
        if i + batch_size > len(train_data): continue
            
        data, targets = get_batch(train_data, i, batch_size)
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(data)
        
        # Compare the final output to the target
        # output shape: [batch, seq_len, features]
        # target shape: [batch, output_window, features]
        loss = criterion(output[:, -1:, :], targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        
        total_loss += loss.item()
        
        log_interval = max(1, (len(train_data) // batch_size) // 5)
        if batch_idx > 0 and batch_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f"Batch {batch_idx:5d}/{len(train_data)//batch_size:5d} | "
                  f"lr {scheduler.get_lr()[0]:02.10f} | "
                  f"{elapsed*1000/log_interval:5.2f} ms | "
                  f"loss {cur_loss:5.7f}")
            total_loss = 0.0
            start_time = time.time()

def evaluate(model, data_source, criterion, batch_size, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(data_source), batch_size):
            if i + batch_size > len(data_source): continue
            data, targets = get_batch(data_source, i, batch_size)
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            total_loss += criterion(output[:, -1:, :], targets).item()
    return total_loss / (len(data_source) / batch_size)

# --- Main Execution ---

# Define your parameters
input_window = 50
output_window = 1
batch_size = 32
num_epochs = 10
split_ratio = 0.8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Data
df = pd.DataFrame(np.random.rand(1000, 3), columns=['A', 'B', 'C'])

# Prepare data
train_data, test_data = get_data(df, split_ratio, input_window, output_window)

# Instantiate the model
input_features = df.shape[1]
d_model = 32  # Model's internal dimension - MUST be even
n_heads = 8   # Number of attention heads - MUST be a divisor of d_model
model = TransformerModel(input_features=input_features, d_model=d_model, nhead=n_heads, num_layers=2).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# --- Training Loop ---
print("Starting Training...")
for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    train_epoch(model, train_data, optimizer, criterion, scheduler, batch_size, device)
    val_loss = evaluate(model, test_data, criterion, batch_size, device)
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | valid loss {val_loss:5.7f}')
    print('-' * 89)
    scheduler.step()
