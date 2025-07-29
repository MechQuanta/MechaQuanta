import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
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
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    """
    Transformer model for time series prediction
    """
    def __init__(self, n_features=3, d_model=64, nhead=8, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, output_window=1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.n_features = n_features
        self.output_window = output_window
        
        # Input projection layer
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_features * output_window)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection[0].weight.data.uniform_(-initrange, initrange)
        self.output_projection[3].weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None):
        """
        Forward pass
        src: (batch_size, seq_len, n_features)
        """
        # Transpose to (seq_len, batch_size, n_features) for transformer
        src = src.transpose(0, 1)
        seq_len, batch_size, _ = src.shape
        
        # Project input to d_model dimensions
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Apply transformer encoder
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)
        
        output = self.transformer_encoder(src, src_mask)
        
        # Use only the last time step for prediction
        output = output[-1, :, :]  # (batch_size, d_model)
        
        # Project to output dimensions
        output = self.output_projection(output)  # (batch_size, n_features * output_window)
        
        # Reshape to (batch_size, output_window, n_features)
        output = output.view(batch_size, self.output_window, self.n_features)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for transformer"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransformerTrainer:
    """
    Training wrapper for the transformer model
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def train_model(self, train_data, val_data, num_epochs=100, learning_rate=0.001):
        """
        Complete training loop
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self._train_epoch(train_data, optimizer, criterion)
            
            # Validation
            val_loss = self._evaluate(val_data, criterion)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_transformer_model.pth')
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}')
    
    def _train_epoch(self, train_data, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        batch_size = 32
        
        for i in range(0, len(train_data) - batch_size, batch_size):
            data, targets = self._get_batch(train_data, i, batch_size)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / (len(train_data) // batch_size)
    
    def _evaluate(self, data_source, criterion):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(data_source) - batch_size, batch_size):
                data, targets = self._get_batch(data_source, i, batch_size)
                output = self.model(data)
                loss = criterion(output, targets)
                total_loss += loss.item()
        
        return total_loss / (len(data_source) // batch_size)
    
    def _get_batch(self, source, i, batch_size):
        """Get batch data"""
        seq_len = min(batch_size, len(source) - i)
        batch_data = source[i:i+seq_len]
        
        inputs = torch.stack([torch.FloatTensor(item[0]) for item in batch_data])
        targets = torch.stack([torch.FloatTensor(item[1]) for item in batch_data])
        
        return inputs.to(self.device), targets.to(self.device)
    
    def predict(self, data):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            data = data.unsqueeze(0).to(self.device)  # Add batch dimension
            prediction = self.model(data)
            return prediction.cpu().numpy()

# Usage example and complete pipeline
def create_and_train_transformer(close_logreturn, train_split=0.6):
    """
    Complete pipeline to create and train transformer
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Prepare data (using the functions from previous artifact)
    from transformer_data_prep import get_data
    train_data, val_data = get_data(close_logreturn, train_split)
    
    print(f'Training sequences: {len(train_data)}')
    print(f'Validation sequences: {len(val_data)}')
    
    # Create model
    model = TransformerModel(
        n_features=3,          # Your 3 features
        d_model=64,            # Model dimension
        nhead=8,               # Number of attention heads
        num_layers=3,          # Number of transformer layers
        dim_feedforward=256,   # Feedforward dimension
        dropout=0.1,           # Dropout rate
        output_window=1        # Predict 1 step ahead
    )
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create trainer
    trainer = TransformerTrainer(model, device)
    
    # Train model
    trainer.train_model(train_data, val_data, num_epochs=100, learning_rate=0.001)
    
    return trainer

# Example usage:
"""
# Assuming you have your close_logreturn data (1000, 3)
trainer = create_and_train_transformer(close_logreturn)

# Make predictions on new data
# new_sequence shape should be (sequence_length, 3)
prediction = trainer.predict(new_sequence)
print(f'Prediction shape: {prediction.shape}')  # Should be (1, 1, 3)
"""
