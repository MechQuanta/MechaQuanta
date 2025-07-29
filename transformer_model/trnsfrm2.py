import torch, math
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional, Dict
from pytorch_model_summary import summary

# Transformer model - Modified for 3 input features
class NoiseLayer(nn.Module):
    def __init__(self, mean : float = 0, std : float = 1e-2):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x : torch.Tensor):
        if self.training:
            noise = Variable(torch.ones_like(x).to(x.device) * self.mean + torch.randn(x.size()).to(x.device) * self.std)
            return x + noise
        else:
            return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, max_len : int = 5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1) # (max_len, 1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() # (d_model // 2, )
        pe[:,0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:,1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1) # shape : (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x:torch.Tensor):
        # x : (seq_len, batch_size, n_features)
        return x + self.pe[:x.size(0), :, :]

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class CNNencoder(nn.Module):
    def __init__(self, input_dim : int, feature_dim : int, kernel_size : int, padding : int, reduction : bool):
        super().__init__()
        dk = 1 if reduction else 0
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=feature_dim, kernel_size= kernel_size + dk, stride = 1, padding = padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size= kernel_size, stride = 1, padding = padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
    def forward(self, x : torch.Tensor):
        return self.encoder(x)

class Transformer(nn.Module):
    def __init__(
        self, 
        n_layers : int = 4, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1,        
        RIN : bool = False,
        input_0D_dim : int = 2,  # days, k_eff, cr_pos
        input_0D_seq_len : int = 20,
        input_ctrl_dim : int = 1,  # Keep for compatibility, but won't be used much
        input_ctrl_seq_len : int = 4,
        output_0D_pred_len : int = 4,
        output_0D_dim : int = 1,  # Predicting only cr_pos
        feature_dim : int = 128,
        range_info : Optional[Dict] = None,
        noise_mean : float = 0,
        noise_std : float = 0.1,
        kernel_size : int = 3,
        use_cnn : bool = False,  # New parameter to enable/disable CNN
        ):
        
        super(Transformer, self).__init__()
        
        # input information
        self.input_0D_dim = input_0D_dim
        self.input_0D_seq_len = input_0D_seq_len
        self.input_ctrl_dim = input_ctrl_dim
        self.input_ctrl_seq_len = input_ctrl_seq_len
        
        # output information
        self.output_0D_pred_len = output_0D_pred_len
        self.output_0D_dim = output_0D_dim
        
        # source mask
        self.src_mask = None
        
        # transformer info
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_cnn = use_cnn
        
        self.RIN = RIN
        
        self.noise = NoiseLayer(mean = noise_mean, std = noise_std)
        
        # Use input_0D_seq_len as the main sequence length since we're focusing on 0D data
        self.input_seq_len = input_0D_seq_len
        
        if kernel_size % 2 == 0:
            print("kernel should be odd number")
            kernel_size += 1
            
        padding = (kernel_size - 1) // 2
        
        if self.use_cnn:
            # CNN approach: Separate encoders for different parts of the data
            self.encoder_input_0D = CNNencoder(input_0D_dim, feature_dim, kernel_size, padding, False)
            # We can skip the ctrl encoder or make it minimal
            self.encoder_input_ctrl = CNNencoder(1, feature_dim//4, kernel_size, padding, False) # Minimal ctrl encoder
        else:
            # Linear approach: Direct linear transformation from input features to model dimension
            self.input_projection = nn.Linear(input_0D_dim, feature_dim)
        
        self.pos = PositionalEncoding(d_model = feature_dim, max_len = self.input_seq_len)
        self.enc = nn.TransformerEncoderLayer(
            d_model = feature_dim, 
            nhead = n_heads, 
            dropout = dropout,
            dim_feedforward = dim_feedforward,
            activation = GELU()
        )
        
        self.trans_enc = nn.TransformerEncoder(self.enc, num_layers=n_layers)
        
        # FC decoder
        # sequence length reduction
        self.lc_seq = nn.Linear(self.input_seq_len, output_0D_pred_len)
        
        # dimension reduction
        self.lc_feat= nn.Linear(feature_dim, output_0D_dim)
        
        # Reversible Instance Normalization
        if self.RIN:
            self.affine_weight_0D = nn.Parameter(torch.ones(1, 1, input_0D_dim))
            self.affine_bias_0D = nn.Parameter(torch.zeros(1, 1, input_0D_dim))
            
        self.range_info = range_info
        
        if range_info:
            self.register_buffer('range_min', torch.Tensor([range_info[key][0] * 0.1 for key in range_info.keys()]))
            self.register_buffer('range_max', torch.Tensor([range_info[key][1] * 10.0 for key in range_info.keys()]))
        else:
            self.range_min = None
            self.range_max = None
            
        # initialize
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1    
        self.lc_feat.bias.data.zero_()
        self.lc_feat.weight.data.uniform_(-initrange, initrange)
        
        self.lc_seq.bias.data.zero_()
        self.lc_seq.weight.data.uniform_(-initrange, initrange)
        
        if not self.use_cnn:
            self.input_projection.bias.data.zero_()
            self.input_projection.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x_0D : torch.Tensor, x_ctrl : torch.Tensor):
        # x_0D shape: (batch_size, seq_len, 3) - [days, k_eff, cr_pos]
        # x_ctrl is kept for compatibility but not used much in this approach
        
        # add noise to robust performance
        x_0D = self.noise(x_0D)
        
        if self.RIN:
            means_0D = x_0D.mean(1, keepdim=True).detach()
            x_0D = x_0D - means_0D
            stdev_0D = torch.sqrt(torch.var(x_0D, dim=1, keepdim=True, unbiased=False) + 1e-3).detach()
            x_0D /= stdev_0D
            x_0D = x_0D * self.affine_weight_0D + self.affine_bias_0D
            
        if self.use_cnn:
            # CNN approach: encoding : (N, T, F) -> (N, T, d_model)
            x_0D_encoded = self.encoder_input_0D(x_0D.permute(0,2,1)).permute(0,2,1)
            
            # For compatibility, we can still use a minimal ctrl input if needed
            # But since our main data is in x_0D, we'll focus on that
            x = x_0D_encoded
        else:
            # Linear approach: Direct projection to feature dimension
            # x_0D: (batch_size, seq_len, input_0D_dim) -> (batch_size, seq_len, feature_dim)
            x = self.input_projection(x_0D)
        
        # (T, N, d_model) - Transformer expects seq_len first
        x = x.permute(1,0,2)
    
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            self.src_mask = self._generate_square_subsequent_mask(len(x)).to(x.device)
            
        # positional encoding for time axis : (T, N, d_model)
        x = self.pos(x)
        
        # transformer encoding layer : (T, N, d_model)
        x = self.trans_enc(x, self.src_mask)
        
        # (N, T, d_model)
        x = x.permute(1,0,2)
    
        # Sequence length reduction: (N, T, d_model) -> (N, pred_len, d_model)
        # (N, d_model, T) -> (N, d_model, pred_len)
        x = self.lc_seq(x.permute(0,2,1))
        x = torch.nn.functional.relu(x)
        
        # (N, pred_len, d_model)
        x = x.permute(0,2,1)
        
        # dimension reduction: (N, pred_len, d_model) -> (N, pred_len, output_dim)
        x = self.lc_feat(x)
    
        # RevIN for considering data distribution shift
        if self.RIN:
            # For output, we only reverse the normalization for the output dimension
            # Since we're predicting cr_pos, we use its statistics
            if self.input_0D_dim >= 3:  # If cr_pos was in input
                cr_pos_mean = means_0D[:, :, 2:3]  # cr_pos mean
                cr_pos_stdev = stdev_0D[:, :, 2:3]  # cr_pos stdev
                affine_weight = self.affine_weight_0D[:, :, 2:3]
                affine_bias = self.affine_bias_0D[:, :, 2:3]
                
                x = x - affine_bias
                x = x / (affine_weight + 1e-6)
                x = x * cr_pos_stdev
                x = x + cr_pos_mean
        
        # clamping : output range
        x = torch.clamp(x, min = -10.0, max = 10.0)
        
        # remove nan value for stability
        x = torch.nan_to_num(x, nan = 0)
        
        return x

    def _generate_square_subsequent_mask(self, size : int):
        mask = (torch.triu(torch.ones(size,size))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def summary(self):
        sample_0D = torch.zeros((1, self.input_0D_seq_len, self.input_0D_dim))
        sample_ctrl = torch.zeros((1, self.input_ctrl_seq_len, self.input_ctrl_dim))
        summary(self, sample_0D, sample_ctrl, batch_size = 1, show_input = True, print_summary=True)
