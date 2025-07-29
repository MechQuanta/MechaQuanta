import torch, math
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional, Dict, List
from pytorch_model_summary import summary
from src.nn_env.transformer import PositionalEncoding, NoiseLayer, GELU

# De-stationary attention module
class DSAttention(nn.Module):
    def __init__(self, mask_flag : bool = True, factor : float = 5, scale = None, dropout : float = 0.1, output_attention : bool = False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, att_mask : torch.Tensor, tau : Optional[torch.Tensor]= None, delta : Optional[torch.Tensor] = None):
        B, L, H, E = q.size()
        _, S, _, D = v.size()
        scale = self.scale or 1.0 / math.sqrt(E)
        
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
        
        scores = torch.einsum("blhe,bshe->bhls", q, k) * tau + delta
        
        if self.mask_flag and att_mask is not None:
            scores.masked_fill_(att_mask.bool(), -torch.inf)
        
        A = self.dropout(torch.softmax(scale * scores, dim = -1))
        V = torch.einsum("bhls,bshd->blhd", A, v)
        
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention : DSAttention, d_model : int, n_heads : int, d_keys : Optional[int] = None, d_values : Optional[int]= None): 
        super().__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.q_proj = nn.Linear(d_model, d_keys * n_heads)
        self.k_proj = nn.Linear(d_model, d_keys * n_heads)
        self.v_proj = nn.Linear(d_model, d_values * n_heads)
        self.out_proj = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, att_mask : torch.Tensor,tau : Optional[torch.Tensor]= None, delta : Optional[torch.Tensor] = None):
        B, L, _ = q.size()
        _, S, _ = k.size()
        H = self.n_heads
        
        q = self.q_proj(q).view(B, L, H, -1)
        k = self.k_proj(k).view(B, S, H, -1)
        v = self.v_proj(v).view(B, S, H, -1)
        
        out, att = self.inner_attention(q,k,v,att_mask,tau,delta)
        out = out.view(B,L,-1)
        out = self.out_proj(out)
        
        return out, att
        
class EncoderLayer(nn.Module):
    def __init__(self, attention : AttentionLayer, d_model : int, d_ff : int, dropout : float = 0.1):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
        
    def forward(self, x : torch.Tensor, att_mask : torch.Tensor, tau : Optional[torch.Tensor]=None, delta : Optional[torch.Tensor] = None):
        x_n, attn = self.attention(x,x,x,att_mask,tau,delta)
        x = self.dropout(x_n) + x
        x = self.norm1(x)
        out = self.dropout(self.activation(self.conv1(x.transpose(-1,1))))
        out = self.dropout(self.conv2(out).transpose(-1,1))
        out = self.norm2(out + x)
        return out, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers:List, norm_layer : Optional[nn.Module]=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
    
    def forward(self, x : torch.Tensor, attn_mask : torch.Tensor, tau : Optional[torch.Tensor]=None, delta : Optional[torch.Tensor] = None):
        # x : (B, L, D)
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask, tau, delta)
            attns.append(attn)
        if self.norm:
            x = self.norm(x)
        return x, attns
    
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):

        x = x + self.dropout(self.self_attention(
            x, x, x,
            att_mask=x_mask,
            tau=tau, delta=None
        )[0])  
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            att_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
  
# Projector : MLP to learn De-stationary factors
class Projector(nn.Module):
    def __init__(self, enc_in : int, seq_len : int, hidden_dim : int, output_dim : int, kernel_size : int = 3):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size = kernel_size, padding = padding, padding_mode='circular', bias = False)
        self.backbone = nn.Sequential(
            nn.Linear(2 * enc_in, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, output_dim, bias = False)
        )
    
    def forward(self, x : torch.Tensor, stats : torch.Tensor):
        # x : (B,S,E)
        # stats : (B,1,E)
        B = x.size()[0]
        # (B,1,E)
        x = self.series_conv(x)
        # (B, 2, E)
        x = torch.cat([x, stats], dim = 1)
        # (B, 2E)
        x = x.view(B, -1)
        # (B, output_dim)
        x = self.backbone(x)
        return x


# Assume these are correctly defined in your project
# from pytorch_model_summary import summary
# from src.nn_env.transformer import PositionalEncoding, NoiseLayer, GELU

# NOTE: All the low-level classes (DSAttention, AttentionLayer, EncoderLayer,
#       DecoderLayer, Encoder, Decoder, Projector) are PERFECT as they are.
#       They are general-purpose and do not need any modification.
#       The only changes are in the main, top-level class that puts them together.

class FissionReactorTransformer(nn.Module):
    """
    A Transformer model adapted to predict the necessary Control Rod Positions
    to guide a nuclear reactor's state (k_eff, Power) along a target trajectory.
    """
    def __init__(
        self,
        # --- Core Fission Parameters ---
        state_features: int = 2,          # Number of state variables (e.g., k_eff, Power_MW)
        control_features: int = 1,        # Number of control variables (e.g., Control_Rod_Position)
        input_seq_len: int = 100,         # Number of historical time steps to look at
        output_pred_len: int = 24,        # Number of future time steps to predict control for

        # --- Transformer Hyperparameters ---
        feature_dim: int = 128,           # The internal working dimension of the model (d_model)
        n_heads: int = 8,                 # Number of attention heads
        n_layers: int = 3,                # Number of Encoder and Decoder layers
        dim_feedforward: int = 512,       # Dimension of the FFN inside Encoder/Decoder layers
        dropout: float = 0.1,
        factor: float = 5.0,              # Factor for DSAttention (can be tuned)
        kernel_size: int = 3,
        ):

        super(FissionReactorTransformer, self).__init__()

        # Store key dimensions
        self.input_state_dim = state_features
        self.input_control_dim = control_features
        self.input_seq_len = input_seq_len
        self.output_pred_len = output_pred_len
        self.output_control_dim = control_features # We predict the control features

        # Masks will be created dynamically in the forward pass
        self.src_mask = None
        self.tgt_dec_mask = None

        # --- CONVOLUTIONAL EMBEDDING FOR ENCODER ---
        # Takes historical state + control data and creates rich embeddings
        self.encoder_input_embedding = nn.Sequential(
            nn.Conv1d(in_channels=self.input_state_dim + self.input_control_dim, out_channels=feature_dim // 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim // 2, out_channels=feature_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        # --- ENCODER ---
        self.enc_pos = PositionalEncoding(d_model=feature_dim)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(mask_flag=False, factor=factor, dropout=dropout), # Encoder sees all history
                        feature_dim, n_heads
                    ),
                    feature_dim, dim_feedforward, dropout
                ) for _ in range(n_layers)
            ],
            norm_layer=nn.LayerNorm(feature_dim)
        )

        # --- DE-STATIONARY LEARNERS (TAU & DELTA) ---
        self.tau_learner = Projector(self.input_state_dim, input_seq_len, feature_dim, n_heads)
        self.delta_learner = Projector(self.input_state_dim, input_seq_len, feature_dim, n_heads)

        # --- CONVOLUTIONAL EMBEDDING FOR DECODER ---
        # Takes the target state + control prompt and creates embeddings
        # Note: The input is the desired future STATE and the ground-truth previous CONTROL action
        self.decoder_input_embedding = nn.Sequential(
            nn.Conv1d(in_channels=self.input_state_dim + self.input_control_dim, out_channels=feature_dim // 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim // 2, out_channels=feature_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        # --- DECODER ---
        self.dec_pos = PositionalEncoding(d_model=feature_dim)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # 1. Masked Self-Attention (reviews its own generated sequence)
                    self_attention=AttentionLayer(
                        DSAttention(mask_flag=True, factor=factor, dropout=dropout),
                        feature_dim, n_heads
                    ),
                    # 2. Cross-Attention (consults the encoder's memory)
                    cross_attention=AttentionLayer(
                        DSAttention(mask_flag=False, factor=factor, dropout=dropout),
                        feature_dim, n_heads
                    ),
                    d_model=feature_dim,
                    d_ff=dim_feedforward,
                    dropout=dropout
                ) for _ in range(n_layers)
            ],
            norm_layer=nn.LayerNorm(feature_dim),
            # Final projection layer to get the predicted control rod position
            projection=nn.Linear(feature_dim, self.output_control_dim)
        )

    def forward(self,
                past_state: torch.Tensor,      # Historical k_eff, Power_MW. Shape: [B, S, F1]
                past_control: torch.Tensor,    # Historical Control_Rod_Position. Shape: [B, S, F2]
                future_state_target: torch.Tensor, # Target trajectory for k_eff, Power_MW. Shape: [B, L, F1]
                future_control_prompt: torch.Tensor # Ground-truth rod position for teacher forcing. Shape: [B, L, F2]
               ):

        # --- PREPARE ENCODER INPUT ---
        # We need to save the original state for the de-stationary learners and for de-normalization
        past_state_original = past_state.clone().detach()
        means = past_state.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(past_state, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        past_state_normalized = (past_state - means) / stdev

        # Combine historical state and control actions
        encoder_input = torch.cat([past_state_normalized, past_control], dim=2) # Shape: [B, S, F1+F2]

        # --- ENCODER PATH ---
        # 1. Create rich embeddings from the input sequence
        # Note the permute for Conv1d, which expects [B, Channels, Length]
        enc_out = self.encoder_input_embedding(encoder_input.permute(0, 2, 1)).permute(0, 2, 1) # Shape: [B, S, D_model]
        # 2. Add positional information
        enc_out = self.enc_pos(enc_out)
        # 3. Generate dynamic tau and delta based on the original data's characteristics
        tau = self.tau_learner(past_state_original, stdev).exp()     # Shape: [B, n_heads]
        delta = self.delta_learner(past_state_original, means) # Shape: [B, n_heads]
        # 4. Run through the encoder stack
        # Encoder uses self-attention, so no mask is typically needed unless for padding
        enc_out, _ = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta) # enc_out is the final memory

        # --- PREPARE DECODER INPUT ---
        # This is the "prompt" for the decoder, combining the target state and the teacher-forcing control input
        decoder_input = torch.cat([future_state_target, future_control_prompt], dim=2) # Shape: [B, L, F1+F2]

        # --- DECODER PATH ---
        # 1. Create rich embeddings for the decoder prompt
        dec_out = self.decoder_input_embedding(decoder_input.permute(0, 2, 1)).permute(0, 2, 1) # Shape: [B, L, D_model]
        # 2. Add positional information for the future timeline
        dec_out = self.dec_pos(dec_out)
        # 3. Create the "no cheating" look-ahead mask for the decoder's self-attention
        if self.tgt_dec_mask is None or self.tgt_dec_mask.size(0) != dec_out.size(1):
            self.tgt_dec_mask = self._generate_square_subsequent_mask(dec_out.size(1)).to(dec_out.device)
        # 4. Run through the decoder stack
        # It receives its own prompt (dec_out), the encoder's memory (enc_out), and the rules (masks, tau, delta)
        predicted_control = self.decoder(
            x=dec_out,
            cross=enc_out,
            x_mask=self.tgt_dec_mask,
            cross_mask=None, # Assuming no padding mask for encoder output
            tau=tau,
            delta=delta
        )

        return predicted_control # Final output is the predicted Control_Rod_Position

    def _generate_square_subsequent_mask(self, size: int):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def summary(self):
        sample_0D = torch.zeros((1, self.input_seq_len, self.input_0D_dim))
        sample_ctrl = torch.zeros((1, self.input_seq_len, self.input_ctrl_dim))
        summary(self, sample_0D, sample_ctrl, batch_size = 1, show_input = True, print_summary=True)
