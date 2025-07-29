import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

# =================================================================================================
# SECTION 1: TRANSFORMER MODEL ARCHITECTURE (Based on provided code)
# =================================================================================================

class NoiseLayer(nn.Module):
    def __init__(self, mean: float = 0, std: float = 1e-2):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x: torch.Tensor):
        if self.training:
            noise = Variable(torch.ones_like(x).to(x.device) * self.mean + torch.randn(x.size()).to(x.device) * self.std)
            return x + noise
        else:
            return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
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
        
    def forward(self, x: torch.Tensor):
        # x : (seq_len, batch_size, n_features)
        return x + self.pe[:x.size(0), :, :]

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class CNNencoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, kernel_size: int, padding: int, reduction: bool):
        super().__init__()
        dk = 1 if reduction else 0
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=feature_dim, kernel_size=kernel_size + dk, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor):
        return self.encoder(x)

class FissionReactorTransformer(nn.Module):
    def __init__(
        self, 
        n_layers: int = 6, 
        n_heads: int = 4, 
        dim_feedforward: int = 1024, 
        dropout: float = 0.1,        
        RIN: bool = False,
        state_features: int = 1,  # k_eff
        input_seq_len: int = 20,
        control_features: int = 1,  # CR_Pos
        control_seq_len: int = 20,  # Same as input for simplicity
        output_pred_len: int = 4,
        feature_dim: int = 128,
        range_info: Optional[Dict] = None,
        noise_mean: float = 0,
        noise_std: float = 0.01,  # Reduced for reactor stability
        kernel_size: int = 3,
        ):
        
        super(FissionReactorTransformer, self).__init__()
        
        # Input information
        self.state_features = state_features
        self.input_seq_len = input_seq_len
        self.control_features = control_features
        self.control_seq_len = control_seq_len
        
        # Output information
        self.output_pred_len = output_pred_len
        self.output_control_dim = control_features
        
        # Source mask
        self.src_mask = None
        
        # Transformer info
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        self.RIN = RIN
        
        self.noise = NoiseLayer(mean=noise_mean, std=noise_std)
        
        # Use minimum sequence length for alignment
        self.effective_seq_len = min(input_seq_len, control_seq_len)
        
        if kernel_size % 2 == 0:
            print("Kernel should be odd number")
            kernel_size += 1
            
        padding = (kernel_size - 1) // 2
        
        # Convolution layers for extracting temporal components
        self.encoder_state = CNNencoder(state_features, feature_dim//2, kernel_size, padding, False)
        self.encoder_control = CNNencoder(control_features, feature_dim//2, kernel_size, padding, 
                                        True if control_seq_len > input_seq_len else False)
        
        # Positional encoding
        self.pos = PositionalEncoding(d_model=feature_dim, max_len=self.effective_seq_len)
        
        # Transformer encoder
        self.enc = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=n_heads, 
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation=GELU()
        )
        
        self.trans_enc = nn.TransformerEncoder(self.enc, num_layers=n_layers)
        
        # Decoder layers
        # Sequence length transformation (input_seq -> output_pred_len)
        self.lc_seq = nn.Linear(self.effective_seq_len, output_pred_len)
        
        # Feature dimension transformation (feature_dim -> control_features)
        self.lc_feat = nn.Linear(feature_dim, control_features)
        
        # Reversible Instance Normalization
        if self.RIN:
            self.affine_weight_state = nn.Parameter(torch.ones(1, 1, state_features))
            self.affine_bias_state = nn.Parameter(torch.zeros(1, 1, state_features))
            
        self.range_info = range_info
        
        if range_info:
            self.register_buffer('range_min', torch.Tensor([range_info[key][0] * 0.1 for key in range_info.keys()]))
            self.register_buffer('range_max', torch.Tensor([range_info[key][1] * 10.0 for key in range_info.keys()]))
        else:
            self.range_min = None
            self.range_max = None
            
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1    
        self.lc_feat.bias.data.zero_()
        self.lc_feat.weight.data.uniform_(-initrange, initrange)
        
        self.lc_seq.bias.data.zero_()
        self.lc_seq.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, past_state, past_control, future_state_target, future_control_prompt):
        """
        Args:
            past_state: [B, seq_len, state_features] - Historical reactor state (k_eff)
            past_control: [B, seq_len, control_features] - Historical control actions (CR_Pos)
            future_state_target: [B, pred_len, state_features] - Target future states (not used in this architecture)
            future_control_prompt: [B, pred_len, control_features] - Ground truth for training
        
        Returns:
            predicted_control: [B, pred_len, control_features] - Predicted control actions
        """
        
        # Add noise to state for robust performance
        x_state = self.noise(past_state)
        
        # Reversible Instance Normalization for state
        if self.RIN:
            means_state = x_state.mean(1, keepdim=True).detach()
            x_state = x_state - means_state
            stdev_state = torch.sqrt(torch.var(x_state, dim=1, keepdim=True, unbiased=False) + 1e-3).detach()
            x_state /= stdev_state
            x_state = x_state * self.affine_weight_state + self.affine_bias_state
            
        # Encoding path
        # Convert to CNN format: (N, T, F) -> (N, F, T)
        x_state_encoded = self.encoder_state(x_state.permute(0, 2, 1)).permute(0, 2, 1)
        x_control_encoded = self.encoder_control(past_control.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Concatenate state and control features
        x = torch.cat([x_state_encoded, x_control_encoded], dim=2)
        
        # Transform to transformer format: (N, T, F) -> (T, N, F)
        x = x.permute(1, 0, 2)
    
        # Generate causal mask if needed
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            self.src_mask = self._generate_square_subsequent_mask(len(x)).to(x.device)
            
        # Add positional encoding
        x = self.pos(x)
        
        # Apply transformer encoder
        x = self.trans_enc(x, self.src_mask)
        
        # Transform back: (T, N, F) -> (N, T, F)
        x = x.permute(1, 0, 2)
    
        # Sequence length transformation: (N, T, F) -> (N, F, T) -> (N, F, pred_len)
        x = self.lc_seq(x.permute(0, 2, 1))
        x = torch.nn.functional.relu(x)
        
        # Transform back: (N, F, pred_len) -> (N, pred_len, F)
        x = x.permute(0, 2, 1)
        
        # Feature dimension reduction: (N, pred_len, F) -> (N, pred_len, control_features)
        x = self.lc_feat(x)
    
        # Reverse RIN if applied
        if self.RIN:
            x = x - self.affine_bias_state
            x = x / (self.affine_weight_state + 1e-6)
            x = x * stdev_state
            x = x + means_state  
        
        # Clamp output to reasonable range for reactor control
        x = torch.clamp(x, min=-10.0, max=30.0)  # Adjusted for CR_Pos range
        
        # Remove NaN values for stability
        x = torch.nan_to_num(x, nan=0)
        
        return x

    def _generate_square_subsequent_mask(self, size: int):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# ==============================================================================
# RAW DATA ENTRY
# ==============================================================================

data = {
'Time': [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336, 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308, 315, 322, 329, 336],
'k_eff': [1.09270183844705, 1.0643749696548, 1.06501276887864, 1.0658765911557, 1.06180038766813, 1.05939349980401, 1.06061460254453, 1.06385649592581, 1.06477423262992, 1.06478464151116, 1.06201129106431, 1.05841462233477, 1.05678163147585, 1.05644009927099, 1.05464991263634, 1.0543471573721, 1.05394490715222, 1.05553289394673, 1.05338643666532, 1.05586014693404, 1.05672194380278, 1.05229195821757, 1.05496112572831, 1.04752530561492, 1.05011441495955, 1.05109371827276, 1.05156443824562, 1.05199315596034, 1.05134895096823, 1.04546335320125, 1.0513824667683, 1.045560739086, 1.04988434349434, 1.0580424994414, 1.04767251714544, 1.04741089930651, 1.04046213447768, 1.04374840316015, 1.04692818231167, 1.04480769790002, 1.04751321635445, 1.04601843509891, 1.04408482752359, 1.04986927081918, 1.04406065359152, 1.04059819770853, 1.04289596663744, 1.0472272891175, 1.04090445088931, 1.0673286902133, 1.03713067973215, 1.03679034618607, 1.0426094883169, 1.04670425755473, 1.04164115609725, 1.03892513086673, 1.03708809828184, 1.03273992260461, 1.03537356214892, 1.04429801280307, 1.03511843560203, 1.03607262743649, 1.03088096530531, 1.02789912626441, 1.03274139788229, 1.03535726273696, 1.03512198278929, 1.0367066249323, 1.03597942696092, 1.03395047422563, 1.03406081274368, 1.02673846225835, 1.03030885504409, 1.03517928408042, 1.03346474192412, 1.03147168904358, 1.03087910496484, 1.03381719144396, 1.02999120365883, 1.03192122430225, 1.02685563868599, 1.03204941028087, 1.0293192309013, 1.03316653516676, 1.02587297134131, 1.02515307828065, 1.03047599228256, 1.0253881244362, 1.03280464583865, 1.03121171095693, 1.0288797053598, 1.02655123520943, 1.02966379821721, 1.03037305596267, 1.02248331784083, 1.02596076918559, 1.02836034086453, 1.02288172395269, 1.05960251916676, 1.03005015716095, 1.03337021146257, 1.02727148188047, 1.0299738748805, 1.02624069671838, 1.02603256853357, 1.02867971888034, 1.02459463916783, 1.02654137613134, 1.01998528641481, 1.0264308249202, 1.02425904562539, 1.02278962899066, 1.02341824842264, 1.02582832080021, 1.02348298737679, 1.02078759896086, 1.02262452368937, 1.02051398780961, 1.02317254805872, 1.0223949413818, 1.02360985817627, 1.02281006901489, 1.0208197805297, 1.02228049531238, 1.02156288779766, 1.02743203987915, 1.02265573259554, 1.0199543067035, 1.01892156755931, 1.02079426922808, 1.0188038210478, 1.02119413616056, 1.01854332984385, 1.01866390144323, 1.01932453474126, 1.01752241092755, 1.01590359738709, 1.01459196324763, 1.0148178794578, 1.01131499879036, 1.01343913987004, 1.01391903088963, 1.01753650216506, 1.01373147822093, 1.01334569738678, 1.01249492223498, 1.01161576914063, 1.03662082932069, 1.00786586339113, 1.01374950801024, 1.00951374228098, 1.00909207453298, 1.00213930628985, 1.00137720643154, 1.00326046751725, 1.00330467617217, 1.00360745568485, 1.00990016979824, 1.00170321503617, 1.00531787370855, 1.00366228087333, 1.00261231741463, 0.995874254281175, 1.00297649496437, 1.0015617629883, 1.00740090708893, 0.999787011774798, 0.999074209680911, 0.99862293032869, 0.999712692068926, 0.994884636509982, 1.0018173931433, 1.00414192944546, 0.99399244129109, 1.00331333626115, 0.997871993848439, 1.00225984584392, 1.00208720623614, 0.996045150869745, 1.00538222434348, 1.00367310097068, 0.996053736518011, 0.995550332471514, 0.9978195769251, 0.994630307145708, 0.996167148206082, 0.99838758280413, 0.994343743263915, 0.997398954673151, 0.992619516439709, 0.998687993985981, 0.993313284135946, 0.992451727943551, 0.993248797018366, 0.99428512682069, 0.995308213107007, 1.01193255680547, 0.994388885331012, 0.994134276241682, 0.987134842244376, 0.99281354790758, 0.997206256966017, 0.985260934527533, 0.985393783290584, 0.987204927309185, 0.985914890261068, 0.980912075236045, 982980790142391, 0.983974360826939, 0.987262978553874, 0.985471930356754, 0.989030517012246, 0.986045788506013, 0.983876521795378, 0.983451875233888, 0.986715107915513, 0.984544743442471, 0.983212610134445, 0.979800139418763, 0.979983046748916, 0.990386734581925, 0.980312169538438, 0.983708376381868, 0.983076334492721, 0.98435783943176, 0.98765844947183, 0.98131314318215, 0.98691112972517, 0.981279570933728, 0.983404859006246, 0.977769690504726, 0.977464719786311, 0.975451317132869, 0.979542876385254, 0.978440378493575, 0.977797699655094, 0.975571057183086, 0.980644393181371, 0.97887716717029, 0.977763790682662, 0.98577099342186, 0.980314978753987, 0.981072590799463, 0.974530001893048, 0.975539105979112, 0.996987048290785, 0.975751889187771, 0.979906743628355, 0.97111328091633, 0.972199690347839, 0.97199364707141, 0.971405969366714, 0.972298854115534, 0.971424761701418, 0.971266983257136, 0.963629948175006, 0.966495992242591, 0.968169346788767, 0.966147277870901, 0.967545947936199, 0.975570748229985, 0.970629171211536, 0.965616534548475, 0.962900644872699, 0.965372969497341, 0.968748960857997, 0.963551769580766, 0.966718469939159, 0.962571650727767, 0.963592707554144, 0.967721335423575, 0.96346253935954, 0.966048761663707, 0.965763076697617, 0.966600813031101, 0.967925245115546, 0.968259332958311, 0.965649998514265, 0.95878133233201, 0.963367224466267, 0.963352864235313, 0.963959003151184, 0.961948470966127, 0.952163790859466, 0.963129750708182, 0.964039998469031, 0.962158080306711, 0.959845028793009, 0.959731511858083, 0.962593100668997, 0.96294800995021, 0.959152652176231, 0.960078827737686, 0.959833821084637],
'CR_Pos': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 13.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 19.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1, 27.1]
}

df = pd.DataFrame(data)
print("DataFrame created successfully!")
print(f"Data shape: {df.shape}")

# ==============================================================================
# DATA PREPARATION FUNCTIONS
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

# =================================================================================================
# TRAINING & EVALUATION FUNCTIONS
# =================================================================================================

def train_per_epoch(train_loader, model, optimizer, loss_fn, device, max_norm_grad):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        past_state, past_control, future_state_target, future_control_prompt = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        output = model(past_state, past_control, future_state_target, future_control_prompt)
        loss = loss_fn(output, future_control_prompt)
        
        if not torch.isfinite(loss): 
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(num_batches, 1)

def valid_per_epoch(valid_loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            past_state, past_control, future_state_target, future_control_prompt = [b.to(device) for b in batch]
            output = model(past_state, past_control, future_state_target, future_control_prompt)
            loss = loss_fn(output, future_control_prompt)
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / max(num_batches, 1)

def train(train_loader, valid_loader, model, optimizer, scheduler, loss_fn, device, num_epoch, verbose):
    history = {'train_loss': [], 'valid_loss': []}
    best_loss = float('inf')
    
    for epoch in tqdm(range(num_epoch), desc="Training"):
        train_loss = train_per_epoch(train_loader, model, optimizer, loss_fn, device, max_norm_grad=1.0)
        valid_loss = valid_per_epoch(valid_loader, model, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        
        if scheduler: 
            scheduler.step()
            
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "fission_best_model.pth")
            
        if verbose and (epoch % verbose == 0):
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
            
    print(f"\nTraining finished. Best validation loss: {best_loss:.4f}")
    return history

def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['valid_loss'], label='Validation Loss', linewidth=2)
    plt.title('Training History - Fission Reactor Control', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =================================================================================================
# MAIN EXECUTION BLOCK
# =================================================================================================

if __name__ == "__main__":
    # --- 1. Hyperparameters & Configuration ---
    # Data Parameters
    INPUT_SEQ_LEN = 20
    OUTPUT_PRED_LEN = 4
    STATE_COLS = ['k_eff']
    CONTROL_COLS = ['CR_Pos']
    
    # Model Parameters
    FEATURE_DIM = 128
    N_HEADS = 8
    N_LAYERS = 3
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    KERNEL_SIZE = 3
    NOISE_STD = 0.01
    USE_RIN = True  # Reversible Instance Normalization
    
    # Training Parameters
    BATCH_SIZE = 8  # Reduced for smaller dataset
    EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Data Preparation ---
    all_samples = create_fission_sliding_windows(df, INPUT_SEQ_LEN, OUTPUT_PRED_LEN, STATE_COLS, CONTROL_COLS)
    print(f"Total samples created: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("Error: No samples could be created. Check your data length and sequence parameters.")
        exit()
    
    train_size = int(0.7 * len(all_samples))
    val_size = int(0.15 * len(all_samples))
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]
    
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}, Test samples: {len(test_samples)}")
    
    if len(train_samples) == 0 or len(val_samples) == 0:
        print("Error: Insufficient data for training/validation splits.")
        exit()
    
    train_dataset = ReactorDataset(train_samples)
    val_dataset = ReactorDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True if device == "cuda" else False)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, pin_memory=True if device == "cuda" else False)
    
    # --- 3. Model & Training Setup ---
    model = FissionReactorTransformer(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        RIN=USE_RIN,
        state_features=len(STATE_COLS),
        input_seq_len=INPUT_SEQ_LEN,
        control_features=len(CONTROL_COLS),
        control_seq_len=INPUT_SEQ_LEN,  # Same as input sequence length
        output_pred_len=OUTPUT_PRED_LEN,
        feature_dim=FEATURE_DIM,
        noise_std=NOISE_STD,
        kernel_size=KERNEL_SIZE
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    print(f"Starting training on {device}...")
    print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    
    # --- 4. Run Training ---
    history = train(train_loader, valid_loader, model, optimizer, scheduler, loss_fn, device, EPOCHS, verbose=2)
    
    # --- 5. Visualize Results ---
    plot_history(history)
    
    # --- 6. Final Test Evaluation ---
    if len(test_samples) > 0:
        test_dataset = ReactorDataset(test_samples)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                               num_workers=NUM_WORKERS, pin_memory=True if device == "cuda" else False)
        
        # Load best model
        model.load_state_dict(torch.load("fission_best_model.pth"))
        test_loss = valid_per_epoch(test_loader, model, loss_fn, device)
        print(f"Final test loss: {test_loss:.4f}")
        
        # Show a sample prediction
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(test_loader))
            past_state, past_control, future_state_target, future_control_prompt = [b.to(device) for b in sample_batch]
            predictions = model(past_state, past_control, future_state_target, future_control_prompt)
            
            # Print first sample prediction vs ground truth
            print("\nSample Prediction vs Ground Truth:")
            print(f"Predicted CR_Pos: {predictions[0].cpu().numpy().flatten()}")
            print(f"Actual CR_Pos:    {future_control_prompt[0].cpu().numpy().flatten()}")
    else:
        print("No test samples available for evaluation.")
