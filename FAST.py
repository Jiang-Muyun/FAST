import os
import random
import sys
import einops
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from einops.layers.torch import Rearrange, Reduce
from transformers import PretrainedConfig


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        attn_output, attn_weights = self.attn(inp_x, inp_x, inp_x)
        x = x + attn_output
        x = x + self.linear(self.layer_norm_2(x))
        return x

class Spatial_Temporal_Tokenizer(nn.Module):
    def __init__(self, electrodes, zone_dict, feature_dim):
        super().__init__()
        self.index_dict = {}
        self.encoders = nn.ModuleDict()
        for area, ch_names in zone_dict.items():
            self.index_dict[area] = torch.tensor([electrodes.index(ch_name) for ch_name in ch_names])
            self.encoders[area] = self.make_encoder(len(self.index_dict[area]), feature_dim)

    def forward(self, x):
        return torch.stack([encoder(x[:, self.index_dict[area]]) for area, encoder in self.encoders.items()], dim=1) # B N F

    def make_encoder(self, in_channels, feature_dim=32):
        F1, F2, F3, F4 = feature_dim//2, feature_dim//3, feature_dim//3, feature_dim
        normFn = nn.BatchNorm2d
        # normFn = nn.Identity
        poolings = (2, 2, 2, 2)
        kernels = (3, 3, 3, 3)
        return nn.Sequential(
            Rearrange('B C T -> B 1 C T'),
            nn.Conv2d(1, F1, (1, kernels[0]), bias=True),
            nn.Conv2d(F1, F1, (in_channels, 1), padding=0, bias=False),
            normFn(F1),
            nn.GELU(),
            nn.MaxPool2d((1, poolings[0]), stride=(1, poolings[0])),
            nn.Conv2d(F1, F2, (1, kernels[1]), bias=False),
            normFn(F2),
            nn.GELU(),
            nn.MaxPool2d((1, poolings[1]), stride=(1, poolings[1])),
            nn.Conv2d(F2, F3, (1, kernels[2]), bias=False),
            normFn(F3),
            nn.GELU(),
            nn.MaxPool2d((1, poolings[2]), stride=(1, poolings[2])),
            nn.Conv2d(F3, F4, (1, kernels[3]), bias=False),
            normFn(F4),
            nn.GELU(),
            nn.MaxPool2d((1, poolings[3]), stride=(1, poolings[3])),
            Reduce('B F 1 T -> B F', 'mean')
        )

class FAST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        electrodes = config.electrodes
        zone_dict = config.zone_dict
        dim_cnn = config.dim_cnn
        dim_token = config.dim_token
        seq_len = config.seq_len
        window_len = config.window_len
        slide_step = config.slide_step
        n_classes = config.n_classes
        num_heads = config.num_heads
        num_layers = config.num_layers
        dropout = config.dropout

        self.n_tokens = (seq_len - window_len) // slide_step + 1
        print(f" > Number of Zones: {len(zone_dict)}")
        print(f" > Number of Electrodes: {len(electrodes)}")
        print(f" > Number of Tokens: {self.n_tokens} (seq_len={seq_len}, window_len={window_len}, slide_step={slide_step})")
        print(f" > CNN Dimension: {dim_cnn}")
        print(f" > Token Dimension: {dim_token}")
        
        self.head = Spatial_Temporal_Tokenizer(electrodes, zone_dict, dim_cnn)
        self.input_layer = nn.Linear(dim_cnn * len(zone_dict), dim_token)
        self.transformer = nn.Sequential(*[AttentionBlock(dim_token, dim_token*2, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_tokens + 1, dim_token))  # +1 for CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_token))  # CLS token
        self.last_layer = nn.Linear(dim_token, n_classes)  # Changed to embed_dim
        self.dropout = nn.Dropout(dropout)

    def forward_head(self, x, step_override = None):
        if step_override is not None:
            slide_step = step_override
        else:
            slide_step = self.config.slide_step
        x = x.unfold(-1, self.config.window_len, slide_step)#.contiguous()
        B, C, N, T = x.shape
        x = einops.rearrange(x, 'B C N T -> (B N) C T')
        feature = self.head(x)
        feature = einops.rearrange(feature, '(B N) Z F -> B N Z F', B=B)
        return feature
    
    def forward(self, x):
        x = self.forward_head(x)
        x = einops.rearrange(x, 'B N Z F -> B N (Z F)')
        x = self.input_layer(x)
        cls_token_expand = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token_expand, x), dim=1)
        x = x + self.pos_embedding[:, :self.n_tokens + 1]
        x = self.dropout(x)
        tokens = self.transformer(x)
        logits = self.last_layer(self.dropout(tokens[:, 0]))
        return logits, tokens

    def loss(self, x, y):
        logits, _ = self.forward(x)
        return nn.CrossEntropyLoss()(logits, y)
    
if __name__ == '__main__':
    
    Example_Electrodes = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 
        'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 
        'CP6', 'TP9', 'TP10', 'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 
        'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7',
        'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'Fpz', 'CPz'
    ]

    Example_Zones = {
        'Pre-frontal': ['AF7', 'Fp1', 'Fpz', 'Fp2', 'AF8', 'AF3', 'AF4'],
        'Frontal': ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
        'Pre-central': ['FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
        'Central': ['C1', 'C2', 'C3', 'Cz', 'C4', 'C5', 'C6'],
        'Post-central': ['CP1', 'CP2', 'CP3', 'CPz', 'CP4', 'CP5', 'CP6'],
        'Temporal': ['T7', 'T8', 'FT7', 'FT8', 'TP7', 'TP8', 'TP9', 'TP10', 'FT9', 'FT10'],
        'Parietal': ['P1', 'P2', 'P3', 'P4', 'Pz', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8'],
        'Occipital': ['O1', 'O2', 'Oz', 'POz'],
    }

    config = PretrainedConfig(
        electrodes=Example_Electrodes,
        zone_dict=Example_Zones,
        dim_cnn=32,
        dim_token=64,
        seq_len=1000,
        window_len=250,
        slide_step=125,
        n_classes=5,
        num_layers=4,
        num_heads=8,
        dropout=0.2,
    )
    model = FAST(config)
    x = torch.randn(10, len(Example_Electrodes), config.seq_len)
    print('Input:', x.shape)
    logits, tokens = model(x)
    print('Output:', logits.shape, tokens.shape)