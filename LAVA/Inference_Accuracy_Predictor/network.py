import numpy as np
import pandas as pd
import os
cuda = '5'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import cv2
from MNv4 import MNV4ConvSmall

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_STEPS = 8
FEATURES = 5
OUTPUT_STEPS = 1
OUTPUT = 176

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=8):
        super().__init__()
        pe = torch.zeros(max_len, model_dim, dtype=torch.float)
        exp_1 = torch.arange(model_dim // 2, dtype=torch.float)
        exp_value = exp_1 / (model_dim / 2)
        alpha = 1 / (10000.0 ** exp_value)
        out = torch.arange(max_len, dtype=torch.float)[:, None] @ alpha[None, :]
        pe[:, 0::2] = torch.sin(out)
        pe[:, 1::2] = torch.cos(out)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x


class CausalAttention(nn.Module):
    def __init__(self, model_dim, num_heads=4):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        out = output.reshape(batch_size, seq_len, -1)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = CausalAttention(model_dim, num_heads)
        self.linear1 = nn.Linear(model_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.self_attn(x))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.dropout(x)


class DC_FeatureExtractor(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.embedding1 = nn.Linear(3, model_dim // 2)
        self.embedding2 = nn.Linear(1, model_dim // 2)
        self.embedding3 = nn.Linear(3, model_dim // 2)

        self.fusion_layer = nn.Sequential(
            nn.Linear(3 * (model_dim // 2), model_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        content = x[:, :, :3]
        motion = x[:, :, 3:4]
        config = x[:, :, 4:7]
        x_emb1 = F.gelu(self.embedding1(content))
        x_emb2 = F.gelu(self.embedding2(motion))
        x_emb3 = F.gelu(self.embedding3(config))

        x_emb = torch.cat([x_emb1, x_emb2, x_emb3], dim=2)
        x_emb = self.fusion_layer(x_emb)
        return x_emb

class VF_FeatureExtractor(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.backbone = MNV4ConvSmall(width_mult=0.5, expand_ratio_scale=0.7)
        self.proj = nn.Sequential(
            nn.Linear(576, model_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x: [batch, 3, 224, 224]
        features = self.backbone(x)  # [batch, 576]
        projected = self.proj(features)  # [batch, model_dim]
        return projected

class FusionFeatureExtractor(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.dc_extractor = DC_FeatureExtractor(model_dim)  # [batch, seq, model_dim]
        self.vf_extractor = VF_FeatureExtractor(model_dim)  # [batch, model_dim]

        self.pos_encoder = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, model_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.fusion_gate = nn.Sequential(
            nn.Linear(model_dim*2, model_dim),
            nn.Sigmoid()
        )

    def forward(self, dc, vf):
        """
        dc: [batch, 8, 5]
        vf: [batch, 3, 224, 224]
        """
        dc_emb = self.dc_extractor(dc)  # [batch,8,128]
        dc_emb = self.pos_encoder(dc_emb)
        for layer in self.layers:
            dc_emb = layer(dc_emb)
        dc_final = dc_emb[:, -1, :]

        vf_final = self.vf_extractor(vf)

        fused_concat = torch.cat([dc_final, vf_final], dim=1)  # [batch, 2*model_dim]
        gate = self.fusion_gate(fused_concat)  # [batch, model_dim]
        fused_out = gate * dc_final + (1-gate) * vf_final  # [batch, model_dim]
        return fused_out


class I_Model(nn.Module):
    def __init__(self, model_dim, output_dim, dropout=0.2):
        super().__init__()
        self.feature_extractor = FusionFeatureExtractor(model_dim, num_heads=8, num_layers=3)
        self.head = nn.Sequential(
            nn.Linear(model_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
    def forward(self, dc, vf):
        features = self.feature_extractor(dc, vf)
        return self.head(features)
