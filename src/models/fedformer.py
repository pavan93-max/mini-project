"""
FEDformer model for time-series forecasting.
Frequency Enhanced Decomposed Transformer for long-term forecasting.
Based on: "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SeriesDecomposition(nn.Module):
    """Series decomposition block for trend and seasonal components."""
    
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Decompose series into trend and seasonal components.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input (B, L, d_model)
            
        Returns:
        --------
        tuple
            (trend, seasonal)
        """
        x_perm = x.permute(0, 2, 1)
        trend = self.moving_avg(x_perm).permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal


class FrequencyBlock(nn.Module):
    """
    Frequency domain enhancement block using FFT.
    Applies transformations in frequency domain.
    """
    
    def __init__(self, d_model: int, modes: int = 64):
        """
        Parameters:
        -----------
        d_model : int
            Model dimension
        modes : int
            Number of frequency modes to keep
        """
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        
        # Frequency domain transformations
        # Initialize as separate real and imaginary parts
        self.complex_weight_real = nn.Parameter(torch.randn(modes, d_model, dtype=torch.float32) * 0.1)
        self.complex_weight_imag = nn.Parameter(torch.randn(modes, d_model, dtype=torch.float32) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain transformation.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input (B, L, d_model)
            
        Returns:
        --------
        torch.Tensor
            Frequency-enhanced output (B, L, d_model)
        """
        B, L, E = x.shape
        
        # FFT along sequence dimension
        x_fft = torch.fft.rfft(x, dim=1)
        
        # Select top modes
        modes = min(self.modes, x_fft.shape[1])
        x_fft_top = x_fft[:, :modes, :]
        
        # Apply frequency domain transformation
        weight = torch.complex(self.complex_weight_real[:modes, :], self.complex_weight_imag[:modes, :])
        weight = weight.unsqueeze(0)  # (1, modes, d_model)
        out_fft = x_fft_top * weight
        
        # Pad back to original length
        out_fft_padded = torch.zeros_like(x_fft)
        out_fft_padded[:, :modes, :] = out_fft
        
        # Inverse FFT
        out = torch.fft.irfft(out_fft_padded, n=L, dim=1)
        
        return out


class FEDformerEncoderLayer(nn.Module):
    """FEDformer encoder layer with frequency enhancement and decomposition."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1, modes: int = 64):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.freq_block = FrequencyBlock(d_model, modes)
        self.decomp1 = SeriesDecomposition()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.decomp2 = SeriesDecomposition()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = x + self.dropout(self.attention(x, x, x)[0])
        x, _ = self.decomp1(x)
        x = self.norm1(x)
        
        # Frequency enhancement
        x = x + self.freq_block(x)
        
        # Feedforward
        x = x + self.feed_forward(x)
        x, _ = self.decomp2(x)
        x = self.norm2(x)
        
        return x


class FEDformerDecoderLayer(nn.Module):
    """FEDformer decoder layer."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1, modes: int = 64):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.freq_block = FrequencyBlock(d_model, modes)
        self.decomp1 = SeriesDecomposition()
        self.decomp2 = SeriesDecomposition()
        self.decomp3 = SeriesDecomposition()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.decomp4 = SeriesDecomposition()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = x + self.dropout(self.self_attention(x, x, x)[0])
        x, _ = self.decomp1(x)
        x = self.norm1(x)
        
        # Cross-attention
        x = x + self.dropout(self.cross_attention(x, enc_out, enc_out)[0])
        x, _ = self.decomp2(x)
        x = self.norm2(x)
        
        # Frequency enhancement
        x = x + self.freq_block(x)
        
        # Feedforward
        x = x + self.feed_forward(x)
        x, _ = self.decomp4(x)
        x = self.norm3(x)
        
        return x


class FEDformerModel(nn.Module):
    """
    FEDformer model for time-series forecasting.
    Frequency Enhanced Decomposed Transformer.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.05,
        modes: int = 64,
        output_size: int = 1,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 1
    ):
        """
        Parameters:
        -----------
        input_size : int
            Number of input features
        d_model : int
            Model dimension
        n_heads : int
            Number of attention heads
        e_layers : int
            Number of encoder layers
        d_layers : int
            Number of decoder layers
        d_ff : int
            Feedforward dimension
        dropout : float
            Dropout rate
        modes : int
            Number of frequency modes
        output_size : int
            Output size
        seq_len : int
            Input sequence length
        label_len : int
            Label length for decoder
        pred_len : int
            Prediction length
        """
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        # Embedding
        self.value_embedding = nn.Linear(input_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        
        # Encoder
        self.encoder = nn.ModuleList([
            FEDformerEncoderLayer(d_model, n_heads, d_ff, dropout, modes)
            for _ in range(e_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            FEDformerDecoderLayer(d_model, n_heads, d_ff, dropout, modes)
            for _ in range(d_layers)
        ])
        
        # Output projection
        self.projection = nn.Linear(d_model, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input (B, seq_len, input_size)
            
        Returns:
        --------
        torch.Tensor
            Predictions (B, pred_len, output_size) or (B, output_size) if pred_len=1
        """
        B, L, _ = x.shape
        
        # Embedding
        x = self.value_embedding(x)
        pos = torch.arange(L, device=x.device).unsqueeze(0).repeat(B, 1)
        x = x + self.position_embedding(pos)
        
        # Encoder
        enc_out = x
        for layer in self.encoder:
            enc_out = layer(enc_out)
        
        # Decoder (using mean pooling for simplicity)
        dec_in = enc_out.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        dec_out = dec_in
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out)
        
        # Projection
        output = self.projection(dec_out)
        
        return output.squeeze(1) if self.pred_len == 1 else output

