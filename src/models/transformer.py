"""
Time-series Transformer for wind power prediction.
Implements patch-based temporal encoding and multi-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Parameters:
        -----------
        d_model : int
            Model dimension
        max_len : int
            Maximum sequence length
        dropout : float
            Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor (seq_len, batch_size, d_model)
            
        Returns:
        --------
        torch.Tensor
            Positionally encoded tensor
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch-based embedding for time-series data.
    Divides sequence into patches and projects to embedding dimension.
    """
    
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int
    ):
        """
        Parameters:
        -----------
        patch_size : int
            Size of each patch
        in_channels : int
            Number of input features
        embed_dim : int
            Embedding dimension
        """
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * in_channels, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x : torch.Tensor
            Input (batch_size, seq_len, in_channels)
            
        Returns:
        --------
        torch.Tensor
            Patched embeddings (batch_size, n_patches, embed_dim)
        """
        batch_size, seq_len, in_channels = x.shape
        
        # Pad if necessary
        if seq_len % self.patch_size != 0:
            pad_len = self.patch_size - (seq_len % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = x.shape[1]
        
        # Reshape to patches
        n_patches = seq_len // self.patch_size
        x = x.reshape(batch_size, n_patches, self.patch_size * in_channels)
        
        # Project to embedding dimension
        x = self.projection(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Parameters:
        -----------
        d_model : int
            Model dimension
        nhead : int
            Number of attention heads
        dim_feedforward : int
            Feedforward dimension
        dropout : float
            Dropout rate
        activation : str
            Activation function
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == 'relu' else F.gelu
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        src : torch.Tensor
            Input (batch_size, seq_len, d_model)
        src_mask : torch.Tensor, optional
            Attention mask
        src_key_padding_mask : torch.Tensor, optional
            Padding mask
            
        Returns:
        --------
        torch.Tensor
            Encoded output
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, 
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TimeSeriesTransformer(nn.Module):
    """
    Time-series Transformer for wind power prediction.
    Uses patch-based encoding and transformer encoder layers.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        patch_size: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        output_size: int = 1,
        use_physics_correction: bool = False
    ):
        """
        Parameters:
        -----------
        input_size : int
            Number of input features
        d_model : int
            Model dimension
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        dim_feedforward : int
            Feedforward dimension
        patch_size : int
            Patch size for patch embedding
        dropout : float
            Dropout rate
        max_seq_len : int
            Maximum sequence length
        output_size : int
            Output size (1 for regression)
        use_physics_correction : bool
            If True, output is physics correction term; if False, direct power prediction
        """
        super().__init__()
        self.use_physics_correction = use_physics_correction
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_size, input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.encoder_layers = encoder_layers
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_size)
        )
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input (batch_size, seq_len, input_size)
        src_mask : torch.Tensor, optional
            Attention mask
            
        Returns:
        --------
        torch.Tensor
            Predictions (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, n_patches, d_model)
        n_patches = x.shape[1]
        
        # Transpose for positional encoding (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        
        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Output projection
        output = self.output_proj(x)  # (batch_size, output_size)
        
        return output.squeeze(-1) if output.shape[-1] == 1 else output

