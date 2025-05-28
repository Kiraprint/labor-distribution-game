import math
from turtle import pen

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional encoding for transformer

        Args:
            d_model: Embedding dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer to store positional encoding
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings

        Args:
            x: Input embeddings [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]

        Returns:
            Embeddings with positional encoding added
        """
        if x.dim() == 3:  # [batch_size, seq_len, embedding_dim]
            x = x + self.pe[: x.size(1), :].unsqueeze(0)
        else:  # [seq_len, embedding_dim]
            x = x + self.pe[: x.size(0), :]

        return self.dropout(x)


class HistoryTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout=0.1):
        """
        Transformer for processing sequential history data

        Args:
            embedding_dim: Dimension of input and output embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super(HistoryTransformer, self).__init__()

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        # Output projection (same dimension as input for residual)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        """
        Forward pass through transformer

        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            mask: Optional attention mask

        Returns:
            Contextualized embeddings [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through transformer
        if mask is not None:
            output = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            output = self.transformer_encoder(x)

        # Project output
        output = self.output_projection(output)

        # Return the embedding for the entire sequence (CLS token equivalent)
        # or the last token's embedding
        if output.dim() == 3:  # [batch_size, seq_len, embedding_dim]
            # Get the embedding for the last token in each sequence
            last_token_embedding = output[:, -1, :]
            return last_token_embedding
        else:  # [seq_len, embedding_dim]
            # Get the embedding for the last token
            return output[-1]
