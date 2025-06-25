import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAttention(nn.Module):
    """
    Simplified conv attention that only applies convolution before softmax.
    Similar to mta_attention_pytorch.py but much simpler.
    """

    def __init__(self, n_heads: int, kernel_size_q: int = 3, kernel_size_k: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.kernel_size_q = kernel_size_q
        self.kernel_size_k = kernel_size_k

        # Convolution weight
        self.conv_weight = torch.nn.parameter.Parameter(
            torch.empty(n_heads, kernel_size_q, kernel_size_k)
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize with identity-like kernel (center weight = 1, others = 0)"""
        with torch.no_grad():
            self.conv_weight.zero_()
            center_q = self.kernel_size_q - 1  # Last row
            center_k = self.kernel_size_k // 2  # Center column
            self.conv_weight[:, center_q, center_k] = 1.0

    def forward(self, Q, K, V, causal=True, softmax_scale=None):
        """
        Forward pass with convolution before softmax
        Args:
            Q: [batch_size, n_heads, seq_len, head_dim]
            K: [batch_size, n_heads, seq_len, head_dim]
            V: [batch_size, n_heads, seq_len, head_dim]
            causal: whether to apply causal masking
            softmax_scale: scaling factor for softmax
        Returns:
            output: [batch_size, n_heads, seq_len, head_dim]
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape

        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        # Compute QK scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, seq, seq]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1)
        mask = mask.bool()
        if causal:
            scores = scores.masked_fill(mask, 0)

        # Apply convolution to scores
        conv_scores = self._apply_convolution(scores)

        # Apply causal mask if needed
        if causal:
            conv_scores = conv_scores.masked_fill(mask, float("-inf"))

        # Apply softmax
        conv_scores_scaled = conv_scores * softmax_scale
        attn_weights = F.softmax(conv_scores_scaled, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        return output

    def _apply_convolution(self, scores):
        """Apply convolution to attention scores"""
        batch_size, n_heads, seq_len, _ = scores.shape

        # Pad scores for convolution
        # For "both" padding strategy like in MTA
        pad_k = (self.kernel_size_k - 1) // 2
        pad_q = self.kernel_size_q - 1

        scores_padded = F.pad(scores, (pad_k, pad_k, pad_q, 0), value=0.0)

        # Apply convolution with grouped convolution
        conv_scores = F.conv2d(
            scores_padded,
            self.conv_weight.unsqueeze(1),  # Add channel dim
            padding=0,
            groups=n_heads,
        )

        return conv_scores