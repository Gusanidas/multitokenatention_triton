import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from conv_attention_triton import TritonConvAttention


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

        return output, scores

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


def compare_conv_attention():
    """Compare ConvAttention PyTorch implementation with Triton kernel"""

    print("=" * 80)
    print("COMPARING CONV ATTENTION PYTORCH vs TRITON")
    print("=" * 80)

    # Test parameters
    batch_size = 2
    n_heads = 8
    seq_len = 128
    head_dim = 64
    kernel_size_q = 3
    kernel_size_k = 3
    causal = True

    print(
        f"Configuration: batch={batch_size}, heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}"
    )
    print(f"Kernel sizes: Q={kernel_size_q}, K={kernel_size_k}, Causal: {causal}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, using CPU")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create test inputs
    Q = torch.randn(
        batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True
    )
    K = torch.randn(
        batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True
    )
    V = torch.randn(
        batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True
    )

    print(f"Input tensors - Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
    print(
        f"Q - Mean abs: {Q.abs().mean().item():.6f}, Max abs: {Q.abs().max().item():.6f}"
    )
    print(
        f"K - Mean abs: {K.abs().mean().item():.6f}, Max abs: {K.abs().max().item():.6f}"
    )

    # Create models
    pytorch_model = ConvAttention(n_heads, kernel_size_q, kernel_size_k).to(device)

    # Create softmax scale
    softmax_scale = 1.0 / (head_dim**0.5)

    print("\nTesting PyTorch implementation...")
    with torch.no_grad():
        pytorch_output, pytorch_qk = pytorch_model(Q, K, V, causal, softmax_scale)

    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"PyTorch QK shape: {pytorch_qk.shape}")

    print("\nTesting Triton implementation...")
    # Create Triton model with same parameters as PyTorch model
    triton_model = TritonConvAttention(
        n_heads, (kernel_size_q, kernel_size_k), causal
    ).to(device)
    # Copy weights from PyTorch model
    triton_model.weight.data = pytorch_model.conv_weight.data.clone()

    with torch.no_grad():
        triton_result = triton_model(Q, K, V, causal, softmax_scale)

        # Handle case where Triton kernel might only return final output
        if isinstance(triton_result, tuple) and len(triton_result) == 2:
            triton_output, triton_qk = triton_result
        elif triton_result is not None:
            triton_output = triton_result
            triton_qk = None
            print(
                "Note: Triton kernel only returned final output, no intermediate values"
            )
        else:
            triton_output = None
            triton_qk = None
            print("Error: Triton kernel returned None")

    if triton_output is not None and hasattr(triton_output, "shape"):
        print(f"Triton output shape: {triton_output.shape}")
    else:
        print("Triton output is None or invalid")

    def compare_tensors(name, tensor1, tensor2):
        """Compare two tensors and print detailed statistics"""
        print(f"\n{name} Comparison:")
        print(f"  PyTorch - Shape: {tensor1.shape}")
        print(f"  Triton - Shape: {tensor2.shape}")
        print(
            f"  PyTorch - Max abs: {tensor1.abs().max().item():.8f}, Mean abs: {tensor1.abs().mean().item():.8f}"
        )
        print(
            f"  Triton - Max abs: {tensor2.abs().max().item():.8f}, Mean abs: {tensor2.abs().mean().item():.8f}"
        )
        # Number of zeros
        print()
        print(
            f"  PyTorch - Number of zeros: {tensor1.eq(0).sum().item()}, percentage: {tensor1.eq(0).sum().item() / tensor1.numel() * 100:.2f}%"
        )
        print(
            f"  Triton - Number of zeros: {tensor2.eq(0).sum().item()}, percentage: {tensor2.eq(0).sum().item() / tensor2.numel() * 100:.2f}%"
        )
        print()
        print(f"  Pytorch - Number of inf: {tensor1.isinf().sum().item()}")
        print(f"  Triton - Number of inf: {tensor2.isinf().sum().item()}")
        if tensor2.isinf().sum().item() > 0:
            # Index of inf
            inf_idx = tensor2.isinf().nonzero()
            print(f"  Inf index: {inf_idx}")
        print()

        diff = tensor1 - tensor2
        print(
            f"  Difference - Max abs: {diff.abs().max().item():.8f}, Mean abs: {diff.abs().mean().item():.8f}"
        )

        rel_error = (diff.abs() / (tensor1.abs() + 1e-8)).mean().item()
        print(f"  Relative error: {rel_error:.8f}")

        # Test with different tolerances
        match_strict = torch.allclose(tensor1, tensor2, rtol=1e-5, atol=1e-7)
        match_loose = torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-5)

        print(f"  Match (strict: rtol=1e-5, atol=1e-7): {match_strict}")
        print(f"  Match (loose: rtol=1e-3, atol=1e-5): {match_loose}")

        # No of small diff value, less than 1e-5
        for threshold in [1e-5, 1e-3, 1e-2, 1e-1]:
            small_diff = (diff.abs() < threshold).sum().item()
            print(
                f"  No of small diff value, less than {threshold}: {small_diff}, percentage: {small_diff / diff.numel() * 100:.2f}%"
            )

        return match_loose

    # Compare outputs
    print("\nForward Pass Comparisons:")
    final_match = compare_tensors("Final Output", pytorch_output, triton_output)

    # Only compare QK if Triton returns it
    if triton_qk is not None:
        print()
        qk_match = compare_tensors("QK", pytorch_qk, triton_qk)
    else:
        qk_match = None
        print("\nQK comparison skipped - Triton kernel doesn't return QK values")

    # Show PyTorch QK values for reference
    print(f"\nPyTorch QK Values (for reference):")
    print(
        f"  QK - Mean abs: {pytorch_qk.abs().mean().item():.8f}, Max abs: {pytorch_qk.abs().max().item():.8f}"
    )

    # Overall summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print("Forward pass matches:")
    print(f"  Final output (loose tolerance): {final_match}")
    if qk_match is not None:
        print(f"  QK scores (loose tolerance): {qk_match}")

    if not final_match:
        print("\nNote: Comparison failed. Possible reasons:")
        print("- Different implementation details between PyTorch and Triton")
        print("- Numerical precision differences")
        print("- Different convolution or masking strategies")
        print("- Acceptable floating-point errors within tolerance")
    else:
        print("\nSuccess: PyTorch and Triton implementations match!")

    # Additional diagnostic information
    print(f"\nDiagnostic Information:")
    print(f"PyTorch conv weight shape: {pytorch_model.conv_weight.shape}")
    print(f"Softmax scale: {softmax_scale}")
    print(f"Device: {device}")

    # Plot output matrices for batch=0, head=0
    print(f"\nPlotting output matrices for batch=0, head=0...")

    if pytorch_output is not None:
        # Extract batch=0, head=0 for PyTorch
        pytorch_out_2d = pytorch_output[0, 0]  # [seq_len, head_dim]
        plot_matrix_colormap(
            pytorch_out_2d,
            title="PyTorch Output (batch=0, head=0)",
            figsize=(10, 8),
            cmap="RdBu_r",
        )

    if triton_output is not None and hasattr(triton_output, "shape"):
        # Extract batch=0, head=0 for Triton
        triton_out_2d = triton_output[0][0]  # [seq_len, head_dim]
        plot_matrix_colormap(
            triton_out_2d,
            title="Triton Output (batch=0, head=0)",
            figsize=(10, 8),
            cmap="RdBu_r",
        )
    else:
        print("Triton output not available for plotting")


def plot_matrix_colormap(
    matrix, title="Matrix Colormap", figsize=(10, 8), cmap="viridis"
):
    """
    Plot a 2D matrix as a colormap
    Args:
        matrix: 2D tensor or numpy array to plot
        title: Title for the plot
        figsize: Figure size (width, height)
        cmap: Colormap to use
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    plt.figure(figsize=figsize)
    plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("K dimension")
    plt.ylabel("Q dimension")
    plt.show()


if __name__ == "__main__":
    compare_conv_attention()
