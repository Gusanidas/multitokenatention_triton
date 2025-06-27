import torch
import traceback
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd
from tabulate import tabulate
from conv_attention_pytorch import ConvAttention
from conv_attention_triton import TritonConvAttention

# Benchmark configurations
BENCHMARK_CONFIGS = [
    # (batch_size, n_heads, seq_len, head_dim)
    (1, 8, 512, 64),
    (1, 8, 1024, 64),
    (1, 16, 1024, 64),
    (2, 8, 1024, 64),
    (2, 32, 1024, 64),
    (8, 8, 1024, 64),
    (2, 8, 1024, 128),
    (1, 8, 2048, 64),
    (1, 16, 2048, 64),
    (1, 8, 4096, 64),
    (1, 8, 8192, 64),
    (1, 8, 4096, 128),
]

WARMUP_RUNS = 3
BENCHMARK_RUNS = 10
DTYPE = torch.bfloat16


class StandardAttention(nn.Module):
    """Standard attention implementation for comparison"""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, Q, K, V, causal=True, softmax_scale=None):
        """
        Standard attention forward pass using PyTorch's Flash Attention
        Args:
            Q: [batch_size, n_heads, seq_len, head_dim]
            K: [batch_size, n_heads, seq_len, head_dim]
            V: [batch_size, n_heads, seq_len, head_dim]
            causal: whether to apply causal masking
            softmax_scale: scaling factor for softmax
        Returns:
            output: [batch_size, n_heads, seq_len, head_dim]
        """
        # Use PyTorch's optimized scaled dot product attention (Flash Attention)
        output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )

        return output


def benchmark_forward_only(
    model,
    Q,
    K,
    V,
    causal=True,
    softmax_scale=None,
    num_runs=BENCHMARK_RUNS,
    warmup=WARMUP_RUNS,
):
    """Benchmark forward pass only"""
    device = Q.device

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            if isinstance(model, ConvAttention):
                _ = model(Q, K, V, causal, softmax_scale)
            elif isinstance(model, TritonConvAttention):
                _ = model(Q, K, V, causal, softmax_scale)
            else:  # Standard attention
                _ = model(Q, K, V, causal, softmax_scale)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            if isinstance(model, ConvAttention):
                _ = model(Q, K, V, causal, softmax_scale)
            elif isinstance(model, TritonConvAttention):
                _ = model(Q, K, V, causal, softmax_scale)
            else:  # Standard attention
                _ = model(Q, K, V, causal, softmax_scale)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    return (end - start) / num_runs * 1000  # ms


def run_benchmarks():
    """Run comprehensive benchmarks comparing PyTorch conv attention, Triton conv attention, and standard attention"""

    print("=" * 100)
    print("BENCHMARKING CONV ATTENTION: PYTORCH vs TRITON vs STANDARD")
    print("=" * 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, benchmarks will not be meaningful")

    print(f"Device: {device}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")
    print(f"Data type: {DTYPE}")
    print()

    results = []

    for batch_size, n_heads, seq_len, head_dim in BENCHMARK_CONFIGS:
        print(
            f"Benchmarking: batch={batch_size}, heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}"
        )

        # Create models
        kernel_size_q = 7
        kernel_size_k = 11
        causal = True
        softmax_scale = 1.0 / (head_dim**0.5)

        conv_attention_pytorch = (
            ConvAttention(n_heads, kernel_size_q, kernel_size_k).to(device).to(DTYPE)
        )
        standard_attention = StandardAttention(n_heads).to(device)

        # Create input tensors
        torch.manual_seed(42)
        Q = torch.randn(
            batch_size, n_heads, seq_len, head_dim, device=device, dtype=DTYPE
        )
        K = torch.randn(
            batch_size, n_heads, seq_len, head_dim, device=device, dtype=DTYPE
        )
        V = torch.randn(
            batch_size, n_heads, seq_len, head_dim, device=device, dtype=DTYPE
        )

        try:
            # Benchmark PyTorch conv attention
            pytorch_time = benchmark_forward_only(
                conv_attention_pytorch, Q, K, V, causal, softmax_scale
            )

            # Benchmark Triton conv attention
            triton_time = None
            try:
                # Set up Triton model with same conv weights
                # triton_model = type(
                #    "TritonModel",
                #    (),
                #    {
                #        "conv_weight": conv_attention_pytorch.conv_weight,
                #        "apply": TritonAttention.apply,
                #    },
                # )()
                triton_model = (
                    TritonConvAttention(n_heads, (kernel_size_q, kernel_size_k))
                    .to(device)
                    .to(DTYPE)
                )

                triton_time = benchmark_forward_only(
                    triton_model, Q, K, V, causal, softmax_scale
                )
            except Exception as e:
                traceback.print_exc()
                print(f"  Error benchmarking Triton: {e}")
                triton_time = None

            # Benchmark standard attention
            standard_time = benchmark_forward_only(
                standard_attention, Q, K, V, causal, softmax_scale
            )

            # Calculate speedups
            pytorch_vs_standard = (
                standard_time / pytorch_time if pytorch_time > 0 else float("inf")
            )
            if triton_time is not None:
                triton_vs_pytorch = (
                    pytorch_time / triton_time if triton_time > 0 else float("inf")
                )
                triton_vs_standard = (
                    standard_time / triton_time if triton_time > 0 else float("inf")
                )
            else:
                triton_vs_pytorch = "ERROR"
                triton_vs_standard = "ERROR"

            results.append(
                {
                    "Batch": batch_size,
                    "Heads": n_heads,
                    "Seq Len": seq_len,
                    "Head Dim": head_dim,
                    "PyTorch Conv (ms)": f"{pytorch_time:.3f}",
                    "Triton Conv (ms)": (
                        f"{triton_time:.3f}" if triton_time is not None else "ERROR"
                    ),
                    "Standard (ms)": f"{standard_time:.3f}",
                    "PyTorch vs Standard": f"{pytorch_vs_standard:.2f}x",
                    "Triton vs PyTorch": (
                        f"{triton_vs_pytorch:.2f}x"
                        if triton_time is not None
                        else "ERROR"
                    ),
                    "Triton vs Standard": (
                        f"{triton_vs_standard:.2f}x"
                        if triton_time is not None
                        else "ERROR"
                    ),
                }
            )

        except Exception as e:
            print(f"  Error benchmarking configuration: {e}")
            # Traceback
            traceback.print_exc()
            results.append(
                {
                    "Batch": batch_size,
                    "Heads": n_heads,
                    "Seq Len": seq_len,
                    "Head Dim": head_dim,
                    "PyTorch Conv (ms)": "ERROR",
                    "Triton Conv (ms)": "ERROR",
                    "Standard (ms)": "ERROR",
                    "PyTorch vs Standard": "ERROR",
                    "Triton vs PyTorch": "ERROR",
                    "Triton vs Standard": "ERROR",
                }
            )

    print("\n" + "=" * 140)
    print("BENCHMARK RESULTS")
    print("=" * 140)
    df = pd.DataFrame(results)
    print(
        tabulate(
            df.values, headers=df.columns.tolist(), tablefmt="grid", showindex=False
        )
    )

    # Calculate average speedups (excluding errors)
    pytorch_vs_standard_speedups = []
    triton_vs_pytorch_speedups = []
    triton_vs_standard_speedups = []

    for result in results:
        try:
            if result["PyTorch vs Standard"] != "ERROR":
                pytorch_vs_standard_speedups.append(
                    float(result["PyTorch vs Standard"].replace("x", ""))
                )
            if result["Triton vs PyTorch"] != "ERROR":
                triton_vs_pytorch_speedups.append(
                    float(result["Triton vs PyTorch"].replace("x", ""))
                )
            if result["Triton vs Standard"] != "ERROR":
                triton_vs_standard_speedups.append(
                    float(result["Triton vs Standard"].replace("x", ""))
                )
        except (ValueError, AttributeError):
            pass

    if pytorch_vs_standard_speedups:
        avg_pytorch_vs_standard = sum(pytorch_vs_standard_speedups) / len(
            pytorch_vs_standard_speedups
        )
        print(
            f"\nAverage PyTorch Conv vs Standard Attention: {avg_pytorch_vs_standard:.2f}x"
        )

    if triton_vs_pytorch_speedups:
        avg_triton_vs_pytorch = sum(triton_vs_pytorch_speedups) / len(
            triton_vs_pytorch_speedups
        )
        print(f"Average Triton Conv vs PyTorch Conv: {avg_triton_vs_pytorch:.2f}x")

    if triton_vs_standard_speedups:
        avg_triton_vs_standard = sum(triton_vs_standard_speedups) / len(
            triton_vs_standard_speedups
        )
        print(
            f"Average Triton Conv vs Standard Attention: {avg_triton_vs_standard:.2f}x"
        )

    softmax_scale = 1.0 / (head_dim**0.5)
    causal = True
    # Memory usage comparison
    if device == "cuda":
        print("\n" + "=" * 80)
        print("MEMORY USAGE ANALYSIS")
        print("=" * 80)

        batch_size, n_heads, seq_len, head_dim = 2, 8, 1024, 64

        try:
            torch.cuda.reset_peak_memory_stats()
            conv_attention_mem = ConvAttention(n_heads, 3, 3).to(device).to(DTYPE)
            Q_mem = torch.randn(
                batch_size, n_heads, seq_len, head_dim, device=device, dtype=DTYPE
            )
            K_mem = torch.randn(
                batch_size, n_heads, seq_len, head_dim, device=device, dtype=DTYPE
            )
            V_mem = torch.randn(
                batch_size, n_heads, seq_len, head_dim, device=device, dtype=DTYPE
            )
            _ = conv_attention_mem(Q_mem, K_mem, V_mem, causal, softmax_scale)
            conv_peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

            torch.cuda.reset_peak_memory_stats()
            standard_attention_mem = StandardAttention(n_heads).to(device)
            _ = standard_attention_mem(Q_mem, K_mem, V_mem, causal, softmax_scale)
            standard_peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

            torch.cuda.reset_peak_memory_stats()
            triton_model = TritonConvAttention(n_heads, (kernel_size_q, kernel_size_k))
            _ = triton_model(Q_mem, K_mem, V_mem, causal, softmax_scale)
            triton_peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"Triton Conv Attention peak memory: {triton_peak_mem:.2f} MB")

            print(
                f"Test configuration: batch={batch_size}, heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}"
            )
            print(f"PyTorch Conv Attention peak memory: {conv_peak_mem:.2f} MB")
            print(f"Standard Attention peak memory: {standard_peak_mem:.2f} MB")
            print(f"Triton Conv Attention peak memory: {triton_peak_mem:.2f} MB")
            print(
                f"Memory ratio (Conv/Standard): {conv_peak_mem/standard_peak_mem:.2f}x"
            )
            print(f"Memory ratio (Conv/Triton): {conv_peak_mem/triton_peak_mem:.2f}x")
            print(
                f"Memory ratio (Triton/Standard): {triton_peak_mem/standard_peak_mem:.2f}x"
            )

        except Exception as e:
            print(f"Error in memory analysis: {e}")


if __name__ == "__main__":
    run_benchmarks()
