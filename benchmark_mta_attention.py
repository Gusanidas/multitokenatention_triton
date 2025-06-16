import torch
import torch.nn as nn
import time
import pandas as pd
from tabulate import tabulate
from mta_attention_pytorch import MTAAttention
from mta_attention_triton import MTAAttentionTriton

# Configuration class to mimic MTA settings
class MTAConfig:
    def __init__(self):
        self.use_mta = True
        self.query_kernel_size = 11
        self.key_kernel_size = 15
        self.after_sm_query_kernel_size = 11
        self.after_sm_key_kernel_size = 15
        self.pre_sm_linear_head = True
        self.post_sm_linear_head = True
        self.pad_key = "both"
        self.init_method = "identity"
        self.dim = 512  

# Benchmark configurations
BENCHMARK_CONFIGS = [
    # (batch_size, n_heads, seq_len, head_dim)
    (1, 16, 512, 64),
    (1, 16, 1024, 128),
    (1, 16, 1024, 64),
    (8, 16, 1024, 64),
    (1, 16, 2048, 64),
    (1, 32, 2048, 64),
    (1, 16, 4096, 64),
]

WARMUP_RUNS = 3
BENCHMARK_RUNS = 10
DROPOUT = 0.0  
DTYPE = torch.bfloat16

def benchmark_forward(model, xq, xk, xv, mask, chunk_start_ids, num_runs=BENCHMARK_RUNS, warmup=WARMUP_RUNS):
    """Benchmark forward pass only"""
    device = xq.device
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(xq, xk, xv, mask, chunk_start_ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(xq, xk, xv, mask, chunk_start_ids)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / num_runs * 1000  # ms

def benchmark_backward(model, xq, xk, xv, mask, chunk_start_ids, num_runs=BENCHMARK_RUNS, warmup=WARMUP_RUNS):
    """Benchmark forward + backward pass"""
    device = xq.device
    
    # Warmup
    for _ in range(warmup):
        xq.grad = None
        xk.grad = None
        xv.grad = None
        model.zero_grad()
        output = model(xq, xk, xv, mask, chunk_start_ids)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        xq.grad = None
        xk.grad = None
        xv.grad = None
        model.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        output = model(xq, xk, xv, mask, chunk_start_ids)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        
        times.append((end - start) * 1000)  # ms
    
    return sum(times) / len(times)

def run_benchmarks():
    """Run comprehensive benchmarks comparing PyTorch and Triton MTA attention implementations"""
    
    print("=" * 80)
    print("BENCHMARKING MTA ATTENTION PYTORCH vs TRITON")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, benchmarks will not be meaningful")
    
    print(f"Device: {device}")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")
    print(f"Dropout: {DROPOUT}")
    print()
    
    results = []
    mta_config = MTAConfig()
    
    for batch_size, n_heads, seq_len, head_dim in BENCHMARK_CONFIGS:
        print(f"Benchmarking: batch={batch_size}, heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}")
        
        model_pytorch = MTAAttention(
            n_heads=n_heads,
            mta=mta_config,
            dropout=DROPOUT
        ).to(device)
        model_pytorch.to(dtype=DTYPE)
        
        model_triton = MTAAttentionTriton(
            n_heads=n_heads,
            mta=mta_config,
            dropout=DROPOUT,
            causal=True,
            use_mask=False,
        ).to(device)
        model_triton.to(dtype=DTYPE)
        
        model_pytorch.reset_mta_parameters()
        model_triton.reset_mta_parameters()
        
        model_triton.copy_mta_parameters(model_pytorch)
        
        model_pytorch.eval()
        model_triton.eval()
        
        torch.manual_seed(123)
        xq = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True, dtype=DTYPE)
        xk = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True, dtype=DTYPE)
        xv = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True, dtype=DTYPE)
        
        xq_triton = xq.clone().detach().requires_grad_(True)
        xk_triton = xk.clone().detach().requires_grad_(True)
        xv_triton = xv.clone().detach().requires_grad_(True)
        
        mask = None  # Let models handle their own masking
        chunk_start_ids = None
        
        try:
            pytorch_fwd_time = benchmark_forward(model_pytorch, xq, xk, xv, mask, chunk_start_ids)
            triton_fwd_time = benchmark_forward(model_triton, xq_triton, xk_triton, xv_triton, mask, chunk_start_ids)
            
            pytorch_bwd_time = benchmark_backward(model_pytorch, xq, xk, xv, mask, chunk_start_ids)
            triton_bwd_time = benchmark_backward(model_triton, xq_triton, xk_triton, xv_triton, mask, chunk_start_ids)
            
            fwd_speedup = pytorch_fwd_time / triton_fwd_time if triton_fwd_time > 0 else float('inf')
            bwd_speedup = pytorch_bwd_time / triton_bwd_time if triton_bwd_time > 0 else float('inf')
            
            results.append({
                'Batch': batch_size,
                'Heads': n_heads,
                'Seq Len': seq_len,
                'Head Dim': head_dim,
                'PyTorch Fwd (ms)': f"{pytorch_fwd_time:.3f}",
                'Triton Fwd (ms)': f"{triton_fwd_time:.3f}",
                'Fwd Speedup': f"{fwd_speedup:.2f}x",
                'PyTorch F+B (ms)': f"{pytorch_bwd_time:.3f}",
                'Triton F+B (ms)': f"{triton_bwd_time:.3f}",
                'F+B Speedup': f"{bwd_speedup:.2f}x",
            })
            
        except Exception as e:
            print(f"  Error benchmarking configuration: {e}")
            results.append({
                'Batch': batch_size,
                'Heads': n_heads,
                'Seq Len': seq_len,
                'Head Dim': head_dim,
                'PyTorch Fwd (ms)': 'ERROR',
                'Triton Fwd (ms)': 'ERROR',
                'Fwd Speedup': 'ERROR',
                'PyTorch F+B (ms)': 'ERROR',
                'Triton F+B (ms)': 'ERROR',
                'F+B Speedup': 'ERROR',
            })
    
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Calculate average speedups (excluding errors)
    fwd_speedups = []
    bwd_speedups = []
    
    for result in results:
        fwd_speedup_str = result['Fwd Speedup']
        bwd_speedup_str = result['F+B Speedup']
        
        if fwd_speedup_str != 'ERROR' and bwd_speedup_str != 'ERROR':
            try:
                fwd_speedups.append(float(fwd_speedup_str.replace('x', '')))
                bwd_speedups.append(float(bwd_speedup_str.replace('x', '')))
            except ValueError:
                pass
    
    if fwd_speedups and bwd_speedups:
        avg_fwd_speedup = sum(fwd_speedups) / len(fwd_speedups)
        avg_bwd_speedup = sum(bwd_speedups) / len(bwd_speedups)
        
        print(f"\nAverage Speedups (successful runs only):")
        print(f"Forward Pass: {avg_fwd_speedup:.2f}x")
        print(f"Forward + Backward: {avg_bwd_speedup:.2f}x")
    
    # Memory usage comparison
    if device == 'cuda':
        print("\n" + "=" * 80)
        print("MEMORY USAGE ANALYSIS")
        print("=" * 80)
        
        batch_size, n_heads, seq_len, head_dim = 2, 8, 2048, 64
        
        try:
            torch.cuda.reset_peak_memory_stats()
            model_pytorch_mem = MTAAttention(n_heads=n_heads, mta=mta_config, dropout=DROPOUT).to(device)
            model_pytorch_mem.reset_mta_parameters()
            xq_mem = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
            xk_mem = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
            xv_mem = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
            _ = model_pytorch_mem(xq_mem, xk_mem, xv_mem, None, None)
            pytorch_peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            torch.cuda.reset_peak_memory_stats()
            model_triton_mem = MTAAttentionTriton(n_heads=n_heads, mta=mta_config, dropout=DROPOUT, causal=True, dtype=DTYPE, use_mask=False).to(device)
            model_triton_mem.reset_mta_parameters()
            _ = model_triton_mem(xq_mem, xk_mem, xv_mem, None, None)
            triton_peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            print(f"Test configuration: batch={batch_size}, heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}")
            print(f"PyTorch peak memory: {pytorch_peak_mem:.2f} MB")
            print(f"Triton peak memory: {triton_peak_mem:.2f} MB")
            print(f"Memory ratio (Triton/PyTorch): {triton_peak_mem/pytorch_peak_mem:.2f}x")
            
        except Exception as e:
            print(f"Error in memory analysis: {e}")
    

if __name__ == "__main__":
    run_benchmarks()