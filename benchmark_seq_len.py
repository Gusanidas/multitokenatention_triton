import torch
import torch.nn as nn
import time
import json
from mta_attention_pytorch import MTAAttention
from mta_attention_triton import MTAAttentionTriton

# Configuration class to mimic MTA settings
class MTAConfig:
    def __init__(self):
        self.use_mta = True
        self.query_kernel_size = 7
        self.key_kernel_size = 7
        self.after_sm_query_kernel_size = 7
        self.after_sm_key_kernel_size = 7
        self.pre_sm_linear_head = True
        self.post_sm_linear_head = True
        self.pad_key = "both"
        self.init_method = "identity"
        self.dim = 512

# Single configuration that will be tested with varying sequence lengths
BATCH_SIZE = 1
N_HEADS = 128
HEAD_DIM = 64
DROPOUT = 0.0
CAUSAL = True
DTYPE = torch.float16
USE_MASK = False

# Sequence length range parameters
SEQ_LEN_START = 2048
SEQ_LEN_END = 4096
SEQ_LEN_STEP = 128

WARMUP_RUNS = 2
BENCHMARK_RUNS = 15

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

def run_sequence_length_benchmarks():
    """Run benchmarks varying sequence length between SEQ_LEN_START and SEQ_LEN_END"""
    
    print("=" * 80)
    print("BENCHMARKING MTA ATTENTION - SEQUENCE LENGTH VARIATION")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, benchmarks will not be meaningful")
    
    print(f"Device: {device}")
    print(f"Configuration: batch={BATCH_SIZE}, heads={N_HEADS}, head_dim={HEAD_DIM}")
    print(f"Sequence length range: {SEQ_LEN_START} to {SEQ_LEN_END} (step={SEQ_LEN_STEP})")
    print(f"Warmup runs: {WARMUP_RUNS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")
    print()
    
    results = []
    mta_config = MTAConfig()
    
    for seq_len in range(SEQ_LEN_START, SEQ_LEN_END + 1, SEQ_LEN_STEP):
        print(f"Benchmarking sequence length: {seq_len}")
        
        model_pytorch = MTAAttention(
            n_heads=N_HEADS,
            mta=mta_config,
            dropout=DROPOUT
        ).to(device)
        model_pytorch.to(dtype=DTYPE)
        
        model_triton = MTAAttentionTriton(
            n_heads=N_HEADS,
            mta=mta_config,
            dropout=DROPOUT,
            causal=CAUSAL,
            dtype=DTYPE,
            use_mask=USE_MASK,
        ).to(device)
        
        model_pytorch.reset_mta_parameters()
        model_triton.reset_mta_parameters()
        
        model_triton.copy_mta_parameters(model_pytorch)
        
        model_pytorch.eval()
        model_triton.eval()
        
        torch.manual_seed(123)
        xq = torch.randn(BATCH_SIZE, N_HEADS, seq_len, HEAD_DIM, device=device, requires_grad=True, dtype=DTYPE)
        xk = torch.randn(BATCH_SIZE, N_HEADS, seq_len, HEAD_DIM, device=device, requires_grad=True, dtype=DTYPE)
        xv = torch.randn(BATCH_SIZE, N_HEADS, seq_len, HEAD_DIM, device=device, requires_grad=True, dtype=DTYPE)
        
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
            
            result = {
                'seq_len': seq_len,
                'batch_size': BATCH_SIZE,
                'n_heads': N_HEADS,
                'head_dim': HEAD_DIM,
                'dropout': DROPOUT,
                'causal': CAUSAL,
                'dtype': str(DTYPE),
                'use_mask': USE_MASK,
                'query_kernel_size': mta_config.query_kernel_size,
                'key_kernel_size': mta_config.key_kernel_size,
                'after_sm_query_kernel_size': mta_config.after_sm_query_kernel_size,
                'after_sm_key_kernel_size': mta_config.after_sm_key_kernel_size,
                'pytorch_fwd_time': pytorch_fwd_time,
                'triton_fwd_time': triton_fwd_time,
                'pytorch_bwd_time': pytorch_bwd_time,
                'triton_bwd_time': triton_bwd_time,
            }
            
            results.append(result)
            print(f"  PyTorch - Fwd: {pytorch_fwd_time:.3f}ms, Bwd: {pytorch_bwd_time:.3f}ms")
            print(f"  Triton  - Fwd: {triton_fwd_time:.3f}ms, Bwd: {triton_bwd_time:.3f}ms")
            
        except Exception as e:
            print(f"  Error benchmarking seq_len={seq_len}: {e}")
            
            result = {
                'seq_len': seq_len,
                'batch_size': BATCH_SIZE,
                'n_heads': N_HEADS,
                'head_dim': HEAD_DIM,
                'dropout': DROPOUT,
                'causal': CAUSAL,
                'dtype': str(DTYPE),
                'use_mask': USE_MASK,
                'query_kernel_size': mta_config.query_kernel_size,
                'key_kernel_size': mta_config.key_kernel_size,
                'after_sm_query_kernel_size': mta_config.after_sm_query_kernel_size,
                'after_sm_key_kernel_size': mta_config.after_sm_key_kernel_size,
                'pytorch_fwd_time': None,
                'triton_fwd_time': None,
                'pytorch_bwd_time': None,
                'triton_bwd_time': None,
            }
            results.append(result)
    
    # Save results to JSONL file
    output_file = 'benchmark_seq_len_results.jsonl'
    with open(output_file, 'a') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nResults saved to {output_file}")
    print(f"Total configurations benchmarked: {len(results)}")
    
    # Print summary statistics
    successful_results = [r for r in results if r['pytorch_fwd_time'] is not None]
    if successful_results:
        print(f"Successful benchmarks: {len(successful_results)}")
        avg_pytorch_fwd = sum(r['pytorch_fwd_time'] for r in successful_results) / len(successful_results)
        avg_triton_fwd = sum(r['triton_fwd_time'] for r in successful_results) / len(successful_results)
        avg_pytorch_bwd = sum(r['pytorch_bwd_time'] for r in successful_results) / len(successful_results)
        avg_triton_bwd = sum(r['triton_bwd_time'] for r in successful_results) / len(successful_results)
        
        print(f"Average PyTorch forward time: {avg_pytorch_fwd:.3f}ms")
        print(f"Average Triton forward time: {avg_triton_fwd:.3f}ms")
        print(f"Average PyTorch backward time: {avg_pytorch_bwd:.3f}ms")
        print(f"Average Triton backward time: {avg_triton_bwd:.3f}ms")
        
        if avg_triton_fwd > 0:
            print(f"Average forward speedup: {avg_pytorch_fwd/avg_triton_fwd:.2f}x")
        if avg_triton_bwd > 0:
            print(f"Average backward speedup: {avg_pytorch_bwd/avg_triton_bwd:.2f}x")

if __name__ == "__main__":
    t0 = time.time()
    run_sequence_length_benchmarks()
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")
