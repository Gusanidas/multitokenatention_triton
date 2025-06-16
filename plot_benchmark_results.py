import json
import matplotlib.pyplot as plt
import numpy as np

def load_benchmark_results(filename='benchmark_seq_len_results.jsonl'):
    """Load benchmark results from JSONL file"""
    results = []
    with open(filename, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def plot_benchmark_results(results, n_heads=None, dtype=None, query_kernel_size=None, key_kernel_size=None):
    """Create plots for benchmark results with optional parameter filtering
    
    Args:
        results: List of benchmark results
        n_heads: Filter for specific number of heads
        dtype: Filter for specific data type
        query_kernel_size: Filter for specific query kernel size
        key_kernel_size: Filter for specific key kernel size
    """
    
    # Filter results based on specified parameters
    filtered_results = results
    if n_heads is not None:
        filtered_results = [r for r in filtered_results if r.get('n_heads') == n_heads]
    if dtype is not None:
        filtered_results = [r for r in filtered_results if r.get('dtype') == dtype]
    if query_kernel_size is not None:
        filtered_results = [r for r in filtered_results if r.get('query_kernel_size') == query_kernel_size]
    if key_kernel_size is not None:
        filtered_results = [r for r in filtered_results if r.get('key_kernel_size') == key_kernel_size]
    
    # Filter out failed benchmarks
    successful_results = [r for r in filtered_results if r['pytorch_fwd_time'] is not None]
    
    if not successful_results:
        print("No successful benchmark results found for the specified parameters!")
        return
    
    # Extract data
    seq_lens = [r['seq_len'] for r in successful_results]
    pytorch_fwd_times = [r['pytorch_fwd_time'] for r in successful_results]
    triton_fwd_times = [r['triton_fwd_time'] for r in successful_results]
    pytorch_bwd_times = [r['pytorch_bwd_time'] for r in successful_results]
    triton_bwd_times = [r['triton_bwd_time'] for r in successful_results]
    
    # Calculate speedups
    fwd_speedups = [pt/tr for pt, tr in zip(pytorch_fwd_times, triton_fwd_times)]
    bwd_speedups = [pt/tr for pt, tr in zip(pytorch_bwd_times, triton_bwd_times)]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Create title with parameter information
    title = 'MTA Attention Benchmark Results: PyTorch vs Triton\n'
    if n_heads is not None:
        title += f'n_heads={n_heads}, '
    if dtype is not None:
        title += f'dtype={dtype}, '
    if query_kernel_size is not None:
        title += f'query_kernel_size={query_kernel_size}, '
    if key_kernel_size is not None:
        title += f'key_kernel_size={key_kernel_size}'
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Forward Times Comparison
    ax1.plot(seq_lens, pytorch_fwd_times, 'b-o', label='PyTorch Forward', linewidth=2, markersize=6)
    ax1.plot(seq_lens, triton_fwd_times, 'r-s', label='Triton Forward', linewidth=2, markersize=6)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Forward Time Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')  # Add logarithmic scale
    
    # Plot 2: Backward Times Comparison
    ax2.plot(seq_lens, pytorch_bwd_times, 'b-^', label='PyTorch Backward', linewidth=2, markersize=6)
    ax2.plot(seq_lens, triton_bwd_times, 'r-d', label='Triton Backward', linewidth=2, markersize=6)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Backward Time Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')  # Add logarithmic scale
    
    # Plot 3: Speedup Comparison
    ax3.plot(seq_lens, fwd_speedups, 'purple', marker='o', label='Forward Speedup', linewidth=2, markersize=6)  # Changed to purple
    ax3.plot(seq_lens, bwd_speedups, 'orange', marker='s', label='Backward Speedup', linewidth=2, markersize=6)  # Changed to orange
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Speedup (PyTorch/Triton)')
    ax3.set_title('Speedup Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('benchmark_results_plot_1.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'benchmark_results_plot_1.png'")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics with parameter information
    print(f"\nSummary Statistics:")
    if n_heads is not None:
        print(f"Number of heads: {n_heads}")
    if dtype is not None:
        print(f"Data type: {dtype}")
    if query_kernel_size is not None:
        print(f"Query kernel size: {query_kernel_size}")
    if key_kernel_size is not None:
        print(f"Key kernel size: {key_kernel_size}")
    print(f"Sequence lengths tested: {min(seq_lens)} to {max(seq_lens)} (step: {seq_lens[1] - seq_lens[0] if len(seq_lens) > 1 else 'N/A'})")
    print(f"Average forward speedup: {np.mean(fwd_speedups):.2f}x")
    print(f"Average backward speedup: {np.mean(bwd_speedups):.2f}x")
    print(f"Max forward speedup: {max(fwd_speedups):.2f}x at seq_len={seq_lens[fwd_speedups.index(max(fwd_speedups))]}")
    print(f"Max backward speedup: {max(bwd_speedups):.2f}x at seq_len={seq_lens[bwd_speedups.index(max(bwd_speedups))]}")

if __name__ == "__main__":
    try:
        results = load_benchmark_results()
        # Example usage with specific parameters
        plot_benchmark_results(
            results,
            n_heads=128,  # Specify desired number of heads
            dtype='torch.float16',  # Specify desired data type
            query_kernel_size=7,  # Specify desired query kernel size
            key_kernel_size=7  # Specify desired key kernel size
        )
    except FileNotFoundError:
        print("benchmark_seq_len_results.jsonl not found. Please run the benchmark script first.")
    except Exception as e:
        print(f"Error plotting results: {e}")