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

def plot_benchmark_results(results):
    """Create plots for benchmark results"""
    
    # Filter out failed benchmarks
    successful_results = [r for r in results if r['pytorch_fwd_time'] is not None]
    
    if not successful_results:
        print("No successful benchmark results found!")
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
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('MTA Attention Benchmark Results: PyTorch vs Triton', fontsize=16)
    
    # Plot 1: PyTorch Forward Time vs Sequence Length
    ax1.plot(seq_lens, pytorch_fwd_times, 'b-o', label='PyTorch Forward', linewidth=2, markersize=6)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('PyTorch Forward Time vs Sequence Length')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Triton Forward Time vs Sequence Length
    ax2.plot(seq_lens, triton_fwd_times, 'r-s', label='Triton Forward', linewidth=2, markersize=6)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Triton Forward Time vs Sequence Length')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: PyTorch Backward Time vs Sequence Length
    ax3.plot(seq_lens, pytorch_bwd_times, 'b-^', label='PyTorch Backward', linewidth=2, markersize=6)
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('PyTorch Backward Time vs Sequence Length')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Triton Backward Time vs Sequence Length
    ax4.plot(seq_lens, triton_bwd_times, 'r-d', label='Triton Backward', linewidth=2, markersize=6)
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Triton Backward Time vs Sequence Length')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Forward Speedup vs Sequence Length
    ax5.plot(seq_lens, fwd_speedups, 'g-o', label='Forward Speedup', linewidth=2, markersize=6)
    ax5.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    ax5.set_xlabel('Sequence Length')
    ax5.set_ylabel('Speedup (PyTorch/Triton)')
    ax5.set_title('Forward Speedup vs Sequence Length')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Backward Speedup vs Sequence Length
    ax6.plot(seq_lens, bwd_speedups, 'g-s', label='Backward Speedup', linewidth=2, markersize=6)
    ax6.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    ax6.set_xlabel('Sequence Length')
    ax6.set_ylabel('Speedup (PyTorch/Triton)')
    ax6.set_title('Backward Speedup vs Sequence Length')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('benchmark_results_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'benchmark_results_plot.png'")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Sequence lengths tested: {min(seq_lens)} to {max(seq_lens)} (step: {seq_lens[1] - seq_lens[0] if len(seq_lens) > 1 else 'N/A'})")
    print(f"Average forward speedup: {np.mean(fwd_speedups):.2f}x")
    print(f"Average backward speedup: {np.mean(bwd_speedups):.2f}x")
    print(f"Max forward speedup: {max(fwd_speedups):.2f}x at seq_len={seq_lens[fwd_speedups.index(max(fwd_speedups))]}")
    print(f"Max backward speedup: {max(bwd_speedups):.2f}x at seq_len={seq_lens[bwd_speedups.index(max(bwd_speedups))]}")

if __name__ == "__main__":
    try:
        results = load_benchmark_results()
        plot_benchmark_results(results)
    except FileNotFoundError:
        print("benchmark_seq_len_results.jsonl not found. Please run the benchmark script first.")
    except Exception as e:
        print(f"Error plotting results: {e}")