import torch
import torch.nn as nn
from mta_attention_pytorch import MTAAttention
from mta_attention_triton import MTAAttentionTriton

# Configuration class to mimic MTA settings
class MTAConfig:
    def __init__(self):
        self.use_mta = True
        self.query_kernel_size = 3
        self.key_kernel_size = 3
        self.after_sm_query_kernel_size = 3
        self.after_sm_key_kernel_size = 3
        self.pre_sm_linear_head = True
        self.post_sm_linear_head = True
        self.pad_key = "both"
        self.init_method = "identity"
        self.dim = 512  

# Hardcoded test parameters
BATCH_SIZE = 2
N_HEADS = 16
SEQ_LEN = 128
HEAD_DIM = 64
DROPOUT = 0.0
CAUSAL = True
DTYPE = torch.float32

def compare_models():
    """Compare outputs and gradients of PyTorch and Triton MTA attention implementations"""
    
    print("=" * 80)
    print("COMPARING MTA ATTENTION PYTORCH vs TRITON")
    print("=" * 80)
    print(f"Configuration: batch={BATCH_SIZE}, heads={N_HEADS}, seq_len={SEQ_LEN}, head_dim={HEAD_DIM}")
    print(f"Dropout: {DROPOUT}")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, using CPU")
    
    mta_config = MTAConfig()
    
    model_pytorch = MTAAttention(
        n_heads=N_HEADS,
        mta=mta_config,
        dropout=DROPOUT,
    ).to(device)
    model_pytorch.to(dtype=DTYPE)
    
    model_triton = MTAAttentionTriton(
        n_heads=N_HEADS,
        mta=mta_config,
        dropout=DROPOUT,
        causal=CAUSAL,
        dtype=DTYPE,
        use_mask=False,
    ).to(device)
    
    model_pytorch.reset_mta_parameters()
    model_triton.reset_mta_parameters()
    
    # Copy parameters from PyTorch to Triton model
    model_triton.copy_mta_parameters(model_pytorch)
    
    # Set models to eval mode to disable dropout for comparison
    model_pytorch.eval()
    model_triton.eval()
    
    torch.manual_seed(123)
    xq = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device=device, requires_grad=True, dtype=DTYPE)
    xk = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device=device, requires_grad=True, dtype=DTYPE)
    xv = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, device=device, requires_grad=True, dtype=DTYPE)
    
    xq_triton = xq.clone().detach().requires_grad_(True)
    xk_triton = xk.clone().detach().requires_grad_(True)
    xv_triton = xv.clone().detach().requires_grad_(True)
    
    print("Testing forward pass...")
    
    mask = None  # Let the model create its own causal mask
    chunk_start_ids = None
    output_pytorch = model_pytorch(xq, xk, xv, mask, chunk_start_ids)
    
    output_triton = model_triton(xq_triton, xk_triton, xv_triton, mask, chunk_start_ids)
    
    def compare_tensors(name, tensor1, tensor2):
        """Compare two tensors and print detailed statistics"""
        print(f"\n{name} Comparison:")
        print(f"  PyTorch tensor shape: {tensor1.shape}")
        print(f"  Triton tensor shape: {tensor2.shape}")
        print(f"  PyTorch - Max abs: {tensor1.abs().max().item():.8f}, Mean abs: {tensor1.abs().mean().item():.8f}")
        print(f"  Triton - Max abs: {tensor2.abs().max().item():.8f}, Mean abs: {tensor2.abs().mean().item():.8f}")
        
        # Count elements < 0.001 for both tensors
        small_tensor1_count = (tensor1.abs() < 0.001).sum().item()
        small_tensor2_count = (tensor2.abs() < 0.001).sum().item()
        print(f"  PyTorch - Elements < 0.001: {small_tensor1_count}/{tensor1.numel()} ({small_tensor1_count/tensor1.numel()*100:.2f}%)")
        print(f"  Triton - Elements < 0.001: {small_tensor2_count}/{tensor2.numel()} ({small_tensor2_count/tensor2.numel()*100:.2f}%)")
        
        diff = tensor1 - tensor2
        print(f"  Difference - Max abs: {diff.abs().max().item():.8f}, Mean abs: {diff.abs().mean().item():.8f}")
        
        # Count elements with small difference
        small_diff_count = (diff.abs() < 0.001).sum().item()
        total_elements = diff.numel()
        print(f"  Elements with diff < 0.001: {small_diff_count}/{total_elements} ({small_diff_count/total_elements*100:.2f}%)")
        
        rel_error = (diff.abs() / (tensor1.abs() + 1e-8)).mean().item()
        print(f"  Relative error: {rel_error:.8f}")
        
        match = torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-5)
        print(f"  Tensors match (rtol=1e-3, atol=1e-5): {match}")
        match = torch.allclose(tensor1, tensor2, rtol=1e-2, atol=1e-1)
        print(f"  Tensors match (rtol=1e-2, atol=1e-1): {match}")
        
        return match
    
    print("Forward Pass Output Comparison:")
    final_match = compare_tensors("Final Output", output_pytorch, output_triton)
    
    print("\nTesting backward pass...")
    
    grad_output = torch.randn_like(output_pytorch)
    
    output_pytorch.backward(grad_output, retain_graph=True)
    grad_xq_pytorch = xq.grad.clone() if xq.grad is not None else None
    grad_xk_pytorch = xk.grad.clone() if xk.grad is not None else None
    grad_xv_pytorch = xv.grad.clone() if xv.grad is not None else None
    
    output_triton.backward(grad_output, retain_graph=True)
    grad_xq_triton = xq_triton.grad.clone() if xq_triton.grad is not None else None
    grad_xk_triton = xk_triton.grad.clone() if xk_triton.grad is not None else None
    grad_xv_triton = xv_triton.grad.clone() if xv_triton.grad is not None else None
    
    print("\nBackward Pass Gradient Comparisons:")
    xq_grad_match = compare_tensors("Gradient w.r.t xq", grad_xq_pytorch, grad_xq_triton) if grad_xq_pytorch is not None else True
    xk_grad_match = compare_tensors("Gradient w.r.t xk", grad_xk_pytorch, grad_xk_triton) if grad_xk_pytorch is not None else True
    xv_grad_match = compare_tensors("Gradient w.r.t xv", grad_xv_pytorch, grad_xv_triton) if grad_xv_pytorch is not None else True
    
    mta_kernel_grad_match = True
    mta_kernel_after_sm_grad_match = True
    if model_pytorch.mta_kernel.grad is not None and model_triton.mta_conv_triton.weight.grad is not None:
        # PyTorch has shape (n_heads, 1, query_sz, key_sz), Triton has (n_heads, query_sz, key_sz)
        pytorch_conv_grad = model_pytorch.mta_kernel.grad.squeeze(1)
        triton_conv_grad = model_triton.mta_conv_triton.weight.grad
        mta_kernel_grad_match = compare_tensors("Gradient w.r.t pre-softmax conv weight", pytorch_conv_grad, triton_conv_grad)
    
    if model_pytorch.mta_kernel_after_sm.grad is not None and model_triton.mta_conv_after_sm_triton.weight.grad is not None:
        # PyTorch has shape (n_heads, 1, query_sz, key_sz), Triton has (n_heads, query_sz, key_sz)
        pytorch_conv_after_sm_grad = model_pytorch.mta_kernel_after_sm.grad.squeeze(1)
        triton_conv_after_sm_grad = model_triton.mta_conv_after_sm_triton.weight.grad
        mta_kernel_after_sm_grad_match = compare_tensors("Gradient w.r.t post-softmax conv weight", pytorch_conv_after_sm_grad, triton_conv_after_sm_grad)
    
    wpsm_grad_match = True
    wposm_grad_match = True
    if model_pytorch.wpsm.weight.grad is not None and model_triton.wpsm_triton.weight.grad is not None:
        wpsm_grad_match = compare_tensors("Gradient w.r.t pre-softmax linear weight", model_pytorch.wpsm.weight.grad, model_triton.wpsm_triton.weight.grad)
    
    if model_pytorch.wposm.weight.grad is not None and model_triton.wposm_triton.weight.grad is not None:
        wposm_grad_match = compare_tensors("Gradient w.r.t post-softmax linear weight", model_pytorch.wposm.weight.grad, model_triton.wposm_triton.weight.grad)
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print("Forward pass matches:")
    print(f"  Final output: {final_match}")
    print("Gradient matches:")
    print(f"  xq gradient: {xq_grad_match}")
    print(f"  xk gradient: {xk_grad_match}")
    print(f"  xv gradient: {xv_grad_match}")
    print(f"  pre-softmax conv weight gradient: {mta_kernel_grad_match}")
    print(f"  post-softmax conv weight gradient: {mta_kernel_after_sm_grad_match}")
    print(f"  pre-softmax linear weight gradient: {wpsm_grad_match}")
    print(f"  post-softmax linear weight gradient: {wposm_grad_match}")
    
    all_match = (final_match and xq_grad_match and xk_grad_match and xv_grad_match and 
                mta_kernel_grad_match and mta_kernel_after_sm_grad_match and 
                wpsm_grad_match and wposm_grad_match)
    print(f"\nAll comparisons passed: {all_match}")
    
    if not all_match:
        print("\nNote: Some comparisons failed. This could be due to:")
        print("- Numerical precision differences between PyTorch and Triton")
        print("- Different implementation details in the kernels")
        print("- Acceptable floating-point errors for the given tolerance")
        print("- Differences in convolution padding or masking implementations")
    
    # Additional statistics
    print("\nDetailed Statistics Summary:")
    print(f"PyTorch final output - mean: {output_pytorch.mean().item():.6f}, std: {output_pytorch.std().item():.6f}")
    print(f"Triton final output - mean: {output_triton.mean().item():.6f}, std: {output_triton.std().item():.6f}")
    
    if grad_xq_pytorch is not None:
        print(f"PyTorch xq grad - mean: {grad_xq_pytorch.mean().item():.6f}, std: {grad_xq_pytorch.std().item():.6f}")
        print(f"Triton xq grad - mean: {grad_xq_triton.mean().item():.6f}, std: {grad_xq_triton.std().item():.6f}")
    
    if grad_xk_pytorch is not None:
        print(f"PyTorch xk grad - mean: {grad_xk_pytorch.mean().item():.6f}, std: {grad_xk_pytorch.std().item():.6f}")
        print(f"Triton xk grad - mean: {grad_xk_triton.mean().item():.6f}, std: {grad_xk_triton.std().item():.6f}")
        
    if grad_xv_pytorch is not None:
        print(f"PyTorch xv grad - mean: {grad_xv_pytorch.mean().item():.6f}, std: {grad_xv_pytorch.std().item():.6f}")
        print(f"Triton xv grad - mean: {grad_xv_triton.mean().item():.6f}, std: {grad_xv_triton.std().item():.6f}")

if __name__ == "__main__":
    compare_models()
