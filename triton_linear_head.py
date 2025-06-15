import torch
import torch.nn as nn
import triton
import triton.language as tl


def get_autotune_config(short=False):
    if short:
        return [
            triton.Config({'block_q': bq, 'p_group_size': pg}, num_stages=ns,
                          num_warps=nw) for bq in [64] for pg in [512] for ns in [3] for nw in [8]
        ]
    else:
        return [
            triton.Config({'block_q': bq, 'p_group_size': pg}, num_stages=ns,
                          num_warps=nw) for bq in [32,64,128,256] for pg in [1,256,512,1024,1536] for ns in [1,2,3] for nw in [1,4,8]
    ]

def get_autotune_config_bwd_input(short=False):
    if short:
        return [
            triton.Config({'block_q': bq, 'p_group_size': pg}, num_stages=3,
                          num_warps=8) for bq in [128] for pg in [256]
    ]
    else:
        return [
            triton.Config({'block_q': bq, 'p_group_size': pg}, num_stages=3,
                          num_warps=8) for bq in [32,64,128,256] for pg in [1,256,512,1024,1536] for ns in [1,2,3] for nw in [1,4,8]
        ]

def get_autotune_config_bwd_weight(short=False):
    if short:
        return [
            triton.Config({'block_q': bq, 'p_group_size': pg}, num_stages=3,
                          num_warps=8) for bq in [128] for pg in [256]
        ]
    else:
        return [
            triton.Config({'block_q': bq, 'p_group_size': pg}, num_stages=3,
                          num_warps=8) for bq in [32,64,128,256] for pg in [1,256,512,1024,1536] for ns in [1,2,3] for nw in [1,4,8]
        ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'L', 'causal']
)
@triton.jit
def linear_head_kernel_forward(
    input_ptr, output_ptr, linear_weight_ptr,
    N, C, P, Q,
    L: tl.constexpr,
    causal: tl.constexpr,
    block_q: tl.constexpr,
    p_group_size: tl.constexpr,
):
    channel_off = tl.arange(0, L)
    linear_weight_block = linear_weight_ptr + channel_off[:, None] + channel_off[None, :]*C
    mask_lw = channel_off < C
    mask_linear_weight = mask_lw[:, None] & mask_lw[None, :]
    linear_weight = tl.load(linear_weight_block, mask=mask_linear_weight, other=0.0)

    mask_channel = channel_off<C

    pid = tl.program_id(0)

    ids_per_batch = tl.cdiv(Q, block_q)*p_group_size
    batch_id = pid // ids_per_batch
    if batch_id >= N:
        return
    pid_2 = pid % ids_per_batch

    start_p_id = pid_2 // tl.cdiv(Q, block_q)
    q_id = pid_2 % tl.cdiv(Q, block_q)
    for p_id in range(0, P, p_group_size):
        p_id = tl.multiple_of(p_id, p_group_size)
        if p_id+start_p_id < P and (not causal or p_id+start_p_id >= q_id*block_q):
            p = start_p_id + p_id
            q = q_id * block_q + tl.arange(0, block_q)
            mask_q = q < Q

            x_off = input_ptr + batch_id * C * P * Q + channel_off[None, :] * P * Q + p * Q + q[:, None]
            mask_x = mask_channel[None, :] & mask_q[:, None]
            x = tl.load(x_off, mask=mask_x, other=0.0)
            y = tl.dot(x, linear_weight)#,input_precision="tf32x3")
            y_off = output_ptr + batch_id * C * P * Q + channel_off[None, :] * P * Q + p * Q + q[:, None]
            if p <= (q_id+1)*block_q and causal:
                mask_pq = p>=q
                mask_x = mask_x & mask_pq[:, None]

            tl.store(y_off, y, mask=mask_x)
            

@triton.autotune(
    configs=get_autotune_config_bwd_input(),
    key=['N', 'C', 'L', 'causal']
)
@triton.jit
def linear_head_kernel_backward_input(
    grad_output_ptr, grad_input_ptr, linear_weight_ptr,
    N, C, P, Q,
    L: tl.constexpr,
    causal: tl.constexpr,
    block_q: tl.constexpr,
    p_group_size: tl.constexpr,
):
    linear_weight_block = linear_weight_ptr + tl.arange(0, L)[:, None] * C + tl.arange(0, L)[None, :]
    linear_weight_t = tl.load(linear_weight_block)
    
    channel_off = tl.arange(0, L)
    mask_channel = channel_off < C
    
    pid = tl.program_id(0)
    ids_per_batch = tl.cdiv(Q, block_q) * p_group_size
    batch_id = pid // ids_per_batch
    
    if batch_id >= N:
        return
    
    pid_2 = pid % ids_per_batch
    start_p_id = pid_2 // tl.cdiv(Q, block_q)
    q_id = pid_2 % tl.cdiv(Q, block_q)
    
    for p_id in range(0, P, p_group_size):
        p_id = tl.multiple_of(p_id, p_group_size)
        if p_id + start_p_id < P and (not causal or p_id+start_p_id >= q_id*block_q):
            p = start_p_id + p_id
            q = q_id * block_q + tl.arange(0, block_q)
            mask_q = q < Q
            if (p <= (q_id+1)*block_q and causal):
                mask_pq = p>=q
                mask_q = mask_q & mask_pq
            
            grad_out_off = grad_output_ptr + batch_id * C * P * Q + channel_off[None, :] * P * Q + p * Q + q[:, None]
            mask_x = mask_channel[None, :] & mask_q[:, None]
            grad_out = tl.load(grad_out_off, mask=mask_x, other=0.0)
            
            grad_in = tl.dot(grad_out, linear_weight_t)
            
            grad_in_off = grad_input_ptr + batch_id * C * P * Q + channel_off[None, :] * P * Q + p * Q + q[:, None]
            tl.store(grad_in_off, grad_in, mask=mask_x)

@triton.autotune(
    configs=get_autotune_config_bwd_weight(),
    key=['N', 'C', 'L', 'causal']
)
@triton.jit
def linear_head_kernel_backward_weight(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    N, C, P, Q,
    L: tl.constexpr,
    causal: tl.constexpr,
    p_group_size: tl.constexpr,
    block_q: tl.constexpr,
):
    pid = tl.program_id(0)
    num_q_ids = tl.cdiv(Q, block_q)
    ids_per_batch = num_q_ids * p_group_size
    batch_id = pid // ids_per_batch
    if batch_id >= N:
        return
    pid_2 = pid % ids_per_batch
    start_p_id = pid_2 // num_q_ids
    q_id = pid_2 % num_q_ids

    channel_off = tl.arange(0, L)
    mask_channel = channel_off < C
    
    acc = tl.zeros((L,L), dtype=tl.float32)
    for p_id in range(0, P, p_group_size):
        p_id = tl.multiple_of(p_id, p_group_size)
        if p_id + start_p_id < P and (not causal or p_id+start_p_id >= q_id*block_q):
            p = start_p_id + p_id
            q = q_id * block_q + tl.arange(0, block_q)
            mask_q = q < Q
            if (p <= (q_id+1)*block_q and causal):
                mask_pq = p>=q
                mask_q = mask_q & mask_pq

            grad_out_off = grad_output_ptr + batch_id * C * P * Q + channel_off[None, :] * P * Q + p * Q + q[:, None]
            mask_output = mask_channel[None, :] & mask_q[:, None]
            grad_out = tl.load(grad_out_off, mask=mask_output, other=0.0)
            
            grad_in_off = input_ptr + batch_id * C * P * Q + channel_off[:, None] * P * Q + p * Q + q[None, :]
            mask_input = mask_channel[:, None] & mask_q[None, :]
            grad_in = tl.load(grad_in_off, mask=mask_input, other=0.0)
            
            acc += tl.dot(grad_in, grad_out, input_precision="tf32x3")
    
    acc_off = grad_weight_ptr + channel_off[None, :] * C + channel_off[:, None]
    mask_acc = channel_off < C
    mask_off = mask_acc[None, :] & mask_acc[None, :]
    tl.atomic_add(acc_off, acc, mask=mask_off)



class TritonLinearHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, causal=True, output_value=0.0):
        N, C, P, Q = input.shape
        L = weight.shape[0]
        assert weight.shape == (L, L), f"Weight must be square matrix, got {weight.shape}"
        assert C == L, f"Channel dimension {C} must match weight dimension {L}"
        L = triton.next_power_of_2(L)
        L = max(L, 16)
        
        if output_value == 0.0:
            output = torch.zeros_like(input)
        else:
            output = torch.full_like(input, output_value)
        
        grid = lambda META: (META['p_group_size']*N*triton.cdiv(Q, META['block_q']),)
        
        linear_head_kernel_forward[grid](
            input, output, weight,
            N, C, P, Q,
            L, causal,
        )
        
        ctx.save_for_backward(input, weight)
        ctx.input_shape = input.shape
        ctx.causal = causal
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        N, C, P, Q = ctx.input_shape
        causal = ctx.causal
        L = triton.next_power_of_2(C)
        L = max(L, 16)
        
        grad_input = None
        grad_weight = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.empty_like(grad_output)
            
            
            grid = lambda META: (META['p_group_size']*N*triton.cdiv(Q, META['block_q']),)
            
            linear_head_kernel_backward_input[grid](
                grad_output, grad_input, weight,
                N, C, P, Q,
                L, causal,
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight, dtype=torch.float64)
            
            
            grid = lambda META: (META['p_group_size']*N*triton.cdiv(Q, META['block_q']),)
            
            linear_head_kernel_backward_weight[grid](
                input, grad_output, grad_weight,
                N, C, P, Q,
                L, causal,
            )
        
        return grad_input, grad_weight, None, None


class TritonLinearHead(nn.Module):
    """
    A PyTorch module that performs linear transformation on the head dimension
    of a 4D tensor (batch, head_dim, seq, seq) using Triton kernels.
    
    Equivalent to: scores.transpose(1, -1) @ weight.T @ scores.transpose(1, -1).T
    """
    
    def __init__(self, head_dim, causal=True, dtype=torch.float32, output_value=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.randn(head_dim, head_dim, dtype=dtype) / (head_dim ** 0.5))
        self.causal = causal
        self.output_value = output_value
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, head_dim, seq, seq)
        
        Returns:
            Tensor of shape (batch, head_dim, seq, seq)
        """
        return TritonLinearHeadFunction.apply(x, self.weight, self.causal, self.output_value)
    
    def extra_repr(self):
        return f'head_dim={self.head_dim}'
