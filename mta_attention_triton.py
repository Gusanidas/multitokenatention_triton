import torch
import torch.nn as nn
import torch.nn.functional as F
from convolution_2d_trit import Conv2dTriton
from triton_linear_head import TritonLinearHead

class MTAAttentionTriton(nn.Module):
    def __init__(self,
                 n_heads: int,
                 mta,
                 dropout: float,
                 causal: bool,
                 dtype: torch.dtype = torch.float32,
                 use_mask: bool = False):
        super().__init__()

        assert mta.use_mta, "MTAAttentionTriton can only be used with MTA"
        assert mta.query_kernel_size is not None and mta.key_kernel_size is not None
        assert mta.pre_sm_linear_head is True
        if not hasattr(mta, 'after_sm_query_kernel_size'):
            mta.after_sm_query_kernel_size = 3
        if not hasattr(mta, 'after_sm_key_kernel_size'):
            mta.after_sm_key_kernel_size = 3
        if not hasattr(mta, 'post_sm_linear_head'):
            mta.post_sm_linear_head = True

        self.n_heads = n_heads
        self.mta = mta
        self.dropout = dropout
        self.causal = causal
        self.use_mask = use_mask
        
        self.mta_conv_triton = Conv2dTriton(
            n_heads, 
            kernel_size=(mta.query_kernel_size, mta.key_kernel_size),
            causal=causal,
            dtype=dtype
        )

        self.wpsm_triton = TritonLinearHead(
            n_heads,
            causal=causal,
            dtype=dtype,
            output_value=float("-inf")
        )
        
        self.mta_conv_after_sm_triton = Conv2dTriton(
            n_heads, 
            kernel_size=(mta.after_sm_query_kernel_size, mta.after_sm_key_kernel_size),
            causal=causal, 
            dtype=dtype
        )
        
        self.wposm_triton = TritonLinearHead(
            n_heads,
            causal=causal,
            dtype=dtype
        )
        
        self.pad_key = mta.pad_key
        self.mta_init_method = mta.init_method
        
        self.dropout = dropout
        
        
    def forward(self, xq, xk, xv, mask, chunk_start_ids):
        assert chunk_start_ids is None, "MTAAttentionTriton does not support chunked attention"
        if self.use_mask:
            mask = self._update_mask(mask=mask, bsz=xq.size(0), xq=xq, xk_size=xk.size(-2))
        else:
            mask = torch.tensor(float("-inf"), device=xq.device)

        head_dim = xq.size(-1)
        scores = torch.matmul(xq, xk.transpose(2, 3)) * torch.rsqrt(
                torch.tensor(head_dim, requires_grad=False, device="cuda")
            )

        scores = self._mta_convolution_triton(scores, mask, self.mta_conv_triton)
        
        scores = self.wpsm_triton(scores)

        if self.use_mask:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        scores = self._mta_convolution_triton(scores, mask, self.mta_conv_after_sm_triton)
        if self.use_mask:
            scores = torch.where(mask == float("-inf"), 0.0, scores)
        
        scores = self.wposm_triton(scores)

        scores = F.dropout(scores, p=self.dropout, training=self.training)
        
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2)
        
        return output

    def _mta_convolution_triton(self, scores, mask, conv_layer):
        """Apply Triton convolution with proper masking"""
        if self.use_mask:
            scores_masked = scores.clone()
            scores_masked[mask == float("-inf")] = 0
        else:
            scores_masked = scores
        
        conv_scores = conv_layer(scores_masked)
        
        return conv_scores

    def _update_mask(
        self, mask: torch.tensor, bsz: int, xq: torch.tensor, xk_size: int
    ):
        if not isinstance(mask, torch.Tensor):
            mask = torch.full((xq.size(-2), xk_size), float("-inf"), device=xq.device)
            mask = torch.triu(mask, diagonal=1).type_as(xq)
            mask = mask.repeat(bsz, self.n_heads, 1, 1)
        else:
            if mask.dtype == torch.bool:
                mask = torch.where(mask, 0.0, float("-inf")).to(xq.dtype)
            if mask.dtype != xq.dtype:
                mask = mask.type(xq.dtype)
            if len(mask.shape) == 2:
                assert mask.size(0) == bsz
                mask = mask.repeat(1, self.n_heads, xq.size(-2), 1)
                mask_i, mask_j = torch.triu_indices(xq.size(-2), xk_size, offset=1)
                mask[:, :, mask_i, mask_j] = float("-inf")
            else:
                if mask.size(0) == 1 and mask.size(1) == 1:
                    # in prefilling mask is defined for 1 head
                    mask = mask.repeat(bsz, self.n_heads, 1, 1)
        return mask

    def reset_mta_parameters(self, init_std=None):
        """Initialize parameters to match PyTorch version"""
        init_std = init_std or (self.mta.dim ** (-0.5))

        # Initialize pre-softmax convolution kernel
        if self.mta_conv_triton.weight is not None:
            self._init_conv_kernel(self.mta_conv_triton.weight, init_std)

        # Initialize post-softmax convolution kernel
        if self.mta_conv_after_sm_triton.weight is not None:
            self._init_conv_kernel(self.mta_conv_after_sm_triton.weight, init_std)

        # Initialize linear heads as identity
        if hasattr(self, 'wpsm_triton'):
            torch.nn.init.eye_(self.wpsm_triton.weight)

        if hasattr(self, 'wposm_triton'):
            torch.nn.init.eye_(self.wposm_triton.weight)

    def _init_conv_kernel(self, kernel, init_std):
        """Initialize convolution kernel based on mta_init_method"""
        if self.mta_init_method == "uniform":
            torch.nn.init.uniform_(kernel.data, a=-1.0, b=1.0)
        elif self.mta_init_method == "normal":
            init_std = 0.3
            torch.nn.init.uniform_(
                kernel.data,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
        elif self.mta_init_method == "diagonal":
            # diagonal kernel pattern
            with torch.no_grad():
                diagonal_kernel = torch.ones(kernel.data.shape)
                _, A, B = kernel.data.shape
                diagonal = (B + 1) // 2 - A
                diagonal_kernel = torch.tril(diagonal_kernel, diagonal=diagonal)
                diagonal_kernel = torch.triu(diagonal_kernel, diagonal=diagonal)
                kernel.data.copy_(diagonal_kernel)
        elif self.mta_init_method == "identity":
            assert self.pad_key == "both"
            # identity kernel pattern
            with torch.no_grad():
                nheads, query_sz, key_sz = kernel.data.shape
                identity_kernel = torch.zeros(nheads, query_sz, key_sz).cuda()
                identity_kernel[:, -1, key_sz // 2] = 1.0
                kernel.data.copy_(identity_kernel)
        elif self.mta_init_method == "const":
            kernel.data.fill_(0.3)
        else:
            raise ValueError(
                f"Unsupported mta_init_method: {self.mta_init_method}"
            )

    def copy_mta_parameters(self, mta_attention):
        """Copy parameters from PyTorch MTA attention"""
        # Copy convolution weights
        if hasattr(mta_attention, 'mta_kernel'):
            # PyTorch kernel has shape (n_heads, 1, query_sz, key_sz)
            # Triton kernel has shape (n_heads, query_sz, key_sz)
            self.mta_conv_triton.weight.data.copy_(mta_attention.mta_kernel.data.squeeze(1))
        
        if hasattr(mta_attention, 'mta_kernel_after_sm'):
            self.mta_conv_after_sm_triton.weight.data.copy_(mta_attention.mta_kernel_after_sm.data.squeeze(1))
        
        # Copy linear weights
        if hasattr(mta_attention, 'wpsm'):
            self.wpsm_triton.weight.data.copy_(mta_attention.wpsm.weight.data)
        
        if hasattr(mta_attention, 'wposm'):
            self.wposm_triton.weight.data.copy_(mta_attention.wposm.weight.data)
