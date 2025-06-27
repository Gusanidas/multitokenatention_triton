import torch
import triton
import triton.language as tl
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import random


@triton.jit
def qk_mult(
    qk_ptr,
    q_block,
    q_start,
    k_ptr,
    k_start,
    stride_k_batch,
    stride_k_head,
    stride_k_row,
    stride_k_col,
    stride_qk_batch,
    stride_qk_head,
    stride_qk_row,
    stride_qk_col,
    batch_id,
    channel_id,
    seq_length: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):

    k_col_idx = tl.arange(0, BLOCK_SIZE_K) + k_start
    mask_k_col_idx = (k_col_idx < seq_length) & (k_col_idx >= 0)
    k_block_ptr = (
        k_ptr
        + batch_id * stride_k_batch
        + channel_id * stride_k_head
        + k_col_idx[None, :] * stride_k_row
        + stride_k_col * tl.arange(0, head_dim)[:, None]
    )

    qk_row_idx = tl.arange(0, BLOCK_SIZE_Q) + q_start
    mask_qk_row_idx = (qk_row_idx < seq_length) & (qk_row_idx >= 0)
    qk_block_ptr = (
        qk_ptr
        + batch_id * stride_qk_batch
        + channel_id * stride_qk_head
        + qk_row_idx[:, None] * stride_qk_row
        + k_col_idx[None, :] * stride_qk_col
    )
    qk_mask = mask_qk_row_idx[:, None] & mask_k_col_idx[None, :]

    k_block = tl.load(
        k_block_ptr, mask=mask_k_col_idx[None, :]
    )  # (head_dim, BLOCK_SIZE_N)

    qk_block = tl.dot(q_block, k_block)  # (BLOCK_SIZE_Q, BLOCK_SIZE_K)

    tl.store(
        qk_block_ptr,
        qk_block,
        mask=qk_mask,
    )


@triton.jit
def qk_conv_fwd_kernel(
    qk_ptr,
    weight_ptr,
    q_block,
    q_start,
    k_ptr,
    k_start,
    stride_k_batch,
    stride_k_head,
    stride_k_row,
    stride_k_col,
    stride_qk_batch,
    stride_qk_head,
    stride_qk_row,
    stride_qk_col,
    kernel_size_q,
    kernel_size_k,
    batch_id,
    channel_id,
    seq_length: tl.constexpr,
    head_dim: tl.constexpr,
    pad_q: tl.constexpr,
    pad_k: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):

    qk_mult(
        qk_ptr,
        q_block,
        q_start,
        k_ptr,
        k_start,
        stride_k_batch,
        stride_k_head,
        stride_k_row,
        stride_k_col,
        stride_qk_batch,
        stride_qk_head,
        stride_qk_row,
        stride_qk_col,
        batch_id,
        channel_id,
        seq_length,
        head_dim,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
    )
    tl.debug_barrier()

    acc = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)

    for kernel_q_idx in range(0, kernel_size_q):
        for kernel_k_idx in range(0, kernel_size_k):
            i = kernel_q_idx
            j = kernel_k_idx

            q_origin = q_start + i - pad_q
            k_origin = k_start + j - pad_k
            q = tl.arange(0, BLOCK_SIZE_Q) + q_origin
            k = tl.arange(0, BLOCK_SIZE_K) + k_origin

            mask_q_left = q >= 0
            mask_q_right = q < seq_length
            mask_k_left = k >= 0
            mask_k_right = k < seq_length
            mask_q = mask_q_left & mask_q_right
            mask_k = mask_k_left & mask_k_right
            mask_input = mask_q[:, None] & mask_k[None, :]
            if causal:
                mask_causal = q[:, None] >= k[None, :]
                mask_input = mask_input & mask_causal

            x_block = (
                qk_ptr
                + batch_id * stride_qk_batch
                + channel_id * stride_qk_head
                + q[:, None] * stride_qk_row
                + k[None, :] * stride_qk_col
            )
            x = tl.load(x_block, mask=mask_input, other=0.0)  # , cache_modifier=".ca")

            w_val = (
                weight_ptr
                + channel_id * kernel_size_q * kernel_size_k
                + i * kernel_size_k
                + j
            )
            w = tl.load(w_val)
            acc += x * w

    o_q_idx = tl.arange(0, BLOCK_SIZE_Q)
    o_k_idx = tl.arange(0, BLOCK_SIZE_K)
    mask_q_idx_left = o_q_idx >= pad_q
    mask_q_idx_right = o_q_idx < BLOCK_SIZE_Q
    mask_k_idx_left = o_k_idx >= pad_k
    mask_k_idx_right = o_k_idx < BLOCK_SIZE_K - pad_k
    mask_q_idx = mask_q_idx_left & mask_q_idx_right
    mask_k_idx = mask_k_idx_left & mask_k_idx_right

    o_q = q_start + o_q_idx
    o_k = k_start + o_k_idx
    mask_q_left = o_q >= 0
    mask_q_right = o_q < seq_length
    mask_k_left = o_k >= 0
    mask_k_right = o_k < seq_length

    mask_o_block = (
        mask_q_left[:, None]
        & mask_q_right[:, None]
        & mask_q_idx[:, None]
        & mask_k_left[None, :]
        & mask_k_right[None, :]
        & mask_k_idx[None, :]
    )
    return acc, mask_o_block
