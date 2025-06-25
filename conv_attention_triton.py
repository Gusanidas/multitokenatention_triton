import torch
from torch import nn

import triton
import triton.language as tl

from qk_conv_kernel import qk_conv_fwd_kernel


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    k_ptr,
    stride_k_batch,
    stride_k_head,
    stride_k_row,
    stride_k_col,
    qk_ptr,
    stride_qk_batch,
    stride_qk_head,
    stride_qk_row,
    stride_qk_col,
    # conv_out_ptr,
    weight_ptr,
    kernel_size_q,
    kernel_size_k,
    V_block_ptr,
    v_ptr,
    stride_v_batch,
    stride_v_head,
    stride_v_seq,
    stride_v_dim,
    block_index_q,
    softmax_scale,
    batch_id,
    channel_id,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    reduced_block_q: tl.constexpr,
    reduced_block_k: tl.constexpr,
    pad_q: tl.constexpr,
    pad_k: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    causal = STAGE < 3

    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal, but reduced_block_k and reduced_block_q may not be multiples of each other
        lo, hi = (
            0,
            ((block_index_q * reduced_block_q) // reduced_block_k) * reduced_block_k,
        )
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = (
            ((block_index_q * reduced_block_q) // reduced_block_k) * reduced_block_k,
            (((block_index_q + 1) * reduced_block_q) // reduced_block_k + 1)
            * reduced_block_k,
        )
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN + reduced_block_k + pad_k

    q_start = block_index_q * reduced_block_q - pad_q

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, reduced_block_k):
        k_start = start_kv - pad_k
        offs_kv = tl.arange(0, BLOCK_SIZE_KV) + k_start

        # Compute the convolution.
        # Returns the convolution output for the tile (start_q, start_q + BLOCK_SIZE_Q) x (start_k, start_k + BLOCK_SIZE_K)
        # But we only use the values in the range (start_q + pad_q, start_q + reduced_block_q) x (start_k + pad_k, start_k + reduced_block_k+pad_k)
        conv_out, mask_conv = qk_conv_fwd_kernel(
            qk_ptr=qk_ptr,
            weight_ptr=weight_ptr,
            q_block=Q_block,
            q_start=q_start,
            k_ptr=k_ptr,
            k_start=k_start,
            stride_k_batch=stride_k_batch,
            stride_k_head=stride_k_head,
            stride_k_row=stride_k_row,
            stride_k_col=stride_k_col,
            stride_qk_batch=stride_qk_batch,
            stride_qk_head=stride_qk_head,
            stride_qk_row=stride_qk_row,
            stride_qk_col=stride_qk_col,
            kernel_size_q=kernel_size_q,
            kernel_size_k=kernel_size_k,
            batch_id=batch_id,
            channel_id=channel_id,
            seq_length=SEQ_LEN,
            head_dim=HEAD_DIM,
            pad_q=pad_q,
            pad_k=pad_k,
            causal=causal,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_K=BLOCK_SIZE_KV,
        )
        # conv_out_offs = (
        #     conv_out_ptr
        #     + batch_id * stride_qk_batch
        #     + channel_id * stride_qk_head
        #     + offs_q[:, None] * stride_qk_row
        #     + offs_kv[None, :] * stride_qk_col
        # )
        # tl.store(conv_out_offs, conv_out, mask=mask_conv)

        if STAGE == 2:
            mask = offs_q[:, None] >= (offs_kv[None, :])
            conv_out = conv_out * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(conv_out, 1))
            conv_out -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(conv_out, 1) * softmax_scale)
            conv_out = conv_out * softmax_scale - m_ij[:, None]

        conv_out = tl.where(mask_conv, conv_out, -1.0e6)
        P_block = tl.math.exp(conv_out)
        l_ij = tl.sum(P_block, 1)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        v_block_ptr = (
            v_ptr
            + batch_id * stride_v_batch
            + channel_id * stride_v_head
            + offs_kv[:, None] * stride_v_seq
            + tl.arange(0, HEAD_DIM)[None, :] * stride_v_dim
        )
        mask_offs_v = (offs_kv[:, None] < SEQ_LEN) & (offs_kv[:, None] >= 0)
        V_block = tl.load(v_block_ptr, mask=mask_offs_v)
        # P_block = P_block.to(tl.float16)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [32, 64, 128]
        for BLOCK_SIZE_KV in [32, 64, 128]
        for num_stages in [2, 3, 4]
        for num_warps in [2, 4, 8]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    QK,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN
    W,  # NUM_HEADS, Q_KERNEL_SIZE, K_KERNEL_SIZE
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    # CONV_OUT,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    stride_QK_batch,
    stride_QK_head,
    stride_QK_seq,
    stride_QK_dim,
    q_kernel_size,
    k_kernel_size,
    pad_q,
    pad_k,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    # tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # We calculate the matrix QK in blocks of size BLOCK_SIZE_Q x BLOCK_SIZE_KV.
    # With this we can calculate values of the convolution of size reduced_block_q x reduced_block_k.
    reduced_block_q = BLOCK_SIZE_Q - pad_q
    reduced_block_k = BLOCK_SIZE_KV - pad_k * 2

    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    offs_q = block_index_q * reduced_block_q + tl.arange(0, BLOCK_SIZE_Q) - pad_q
    mask_offs_q = (offs_q < SEQ_LEN) & (offs_q >= 0)
    q_block_ptr = (
        Q
        + qvk_offset
        + offs_q[:, None] * stride_Q_seq
        + tl.arange(0, HEAD_DIM)[None, :] * stride_Q_dim
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    block_idx = tl.arange(0, BLOCK_SIZE_Q)
    block_mask = block_idx >= pad_q
    offs_o = block_index_q * reduced_block_q - pad_q + tl.arange(0, BLOCK_SIZE_Q)
    mask_offs_o = (offs_o < SEQ_LEN) & (offs_o >= 0) & block_mask
    o_block_ptr = (
        O
        + qvk_offset
        + offs_o[:, None] * stride_O_seq
        + tl.arange(0, HEAD_DIM)[None, :] * stride_O_dim
    )

    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")  # TODO: Handle this
    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0  # TODO: Handle this as well
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    Q_block = tl.load(q_block_ptr, mask=mask_offs_q[:, None])

    if STAGE == 1 or STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block=O_block,
            l_i=l_i,
            m_i=m_i,
            Q_block=Q_block,
            k_ptr=K,
            stride_k_batch=stride_K_batch,
            stride_k_head=stride_K_head,
            stride_k_row=stride_K_seq,
            stride_k_col=stride_K_dim,
            qk_ptr=QK,
            stride_qk_batch=stride_QK_batch,
            stride_qk_head=stride_QK_head,
            stride_qk_row=stride_QK_seq,
            stride_qk_col=stride_QK_dim,
            # conv_out_ptr=CONV_OUT,
            weight_ptr=W,
            kernel_size_q=q_kernel_size,
            kernel_size_k=k_kernel_size,
            V_block_ptr=V_block_ptr,
            v_ptr=V,
            stride_v_batch=stride_V_batch,
            stride_v_head=stride_V_head,
            stride_v_seq=stride_V_seq,
            stride_v_dim=stride_V_dim,
            block_index_q=block_index_q,
            softmax_scale=softmax_scale,
            batch_id=index_batch,
            channel_id=index_head,
            HEAD_DIM=HEAD_DIM,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
            reduced_block_q=reduced_block_q,
            reduced_block_k=reduced_block_k,
            pad_q=pad_q,
            pad_k=pad_k,
            STAGE=4 - STAGE,
            offs_q=offs_q,
            offs_kv=offs_kv,
            SEQ_LEN=SEQ_LEN,
        )

    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block=O_block,
            l_i=l_i,
            m_i=m_i,
            Q_block=Q_block,
            k_ptr=K,
            stride_k_batch=stride_K_batch,
            stride_k_head=stride_K_head,
            stride_k_row=stride_K_seq,
            stride_k_col=stride_K_dim,
            qk_ptr=QK,
            stride_qk_batch=stride_QK_batch,
            stride_qk_head=stride_QK_head,
            stride_qk_row=stride_QK_seq,
            stride_qk_col=stride_QK_dim,
            # conv_out_ptr=CONV_OUT,
            weight_ptr=W,
            kernel_size_q=q_kernel_size,
            kernel_size_k=k_kernel_size,
            V_block_ptr=V_block_ptr,
            v_ptr=V,
            stride_v_batch=stride_V_batch,
            stride_v_head=stride_V_head,
            stride_v_seq=stride_V_seq,
            stride_v_dim=stride_V_dim,
            block_index_q=block_index_q,
            softmax_scale=softmax_scale,
            batch_id=index_batch,
            channel_id=index_head,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_KV=BLOCK_SIZE_KV,
            reduced_block_q=reduced_block_q,
            reduced_block_k=reduced_block_k,
            pad_q=pad_q,
            pad_k=pad_k,
            STAGE=2,
            offs_q=offs_q,
            offs_kv=offs_kv,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
        )
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    # tl.store(m_ptrs, m_i) # TODO: Necessary for the backward pass
    tl.store(o_block_ptr, O_block.to(O.type.element_ty), mask=mask_offs_o[:, None])


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, W, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        CHANNEL_DIM, Q_KERNEL_SIZE, K_KERNEL_SIZE = W.shape

        assert CHANNEL_DIM == NUM_HEADS
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        QK = torch.zeros(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN),
            device=Q.device,
            dtype=Q.dtype,
        )
        # CONV_OUT = torch.zeros(
        #    (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN),
        #    device=Q.device,
        #    dtype=Q.dtype,
        # )
        QK = QK.contiguous()
        # CONV_OUT = CONV_OUT.contiguous()
        W = W.contiguous()
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        pad_q = Q_KERNEL_SIZE - 1
        pad_k = (K_KERNEL_SIZE - 1) // 2

        O = torch.zeros_like(Q)
        stage = 3 if causal else 1

        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"] - pad_q) + 1,
            BATCH_SIZE * NUM_HEADS,
            1,
        )

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            QK=QK,
            W=W,
            # CONV_OUT=CONV_OUT,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            stride_QK_batch=QK.stride(0),
            stride_QK_head=QK.stride(1),
            stride_QK_seq=QK.stride(2),
            stride_QK_dim=QK.stride(3),
            q_kernel_size=Q_KERNEL_SIZE,
            k_kernel_size=K_KERNEL_SIZE,
            pad_q=pad_q,
            pad_k=pad_k,
            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O, QK

    @staticmethod
    def backward(ctx, dO):

        # TODO: Implement this
        Q, K, V, O, M = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        return dQ, dK, dV, None, None


class TritonConvAttention(nn.Module):
    def __init__(self, channels, kernel_size, causal=False):
        super(TritonConvAttention, self).__init__()

        self.channels = channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.causal = causal

        self.weight = nn.Parameter(
            torch.randn(channels, self.kernel_size[0], self.kernel_size[1])
        )

        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.weight.zero_()
            center_x = self.kernel_size[0] // 2
            center_y = self.kernel_size[1] // 2
            self.weight[:, center_x, center_y] = 1.0

    def forward(self, Q, K, V, causal, softmax_scale):
        return TritonAttention.apply(Q, K, V, self.weight, causal, softmax_scale)
