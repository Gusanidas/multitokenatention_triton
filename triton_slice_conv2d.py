import torch
import triton
import triton.language as tl
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import random

SHORT_CONFIG = True


def get_autotune_config(short=SHORT_CONFIG):
    if short:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": bm,
                    "BLOCK_SIZE_N": bn,
                },
                num_stages=3,
                num_warps=8,
            )
            for bm in [32]
            for bn in [32]
        ]
    else:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": bm,
                    "BLOCK_SIZE_N": bn,
                },
                num_stages=3,
                num_warps=8,
            )
            for bm in [32, 64, 128, 256]
            for bn in [32, 64, 128, 256]
        ]


def get_autotune_config_bwd_input(short=SHORT_CONFIG):
    if short:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": bm,
                    "BLOCK_SIZE_N": bn,
                    "GROUP_SIZE_N": gm,
                    "SIZE_RS_BLOCKS": s,
                },
                num_stages=3,
                num_warps=8,
            )
            for bm in [64]
            for bn in [64]
            for gm in [1]
            for s in [1]
        ]
    else:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": bm,
                    "BLOCK_SIZE_N": bn,
                    "GROUP_SIZE_N": gm,
                    "SIZE_RS_BLOCKS": s,
                },
                num_stages=3,
                num_warps=8,
            )
            for bm in [32, 64, 128, 256]
            for bn in [32, 64, 128, 256]
            for gm in [1, 2, 4, 8, 16, 32]
            for s in [1, 3, 16, 32]
        ]


def get_autotune_config_bwd_weight(short=SHORT_CONFIG):
    if short:
        return [
            triton.Config(
                {"BLOCK_SIZE": bs, "ID_BLOCK_SIZE": ib}, num_stages=3, num_warps=8
            )
            for bs in [64]
            for ib in [16]
        ]
    else:
        return [
            triton.Config(
                {"BLOCK_SIZE": bs, "ID_BLOCK_SIZE": ib}, num_stages=3, num_warps=8
            )
            for bs in [32, 64, 128, 256]
            for ib in [1, 3, 16, 32, 64, 128, 256]
        ]


@triton.jit
def _indicator_(idx_m, idx_n, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    m = tl.arange(0, BLOCK_SIZE_M)
    n = tl.arange(0, BLOCK_SIZE_N)
    m_mask = tl.where(m == idx_m, 1, 0)
    n_mask = tl.where(n == idx_n, 1, 0)
    return m_mask[:, None] * n_mask[None, :]


@triton.jit
def _take_slice_(
    x, idx_m, idx_n, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    ind = _indicator_(idx_m, idx_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    y = tl.sum(x * ind)
    return y


@triton.jit
def _put_slice_(
    x, idx_m, idx_n, input_slice, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    ind = _indicator_(idx_m, idx_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
    y = tl.where(ind == 1, x + input_slice, x)
    return y


@triton.autotune(configs=get_autotune_config(), key=["C", "R", "S", "pad_h", "pad_w"])
@triton.jit
def conv_fwd_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    C,
    H,
    W,
    pad_h,
    pad_w,
    R: tl.constexpr,
    S: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    batch_head_pid = tl.program_id(0)
    batch_id = batch_head_pid // C
    channel_id = batch_head_pid % C

    hw_pid = tl.program_id(1)

    reduced_block_m = BLOCK_SIZE_M - pad_h
    reduced_block_n = BLOCK_SIZE_N - pad_w * 2

    num_ids_n = tl.cdiv(W, reduced_block_n)
    num_ids_m = tl.cdiv(H, reduced_block_m)
    pid_n = hw_pid % num_ids_n
    pid_m = hw_pid // num_ids_n

    h_start = pid_m * reduced_block_m - pad_h
    w_start = pid_n * reduced_block_n - pad_w

    if pid_m >= num_ids_m or pid_n >= num_ids_n:
        return

    x_h = tl.arange(0, BLOCK_SIZE_M) + h_start
    x_w = tl.arange(0, BLOCK_SIZE_N) + w_start
    mask_x_h = (x_h >= 0) & (x_h < H)
    mask_x_w = (x_w >= 0) & (x_w < W)
    mask_x = mask_x_h[:, None] & mask_x_w[None, :]

    x = tl.load(
        input_ptr
        + batch_id * H * W * C
        + channel_id * H * W
        + x_h[:, None] * W
        + x_w[None, :],
        mask=mask_x,
        other=0.0,
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for out_h_idx in range(0, reduced_block_m):
        out_h = out_h_idx + pad_h
        for out_w_idx in range(0, reduced_block_n):
            out_w = out_w_idx + pad_w
            out = 0.0
            for kernel_q_idx in range(0, R):
                for kernel_k_idx in range(0, S):
                    w_val = (
                        weight_ptr
                        + channel_id * R * S
                        + kernel_q_idx * S
                        + kernel_k_idx
                    )
                    w = tl.load(w_val)
                    out += (
                        _take_slice_(
                            x,
                            out_h_idx + kernel_q_idx,
                            out_w_idx + kernel_k_idx,
                            BLOCK_SIZE_M,
                            BLOCK_SIZE_N,
                        )
                        * w
                    )
            acc = _put_slice_(acc, out_h, out_w, out, BLOCK_SIZE_M, BLOCK_SIZE_N)

    o_h_idx = tl.arange(0, BLOCK_SIZE_M)
    mask_h_idx = o_h_idx >= pad_h

    o_w_idx = tl.arange(0, BLOCK_SIZE_N)
    mask_w_idx = (o_w_idx >= pad_w) & (o_w_idx < BLOCK_SIZE_N - pad_w)

    o_h = h_start + tl.arange(0, BLOCK_SIZE_M)
    o_w = w_start + tl.arange(0, BLOCK_SIZE_N)
    mask_h_left = o_h >= 0
    mask_h_right = o_h < H
    mask_w_left = o_w >= 0
    mask_w_right = o_w < W

    mask_o = (
        mask_h_idx[:, None]
        & mask_h_left[:, None]
        & mask_h_right[:, None]
        & mask_w_left[None, :]
        & mask_w_right[None, :]
        & mask_w_idx[None, :]
    )

    o_block = (
        output_ptr
        + batch_id * C * H * W
        + channel_id * H * W
        + o_h[:, None] * W
        + o_w[None, :]
    )
    if causal:
        mask_causal = o_h[:, None] >= o_w[None, :]
        mask_o = mask_o & mask_causal

    tl.store(o_block, acc, mask=mask_o)


# Backward kernel for input gradient (dx)
@triton.autotune(
    configs=get_autotune_config_bwd_input(), key=["C", "R", "S", "pad_h", "pad_w"]
)
@triton.jit
def conv_bwd_input_kernel(
    dx_ptr,
    dout_ptr,
    weight_ptr,
    C,
    H,
    W,
    P,
    Q,
    pad_h,
    pad_w,
    R: tl.constexpr,
    S: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    SIZE_RS_BLOCKS: tl.constexpr,
):
    batch_head_pid = tl.program_id(0)
    batch_id = batch_head_pid // C
    channel_id = batch_head_pid % C

    hw_pid = tl.program_id(1)

    num_ids_n = tl.cdiv(W, BLOCK_SIZE_N)
    num_ids_m = tl.cdiv(H, BLOCK_SIZE_M)
    num_ids_in_group = GROUP_SIZE_N * num_ids_m
    group_id = hw_pid // num_ids_in_group
    first_pid_n = group_id * GROUP_SIZE_N
    group_size_n = min(num_ids_n - first_pid_n, GROUP_SIZE_N)
    pid_n = first_pid_n + ((hw_pid % num_ids_in_group) % group_size_n)
    pid_m = (hw_pid % num_ids_in_group) // group_size_n
    if pid_m * BLOCK_SIZE_M < pid_n * BLOCK_SIZE_N and causal:
        return

    h_start = pid_m * BLOCK_SIZE_M
    w_start = pid_n * BLOCK_SIZE_N

    if pid_m >= num_ids_m or pid_n >= num_ids_n:
        return

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)

    pid_2 = tl.program_id(2)
    for id_2 in range(0, R * S, SIZE_RS_BLOCKS):
        id_2 = tl.multiple_of(id_2, SIZE_RS_BLOCKS)
        ii = id_2 + pid_2
        if ii < R * S:
            i = ii // S
            j = ii % S
            out_h_start = h_start - i + pad_h
            out_w_start = w_start - j + pad_w

            out_h = tl.arange(0, BLOCK_SIZE_M) + out_h_start
            out_w = tl.arange(0, BLOCK_SIZE_N) + out_w_start

            mask_out_h = (out_h >= 0) & (out_h < P)
            mask_out_w = (out_w >= 0) & (out_w < Q)
            mask_out = mask_out_h[:, None] & mask_out_w[None, :]
            if causal:
                mask_causal = out_h[:, None] >= out_w[None, :]
                mask_out = mask_out & mask_causal

            dout_block = (
                dout_ptr
                + batch_id * C * P * Q
                + channel_id * P * Q
                + out_h[:, None] * Q
                + out_w[None, :]
            )
            dout = tl.load(dout_block, mask=mask_out, other=0.0)

            # Load corresponding weight (flipped for convolution transpose)
            w_idx = weight_ptr + channel_id * R * S + i * S + j
            w = tl.load(w_idx)

            acc += dout * w

    in_h = h_start + tl.arange(0, BLOCK_SIZE_M)
    in_w = w_start + tl.arange(0, BLOCK_SIZE_N)
    mask_in_h = (in_h >= 0) & (in_h < H)
    mask_in_w = (in_w >= 0) & (in_w < W)
    mask_in = mask_in_h[:, None] & mask_in_w[None, :]
    if causal:
        mask_causal = in_h[:, None] >= in_w[None, :]
        mask_in = mask_in & mask_causal

    dx_block = (
        dx_ptr
        + batch_id * H * W * C
        + channel_id * H * W
        + in_h[:, None] * W
        + in_w[None, :]
    )
    if SIZE_RS_BLOCKS == 1:
        tl.store(dx_block, acc, mask=mask_in)
    else:
        tl.atomic_add(dx_block, acc, mask=mask_in)


# Backward kernel for weight gradient (dw)
@triton.autotune(
    configs=get_autotune_config_bwd_weight(), key=["C", "R", "S", "pad_h", "pad_w"]
)
@triton.jit
def conv_bwd_weight_kernel(
    dw_ptr,
    input_ptr,
    dout_ptr,
    N,
    C,
    H,
    W,
    R,
    S,
    pad_h,
    pad_w,
    P: tl.constexpr,
    Q: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ID_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    channel_id = pid // (R * S)
    rs_id = pid % (R * S)
    r_id = rs_id // S
    s_id = rs_id % S

    if channel_id >= C:
        return

    acc = tl.zeros((1,), dtype=tl.float64)
    pid_2 = tl.program_id(1)
    num_p_ids = tl.cdiv(P, BLOCK_SIZE)
    num_q_ids = tl.cdiv(Q, BLOCK_SIZE)
    ids_per_batch = num_p_ids * num_q_ids
    total_ids = N * ids_per_batch

    for start_id in range(0, total_ids, ID_BLOCK_SIZE):
        start_id = tl.multiple_of(start_id, ID_BLOCK_SIZE)
        id_2 = start_id + pid_2
        if id_2 < total_ids:
            batch_id = id_2 // ids_per_batch
            out_h_s = (id_2 % ids_per_batch) // num_q_ids
            out_w_s = (id_2 % ids_per_batch) % num_q_ids
            out_h_s = out_h_s * BLOCK_SIZE
            out_w_s = out_w_s * BLOCK_SIZE
            if out_h_s >= out_w_s or not causal:

                out_h = out_h_s + tl.arange(0, BLOCK_SIZE)
                out_w = out_w_s + tl.arange(0, BLOCK_SIZE)

                in_h = out_h + r_id - pad_h
                in_w = out_w + s_id - pad_w

                mask_in_h = (in_h >= 0) & (in_h < H)
                mask_in_w = (in_w >= 0) & (in_w < W)
                mask_in = mask_in_h[:, None] & mask_in_w[None, :]
                if causal:
                    mask_causal = in_h[:, None] >= in_w[None, :]
                    mask_in = mask_in & mask_causal

                mask_out_h = (out_h >= 0) & (out_h < P)
                mask_out_w = (out_w >= 0) & (out_w < Q)
                mask_out = mask_out_h[:, None] & mask_out_w[None, :]
                if causal:
                    mask_causal = out_h[:, None] >= out_w[None, :]
                    mask_out = mask_out & mask_causal

                x_idx = (
                    batch_id * H * W * C
                    + channel_id * H * W
                    + in_h[:, None] * W
                    + in_w[None, :]
                )
                x = tl.load(input_ptr + x_idx, mask=mask_in, other=0.0)

                dout_idx = (
                    batch_id * C * P * Q
                    + channel_id * P * Q
                    + out_h[:, None] * Q
                    + out_w[None, :]
                )
                dout = tl.load(dout_ptr + dout_idx, mask=mask_out, other=0.0)

                acc += tl.sum(x * dout)

    dw_idx = channel_id * R * S + r_id[:, None] * S + s_id[None, :]
    if ID_BLOCK_SIZE == 1:
        tl.store(dw_ptr + dw_idx, acc)
    else:
        tl.atomic_add(dw_ptr + dw_idx, acc)


class Conv2dTritonFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, causal):

        N, C, H, W = input.shape
        C_out, R, S = weight.shape
        assert (
            C == C_out
        ), "This implementation assumes depthwise convolution (C_in == C_out)"

        P = H  # Output height (same as input for stride=1)
        Q = W  # Output width (same as input for stride=1)
        pad_h = R - 1
        pad_w = (S - 1) // 2

        output = torch.zeros(N, C, P, Q, device=input.device, dtype=input.dtype)

        # Grid configuration
        grid = lambda META: (
            N * C,
            (triton.cdiv(H, META["BLOCK_SIZE_M"] - pad_h) + 1)
            * (triton.cdiv(W, META["BLOCK_SIZE_N"] - pad_w) + 1),
        )

        conv_fwd_kernel[grid](
            output,
            input,
            weight,
            C,
            H,
            W,
            pad_h,
            pad_w,
            R=R,
            S=S,
            causal=causal,
        )

        ctx.save_for_backward(input, weight)
        ctx.padding = (pad_h, pad_w)
        ctx.input_shape = input.shape
        ctx.causal = causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        N, C, H, W = ctx.input_shape
        C_out, R, S = weight.shape
        pad_h, pad_w = ctx.padding
        P, Q = grad_output.shape[2], grad_output.shape[3]

        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            # Gradient w.r.t input
            grad_input = torch.zeros_like(input)

            grid = lambda META: (
                N * C,
                triton.cdiv(H, META["BLOCK_SIZE_M"])
                * triton.cdiv(W, META["BLOCK_SIZE_N"]),
                META["SIZE_RS_BLOCKS"],
            )

            conv_bwd_input_kernel[grid](
                grad_input,
                grad_output,
                weight,
                C,
                H,
                W,
                P,
                Q,
                pad_h,
                pad_w,
                R=R,
                S=S,
                causal=ctx.causal,
            )

        if ctx.needs_input_grad[1]:
            # Gradient w.r.t weight
            grad_weight = torch.zeros_like(weight, dtype=torch.float64)

            grid = lambda META: (C * R * S, META["ID_BLOCK_SIZE"])

            conv_bwd_weight_kernel[grid](
                grad_weight,
                input,
                grad_output,
                N,
                C,
                H,
                W,
                R,
                S,
                pad_h,
                pad_w,
                P=P,
                Q=Q,
                causal=ctx.causal,
            )

        return grad_input, grad_weight, None


class Conv2dTriton(nn.Module):
    def __init__(self, channels, kernel_size, causal=False):
        super(Conv2dTriton, self).__init__()

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

    def forward(self, x):
        return Conv2dTritonFunction.apply(x, self.weight, self.causal)


def add_causal_mask(x):
    mask = torch.tril(torch.ones_like(x))
    return x * mask


class PytorchConv(nn.Module):
    def __init__(self, channels, kernel_size, causal=False):
        super(PytorchConv, self).__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.weight = nn.Parameter(
            torch.randn(channels, 1, self.kernel_size[0], self.kernel_size[1])
        )
        self.causal = causal

    def forward(self, x):
        return pytorch_conv(x, self.weight, self.causal)


def pytorch_conv(x, weight, causal=False):
    qdim, kdim = weight.shape[-2], weight.shape[-1]
    scores_padded = torch.nn.functional.pad(
        x, ((kdim - 1) // 2, (kdim - 1) // 2, qdim - 1, 0), value=0.0
    )
    groups = x.shape[1]
    conv_scores = F.conv2d(
        scores_padded,
        weight,
        padding=0,
        groups=groups,
    )
    if causal:
        conv_scores = add_causal_mask(conv_scores)
    return conv_scores
