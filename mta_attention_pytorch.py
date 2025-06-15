import torch
import torch.nn as nn
import torch.nn.functional as F

class MTAAttention(nn.Module):
    def __init__(self,
                 n_heads: int,
                 mta,
                 dropout: float):
        super().__init__()

        assert mta.use_mta, "MTAAttention can only be used with MTA"
        assert mta.query_kernel_size is not None and mta.key_kernel_size is not None
        assert mta.pre_sm_linear_head is True
        # Set default values if not provided
        if not hasattr(mta, 'after_sm_query_kernel_size'):
            mta.after_sm_query_kernel_size = 3
        if not hasattr(mta, 'after_sm_key_kernel_size'):
            mta.after_sm_key_kernel_size = 3
        if not hasattr(mta, 'post_sm_linear_head'):
            mta.post_sm_linear_head = True

        self.n_heads = n_heads
        self.mta = mta
        self.dropout = dropout

        self.mta_kernel = torch.nn.parameter.Parameter(
            torch.empty(
                self.n_heads, 1, mta.query_kernel_size, mta.key_kernel_size
            )
        )

        self.wpsm = nn.Linear(
            n_heads,
            n_heads,
            bias=False,
        )
        
        
        self.mta_kernel_after_sm = torch.nn.parameter.Parameter(
            torch.empty(
                self.n_heads, 1, mta.after_sm_query_kernel_size, mta.after_sm_key_kernel_size
            )
        )
        
        self.wposm = nn.Linear(
            n_heads,
            n_heads,
            bias=False,
        )
        
        self.pad_key = mta.pad_key
        self.mta_init_method = mta.init_method
        
        self.dropout = dropout
        
        
    def forward(self, xq, xk, xv, mask, chunk_start_ids):
        assert chunk_start_ids is None, "MTAAttention does not support chunked attention"
        mask = self._update_mask(mask=mask, bsz=xq.size(0), xq=xq, xk_size=xk.size(-2))

        head_dim = xq.size(-1)
        scores = torch.matmul(xq, xk.transpose(2, 3)) * torch.rsqrt(
                torch.tensor(head_dim, requires_grad=False, device="cuda")
            )

        scores = self._mta_convolution(scores, mask, chunk_start_ids, self.mta_kernel)
        scores = self.wpsm(scores.transpose(1, -1)).transpose(1, -1)

        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        scores = self._mta_convolution(scores, mask, chunk_start_ids, self.mta_kernel_after_sm)
        scores = torch.where(mask == float("-inf"), 0.0, scores)
        
        scores = self.wposm(scores.transpose(1, -1)).transpose(1, -1)

        scores = F.dropout(scores, p=self.dropout, training=self.training)
        
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2)
        
        return output

    def _mta_convolution(self, scores, mask, chunk_start_ids, kernel):
        assert chunk_start_ids is None, "MTAAttention does not support chunked attention"
        scores = torch.where(mask == float("-inf"), 0.0, scores)
        n_loc_heads, head_kernel_size, qdim, kdim = kernel.shape
        assert n_loc_heads == self.n_heads

        if self.pad_key == "left":
            scores_padded = torch.nn.functional.pad(
                scores, (kdim - 1, 0, qdim - 1, 0), value=0.0
            )
        elif self.pad_key == "right":
            scores_padded = torch.nn.functional.pad(
                scores, (0, kdim - 1, qdim - 1, 0), value=0.0
            )
        elif self.pad_key == "both":
            assert (kdim - 1) % 2 == 0
            scores_padded = torch.nn.functional.pad(
                scores, ((kdim - 1) // 2, (kdim - 1) // 2, qdim - 1, 0), value=0.0
            )
            
        conv_scores = F.conv2d(
            scores_padded,
            kernel,
            padding=0,
            groups=self.n_heads // head_kernel_size,
        )
        del scores_padded
        return conv_scores

    def _update_mask(
        self, mask: torch.tensor, bsz: int, xq: torch.tensor, xk_size: int
    ):
        if not isinstance(mask, torch.Tensor):
            # causal mask
            mask = torch.full((xq.size(-2), xk_size), float("-inf"), device=xq.device)
            mask = torch.triu(mask, diagonal=1).type_as(xq)
            mask = mask.repeat(bsz, self.n_heads, 1, 1)
        else:
            # generation task, mask is provided and reflects concatenated docs
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
        init_std = init_std or (self.mta.dim ** (-0.5))

        if self.mta_kernel is not None:
            if self.mta_init_method == "uniform":
                torch.nn.init.uniform_(self.mta_kernel.data, a=-1.0, b=1.0)
            elif self.mta_init_method == "normal":
                init_std = 0.3
                torch.nn.init.uniform_(
                    self.mta_kernel.data,
                    mean=0.0,
                    std=init_std,
                    a=-3 * init_std,
                    b=3 * init_std,
                )
            elif self.mta_init_method == "diagonal":
                # diagonal of the form
                # 1 0 0 0 0
                # 0 1 0 0 0
                # 0 0 1 0 0
                with torch.no_grad():
                    diagonal_kernel = torch.ones(self.mta_kernel.data.shape)
                    _, _, A, B = self.mta_kernel.data.shape
                    diagonal = (B + 1) // 2 - A
                    diagonal_kernel = torch.tril(diagonal_kernel, diagonal=diagonal)
                    diagonal_kernel = torch.triu(diagonal_kernel, diagonal=diagonal)
                    self.mta_kernel.data.copy_(diagonal_kernel)
            elif self.mta_init_method == "identity":
                assert self.pad_key == "both"
                # identity kernel of the form
                # 0 0 0 0 0
                # 0 0 0 0 0
                # 0 0 1 0 0
                with torch.no_grad():
                    nheads, head_sz, query_sz, key_sz = self.mta_kernel.data.shape
                    identity_kernel = torch.zeros(
                        nheads, head_sz, query_sz, key_sz
                    ).cuda()
                    if head_sz == 1:
                        identity_kernel[:, :, -1, key_sz // 2] = 1.0
                    else:
                        # it is bit complicated with head conv
                        # weight to other heads should be zero
                        identity_kernel = identity_kernel.view(
                            nheads // head_sz, head_sz, head_sz, query_sz, key_sz
                        )
                        for i in range(head_sz):
                            identity_kernel[:, i, i, -1, key_sz // 2] = 1.0
                        identity_kernel = identity_kernel.view(
                            nheads, head_sz, query_sz, key_sz
                        )
                    self.mta_kernel.data.copy_(identity_kernel)
            elif self.mta_init_method == "const":
                self.mta_kernel.data.fill_(0.3)
            else:
                raise ValueError(
                    f"Unsopperted mta_init_method: {self.mta_init_method}"
                )

        if True:
            assert self.mta_init_method == "identity"
            assert self.pad_key == "both"
            with torch.no_grad():
                (
                    nheads,
                    head_sz,
                    query_sz,
                    key_sz,
                ) = self.mta_kernel_after_sm.data.shape
                identity_kernel = torch.zeros(
                    nheads, head_sz, query_sz, key_sz
                ).cuda()
                if head_sz == 1:
                    identity_kernel[:, :, -1, key_sz // 2] = 1.0
                else:
                    # it is bit complicated with head conv
                    # weight to other heads should be zero
                    identity_kernel = identity_kernel.view(
                        nheads // head_sz, head_sz, head_sz, query_sz, key_sz
                    )
                    for i in range(head_sz):
                        identity_kernel[:, i, i, -1, key_sz // 2] = 1.0
                    identity_kernel = identity_kernel.view(
                        nheads, head_sz, query_sz, key_sz
                    )
                self.mta_kernel_after_sm.data.copy_(identity_kernel)

        if True:
            identity_kernel = torch.eye(self.n_heads).cuda()
            self.wpsm.weight.data.copy_(identity_kernel)

        if True:
            identity_kernel = torch.eye(self.n_heads).cuda()
            self.wposm.weight.data.copy_(identity_kernel)


    def copy_mta_parameters(self, mta_attention):
        self.mta_kernel.data.copy_(mta_attention.mta_kernel.data)
        self.mta_kernel_after_sm.data.copy_(mta_attention.mta_kernel_after_sm.data)
        self.wpsm.weight.data.copy_(mta_attention.wpsm.weight.data)
        self.wposm.weight.data.copy_(mta_attention.wposm.weight.data)
        
        
    




