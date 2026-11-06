"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda


@triton.jit
def group_mean_kernel(
    q_ptr,          
    q_out_ptr,      
    qm_out_ptr,     
    perm_ptr,       # None: input order; else the rows of each group are read through it
    B, H, L, D: tl.constexpr,    
    stride_qb, stride_qh, stride_ql, stride_qd,  
    stride_ob, stride_oh, stride_ol, stride_od,
    stride_qmb, stride_qmh, stride_qml, stride_qmd,  
    stride_pb, stride_ph, stride_pl,
    GROUP_SIZE: tl.constexpr,
    PERMUTE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    
    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    valid = offsets < L
    rows = offsets
    if PERMUTE:
        rows = tl.load(perm_ptr + pid_b * stride_pb + pid_h * stride_ph + offsets * stride_pl, mask=valid, other=0)
    q_offsets = pid_b * stride_qb + pid_h * stride_qh + rows[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets, mask=valid[:, None], other=0.0)
    
    # Mean of the real rows, rounded to the stored type so the rows minus their
    # stored mean reproduce the input exactly.
    qm_group = tl.sum(q_group.to(tl.float32), axis=0) / tl.minimum(L - group_start, GROUP_SIZE)
    qm_group = qm_group.to(qm_out_ptr.dtype.element_ty).to(tl.float32)
    
    q_group = tl.where(valid[:, None], q_group - qm_group, 0.0)
    out_offsets = pid_b * stride_ob + pid_h * stride_oh + offsets[:, None] * stride_ol + tl.arange(0, D)[None, :] * stride_od
    tl.store(q_out_ptr + out_offsets, q_group)

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor, perm: torch.Tensor = None):
    """Per-128-row means and the demeaned rows. With ``perm`` (B, H, L) the rows
    are gathered through it and the output is padded to a multiple of 128 with
    zero rows; the means are over the real rows."""
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = (L + GROUP_SIZE - 1) // GROUP_SIZE
    
    q_out = torch.empty(B, H, num_groups * GROUP_SIZE, D, device=q.device, dtype=q.dtype)
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype) 
    
    grid = (B, H, num_groups)
    
    group_mean_kernel[grid](
        q, q_out, qm, perm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        *((0, 0, 0) if perm is None else perm.stride()),
        GROUP_SIZE=GROUP_SIZE,
        PERMUTE=perm is not None
    )
    return q_out, qm


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):

    def pad_128(x):
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
    
    k -= k.mean(dim=-2, keepdim=True)  
    q, k, v = map(lambda x: pad_128(x), [q, k, v])
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k, v, delta_s

def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False, 
                         per_block_mean: bool = True,
                         is_bf16: bool = True,
                         mu: torch.Tensor = None
                        ):
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, mu, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16)


def fp4_attention(q, k, v, delta_s, QL, KL, is_causal = False, per_block_mean = True, mu = None):
    """Quantize the preprocessed (padded) q, k, v to NVFP4 and run the kernel for the
    ``QL`` queries and ``KL`` keys; ``mu`` are the V-Smooth tile means of ``v`` when
    ``v`` holds tile-mean residuals."""
    is_bf16 = q.dtype == torch.bfloat16
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    o_fp4 = blockscaled_fp4_attn(
    qlist_from_cuda,
    klist_from_cuda, 
    vlist_from_cuda,
    delta_s,
    KL,
    is_causal,
    per_block_mean,
    is_bf16,
    mu
    )[0][:, :, :QL, :].contiguous()
    return o_fp4


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, **kwargs):
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    return fp4_attention(q, k, v, delta_s, QL, KL, is_causal, per_block_mean)

def token_permutation(v: torch.Tensor, num_clusters: int = 64, iters: int = 3, generator: torch.Generator = None):
    """VC-Attention value grouping: k-means over the value rows of every batch and
    head, then the tokens sorted by cluster, so that 128-token tiles hold similar
    values. Returns the int64 permutation (B, H, L); K and V are read through it."""
    B, H, L, D = v.shape
    num_clusters = min(num_clusters, L)
    x = v.reshape(B * H, L, D).float()
    seed_rows = torch.randperm(L, device=v.device, generator=generator)[:num_clusters]
    centroids = x[:, seed_rows]  # one draw for every batch element and head
    x_sq = (x * x).sum(-1, keepdim=True)
    for _ in range(iters):
        c_sq = (centroids * centroids).sum(-1).unsqueeze(1)
        labels = (x_sq + c_sq - 2.0 * torch.bmm(x, centroids.transpose(1, 2))).argmin(dim=-1)
        members = F.one_hot(labels, num_clusters).to(x.dtype).transpose(1, 2)  # deterministic, unlike scatter_add_
        counts = members.sum(-1, keepdim=True)
        centroids = torch.where(counts > 0, torch.bmm(members, x) / counts.clamp_min(1.0), centroids)
    return torch.argsort(labels, dim=-1, stable=True).view(B, H, L)


def sageattn3_blackwell_vsmooth(q, k, v, perm, per_block_mean = True, **kwargs):
    """VC-Attention V-Smooth on the 4-bit kernel (paper Table 2, RTX PRO 6000 block).

    K and V are read in the order of ``perm`` (``token_permutation``), the values
    are stored as NVFP4 residuals around their 128-token tile means and the kernel
    adds each tile's mean back weighted by the row's probability mass. Attention is
    invariant to a common key and value permutation, so the output is unchanged
    in meaning; a causal mask is not supported. ``q, k, v`` are FP16/BF16
    ``(batch_size, head_num, seq_len, head_dim)``.
    """
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v)
    QL = q.size(2)
    KL = k.size(2)
    k = torch.gather(k, 2, perm.unsqueeze(-1).expand(-1, -1, -1, k.size(-1)))
    # The values are gathered, demeaned per tile and padded in one pass.
    v, mu = triton_group_mean(v, perm)
    q, k, _, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    return fp4_attention(q, k, v, delta_s, QL, KL, False, per_block_mean, mu.float().contiguous())
