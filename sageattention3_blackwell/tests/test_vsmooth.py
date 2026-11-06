"""Tests of VC-Attention V-Smooth on the SageAttention3 Blackwell kernel.

    pytest sageattention3_blackwell/tests/test_vsmooth.py

The reference is fp32 attention. On Gaussian data the NVFP4 kernel itself is
far from fp32, so the checks that use random data bound the V-Smooth error by
the plain kernel's error on the same data, and the sharp checks use values that
are constant within a tile under the grouping: their residuals are exactly
zero, the output is the probability-mass weighted sum of the tile means and
any error in the units, rescaling or masking of the added means shows directly.
"""

import pytest
import torch
import torch.nn.functional as F

from sageattn3 import sageattn3_blackwell, sageattn3_blackwell_vsmooth, token_permutation
from sageattn3.api import triton_group_mean

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="the SageAttention3 Blackwell kernel runs on SM 12.0",
)

GROUP = 128


def ref_attention(q, k, v, chunk=4096):
    """fp32 attention, chunked over the queries so long sequences fit in memory."""
    q, k, v = q.float(), k.float(), v.float()
    return torch.cat([F.scaled_dot_product_attention(q[:, :, i:i + chunk], k, v) for i in range(0, q.size(2), chunk)], dim=2)


def rel_error(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def random_perm(B, H, L, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.argsort(torch.rand(B, H, L, device="cuda", generator=g), dim=-1)


def tile_constant_values(perm, means, dtype=torch.bfloat16):
    """Values equal to ``means[b, h, t]`` for every token of tile ``t`` under ``perm``."""
    B, H, L = perm.shape
    grouped = means.repeat_interleave(GROUP, dim=2)[:, :, :L]
    v = torch.empty_like(grouped)
    v.scatter_(2, perm.unsqueeze(-1).expand(-1, -1, -1, means.size(-1)), grouped)
    return v.to(dtype)


def peaked_qk(B, H, QL, KL, D, scale=3.0, dtype=torch.bfloat16, seed=0):
    """Scores whose row maximum moves from tile to tile, so the online rescaling runs."""
    torch.manual_seed(seed)
    k = torch.randn(B, H, KL, D, device="cuda")
    q = scale * k[:, :, torch.randint(0, KL, (QL,), device="cuda")] + torch.randn(B, H, QL, D, device="cuda")
    return q.to(dtype), k.to(dtype)


# --------------------------------------------------------------------------- #
# Exact structure: tile-constant values                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("L", [128, 129, 383, 2000, 4097])
@pytest.mark.parametrize("D", [64, 128])
def test_means_conserve_probability_mass(L, D):
    """With every value equal to one, each output row is the sum of the per-tile
    masses over the row sum, which must be exactly one: the added means share the
    units and the online rescaling of the row sum, and the tail tile's pad rows
    carry no mass."""
    B, H = 2, 3
    q, k = peaked_qk(B, H, L, L, D, scale=8.0)
    perm = random_perm(B, H, L)
    v = torch.ones(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    out = sageattn3_blackwell_vsmooth(q, k, v, perm)
    assert out.shape == (B, H, L, D)
    assert torch.isfinite(out).all()
    assert (out.float() - 1.0).abs().max().item() < 2e-3


@pytest.mark.parametrize("L", [256, 1000, 8191])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tile_means_weighted_by_tile_mass(L, dtype):
    """Values constant within each tile under the grouping, with large tile means
    of both signs: the output is the mass-weighted sum of the means, so only the
    probabilities carry error. The plain kernel sees the same NVFP4 scores and
    exact per-16-key V blocks but also rounds P to NVFP4, so V-Smooth must not be
    worse than it on the same grouped data."""
    B, H, D = 1, 4, 128
    q, k = peaked_qk(B, H, L, L, D, dtype=dtype)
    perm = random_perm(B, H, L)
    T = (L + GROUP - 1) // GROUP
    means = 20.0 * torch.randn(B, H, T, D, device="cuda")
    v = tile_constant_values(perm, means, dtype)
    ref = ref_attention(q, k, v)
    kg = torch.gather(k, 2, perm.unsqueeze(-1).expand(-1, -1, -1, D))
    vg = torch.gather(v, 2, perm.unsqueeze(-1).expand(-1, -1, -1, D))
    plain = rel_error(sageattn3_blackwell(q, kg, vg), ref)
    smooth = rel_error(sageattn3_blackwell_vsmooth(q, k, v, perm), ref)
    assert smooth < 0.1
    assert smooth <= plain


# --------------------------------------------------------------------------- #
# The point of V-Smooth: large tile means leave the residual quantization      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("offset", [10.0, 100.0, 1000.0])
def test_large_tile_offsets_removed(offset):
    """Residuals of unit variance around tile means of growing size. The plain
    kernel's NVFP4 V loses the residual against the offset; V-Smooth must stay at
    the error of the offset-free data whatever the offset."""
    B, H, L, D = 1, 4, 2000, 128
    q, k = peaked_qk(B, H, L, L, D)
    perm = random_perm(B, H, L)
    T = (L + GROUP - 1) // GROUP
    torch.manual_seed(1)
    residual = torch.randn(B, H, L, D, device="cuda")
    means = offset * torch.randn(B, H, T, D, device="cuda")
    v = (residual + tile_constant_values(perm, means, torch.float32)).to(torch.bfloat16)
    ref = ref_attention(q, k, v)
    plain = rel_error(sageattn3_blackwell(q, k, v), ref)
    smooth = rel_error(sageattn3_blackwell_vsmooth(q, k, v, perm), ref)
    centered = rel_error(sageattn3_blackwell(q, k, residual.to(torch.bfloat16)), ref_attention(q, k, residual))
    assert smooth < 0.6 * plain
    assert smooth < centered * 1.3 + 0.01


def test_clustered_values_beat_plain_kernel():
    """The paper's setting: values from a few clusters, grouped by k-means. The
    grouped V-Smooth error must be well below the plain kernel's."""
    B, H, L, D = 1, 2, 2000, 128
    q, k = peaked_qk(B, H, L, L, D)
    torch.manual_seed(2)
    centers = 4.0 * torch.randn(B, H, 8, D, device="cuda")
    v = (centers[:, :, torch.randint(0, 8, (L,), device="cuda")] + 0.5 * torch.randn(B, H, L, D, device="cuda")).to(torch.bfloat16)
    ref = ref_attention(q, k, v)
    perm = token_permutation(v, num_clusters=8, generator=torch.Generator(device="cuda").manual_seed(0))
    plain = rel_error(sageattn3_blackwell(q, k, v), ref)
    smooth = rel_error(sageattn3_blackwell_vsmooth(q, k, v, perm), ref)
    assert smooth < 0.85 * plain


# --------------------------------------------------------------------------- #
# Shapes, dtypes, options                                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("B,H,QL,KL,D", [
    (1, 1, 1, 1, 128),
    (1, 1, 1, 127, 64),
    (3, 5, 129, 1, 128),
    (1, 2, 2000, 129, 128),
    (2, 1, 100, 2000, 64),
    (1, 3, 4097, 4096, 128),
    (1, 1, 128, 100000, 128),
])
@pytest.mark.parametrize("per_block_mean", [True, False])
def test_shapes_and_options(B, H, QL, KL, D, per_block_mean):
    """Query and key lengths of every residue modulo 128, both head sizes and
    both delta_s modes: finite output of the unpadded shape, no worse than the
    plain kernel on the same (random, ungrouped) data."""
    q, k = peaked_qk(B, H, QL, KL, D)
    torch.manual_seed(3)
    v = torch.randn(B, H, KL, D, device="cuda", dtype=torch.bfloat16)
    perm = random_perm(B, H, KL)
    out = sageattn3_blackwell_vsmooth(q, k, v, perm, per_block_mean=per_block_mean)
    assert out.shape == (B, H, QL, D) and out.dtype == q.dtype
    assert torch.isfinite(out).all()
    ref = ref_attention(q, k, v)
    plain = rel_error(sageattn3_blackwell(q, k, v, per_block_mean=per_block_mean), ref)
    assert rel_error(out, ref) < plain * 1.25 + 0.01


def test_fp16_matches_bf16_path():
    B, H, L, D = 1, 2, 1000, 128
    q, k = peaked_qk(B, H, L, L, D, dtype=torch.float16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    perm = random_perm(B, H, L)
    out16 = sageattn3_blackwell_vsmooth(q, k, v, perm)
    outbf = sageattn3_blackwell_vsmooth(q.bfloat16(), k.bfloat16(), v.bfloat16(), perm)
    assert out16.dtype == torch.float16
    ref = ref_attention(q, k, v)
    assert abs(rel_error(out16, ref) - rel_error(outbf, ref)) < 0.02


def test_identity_permutation_equals_grouping_in_input_order():
    """With the identity permutation the grouped path must give the same result
    as grouping the input order itself; the gather and the scatter of the means
    through ``perm`` are then no-ops."""
    B, H, L, D = 1, 2, 1000, 128
    q, k = peaked_qk(B, H, L, L, D)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) + 5.0
    identity = torch.arange(L, device="cuda").expand(B, H, L)
    a = sageattn3_blackwell_vsmooth(q, k, v, identity)
    b = sageattn3_blackwell_vsmooth(q, k, v, identity.clone())
    assert torch.equal(a, b)


def test_deterministic():
    B, H, L, D = 2, 4, 3000, 128
    q, k = peaked_qk(B, H, L, L, D)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    perm = random_perm(B, H, L)
    a = sageattn3_blackwell_vsmooth(q, k, v, perm)
    b = sageattn3_blackwell_vsmooth(q, k, v, perm)
    assert torch.equal(a, b)


def test_inputs_not_modified():
    B, H, L, D = 1, 2, 1000, 128
    q, k = peaked_qk(B, H, L, L, D)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    perm = random_perm(B, H, L)
    q0, k0, v0, p0 = q.clone(), k.clone(), v.clone(), perm.clone()
    sageattn3_blackwell_vsmooth(q, k, v, perm)
    assert torch.equal(q, q0) and torch.equal(k, k0) and torch.equal(v, v0) and torch.equal(perm, p0)


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("L", [128, 129, 1000, 2047])
@pytest.mark.parametrize("permute", [False, True])
def test_group_mean_kernel(L, permute):
    """The gathered, demeaned and padded values and the means match torch."""
    B, H, D = 2, 3, 128
    torch.manual_seed(4)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 3 + 2
    perm = random_perm(B, H, L) if permute else None
    out, mu = triton_group_mean(v, perm)
    T = (L + GROUP - 1) // GROUP
    assert out.shape == (B, H, T * GROUP, D) and mu.shape == (B, H, T, D)
    g = v if perm is None else torch.gather(v, 2, perm.unsqueeze(-1).expand(-1, -1, -1, D))
    g = F.pad(g.float(), (0, 0, 0, T * GROUP - L))
    counts = torch.full((T,), GROUP, device="cuda", dtype=torch.float32)
    counts[-1] = L - (T - 1) * GROUP
    mu_ref = g.view(B, H, T, GROUP, D).sum(3) / counts[None, None, :, None]
    assert torch.allclose(mu.float(), mu_ref, atol=0, rtol=2 ** -8)  # bf16 rounding of the stored mean
    res_ref = (g.view(B, H, T, GROUP, D) - mu_ref.unsqueeze(3)).view(B, H, T * GROUP, D)
    res_ref[:, :, L:] = 0
    assert torch.allclose(out.float(), res_ref, atol=4e-2, rtol=1e-2)
    assert (out[:, :, L:] == 0).all()


@pytest.mark.parametrize("L,num_clusters", [(1, 64), (50, 64), (2000, 8), (2000, 64), (75600, 64)])
def test_token_permutation_is_a_permutation(L, num_clusters):
    B, H, D = 2, 3, 128
    torch.manual_seed(5)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(0)
    perm = token_permutation(v, num_clusters=num_clusters, generator=g)
    assert perm.shape == (B, H, L) and perm.dtype == torch.int64
    assert torch.equal(torch.sort(perm, dim=-1).values, torch.arange(L, device="cuda").expand(B, H, L))
    again = token_permutation(v, num_clusters=num_clusters, generator=torch.Generator(device="cuda").manual_seed(0))
    assert torch.equal(perm, again)


def test_token_permutation_groups_clusters():
    """Eight well separated clusters: under the permutation the cluster label
    changes at most once per k-means group (64 groups, each inside one cluster),
    for the whole batch and for one head on its own."""
    B, H, L, D = 2, 2, 4096, 128
    torch.manual_seed(6)
    labels = torch.randint(0, 8, (B, H, L), device="cuda")
    centers = 6.0 * torch.randn(B, H, 8, D, device="cuda")
    v = (torch.gather(centers, 2, labels.unsqueeze(-1).expand(-1, -1, -1, D)) + 0.3 * torch.randn(B, H, L, D, device="cuda")).to(torch.bfloat16)
    perm = token_permutation(v, generator=torch.Generator(device="cuda").manual_seed(0))
    grouped = torch.gather(labels, 2, perm)
    changes = (grouped[..., 1:] != grouped[..., :-1]).sum(-1)
    assert changes.max().item() < 64, changes.tolist()
    alone = token_permutation(v[1:, 1:], generator=torch.Generator(device="cuda").manual_seed(0))
    assert (torch.gather(labels[1:, 1:], 2, alone).diff(dim=-1) != 0).sum().item() < 64


# --------------------------------------------------------------------------- #
# Rejected inputs                                                              #
# --------------------------------------------------------------------------- #

def test_rejects_causal_and_bad_means():
    from sageattn3.api import fp4_attention, preprocess_qkv
    B, H, L, D = 1, 1, 256, 128
    q, k = peaked_qk(B, H, L, L, D)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16)
    q, k, v, delta_s = preprocess_qkv(q, k, v, True)
    mu = torch.zeros(B, H, L // GROUP, D, device="cuda")
    fp4_attention(q, k, v, delta_s, L, L, mu=mu)
    with pytest.raises(RuntimeError):
        fp4_attention(q, k, v, delta_s, L, L, is_causal=True, mu=mu)
    with pytest.raises(RuntimeError):
        fp4_attention(q, k, v, delta_s, L, L, mu=mu.to(torch.bfloat16))
    with pytest.raises(RuntimeError):
        fp4_attention(q, k, v, delta_s, L, L, mu=mu[:, :, :1])


# --------------------------------------------------------------------------- #
# Production shape                                                             #
# --------------------------------------------------------------------------- #

def test_long_sequence_tile_means():
    """A video-generation length, odd, with tile-constant values: the output is
    the mass-weighted sum of 591 tile means against chunked fp32 attention."""
    B, H, L, D = 1, 8, 75601, 128
    q, k = peaked_qk(B, H, L, L, D)
    perm = random_perm(B, H, L)
    T = (L + GROUP - 1) // GROUP
    means = 20.0 * torch.randn(B, H, T, D, device="cuda")
    v = tile_constant_values(perm, means)
    out = sageattn3_blackwell_vsmooth(q, k, v, perm)
    assert out.shape == (B, H, L, D) and torch.isfinite(out).all()
    assert rel_error(out, ref_attention(q, k, v)) < 0.05
