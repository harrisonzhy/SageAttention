import torch
import argparse

from sageattn3 import sageattn3_blackwell, sageattn3_blackwell_vsmooth, token_permutation

parser = argparse.ArgumentParser(description='Benchmark SageAttention3 with VC-Attention V-Smooth')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_heads', type=int, default=40, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--seq_len', type=int, default=75600, help='Sequence length')
args = parser.parse_args()

print(f"SageAttention3 V-Smooth Benchmark")
print(f"batch: {args.batch_size}, head: {args.num_heads}, headdim: {args.head_dim}, seq_len: {args.seq_len}")

def relative_error(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()

def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[iters // 2]

# Accuracy: peaked attention over values that fall into a few clusters, the
# structure the value grouping exploits.
torch.manual_seed(0)
B, H, L, D = 1, 2, 2000, args.head_dim
k = torch.randn(B, H, L, D, device="cuda")
q = (3.0 * k).to(torch.bfloat16)
k = k.to(torch.bfloat16)
centers = torch.randn(B, H, 8, D, device="cuda") * 4
v = (centers[:, :, torch.randint(0, 8, (L,), device="cuda")] + 0.5 * torch.randn(B, H, L, D, device="cuda")).to(torch.bfloat16)
ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
perm = token_permutation(v, num_clusters=8, generator=torch.Generator(device="cuda").manual_seed(0))
print(f"relative error vs fp32: plain {relative_error(sageattn3_blackwell(q.clone(), k.clone(), v), ref):.4f}, "
      f"vsmooth {relative_error(sageattn3_blackwell_vsmooth(q.clone(), k.clone(), v, perm), ref):.4f}")

# Speed, including the preprocessing of each path.
B, H, L, D = args.batch_size, args.num_heads, args.seq_len, args.head_dim
q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
perm = token_permutation(v)
print(f"plain:   {bench(lambda: sageattn3_blackwell(q, k, v)):.3f} ms")
print(f"vsmooth: {bench(lambda: sageattn3_blackwell_vsmooth(q, k, v, perm)):.3f} ms")
