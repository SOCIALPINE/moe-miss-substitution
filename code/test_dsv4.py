"""Check mr_run.py's DeepSeek-V4 adapter against the official inference/model.py (ref_dsv4/model.py) on a tiny
random V4: hyper-connections, hash layer, sliding window + compressed KV (ratio 4 with overlap + lightning
indexer, and a non-overlapping ratio), attention sinks, YaRN, FP8 weights + ue8m0 scales, packed FP4 experts.
The reference's tilelang kernels are replaced by exact torch equivalents (below). The checkpoint is written both
with HF-style names and with native names, each in 2 shards with an index.  Runs on CPU with torch >= 2.4."""
import os, sys, json, math, types, struct, shutil, subprocess
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"
if not hasattr(torch, "float4_e2m1fn_x2"):      # old torch: packed fp4 lives in int8, e8m0 scales in uint8
    torch.float4_e2m1fn_x2 = torch.int8
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.uint8
if True:                        # integer / packed stand-ins cannot require grad
    _P = torch.nn.Parameter

    class _Param(_P):
        def __new__(cls, data=None, requires_grad=True):
            if data is not None and not (data.is_floating_point() or data.is_complex()):
                requires_grad = False
            return _P.__new__(cls, data, requires_grad)
    torch.nn.Parameter = _Param
FP4 = torch.float4_e2m1fn_x2
E8 = torch.float8_e8m0fnu
TAB = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def u8(t):
    return t.view(torch.uint8)


def e8(s):                                  # e8m0 bytes / fp32 scales -> fp32
    return torch.exp2(u8(s).float() - 127) if s.dtype in (torch.uint8, E8) else s.float()


def p2c(v):
    m, e = torch.frexp(v)
    return torch.ldexp(torch.ones_like(v), torch.where(m == 0.5, e - 1, e))


# ------------------------------------------------------------------ torch stand-ins for inference/kernel.py
def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    z = x.contiguous()
    xf = z.float().view(*z.shape[:-1], N // block_size, block_size)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = p2c(amax * (1.0 / 448.0)) if scale_fmt is not None else amax * (1.0 / 448.0)
    y = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s).view(z.shape).to(x.dtype))
        return x
    return y.view(z.shape), s.squeeze(-1)


def _gemm(a, a_s, bdeq):
    K = a.size(-1)
    ad = (a.float().view(-1, K // 128, 128) * a_s.float().reshape(-1, K // 128, 1)).view(-1, K)
    return (ad @ bdeq.t()).to(torch.get_default_dtype()).view(*a.shape[:-1], -1)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    n, k = b.shape
    sc = e8(b_s).repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return _gemm(a, a_s, b.float() * sc)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    x = u8(b)                                   # [N, K/2], two e2m1 per byte, low nibble first (inference/convert.py)
    v = torch.stack([TAB[(x & 15).long()], TAB[(x >> 4).long()]], -1).flatten(1)
    return _gemm(a, a_s, v * e8(b_s).repeat_interleave(32, 1))


def fp4_act_quant(x, block_size=32, inplace=False):
    assert inplace
    return x                                    # only used by the indexer; irrelevant when it selects everything


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b = q.size(0)
    idx = topk_idxs.long()
    valid = idx >= 0
    g = kv[torch.arange(b)[:, None, None], idx.clamp(min=0)].float()          # [b, m, k, d]
    s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g) * softmax_scale
    s = s.masked_fill(~valid[:, :, None, :], float("-inf"))
    sink = attn_sink.float().view(1, 1, -1)
    mx = torch.maximum(s.amax(-1), sink)
    p = torch.exp(s - mx[..., None])
    den = p.sum(-1) + torch.exp(sink - mx)
    o = torch.einsum("bmhk,bmkd->bmhd", p.to(torch.bfloat16).float(), g) / den[..., None]
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    h = hc_mult
    pre = torch.sigmoid(mixes[..., :h] * hc_scale[0] + hc_base[:h]) + eps
    post = 2 * torch.sigmoid(mixes[..., h:2 * h] * hc_scale[1] + hc_base[h:2 * h])
    c = mixes[..., 2 * h:] * hc_scale[2] + hc_base[2 * h:]
    c = c.view(*c.shape[:-1], h, h)
    c = torch.exp(c - c.amax(-1, keepdim=True))
    c = c / c.sum(-1, keepdim=True) + eps
    c = c / (c.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        c = c / (c.sum(-1, keepdim=True) + eps)
        c = c / (c.sum(-2, keepdim=True) + eps)
    return pre, post, c


kern = types.ModuleType("kernel")
for f in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn):
    setattr(kern, f.__name__, f)
sys.modules["kernel"] = kern
fht = types.ModuleType("fast_hadamard_transform")
fht.hadamard_transform = lambda x, scale=1.0: x
sys.modules["fast_hadamard_transform"] = fht
sys.path.insert(0, os.path.join(HERE, "ref_dsv4"))
import model as R                                                  # noqa: E402

# ------------------------------------------------------------------ tiny model
S_ALL, T, P, V = 6, 64, 16, 512
cfg = dict(max_batch_size=3, max_seq_len=T, dtype="fp8", scale_fmt="ue8m0", expert_dtype="fp4", scale_dtype="fp8",
           vocab_size=V, dim=256, moe_inter_dim=128, n_layers=4, n_hash_layers=1, n_mtp_layers=1, n_heads=4,
           n_routed_experts=16, n_shared_experts=1, n_activated_experts=3, score_func="sqrtsoftplus", route_scale=1.5,
           swiglu_limit=10.0, q_lora_rank=128, head_dim=128, rope_head_dim=64, norm_eps=1e-6, o_groups=2,
           o_lora_rank=64, window_size=16, compress_ratios=[0, 4, 8, 4, 0], compress_rope_theta=160000.0,
           original_seq_len=256, rope_theta=10000.0, rope_factor=16, beta_fast=32, beta_slow=1, index_n_heads=4,
           index_head_dim=128, index_topk=T // 4, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
args = R.ModelArgs(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()})
m = R.Transformer(args).eval()
R.ParallelHead.get_logits = lambda self, x: F.linear(x.float(), self.weight)      # all positions, not only the last


def q8(w):          # fp32 [o,i] -> fp8 [o,i] + e8m0 byte scale per 128x128 block
    o, i = w.shape
    bo, bi = math.ceil(o / 128), math.ceil(i / 128)
    wp = F.pad(w, (0, bi * 128 - i, 0, bo * 128 - o)).view(bo, 128, bi, 128)
    s = p2c(wp.abs().amax((1, 3)).clamp(min=1e-8) / 448.0)
    q = (wp / s[:, None, :, None]).view(bo * 128, bi * 128)[:o, :i].to(torch.float8_e4m3fn)
    return q, (torch.log2(s).round() + 127).to(torch.uint8)


def bfr(t):         # bf16-representable fp32
    return t.bfloat16().float()


stored = {}         # native name -> (tensor to save, safetensors dtype)
g = torch.Generator().manual_seed(1)
rn = lambda *sh: torch.randn(*sh, generator=g)
with torch.no_grad():
    for mn, mod in m.named_modules():
        if isinstance(mod, R.Linear):
            w = mod.weight
            o, i = mod.out_features, mod.in_features
            if w.dtype == torch.float8_e4m3fn:
                q, s = q8(rn(o, i) / math.sqrt(i))
                w.data = q.clone(); mod.scale.data = s.clone().view(mod.scale.dtype)
                stored[mn + ".weight"] = (q, "F8_E4M3"); stored[mn + ".scale"] = (s, "F8_E8M0")
            elif w.dtype == FP4:
                b = torch.randint(0, 256, (o, i // 2), generator=g, dtype=torch.int64).to(torch.uint8)
                s = torch.randint(118, 122, (o, i // 32), generator=g, dtype=torch.int64).to(torch.uint8)
                w.data = b.clone().view(w.dtype); mod.scale.data = s.clone().view(mod.scale.dtype)
                stored[mn + ".weight"] = (b.view(torch.int8), "I8"); stored[mn + ".scale"] = (s, "F8_E8M0")
            elif mn.endswith("wo_a"):             # bf16 in the model, FP8 + scale in the HF checkpoint
                q, s = q8(rn(o, i) / math.sqrt(i))
                w.data.copy_((q.float() * torch.exp2(s.float() - 127).repeat_interleave(128, 0)[:o]
                              .repeat_interleave(128, 1)[:, :i]).to(w.dtype))
                stored[mn + ".weight"] = (q, "F8_E4M3"); stored[mn + ".scale"] = (s, "F8_E8M0")
            else:                                  # compressor wkv/wgate (fp32 param, bf16 in ckpt), weights_proj
                w.data.copy_(bfr(rn(o, i) / math.sqrt(i)))
                stored[mn + ".weight"] = (w.data.bfloat16(), "BF16")
    for n, p in m.named_parameters():
        if n in stored or n.endswith(".scale") and n[:-6] + ".weight" in stored:
            continue
        leaf = n.split(".")[-1]
        if n.endswith("tid2eid"):
            p.data.copy_(torch.stack([torch.randperm(args.n_routed_experts, generator=g)[:args.n_activated_experts]
                                      for _ in range(V)]).int())
            stored[n] = (p.data.clone(), "I32")
            p.data = p.data.long()          # old torch gather() wants int64 indices; same values
            continue
        if leaf.startswith("hc_") and leaf.endswith("_fn"):
            v = rn(*p.shape) * 0.02
        elif leaf.startswith("hc_") and leaf.endswith("_scale"):
            v = 1 + 0.1 * rn(*p.shape)
        elif leaf.startswith("hc_") and leaf.endswith("_base"):
            v = 0.5 * rn(*p.shape)
        elif leaf == "attn_sink":
            v = rn(*p.shape)
        elif leaf == "ape":
            v = 0.3 * rn(*p.shape)
        elif n.endswith("gate.weight"):
            v = bfr(2 * rn(*p.shape) / math.sqrt(p.shape[1]))
        elif n.endswith("gate.bias"):
            v = 0.1 * rn(*p.shape)
        elif n == "embed.weight":
            v = bfr(rn(*p.shape))
        elif n == "head.weight":
            v = bfr(2 * rn(*p.shape) / math.sqrt(p.shape[1]))
        elif leaf == "weight" and p.dim() == 1:                  # RMSNorm
            v = bfr(1 + 0.1 * rn(*p.shape))
        else:
            raise RuntimeError("unhandled param " + n)
        p.data.copy_(v.to(p.dtype))
        dt = "BF16" if (leaf == "weight" and (p.dim() == 1 or n in ("head.weight",))) or n.endswith("gate.weight") \
            or n == "embed.weight" else "F32"
        stored[n] = (p.data.bfloat16() if dt == "BF16" else p.data.float().clone(), dt)

# ------------------------------------------------------------------ reference forward
tokens = torch.randint(0, V, (S_ALL, T), generator=g)
sel = [1, 3, 5]
with torch.no_grad():
    lg_ref = m(tokens[sel]).float()                      # [3, T, V]
lp_ref = torch.log_softmax(lg_ref[:, P:T - 1], -1).reshape(-1, V)
ppl_ref = float(torch.exp(-lp_ref.gather(1, tokens[sel][:, P + 1:].reshape(-1, 1)).mean()))
torch.set_default_dtype(torch.float32)
del sys.modules["fast_hadamard_transform"]          # transformers probes this package's __spec__
print("reference ppl", ppl_ref)

# ------------------------------------------------------------------ checkpoints (raw safetensors writer)
HF = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "embed": "embed_tokens",
      "head": "lm_head", "w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def hf_name(n):
    parts = n.split(".")
    parts = [HF.get(x, x) for x in parts]
    n = ".".join(parts).replace(".attn.", ".self_attn.").replace(".ffn.", ".mlp.")
    if n.endswith(".scale") and not n.split(".")[-2].startswith("hc_"):
        n = n[:-len(".scale")] + ".weight_scale_inv"
    n = n.replace("gate.bias", "gate.e_score_correction_bias")
    return n if n.startswith("lm_head") else "model." + n


def write_ckpt(d, rename):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "inference"))
    json.dump(cfg, open(os.path.join(d, "inference", "config.json"), "w"))
    names = sorted(stored)
    wmap = {}
    for si, part in enumerate((names[::2], names[1::2])):
        fn = f"model-{si+1:05d}-of-00002.safetensors"
        hdr, blobs, off = {}, [], 0
        for n in part:
            t, dt = stored[n]
            b = t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16 else \
                t.contiguous().view(torch.int16).numpy().tobytes()
            k = rename(n)
            hdr[k] = {"dtype": dt, "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
            blobs.append(b); off += len(b)
            wmap[k] = fn
        h = json.dumps(hdr).encode()
        h += b" " * ((8 - len(h) % 8) % 8)
        with open(os.path.join(d, fn), "wb") as f:
            f.write(struct.pack("<Q", len(h))); f.write(h)
            for b in blobs:
                f.write(b)
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(d, "model.safetensors.index.json"), "w"))


base = "/tmp/moe_test/tiny_dsv4"
dirs = [(base + "_hf", hf_name), (base + "_native", lambda n: n)]
for d, rn_ in dirs:
    write_ckpt(d, rn_)
np.savez(base + "_sessions.npz", tokens=tokens.numpy().astype(np.int32), domain=np.array([0, 0, 1, 1, 2, 2]),
         domain_names=np.array(["a", "b", "c"]), prompt_len=np.array(P))

# FP4 layout cross-check against the official lossless converter (e2m1 -> e4m3 with 128x128 block scales)
sys.path.insert(0, os.path.join(HERE, "ref_dsv4"))
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda x, **k: x, trange=range))
import convert as CV                                                # noqa: E402
sys.path.insert(0, HERE)
from mr_run import deq_fp4                                         # noqa: E402
wb = torch.randint(0, 256, (256, 128), generator=g, dtype=torch.int64).to(torch.uint8)
sb = torch.randint(118, 126, (256, 8), generator=g, dtype=torch.int64).to(torch.uint8)
if E8 is torch.uint8:           # old torch: feed/receive real fp32 scale values instead of e8m0 bytes
    torch.float8_e8m0fnu = torch.float32
    x8, s8 = CV.cast_e2m1fn_to_e4m3fn(wb.view(torch.int8), torch.exp2(sb.float() - 127))
    torch.float8_e8m0fnu = torch.uint8
else:
    x8, s8 = CV.cast_e2m1fn_to_e4m3fn(wb.view(torch.int8), sb.view(E8))
ref8 = x8.float() * e8(s8).repeat_interleave(128, 0).repeat_interleave(128, 1)
assert torch.equal(ref8, deq_fp4(wb, sb, torch.float32)), "FP4 layout mismatch vs inference/convert.py"
print("FP4 layout matches inference/convert.py")

# ------------------------------------------------------------------ adapter runs
for d, _ in dirs:
    out = d + "/mr"
    env = dict(os.environ, MR_DUMP_BASE_LOGP=d + "/base_lp.npy")
    r = subprocess.run([sys.executable, os.path.join(HERE, "mr_run.py"), "--adapter", "deepseek_v4", "--model_dir", d,
                        "--data", base + "_sessions.npz", "--out", out, "--device", DEV, "--attn_batch", "2",
                        "--chunk", "64", "--per_domain", "1",
                        "--configs", "1.0:lru:B,1.0:lru:C,0.5:lru:B,0.5:lru:C@0.3"],
                       capture_output=True, text=True, env=env)
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0
    st = json.load(open(out + "/mr.json"))["streams"]
    lp = torch.from_numpy(np.load(d + "/base_lp.npy"))
    kl = float((lp_ref.exp() * (lp_ref - lp)).sum(-1).mean())
    agree = float((lp_ref.argmax(-1) == lp.argmax(-1)).float().mean())
    print(d, "ppl ref", ppl_ref, "adapter", st["base"]["ppl"]["ALL"], "KL(ref||adapter)", kl, "argmax agree", agree,
          "max|dlogp|", float((lp - lp_ref).abs().max()))
    assert abs(ppl_ref - st["base"]["ppl"]["ALL"]) / ppl_ref < 2e-2 and kl < 5e-3 and agree > 0.97
    for k in ("1.0:lru:B", "1.0:lru:C"):
        assert st[k]["kl_nats"] < 1e-9 and st[k]["routing"]["subst_rate"] == 0, k
    r5 = st["0.5:lru:B"]["routing"]
    assert st["0.5:lru:B"]["kl_nats"] > 0 and r5["subst_rate"] > 0 and r5["hash_stall_experts_per_tok"] > 0
    assert st["0.5:lru:C@0.3"]["routing"]["tau"] == 0.3
print("DSV4 ADAPTER TEST PASSED")
