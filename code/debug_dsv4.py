"""Module-by-module comparison of the DSV4 adapter against ref_dsv4/model.py on the tiny test model."""
import sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dsv4.py")).read()
pre, rest = src.split("# ------------------------------------------------------------------ reference forward")
ckpt_part = rest.split("# ------------------------------------------------------------------ checkpoints (raw safetensors writer)")[1]
ckpt_part = ckpt_part.split("# FP4 layout cross-check")[0]
exec(pre)
caps = {}


def hook(name):
    def f(mod, inp, out):
        caps[name] = ([a.detach().clone() if torch.is_tensor(a) else a for a in inp],
                      out.detach().clone() if torch.is_tensor(out) else out)
    return f


for L, blk in enumerate(m.layers):
    blk.attn.register_forward_hook(hook(f"attn{L}"))
    blk.ffn.register_forward_hook(hook(f"ffn{L}"))
    blk.register_forward_hook(hook(f"blk{L}"))
    if blk.attn.compress_ratio:
        blk.attn.compressor.register_forward_hook(hook(f"comp{L}"))
tokens = torch.randint(0, V, (S_ALL, T), generator=g)
sel = [1, 3, 5]
with torch.no_grad():
    lg_ref = m(tokens[sel]).float()
torch.set_default_dtype(torch.float32)
del sys.modules["fast_hadamard_transform"]
exec(ckpt_part)
d = "/tmp/moe_test/tiny_dsv4_dbg"
write_ckpt(d, lambda n: n)
sys.path.insert(0, HERE)
import mr_run as MR
from masked_route import simulate
A = MR.DSV4(d, torch.device("cpu"), torch.bfloat16, None, T)
A.tok = tokens[sel].reshape(-1)
ck = MR.RawCkpt(d)


def cmp(name, a, b):
    a, b = a.float(), b.float()
    print(f"  {name:28s} max|d| {float((a-b).abs().max()):.3e}  rel {float((a-b).norm()/(b.norm()+1e-12)):.3e}")


with torch.no_grad():
    for L in range(args.n_layers):
        o = A.build_layer(L, A.load_layer(ck, L))
        blk = m.layers[L]
        print(f"layer {L} ratio {o.ratio} hash {o.hash}")
        # ---- attention on the reference's own normalised input
        (x_in, start_pos), a_ref = caps[f"attn{L}"]
        x = x_in
        rd, fc = A.rd, A.fc[1 if o.ratio else 0]
        qr = MR.rms(A._lin_f(x, o.wq_a), o.q_norm, A.eps)
        q = A._lin_f(qr, o.wq_b).unflatten(-1, (A.nh, A.hd))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + A.eps)
        MR.rope_(q[..., -rd:], fc)
        kv = MR.rms(A._lin_f(x, o.wkv), o.kv_norm, A.eps)
        MR.rope_(kv[..., -rd:], fc)
        kv[..., :-rd] = MR.fp8_sim(kv[..., :-rd], 64)
        cmp("freqs vs ref", torch.view_as_real(fc), torch.view_as_real(blk.attn.freqs_cis[:T]))
        if o.ratio:
            kvc = A._compress(x, o, fc)
            cmp("compressed kv", kvc, caps[f"comp{L}"][1])
        else:
            kvc = None
        a = A._attend(q, kv, kvc, o)
        MR.rope_(a[..., -rd:], fc, inverse=True)
        a = torch.einsum("bsgd,grd->bsgr", a.view(3, T, A.ng, -1), o.wo_a)
        y = A._lin_f(a.flatten(2), o.wo_b)
        cmp("attn out", y, a_ref)
        # ---- block-level hc on the reference's block input
        (xb_in, _, _), blk_ref = caps[f"blk{L}"]
        xb = xb_in.clone()
        A.attn_(xb, o)
        # ---- ffn on the reference's own normalised input
        (f_in, ids), f_ref = caps[f"ffn{L}"]
        h = f_in.reshape(-1, A.H)
        z, s, forced = A.route(h, o, 0)
        E, K = A.E, A.K
        kw = dict(wsc=s.view(3, T, E), wscale=A.wscale, forced=None if forced is None else forced.view(3, T, K))
        ti, tw, _ = simulate(z.view(3, T, E), "exact", None, E, 16, K, 0, 0, **kw)
        ti, tw = ti.view(-1, K), tw.view(-1, K)
        ref_w, ref_i = blk.ffn.gate(h, ids.flatten())
        same = (ti.sort(-1).values == ref_i.long().sort(-1).values).all(-1).float().mean()
        print(f"  routing identical sets: {float(same):.4f}")
        out = torch.zeros(h.shape[0], A.H)
        for e in range(E):
            r_, c_ = torch.where(ti == e)
            if len(r_):
                out.index_add_(0, r_, A.expert(h[r_], e, o, tw[r_, c_]).float())
        out += A.shared(h, o).float()
        cmp("ffn out", out.view(3, T, -1).to(torch.bfloat16), f_ref)
        # full block from the reference block input
        xh, ctx = A.moe_in(xb.view(-1, *A.hs), o)
        cmp("attn half-block (hc)", xb, blk_ref if False else xb)   # placeholder
        # full block
        xf = xb.view(-1, *A.hs)
        z, s, forced = A.route(xh, o, 0)
        kw = dict(wsc=s.view(3, T, E), wscale=A.wscale, forced=None if forced is None else forced.view(3, T, K))
        ti, tw, _ = simulate(z.view(3, T, E), "exact", None, E, 16, K, 0, 0, **kw)
        ti, tw = ti.view(-1, K), tw.view(-1, K)
        out = torch.zeros(xh.shape[0], A.H)
        for e in range(E):
            r_, c_ = torch.where(ti == e)
            if len(r_):
                out.index_add_(0, r_, A.expert(xh[r_], e, o, tw[r_, c_]).float())
        out += A.shared(xh, o).float()
        A.moe_out(xf, out, ctx)
        cmp("block out", xb, blk_ref)
