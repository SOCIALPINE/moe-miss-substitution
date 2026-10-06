"""Model-agnostic cache-constrained routing runner (same method as masked_route.py, adapters per model).

An adapter provides: config numbers (H, E, K, NL), the per-token hidden shape `hs` (H,) or e.g. (hc, H) for
hyper-connections, embedding, a per-layer weight loader (CPU, background thread), a per-layer builder (GPU), the
attention/mixer residual update, the MoE input (normalised h + residual context), router scores, one expert's
forward on a token subset, an optional shared expert, the MoE residual update, and the final norm + lm_head.
Routing weights: softmax over the selected logits by default (Qwen3-Next, gpt-oss); adapters may instead return
separate weight scores (DeepSeek-V4: sqrtsoftplus, renormalised, x route_scale) and forced sets (hash layers).
"""
import os, sys, json, time, argparse, threading, queue, struct, contextlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_next import Ckpt        # noqa: E402  (safetensors reader that waits for downloading shards)
from masked_route import simulate  # noqa: E402  (causal cache simulation, modes exact/B/C)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class AdapterBase:
    hs = None                 # per-token hidden shape (set in __init__)
    weights_inside = False    # expert() applies the routing weight itself
    wscale = 1.0

    def route(self, h, o, c0):
        return self.logits(h, o).float(), None, None

    def moe_out(self, xc, out, ctx):
        xc.add_(out.to(xc.dtype))


# ============================================================================ gpt-oss adapter
class GptOss(AdapterBase):
    def __init__(self, model_dir, dev, dt, dl_flag_dir, T):
        from transformers import AutoConfig
        from transformers.models.gpt_oss import modeling_gpt_oss as M
        self.M = M
        cfg = AutoConfig.from_pretrained(model_dir)
        cfg._attn_implementation = "eager"
        self.cfg, self.dev, self.dt = cfg, dev, dt
        self.H, self.E, self.K, self.NL = cfg.hidden_size, cfg.num_local_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
        self.hs = (self.H,)
        self.I = cfg.intermediate_size
        self.alpha, self.limit = 1.702, float(getattr(cfg, "swiglu_limit", 7.0))
        self.ck = Ckpt(model_dir, dl_flag_dir)
        rot = M.GptOssRotaryEmbedding(config=cfg, device=dev)
        dummy = torch.zeros(1, T, self.H, dtype=dt, device=dev)
        pos = torch.arange(T, device=dev)[None]
        self.pe = rot(dummy, pos)
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        kw = dict(config=cfg, input_embeds=dummy, attention_mask=None, cache_position=pos[0], past_key_values=None)
        self.masks = {"full_attention": create_causal_mask(**kw),
                      "sliding_attention": create_sliding_window_causal_mask(**kw)}
        self.name = "gpt_oss"

    def embed(self, tok):
        w = self.ck.get("model.embed_tokens.weight").to(self.dev, self.dt)
        return w[tok]

    def load_layer(self, ck, L):
        p = f"model.layers.{L}."
        g = lambda n: ck.get(p + n)
        W = {"attn": {k: g("self_attn." + k).to(self.dt) for k in
                      ("q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias", "v_proj.weight", "v_proj.bias",
                       "o_proj.weight", "o_proj.bias", "sinks")},
             "in_norm": g("input_layernorm.weight").float(), "post_norm": g("post_attention_layernorm.weight").float(),
             "rw": g("mlp.router.weight").to(self.dt), "rb": g("mlp.router.bias").to(self.dt),
             "gub": g("mlp.experts.gate_up_proj_bias").to(self.dt), "dnb": g("mlp.experts.down_proj_bias").to(self.dt)}
        for proj in ("gate_up_proj", "down_proj"):
            if (p + f"mlp.experts.{proj}_blocks") in ck.wmap:
                W[proj + "_blocks"] = g(f"mlp.experts.{proj}_blocks")
                W[proj + "_scales"] = g(f"mlp.experts.{proj}_scales")
            else:
                W[proj] = g(f"mlp.experts.{proj}").to(self.dt)
        return W

    def build_layer(self, L, W):
        M, dev, dt = self.M, self.dev, self.dt
        with torch.device("meta"):
            attn = M.GptOssAttention(self.cfg, L)
        attn.load_state_dict({k: v.to(dev) for k, v in W["attn"].items()}, strict=True, assign=True)
        o = type("LayerObj", (), {})()
        o.attn = attn.eval()
        o.ltype = self.cfg.layer_types[L]
        o.in_norm = self._norm(W["in_norm"])
        o.post_norm = self._norm(W["post_norm"])
        o.rw, o.rb = W["rw"].to(dev), W["rb"].to(dev)
        o.gub, o.dnb = W["gub"].to(dev), W["dnb"].to(dev)
        for proj in ("gate_up_proj", "down_proj"):
            if proj + "_blocks" in W:
                from transformers.integrations.mxfp4 import convert_moe_packed_tensors
                t = convert_moe_packed_tensors(W[proj + "_blocks"].to(dev), W[proj + "_scales"].to(dev), dtype=dt)
            else:
                t = W[proj].to(dev)
            setattr(o, proj, t.contiguous())
        return o

    def _norm(self, w):
        with torch.device("meta"):
            n = self.M.GptOssRMSNorm(w.shape[0], eps=self.cfg.rms_norm_eps)
        n.load_state_dict({"weight": w.to(self.dev)}, strict=True, assign=True)
        return n

    def attn_(self, xb, o):
        h = o.in_norm(xb)
        out, _ = o.attn(hidden_states=h, position_embeddings=self.pe, attention_mask=self.masks[o.ltype])
        xb.add_(out)

    def moe_in(self, x, o):
        return o.post_norm(x), None

    def logits(self, h, o):
        return F.linear(h, o.rw, o.rb)

    def expert(self, hh, e, o, ww=None):
        gu = hh @ o.gate_up_proj[e] + o.gub[e]
        gate, up = gu[..., ::2], gu[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return ((up + 1) * glu) @ o.down_proj[e] + o.dnb[e]

    def shared(self, h, o):
        return None

    def head(self):
        fn = self._norm(self.ck.get("model.norm.weight").float())
        lm = self.ck.get("lm_head.weight").to(self.dev, self.dt)
        return lambda x: F.linear(fn(x), lm)

    def expert_bytes_4bit(self):          # MXFP4: 4 bits + 1 scale byte / 32 weights, plus biases
        n = 3 * self.H * self.I
        return n * (4 / 8 + 1 / 32)


# ============================================================================ DeepSeek-V4 adapter
class RawCkpt(Ckpt):
    """Reads tensors straight from the safetensors bytes, so FP8-E8M0 scales / packed FP4 load on any torch."""
    DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16, "F8_E4M3": torch.float8_e4m3fn,
          "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
          "F8_E8M0": torch.uint8, "BOOL": torch.bool}

    def _hdr(self, fn):
        if fn not in self.h:
            path = os.path.join(self.d, fn)
            self._wait(path)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            self.h[fn] = (path, 8 + n, hdr)
        return self.h[fn]

    def meta(self, name):
        return self._hdr(self.wmap[name])[2][name]

    def get(self, name):
        path, base, hdr = self._hdr(self.wmap[name])
        m = hdr[name]
        s, e = m["data_offsets"]
        buf = torch.empty(e - s, dtype=torch.uint8)
        with open(path, "rb") as f:
            f.seek(base + s)
            f.readinto(buf.numpy())
        if m["dtype"] not in self.DT:        # e.g. a packed 4-bit dtype: raw bytes, caller reshapes
            return buf
        return buf.view(self.DT[m["dtype"]]).view(m["shape"])


V4_MAP = {"embed_tokens": "embed", "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
          "q_a_proj": "wq_a", "q_a_layernorm": "q_norm", "q_b_proj": "wq_b", "kv_a_layernorm": "kv_norm",
          "o_proj": "wo", "gate_proj": "w1", "down_proj": "w2", "up_proj": "w3", "lm_head": "head"}


def v4_native(name):
    """Checkpoint name -> DeepSeek-V4 inference-module path (same renaming as the official inference/convert.py)."""
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("self_attn", "attn").replace("mlp", "ffn").replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    parts = name.split(".")
    key = parts[-1] if any(x in name for x in ("hc", "attn_sink", "tie2eid", "ape")) else (parts[-2] if len(parts) > 1 else parts[0])
    if key in V4_MAP:
        name = name.replace(key, V4_MAP[key])
    return name


FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def e8m0(s):
    return torch.exp2(s.float() - 127.0) if s.dtype == torch.uint8 else s.float()


def pow2_ceil(v):
    """2**ceil(log2 v) for v > 0, exact (the kernels' IEEE bit trick)."""
    m, e = torch.frexp(v)
    return torch.ldexp(torch.ones_like(v), torch.where(m == 0.5, e - 1, e))


def fp8_sim(x, block):
    """Block-wise FP8-E4M3 quantise->dequantise with power-of-2 (ue8m0) scales, as kernel.act_quant does."""
    shp = x.shape
    xf = x.float().reshape(*shp[:-1], shp[-1] // block, block)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = pow2_ceil(amax * (1.0 / 448.0))
    y = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.reshape(shp).to(x.dtype)


def deq_fp8(w, s, dt):
    o, i = w.shape
    sc = e8m0(s).repeat_interleave(128, 0)[:o].repeat_interleave(128, 1)[:, :i]
    return (w.float() * sc).to(dt)


def deq_fp4(w, s, dt):
    """packed e2m1 [out, in/2] (low nibble = even column) x ue8m0 scale per 32 inputs -> [out, in]"""
    wb = w.view(torch.uint8)
    lut = torch.tensor(FP4_TABLE, dtype=torch.float32, device=w.device)
    v = torch.stack([lut[(wb & 15).long()], lut[(wb >> 4).long()]], -1).flatten(-2)
    return (v * e8m0(s).repeat_interleave(32, -1)).to(dt)


def rms(x, w, eps):
    xf = x.float()
    return (w * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))).to(x.dtype)


def rope_(x, fc, inverse=False):
    """in-place rotary on the last dims (interleaved pairs), as inference/model.apply_rotary_emb"""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    f = fc.conj() if inverse else fc
    f = f.view(1, xc.size(1), xc.size(-1)) if xc.ndim == 3 else f.view(1, xc.size(1), 1, xc.size(-1))
    x.copy_(torch.view_as_real(xc * f).flatten(-2))


def freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    import math

    def corr_dim(nrot):
        return dim * math.log(original_seq_len / (nrot * 2 * math.pi)) / (2 * math.log(base))
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = max(math.floor(corr_dim(beta_fast)), 0), min(math.ceil(corr_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    f = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(f), f)


@contextlib.contextmanager
def tf32():
    """bf16-valued fp32 operands: TF32 products are exact, accumulation fp32 (= bf16 GEMM with fp32 output)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class DSV4(AdapterBase):
    """DeepSeek-V4 (inference/model.py semantics, prefill of whole sessions): hyper-connections (hc_mult copies of
    the residual, Sinkhorn mixing), MQA attention = 128-token sliding window + compressed KV (ratio 4 with overlap,
    or 128), attention sinks, sqrtsoftplus router with bias + hash-routed first layers, FP8 weights / FP4 experts
    with FP8 activation quantisation (emulated exactly: power-of-2 scales, bf16-exact dequantised GEMMs).
    With T <= 4*index_topk the lightning indexer selects every causal compressed entry, so it is skipped."""
    weights_inside = True

    def __init__(self, model_dir, dev, dt, dl_flag_dir, T):
        c = dict(norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20, n_hash_layers=0, swiglu_limit=0.0, route_scale=1.0,
                 score_func="sqrtsoftplus", beta_fast=32, beta_slow=1, original_seq_len=0, rope_factor=40,
                 compress_rope_theta=40000.0, rope_theta=10000.0, window_size=128, index_topk=512, hc_mult=4)
        for p in (os.path.join(model_dir, "inference", "config.json"), os.path.join(model_dir, "config_v4.json")):
            if os.path.exists(p):
                c.update(json.load(open(p)))
                break
        else:
            raise FileNotFoundError("need inference/config.json (DeepSeek-V4 ModelArgs)")
        self.c, self.dev, self.dt = c, dev, torch.bfloat16
        self.H, self.E, self.K, self.NL = c["dim"], c["n_routed_experts"], c["n_activated_experts"], c["n_layers"]
        self.I, self.hc = c["moe_inter_dim"], c["hc_mult"]
        self.hs = (self.hc, self.H)
        self.eps, self.hc_eps = c["norm_eps"], c["hc_eps"]
        self.nh, self.hd, self.rd = c["n_heads"], c["head_dim"], c["rope_head_dim"]
        self.ng, self.win = c["o_groups"], c["window_size"]
        self.ratios = c["compress_ratios"]
        self.lim = float(c["swiglu_limit"])
        self.wscale = float(c["route_scale"])
        assert c["score_func"] in ("sqrtsoftplus", "sigmoid"), c["score_func"]
        if any(r == 4 for r in self.ratios[:self.NL]) and T // 4 > c["index_topk"]:
            raise NotImplementedError("sessions longer than 4*index_topk need the lightning indexer")
        self.fc = [freqs_cis(self.rd, T, 0, c["rope_theta"], c["rope_factor"], c["beta_fast"], c["beta_slow"]).to(dev),
                   freqs_cis(self.rd, T, c["original_seq_len"], c["compress_rope_theta"], c["rope_factor"],
                             c["beta_fast"], c["beta_slow"]).to(dev)]
        self.ck = RawCkpt(model_dir, dl_flag_dir)
        self.nm = {}
        for k in self.ck.wmap:
            n = v4_native(k)
            if not n.startswith("mtp."):
                self.nm[n] = k
        need = ["embed.weight", "norm.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"]
        for L in sorted({0, self.NL - 1} | {i for i in range(self.NL) if self.ratios[i]}.intersection({2, 3})):
            need += self._layer_names(L)
        miss = [n for n in need if n not in self.nm]
        if miss:
            log("DSV4 checkpoint names (layer 0 / top level):")
            for k in sorted(self.ck.wmap):
                if ".layers.0." in "." + k or "layers." not in k:
                    if "experts." not in k or ".experts.0." in k:
                        log("   ", k, "->", v4_native(k))
            raise KeyError(f"missing V4 tensors: {miss[:12]}")
        self.tok = None
        self.name = "deepseek_v4"

    def _layer_names(self, L):
        p = f"layers.{L}."
        n = [p + s for s in ("attn_norm.weight", "ffn_norm.weight", "hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                             "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale", "attn.wq_a.weight", "attn.q_norm.weight",
                             "attn.wq_b.weight", "attn.wkv.weight", "attn.kv_norm.weight", "attn.wo_a.weight",
                             "attn.wo_b.weight", "attn.attn_sink", "ffn.gate.weight",
                             "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight",
                             "ffn.shared_experts.w3.weight", "ffn.experts.0.w1.weight", "ffn.experts.0.w1.scale")]
        n.append(p + ("ffn.gate.tid2eid" if L < self.c["n_hash_layers"] else "ffn.gate.bias"))
        if self.ratios[L]:
            n += [p + "attn.compressor." + s for s in ("ape", "wkv.weight", "wgate.weight", "norm.weight")]
        return n

    # ---------------------------------------------------------------- loading
    def _g(self, ck, n):
        return ck.get(self.nm[n])

    def _lin(self, ck, pre):
        w = self._g(ck, pre + ".weight")
        s = self._g(ck, pre + ".scale") if (pre + ".scale") in self.nm else None
        if s is not None and w.dtype == torch.uint8 and w.dim() == 1:   # packed 4-bit stored with its own dtype
            w = w.view(s.shape[0], -1)
        return (w, s)

    def embed(self, tok):
        w = self._g(self.ck, "embed.weight").to(self.dev, self.dt)
        x = w[tok]
        return x.unsqueeze(-2).repeat(*([1] * (x.dim() - 1)), self.hc, 1)

    def load_layer(self, ck, L):
        p = f"layers.{L}."
        g = lambda n: self._g(ck, p + n)
        W = {"attn_norm": g("attn_norm.weight"), "ffn_norm": g("ffn_norm.weight"),
             "q_norm": g("attn.q_norm.weight"), "kv_norm": g("attn.kv_norm.weight"), "sink": g("attn.attn_sink"),
             "gw": g("ffn.gate.weight")}
        for k in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
            W[k] = g(k)
        for k in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            W[k], W[k + "_s"] = self._lin(ck, p + "attn." + k)
        if L < self.c["n_hash_layers"]:
            W["tid2eid"] = g("ffn.gate.tid2eid")
        else:
            W["gb"] = g("ffn.gate.bias")
        if self.ratios[L]:
            for k in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                W["c_" + k] = g("attn.compressor." + k)
        for k in ("w1", "w2", "w3"):
            W["s" + k], W["s" + k + "_s"] = self._lin(ck, p + "ffn.shared_experts." + k)
            ws = [self._lin(ck, p + f"ffn.experts.{e}.{k}") for e in range(self.E)]
            W["e" + k] = torch.stack([a for a, _ in ws])
            W["e" + k + "_s"] = torch.stack([b for _, b in ws]) if ws[0][1] is not None else None
        return {k: v for k, v in W.items() if v is not None}

    def build_layer(self, L, W):
        dev, bf = self.dev, torch.bfloat16
        o = type("LayerObj", (), {})()
        f32 = lambda k: W[k].to(dev).float()

        def lin(k):
            w, s = W[k].to(dev), (W[k + "_s"].to(dev) if (k + "_s") in W else None)
            if s is None:
                return (w.to(bf), False)
            return (deq_fp8(w, s, bf) if w.dtype == torch.float8_e4m3fn else deq_fp4(w, s, bf), True)
        o.ratio, o.hash = self.ratios[L], L < self.c["n_hash_layers"]
        o.attn_norm, o.ffn_norm, o.q_norm, o.kv_norm = f32("attn_norm"), f32("ffn_norm"), f32("q_norm"), f32("kv_norm")
        o.sink = f32("sink")
        o.hc_attn = (f32("hc_attn_fn"), f32("hc_attn_scale"), f32("hc_attn_base"))
        o.hc_ffn = (f32("hc_ffn_fn"), f32("hc_ffn_scale"), f32("hc_ffn_base"))
        o.wq_a, o.wq_b, o.wkv, o.wo_b = lin("wq_a"), lin("wq_b"), lin("wkv"), lin("wo_b")
        o.wo_a = lin("wo_a")[0].view(self.ng, -1, self.nh * self.hd // self.ng)      # bf16 einsum, no act-quant
        if o.ratio:
            o.c_ape, o.c_wkv, o.c_wgate, o.c_norm = f32("c_ape"), f32("c_wkv.weight"), f32("c_wgate.weight"), f32("c_norm.weight")
        o.gw = f32("gw")
        o.gb = f32("gb") if "gb" in W else None
        o.tid2eid = W["tid2eid"].to(dev).long() if "tid2eid" in W else None
        o.s1, o.s2, o.s3 = lin("sw1"), lin("sw2"), lin("sw3")
        for k in ("w1", "w2", "w3"):
            w, s = W["e" + k], W.get("e" + k + "_s")
            out = None
            for e in range(self.E):
                we = w[e].to(dev)
                if s is None:
                    d = we.to(bf)
                else:
                    se = s[e].to(dev)
                    d = deq_fp8(we, se, bf) if we.dtype == torch.float8_e4m3fn else deq_fp4(we, se, bf)
                if out is None:
                    out = torch.empty(self.E, *d.shape, dtype=bf, device=dev)
                out[e] = d
            setattr(o, "e" + k, out)
        o.eq = "ew1_s" in W
        return o

    # ---------------------------------------------------------------- compute
    @staticmethod
    def _lin_f(x, wq):
        w, q = wq
        return F.linear(fp8_sim(x, 128) if q else x, w)

    def _sinkhorn(self, mixes, scale, base):
        hc = self.hc
        pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + self.hc_eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
        comb = torch.softmax(comb, -1) + self.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.c["hc_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    def _hc_pre(self, x, hcw):
        fn, scale, base = hcw
        xf = x.flatten(-2).float()
        mixes = F.linear(xf, fn) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        pre, post, comb = self._sinkhorn(mixes, scale, base)
        y = (pre.unsqueeze(-1) * xf.view(x.shape)).sum(-2)
        return y.to(x.dtype), post, comb

    @staticmethod
    def _hc_post(y, res, post, comb):
        mix = torch.einsum("...jk,...jd->...kd", comb, res.float())
        return (post.unsqueeze(-1) * y.unsqueeze(-2) + mix).to(y.dtype)

    def _compress(self, x, o, fc):
        r, d, rd = o.ratio, self.hd, self.rd
        xf = x.float()
        kv, sc = F.linear(xf, o.c_wkv), F.linear(xf, o.c_wgate)
        T = x.size(1)
        cut = T - T % r
        kv = kv[:, :cut].unflatten(1, (-1, r))
        sc = sc[:, :cut].unflatten(1, (-1, r)) + o.c_ape
        if r == 4:                       # overlapping windows: previous block (first half dims) + current block
            def ovl(t, v):
                b, s = t.shape[:2]
                n = t.new_full((b, s, 2 * r, d), v)
                n[:, :, r:] = t[:, :, :, d:]
                n[:, 1:, :r] = t[:, :-1, :, :d]
                return n
            kv, sc = ovl(kv, 0.0), ovl(sc, float("-inf"))
        kv = (kv * sc.softmax(dim=2)).sum(dim=2)
        kv = rms(kv.to(torch.bfloat16), o.c_norm, self.eps)
        rope_(kv[..., -rd:], fc[:cut:r])
        kv[..., :-rd] = fp8_sim(kv[..., :-rd], 64)
        return kv

    def _attend(self, q, kv, kvc, o):
        b, T = q.shape[:2]
        r, win = o.ratio, self.win
        scale = self.hd ** -0.5
        out = torch.empty_like(q)
        pos = torch.arange(T, device=q.device)
        sink = o.sink.view(1, 1, -1)
        B = 256
        for q0 in range(0, T, B):
            q1 = min(T, q0 + B)
            k0 = max(0, q0 - win + 1)
            keys = kv[:, k0:q1]
            i, j = pos[q0:q1], pos[k0:q1]
            valid = (j[None] <= i[:, None]) & (j[None] > i[:, None] - win)
            if kvc is not None:
                nc = q1 // r
                keys = torch.cat([keys, kvc[:, :nc]], 1)
                valid = torch.cat([valid, torch.arange(nc, device=q.device)[None] < ((i + 1) // r)[:, None]], 1)
            kf = keys.float()
            with tf32():
                s = torch.einsum("bqhd,bkd->bqhk", q[:, q0:q1].float(), kf) * scale
            s = s.masked_fill(~valid[None, :, None, :], float("-inf"))
            m = torch.maximum(s.amax(-1), sink)
            p = torch.exp(s - m.unsqueeze(-1))
            den = p.sum(-1) + torch.exp(sink - m)
            with tf32():
                ov = torch.einsum("bqhk,bkd->bqhd", p.to(torch.bfloat16).float(), kf)
            out[:, q0:q1] = (ov / den.unsqueeze(-1)).to(q.dtype)
        return out

    def attn_(self, xb, o):
        b, T = xb.shape[:2]
        rd = self.rd
        fc = self.fc[1 if o.ratio else 0]
        x, post, comb = self._hc_pre(xb, o.hc_attn)
        x = rms(x, o.attn_norm, self.eps)
        qr = rms(self._lin_f(x, o.wq_a), o.q_norm, self.eps)
        q = self._lin_f(qr, o.wq_b).unflatten(-1, (self.nh, self.hd))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        rope_(q[..., -rd:], fc)
        kv = rms(self._lin_f(x, o.wkv), o.kv_norm, self.eps)
        rope_(kv[..., -rd:], fc)
        kv[..., :-rd] = fp8_sim(kv[..., :-rd], 64)
        kvc = self._compress(x, o, fc) if o.ratio else None
        a = self._attend(q, kv, kvc, o)
        rope_(a[..., -rd:], fc, inverse=True)
        a = torch.einsum("bsgd,grd->bsgr", a.view(b, T, self.ng, -1), o.wo_a)
        y = self._lin_f(a.flatten(2), o.wo_b)
        xb.copy_(self._hc_post(y, xb, post, comb))

    def moe_in(self, x, o):
        y, post, comb = self._hc_pre(x, o.hc_ffn)
        return rms(y, o.ffn_norm, self.eps), (post, comb)

    def route(self, h, o, c0):
        l = F.linear(h.float(), o.gw)
        s = F.softplus(l).sqrt() if self.c["score_func"] == "sqrtsoftplus" else torch.sigmoid(l)
        z = s + o.gb if o.gb is not None else s
        forced = o.tid2eid[self.tok[c0:c0 + h.shape[0]]] if o.hash else None
        return z, s, forced

    def _ffn(self, x, w1, w2, w3, q, ww):
        xq = fp8_sim(x, 128) if q else x
        g = F.linear(xq, w1).float()
        u = F.linear(xq, w3).float()
        if self.lim > 0:
            u = u.clamp(min=-self.lim, max=self.lim)
            g = g.clamp(max=self.lim)
        y = F.silu(g) * u
        if ww is not None:
            y = ww[:, None] * y
        y = y.to(x.dtype)
        return F.linear(fp8_sim(y, 128) if q else y, w2)

    def expert(self, hh, e, o, ww=None):
        return self._ffn(hh, o.ew1[e], o.ew2[e], o.ew3[e], o.eq, ww)

    def shared(self, h, o):
        return self._ffn(h, o.s1[0], o.s2[0], o.s3[0], o.s1[1], None)

    def moe_out(self, xc, out, ctx):
        post, comb = ctx
        xc.copy_(self._hc_post(out.to(xc.dtype), xc, post, comb))

    def head(self):
        g = lambda n: self._g(self.ck, n).to(self.dev).float()
        fn, base, scale, nw = g("hc_head_fn"), g("hc_head_base"), g("hc_head_scale"), g("norm.weight")
        W = g("head.weight")

        def f(x):
            xf = x.flatten(-2).float()
            mixes = F.linear(xf, fn) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
            pre = torch.sigmoid(mixes * scale + base) + self.hc_eps
            y = (pre.unsqueeze(-1) * xf.view(x.shape)).sum(-2).to(x.dtype)
            return F.linear(rms(y, nw, self.eps).float(), W)
        return f

    def expert_bytes_4bit(self):          # FP4 e2m1 + one ue8m0 scale byte per 32 weights
        return 3 * self.H * self.I * (4 / 8 + 1 / 32)


ADAPTERS = {"gpt_oss": GptOss, "deepseek_v4": DSV4}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=list(ADAPTERS))
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--attn_batch", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--per_domain", type=int, default=8)
    ap.add_argument("--configs", default="0.25:lru:B,0.25:lru:C,0.125:lru:B,0.125:lru:C",
                    help="frac:policy:mode[@tau][/fallback], mode B or C, fallback subst|drop|renorm|shared")
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=1 / 64)
    ap.add_argument("--dl_flag_dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    t_start = time.time()

    dat = np.load(args.data, allow_pickle=True)
    tokens_all = dat["tokens"].astype(np.int64)
    dom_all = dat["domain"].astype(np.int64)
    names = [str(s) for s in dat["domain_names"]]
    P = int(dat["prompt_len"])
    sel = []
    for d in range(len(names)):
        sel += list(np.where(dom_all == d)[0][1::2][:args.per_domain])
    sel = np.array(sel)
    tokens = tokens_all[sel]; dom = dom_all[sel]
    S, T = tokens.shape
    N = S * T

    A = ADAPTERS[args.adapter](args.model_dir, dev, dt, args.dl_flag_dir, T)
    H, E, K, NL, hs = A.H, A.E, A.K, A.NL, A.hs
    streams = [("base", None)]
    for c in args.configs.split(","):
        fr, pol, mode = c.split(":")
        tau, fb = args.tau, "subst"
        if "/" in mode:
            mode, fb = mode.split("/")
        if "@" in mode:
            mode, t_ = mode.split("@")
            tau = float(t_)
        assert mode in ("B", "C") and fb in ("subst", "drop", "renorm", "shared"), c
        streams.append((c, (max(int(round(float(fr) * E)), K + 1), pol, mode, tau, fb)))
    log(f"{A.name}: layers={NL} experts={E} top{K} hidden={hs} | sessions={S} x {T} (prompt {P}) | "
        f"streams={[s[0] for s in streams]}")

    tok_t = torch.from_numpy(tokens).to(dev)
    A.tok = tok_t.view(N)
    x0 = A.embed(tok_t)
    xs = [x0.clone() for _ in streams]
    del x0
    stats = {s[0]: [] for s in streams if s[1] is not None}

    qq = queue.Queue(maxsize=2)

    def loader():
        try:
            ck2 = type(A.ck)(args.model_dir, args.dl_flag_dir)
            for L in range(NL):
                W = A.load_layer(ck2, L)
                if dev.type == "cuda":
                    W = {k: ({kk: vv.pin_memory() for kk, vv in v.items()} if isinstance(v, dict) else v.pin_memory())
                         for k, v in W.items()}
                qq.put(W)
        except Exception as e:
            qq.put(e)
    threading.Thread(target=loader, daemon=True).start()

    CH = args.chunk
    for L in range(NL):
        tl0 = time.time()
        W = qq.get()
        if isinstance(W, Exception):
            raise W
        o = A.build_layer(L, W)
        del W
        hashL = bool(getattr(o, "hash", False))
        t0 = time.time()
        line = []
        for si_, (name, spec) in enumerate(streams):
            x = xs[si_]
            for b0 in range(0, S, args.attn_batch):
                A.attn_(x[b0:b0 + args.attn_batch], o)
            xf = x.view(N, *hs)
            lg = torch.empty(N, E, dtype=torch.float32, device=dev)
            wsc = forced = None
            for c0 in range(0, N, CH):
                h, _ = A.moe_in(xf[c0:c0 + CH], o)
                z, w_, f_ = A.route(h, o, c0)
                n = z.shape[0]
                lg[c0:c0 + n] = z
                if w_ is not None:
                    if wsc is None:
                        wsc = torch.empty(N, E, dtype=torch.float32, device=dev)
                    wsc[c0:c0 + n] = w_
                if f_ is not None:
                    if forced is None:
                        forced = torch.empty(N, K, dtype=torch.long, device=dev)
                    forced[c0:c0 + n] = f_
                del h
            kw = dict(wsc=None if wsc is None else wsc.view(S, T, E), wscale=A.wscale,
                      forced=None if forced is None else forced.view(S, T, K))
            shx = None
            if spec is None:
                ti, tw, _ = simulate(lg.view(S, T, E), "exact", None, E, P, K, 0, 0, **kw)
            else:
                cap, pol, mode, tau, fb = spec
                if hashL and fb == "subst":   # hash sets are fixed by the token id: no substitution, misses loaded
                    mode, tau = "C", -1.0
                ti, tw, st = simulate(lg.view(S, T, E), mode, pol, cap, P, K, tau, args.alpha, fallback=fb, **kw)
                shx = st.pop("shared_extra", None)
                if shx is not None:
                    shx = shx.view(N)
                st["hash"] = hashL
                stats[name].append(st)
                line.append(f"{name} hit {st['wanted_hit']:.3f} sub {st['subst']:.3f}/{st['subst_mass']:.3f}")
            del lg, wsc, forced
            ti = ti.view(N, K); tw = tw.view(N, K)
            for c0 in range(0, N, CH):
                xc = xf[c0:c0 + CH]
                n = xc.shape[0]
                h, ctx = A.moe_in(xc, o)
                tic = ti[c0:c0 + n]; twc = tw[c0:c0 + n]
                flat = tic.reshape(-1)
                order = torch.argsort(flat, stable=True)
                cnts = torch.bincount(flat, minlength=E).tolist()
                tok_s = order // K
                w_s = twc.reshape(-1)[order]
                out = torch.zeros(n, H, dtype=torch.float32, device=dev)
                off = 0
                for e in range(E):
                    c = cnts[e]
                    if c == 0:
                        continue
                    t_ = tok_s[off:off + c]
                    ww = w_s[off:off + c]
                    off += c
                    if A.weights_inside:
                        y = A.expert(h.index_select(0, t_), e, o, ww).float()
                    else:
                        y = A.expert(h.index_select(0, t_), e, o).float() * ww[:, None]
                    out.index_add_(0, t_, y)
                sh = A.shared(h, o)
                if sh is not None:
                    out += sh.float() if shx is None else sh.float() * (1 + shx[c0:c0 + n, None])
                A.moe_out(xc, out, ctx)
                del out, h, ctx
            del ti, tw, shx
        del o
        if dev.type == "cuda":
            torch.cuda.synchronize()
        log(f"layer {L+1}/{NL} {time.time()-t0:.1f}s wait {t0-tl0:.1f}s{' [hash]' if hashL else ''} | " + " | ".join(line))

    # ---------------------------------------------------------------- outputs on decode positions
    head = A.head()
    labels = torch.full((S, T), -100, dtype=torch.long, device=dev)
    labels[:, P:T - 1] = tok_t[:, P + 1:]
    labels = labels.view(N)
    ns = len(streams)
    nll = np.zeros((ns, S)); kl = np.zeros((ns, S)); agree = np.zeros((ns, S)); acc1 = np.zeros((ns, S))
    cnt = np.zeros(S)
    LC = 512
    dump = [] if os.environ.get("MR_DUMP_BASE_LOGP") else None
    for c0 in range(0, N, LC):
        lb = labels[c0:c0 + LC]
        valid = lb != -100
        if not bool(valid.any()):
            continue
        vs = (torch.arange(c0, c0 + lb.shape[0], device=dev) // T)[valid].cpu().numpy()
        lbv = lb[valid]
        np.add.at(cnt, vs, 1)
        for si_ in range(ns):
            lp = F.log_softmax(head(xs[si_].view(N, *hs)[c0:c0 + LC][valid]).float(), -1)
            am = lp.argmax(-1)
            if si_ == 0:
                base_lp, base_am = lp, am
                if dump is not None:
                    dump.append(lp.cpu().numpy())
            np.add.at(nll[si_], vs, (-lp.gather(1, lbv[:, None])[:, 0]).double().cpu().numpy())
            np.add.at(acc1[si_], vs, (am == lbv).double().cpu().numpy())
            np.add.at(agree[si_], vs, (am == base_am).double().cpu().numpy())
            np.add.at(kl[si_], vs, (base_lp.exp() * (base_lp - lp)).sum(-1).double().cpu().numpy())

    if dump is not None:
        np.save(os.environ["MR_DUMP_BASE_LOGP"], np.concatenate(dump))

    def dom_avg(v):
        out = {"ALL": float(v.sum() / cnt.sum())}
        for d, nm in enumerate(names):
            m = dom == d
            if m.any():
                out[nm] = float(v[m].sum() / cnt[m].sum())
        return out
    rnd = lambda dct, k: {a: round(b, k) for a, b in dct.items()}
    exp_mb = A.expert_bytes_4bit() / 1e6
    res = {"meta": {"adapter": A.name, "sessions": S, "T": T, "prompt": P, "layers": NL, "E": E, "K": K,
                    "tau": args.tau, "expert_mb_4bit": exp_mb, "domains": names, "seconds": time.time() - t_start},
           "streams": {}}
    base_nll = dom_avg(nll[0])
    for si_, (name, spec) in enumerate(streams):
        e = {"ppl": {k: round(float(np.exp(v)), 4) for k, v in dom_avg(nll[si_]).items()},
             "dppl_pct": {k: round(100 * (float(np.exp(v - base_nll[k])) - 1), 3) for k, v in dom_avg(nll[si_]).items()},
             "kl_nats": round(dom_avg(kl[si_])["ALL"], 5), "argmax_agree": round(dom_avg(agree[si_])["ALL"], 5),
             "top1_acc": round(dom_avg(acc1[si_])["ALL"], 5),
             "kl_dom": rnd(dom_avg(kl[si_]), 5), "agree_dom": rnd(dom_avg(agree[si_]), 5),
             "top1_dom": rnd(dom_avg(acc1[si_]), 5)}
        if spec is not None:
            sts = stats[name]
            m = {k: float(np.sum([s[k] for s in sts])) for k in ("stall_per_tok", "bg_per_tok")}
            mh = float(np.sum([s["stall_per_tok"] for s in sts if s["hash"]]))
            nr = [s for s in sts if not s["hash"]]
            e["routing"] = {"tau": spec[3], "fallback": spec[4],
                            "subst_rate_all": round(float(np.mean([s["subst"] for s in sts])), 4),
                            "wanted_hit": round(float(np.mean([s["wanted_hit"] for s in sts])), 4),
                            "subst_rate": round(float(np.mean([s["subst"] for s in nr])), 4),
                            "subst_gate_mass": round(float(np.mean([s["subst_mass"] for s in nr])), 4),
                            "stall_experts_per_tok": round(m["stall_per_tok"], 3),
                            "hash_stall_experts_per_tok": round(mh, 3),
                            "bg_experts_per_tok": round(m["bg_per_tok"], 3),
                            "stall_mb_per_tok": round(m["stall_per_tok"] * exp_mb, 2),
                            "bg_mb_per_tok": round(m["bg_per_tok"] * exp_mb, 2),
                            "subst_rate_per_layer": [round(s["subst"], 4) for s in sts],
                            "hit_per_layer": [round(s["wanted_hit"], 4) for s in sts]}
        res["streams"][name] = e
    compact = {}
    for name, e in res["streams"].items():
        c = dict(e)
        if "routing" in c:
            c["routing"] = {k: v for k, v in c["routing"].items() if not k.endswith("_per_layer")}
        compact[name] = c
        log(f"{name}: " + json.dumps(c, separators=(",", ":")))
    json.dump(res, open(os.path.join(args.out, "mr.json"), "w"), indent=1)
    print("SECTION_MR " + json.dumps({"meta": res["meta"], "streams": compact}, separators=(",", ":")), flush=True)
    log(f"done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
