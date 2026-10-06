"""Layer-wise exact forward of a Qwen3-MoE checkpoint that records, for every token:
  * the real top-k routing (expert ids + normalized weights) at every layer
  * how well "look-ahead gating" (apply a FUTURE layer's norm+router to the CURRENT
    residual stream) predicts the real routing, at several distances
  * per-domain next-token loss (sanity check that the forward is correct)

Layer-wise = each layer's weights are read from disk exactly once and applied to ALL
sessions before moving on, so GPU memory needed is ~one layer + activations.
The math mirrors transformers' Qwen3MoeForCausalLM (verified by test_local.py).

Distances D are measured in half-layers along the residual stream:
  position p=2L   : residual entering layer L (before attention)
  position p=2L+1 : residual after layer L's attention (= router input of layer L)
  target layer T's router sits at position 2T+1, so D = 2T+1-p.
  D=0 is the real router (sanity: recall must be 1.0), D=1 = before own attention,
  D=2 = one full layer ahead, D=4 = two layers ahead, ...
"""
import os, sys, json, time, argparse, threading, queue
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class Ckpt:
    def __init__(self, d):
        self.d = d
        p = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(p):
            self.wmap = json.load(open(p))["weight_map"]
        else:
            fn = "model.safetensors"
            f = safe_open(os.path.join(d, fn), framework="pt", device="cpu")
            self.wmap = {k: fn for k in f.keys()}
        self.h = {}

    def get(self, name):
        fn = self.wmap[name]
        if fn not in self.h:
            self.h[fn] = safe_open(os.path.join(self.d, fn), framework="pt", device="cpu")
        return self.h[fn].get_tensor(name)


ATTN = ["input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate.weight"]


def load_layer(ck, L, E, dtype, pin):
    p = f"model.layers.{L}."
    d = {n: ck.get(p + n).to(dtype) for n in ATTN}
    g = torch.stack([ck.get(f"{p}mlp.experts.{e}.gate_proj.weight") for e in range(E)])
    u = torch.stack([ck.get(f"{p}mlp.experts.{e}.up_proj.weight") for e in range(E)])
    d["gu"] = torch.cat([g, u], dim=1).to(dtype).contiguous()          # [E, 2I, H]
    del g, u
    d["down"] = torch.stack([ck.get(f"{p}mlp.experts.{e}.down_proj.weight")
                             for e in range(E)]).to(dtype).contiguous()  # [E, H, I]
    if pin:
        d = {k: v.pin_memory() for k, v in d.items()}
    return d


def rmsnorm(x, w, eps):
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--attn_batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--dists", default="0,1,2,3,4,6,8")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    t_start = time.time()

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    H = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
    HD = cfg.get("head_dim") or H // NH
    E = cfg["num_experts"]; K = cfg["num_experts_per_tok"]; NL = cfg["num_hidden_layers"]
    I = cfg["moe_intermediate_size"]; eps = cfg["rms_norm_eps"]; theta = float(cfg["rope_theta"])
    norm_topk = cfg.get("norm_topk_prob", True)
    assert not cfg.get("mlp_only_layers"), "dense layers not supported"
    assert cfg.get("decoder_sparse_step", 1) == 1
    assert cfg.get("rope_scaling") in (None, {}), "rope scaling not implemented"
    log(f"cfg: layers={NL} experts={E} top{K} hidden={H} heads={NH}/{NKV}x{HD} moe_I={I}")

    dat = np.load(args.data, allow_pickle=True)
    tokens = dat["tokens"].astype(np.int64)
    dom = dat["domain"].astype(np.int64)
    names = [str(s) for s in dat["domain_names"]]
    S, T = tokens.shape
    N = S * T
    ndom = len(names)
    log(f"data: {S} sessions x {T} tokens = {N} tokens, domains={names}")

    ck = Ckpt(args.model_dir)
    post_w = torch.stack([ck.get(f"model.layers.{L}.post_attention_layernorm.weight")
                          for L in range(NL)]).to(dev, dt)
    gate_w = torch.stack([ck.get(f"model.layers.{L}.mlp.gate.weight")
                          for L in range(NL)]).to(dev, dt)
    emb = ck.get("model.embed_tokens.weight").to(dev, dt)
    tok_t = torch.from_numpy(tokens).to(dev)
    x = emb[tok_t]                              # [S, T, H]
    del emb
    xf = x.view(N, H)
    dom_tok = torch.from_numpy(np.repeat(dom, T)).to(dev)
    ntok_dom = np.bincount(dom, minlength=ndom) * T

    # RoPE (same as HF default rope, attention_scaling=1)
    inv_freq = 1.0 / (theta ** (torch.arange(0, HD, 2, dtype=torch.int64).float().to(dev) / HD))
    pos = torch.arange(T, device=dev).float()
    fr = torch.outer(pos, inv_freq)
    embr = torch.cat([fr, fr], dim=-1)
    cos, sin = embr.cos().to(dt), embr.sin().to(dt)   # [T, HD]

    dists = [int(v) for v in args.dists.split(",")]
    M_LIST = [m for m in (8, 16, 32) if m <= E and m >= 1]
    if not M_LIST or M_LIST[0] != K:
        M_LIST = sorted(set([K] + [m for m in (2 * K, 4 * K) if m <= E]))
    TOPM = max(M_LIST)
    rec = {D: np.zeros((NL, len(M_LIST), ndom)) for D in dists}
    pending = {}
    CH = args.chunk

    trace_idx = np.zeros((NL, N, K), np.uint8)
    trace_w = np.zeros((NL, N, K), np.float16)

    def route(xc, L):
        h = rmsnorm(xc, post_w[L], eps)
        lg = F.linear(h, gate_w[L])
        pr = F.softmax(lg, dim=-1, dtype=torch.float)
        return h, pr

    def stash(p):
        for D in dists:
            if (p + D - 1) % 2:
                continue
            Tl = (p + D - 1) // 2
            if Tl < 0 or Tl >= NL:
                continue
            out = torch.empty(N, TOPM, dtype=torch.uint8, device=dev)
            for c0 in range(0, N, CH):
                _, pr = route(xf[c0:c0 + CH], Tl)
                out[c0:c0 + CH] = pr.topk(TOPM, dim=-1).indices.to(torch.uint8)
            pending[(Tl, D)] = out

    def eval_pending(L, ti_all):
        for key in [k for k in pending if k[0] == L]:
            P = pending.pop(key)
            D = key[1]
            for c0 in range(0, N, CH):
                A = torch.zeros(min(CH, N - c0), E, dtype=torch.bool, device=dev)
                A.scatter_(1, ti_all[c0:c0 + CH], True)
                hitc = A.gather(1, P[c0:c0 + CH].long()).to(torch.int32).cumsum(1)
                for mi, m in enumerate(M_LIST):
                    s = torch.zeros(ndom, device=dev, dtype=torch.float64)
                    s.index_add_(0, dom_tok[c0:c0 + CH], hitc[:, m - 1].double())
                    rec[D][L, mi] += s.cpu().numpy()

    # background weight loader (one layer ahead)
    qq = queue.Queue(maxsize=2)
    pin = dev.type == "cuda"

    def loader():
        try:
            ck2 = Ckpt(args.model_dir)
            for L in range(NL):
                qq.put(load_layer(ck2, L, E, dt, pin))
        except Exception as e:  # propagate
            qq.put(e)
    threading.Thread(target=loader, daemon=True).start()

    rep = NH // NKV
    t_load = 0.0
    for L in range(NL):
        tl0 = time.time()
        W = qq.get()
        if isinstance(W, Exception):
            raise W
        W = {k: v.to(dev, non_blocking=True) for k, v in W.items()}
        t_load += time.time() - tl0
        t0 = time.time()

        stash(2 * L)                            # predictions from pre-attention residual
        # ---- attention ----
        for b0 in range(0, S, args.attn_batch):
            xb = x[b0:b0 + args.attn_batch]
            bs = xb.shape[0]
            h = rmsnorm(xb, W["input_layernorm.weight"], eps)
            q = F.linear(h, W["self_attn.q_proj.weight"]).view(bs, T, NH, HD)
            k = F.linear(h, W["self_attn.k_proj.weight"]).view(bs, T, NKV, HD)
            v = F.linear(h, W["self_attn.v_proj.weight"]).view(bs, T, NKV, HD)
            q = rmsnorm(q, W["self_attn.q_norm.weight"], eps).transpose(1, 2)
            k = rmsnorm(k, W["self_attn.k_norm.weight"], eps).transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(bs, T, NH * HD)
            xb.add_(F.linear(a, W["self_attn.o_proj.weight"]))
            del h, q, k, v, a
        stash(2 * L + 1)                        # predictions from router input (D=0 is exact)

        # ---- MoE ----
        ti_all = torch.empty(N, K, dtype=torch.long, device=dev)
        GU, DN = W["gu"], W["down"]
        for c0 in range(0, N, CH):
            xc = xf[c0:c0 + CH]
            n = xc.shape[0]
            h, pr = route(xc, L)
            tw, ti = pr.topk(K, dim=-1)
            if norm_topk:
                tw = tw / tw.sum(dim=-1, keepdim=True)
            ti_all[c0:c0 + n] = ti
            trace_idx[L, c0:c0 + n] = ti.to(torch.uint8).cpu().numpy()
            trace_w[L, c0:c0 + n] = tw.to(torch.float16).cpu().numpy()
            tw = tw.to(dt)
            flat = ti.reshape(-1)
            order = torch.argsort(flat, stable=True)
            cnts = torch.bincount(flat, minlength=E).tolist()
            tok_s = order // K
            w_s = tw.reshape(-1)[order]
            out = torch.zeros(n, H, dtype=torch.float32, device=dev)
            off = 0
            for e in range(E):
                c = cnts[e]
                if c == 0:
                    continue
                t_ = tok_s[off:off + c]
                ww = w_s[off:off + c]
                off += c
                hh = h.index_select(0, t_)
                gu = F.linear(hh, GU[e])
                act = F.silu(gu[:, :I]) * gu[:, I:]
                y = F.linear(act, DN[e])
                out.index_add_(0, t_, (y * ww[:, None]).float())
            xc.add_(out.to(dt))
            del out, h, pr
        eval_pending(L, ti_all)
        del W, GU, DN
        if dev.type == "cuda":
            torch.cuda.synchronize()
        log(f"layer {L+1}/{NL}: compute {time.time()-t0:.1f}s, waited-for-weights {t0-tl0:.1f}s")

    # ---- final norm + lm_head: per-domain loss (sanity) ----
    norm_w = ck.get("model.norm.weight").to(dev, dt)
    lm = ck.get("lm_head.weight").to(dev, dt)
    labels = torch.full((S, T), -100, dtype=torch.long, device=dev)
    labels[:, :-1] = tok_t[:, 1:]
    labels = labels.view(N)
    nll_dom = np.zeros(ndom); cnt_dom = np.zeros(ndom)
    nll_sess = np.zeros(S)
    LC = 2048 if dev.type == "cuda" else 256
    for c0 in range(0, N, LC):
        hh = rmsnorm(xf[c0:c0 + LC], norm_w, eps)
        lg = F.linear(hh, lm).float()
        lb = labels[c0:c0 + LC]
        l = F.cross_entropy(lg, lb, reduction="none", ignore_index=-100)
        valid = (lb != -100).double()
        ln = (l.double() * valid).cpu().numpy()
        vv = valid.cpu().numpy()
        sid = (np.arange(c0, c0 + len(ln)) // T)
        np.add.at(nll_sess, sid, ln)
        np.add.at(nll_dom, dom[sid], ln)
        np.add.at(cnt_dom, dom[sid], vv)
    nll_sess /= (T - 1)

    # ---- save ----
    idx4 = trace_idx.reshape(NL, S, T, K).transpose(1, 2, 0, 3)
    w4 = trace_w.reshape(NL, S, T, K).transpose(1, 2, 0, 3)
    np.save(os.path.join(args.out, "trace_idx.npy"), np.ascontiguousarray(idx4))
    np.save(os.path.join(args.out, "trace_w.npy"), np.ascontiguousarray(w4))
    np.save(os.path.join(args.out, "nll_sess.npy"), nll_sess)

    look = {}
    for D in dists:
        r = rec[D] / (np.maximum(ntok_dom, 1)[None, None, :] * K)      # [NL, M, dom]
        valid_layers = [L for L in range(NL) if (2 * L + 1 - D) >= 0]
        tot = rec[D][valid_layers].sum(axis=(0, 2)) / (len(valid_layers) * N * K)
        look[str(D)] = {
            "m_list": M_LIST,
            "recall_mean": tot.tolist(),                                    # avg over layers & tokens
            "recall_per_layer": (rec[D].sum(axis=2) / (N * K)).tolist(),  # [NL][M]
            "recall_per_domain": {names[d]: (rec[D][valid_layers, :, d].sum(0) /
                                    (len(valid_layers) * max(ntok_dom[d], 1) * K)).tolist()
                                  for d in range(ndom)},
            "n_layers": len(valid_layers),
        }
    stats = {
        "model_cfg": {"layers": NL, "experts": E, "top_k": K, "hidden": H},
        "S": S, "T": T, "domains": names,
        "ppl_per_domain": {names[d]: float(np.exp(nll_dom[d] / max(cnt_dom[d], 1))) for d in range(ndom)},
        "lookahead": look,
        "seconds": time.time() - t_start,
        "seconds_waiting_for_weights": t_load,
    }
    json.dump(stats, open(os.path.join(args.out, "trace_stats.json"), "w"), indent=1)
    log("ppl:", {k: round(v, 3) for k, v in stats["ppl_per_domain"].items()})
    for D in dists:
        log(f"lookahead D={D}: recall@{M_LIST} = {[round(v, 4) for v in look[str(D)]['recall_mean']]}")
    log(f"trace done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
