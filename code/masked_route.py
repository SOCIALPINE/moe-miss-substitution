"""Cache-constrained routing on Qwen3-Next: the predictor keeps a resident expert set in RAM, the model's own
router picks the exact experts every token, but only among what is resident.

Streams (all run layer-wise with weights loaded once per layer):
  base              : original routing everywhere
  cap:policy:mode   : prompt (prefill, first P tokens) uses original routing; decode tokens use the cache:
      resident set M (cap experts per layer) initialised from the prompt's routing, then per token t:
        wanted = true top-K of the router logits
        B : selected = top-K of the logits restricted to M (missing experts are substituted, never waited for)
        C : like B, but a missing expert is loaded on demand (stall) if it is the router's rank-1 choice or
            its exact normalised gate weight >= tau
        every missing wanted expert is loaded (B: in the background, C: on demand or background) so M keeps
        following the router; eviction by policy (lru = recency, fe = EMA of wanted usage), wanted never evicted.
Measured on decode positions: perplexity, KL(base || stream) of the next-token distribution, argmax agreement,
wanted-hit rate, substitution rate / gate mass, stall and background loads -> speed estimate.
"""
import os, sys, json, time, argparse, threading, queue
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_next import modeling_qwen3_next as QN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_next import Ckpt, load_layer, build_mixer, make_norm   # noqa: E402


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def topk_mask(x, c):
    return torch.zeros_like(x, dtype=torch.bool).scatter_(-1, x.topk(c, dim=-1).indices, True)


def simulate(lg, mode, pol, cap, P, K, tau, alpha, wsc=None, wscale=1.0, forced=None, fallback="subst"):
    """lg [S,T,E] fp32 router selection scores of one stream at one layer -> selected idx/weights [S,T,K] + stats.
    Default (wsc None): gate weights = softmax over the selected logits (Qwen3-Next, gpt-oss).
    wsc [S,T,E]: gate weights = wsc[selected] / sum * wscale (DeepSeek-V4 sqrtsoftplus scores; selection uses lg).
    forced [S,T,K]: the wanted set is fixed (hash-routed layers); lg only ranks substitutes.
    fallback for a missing expert that is not waited for:
      subst  : the router's next best resident expert takes the slot (top-K restricted to resident)
      drop   : the slot is left empty (weight 0), remaining weights unchanged
      renorm : the slot is left empty, remaining routed weights rescaled to the original total
      shared : the slot is left empty and its weight goes to the shared expert (st['shared_extra'] [S,T])"""
    S, T, E = lg.shape
    dev = lg.device
    if forced is not None:
        lg = lg + 1e4 * torch.zeros_like(lg).scatter_(-1, forced, 1.0)

    def gw(idx, vals, t0, t1):
        if wsc is None:
            return torch.softmax(vals, -1)
        g = wsc[:, t0:t1].gather(-1, idx)
        return g / g.sum(-1, keepdim=True) * wscale

    sel_idx = torch.empty(S, T, K, dtype=torch.long, device=dev)
    sel_w = torch.empty(S, T, K, dtype=torch.float32, device=dev)
    top = lg[:, :P].topk(K, dim=-1)
    sel_idx[:, :P] = top.indices
    sel_w[:, :P] = gw(top.indices, top.values, 0, P)
    st = {k: 0.0 for k in ("wanted_hit", "subst", "subst_mass", "stall", "bg")}
    if mode == "exact":
        top = lg[:, P:].topk(K, dim=-1)
        sel_idx[:, P:] = top.indices
        sel_w[:, P:] = gw(top.indices, top.values, P, T)
        return sel_idx, sel_w, st
    Up = torch.zeros(S, P, E, dtype=torch.bool, device=dev).scatter_(2, top.indices, True)
    tt = torch.arange(P, device=dev, dtype=torch.float32).view(1, P, 1)
    rec = torch.where(Up, tt, torch.full_like(tt, -1.0)).amax(1)                 # last use in the prompt
    f = torch.zeros(S, E, device=dev)
    if pol == "fe":
        wts = (1 - alpha) ** torch.arange(P - 1, -1, -1, device=dev, dtype=torch.float32)
        f = alpha * (Up.float() * wts.view(1, P, 1)).sum(1)                       # EMA over the prompt
    pri = rec if pol == "lru" else f + 1e-6 * rec
    M = topk_mask(pri, cap)
    ninf = torch.tensor(float("-inf"), device=dev)
    pinf = torch.tensor(float("inf"), device=dev)
    acc = torch.zeros(5, dtype=torch.float64, device=dev)
    shx = torch.zeros(S, T, device=dev) if fallback == "shared" else None
    for t in range(P, T):
        z = lg[:, t]
        wv, wi = z.topk(K, dim=-1)
        W = torch.zeros(S, E, dtype=torch.bool, device=dev).scatter_(1, wi, True)
        if wsc is None:
            wn = torch.softmax(wv, -1)
        else:
            g = wsc[:, t].gather(1, wi)
            wn = g / g.sum(-1, keepdim=True)
        wmass = torch.zeros(S, E, device=dev).scatter_(1, wi, wn)
        miss = W & ~M
        if mode == "B":
            must = torch.zeros_like(miss)
        else:
            r1 = torch.zeros_like(W).scatter_(1, wi[:, :1], True)
            must = miss & (r1 | (wmass >= tau))
        allowed = M | must
        if fallback == "subst":
            sv, si = torch.where(allowed, z, ninf).topk(K, dim=-1)
            sel_idx[:, t] = si
            sel_w[:, t] = gw(si[:, None], sv[:, None], t, t + 1)[:, 0]
            Sm = torch.zeros_like(W).scatter_(1, si, True)
        else:
            Sm = W & allowed
            wfull = gw(wi[:, None], wv[:, None], t, t + 1)[:, 0]
            w = torch.where(Sm.gather(1, wi), wfull, torch.zeros_like(wfull))
            if fallback == "renorm":
                w = w / w.sum(-1, keepdim=True).clamp_min(1e-12) * wfull.sum(-1, keepdim=True)
            elif fallback == "shared":
                shx[:, t] = (wfull - w).sum(-1)
            sel_idx[:, t] = wi
            sel_w[:, t] = w
        notsel = W & ~Sm
        acc += torch.stack([(W & M).sum().double(), notsel.sum().double(), (wmass * notsel).sum().double(),
                            must.sum().double(), (miss & ~must).sum().double()])
        rec = torch.where(W | Sm, torch.full_like(rec, float(t)), rec)
        if pol == "fe":
            f = f * (1 - alpha) + alpha * W.float()
            pri = f + 1e-6 * rec
        else:
            pri = rec
        Mn = M | miss
        M = topk_mask(torch.where(Mn, torch.where(W, pinf, pri), ninf), cap)
    a = acc.tolist()
    n = S * (T - P)
    st = {"wanted_hit": a[0] / (n * K), "subst": a[1] / (n * K), "subst_mass": a[2] / n,
          "stall_per_tok": a[3] / n, "bg_per_tok": a[4] / n}
    if shx is not None:
        st["shared_extra"] = shx
    return sel_idx, sel_w, st


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--attn_batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--per_domain", type=int, default=8)
    ap.add_argument("--configs", default="0.25:lru:B,0.25:lru:C,0.25:fe:B,0.25:fe:C,0.125:lru:B,0.125:lru:C")
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=1 / 64)
    ap.add_argument("--t_c", type=float, default=21.0)
    ap.add_argument("--expert_mb", type=float, default=1.8)
    ap.add_argument("--dl_flag_dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    t_start = time.time()

    cfg = AutoConfig.from_pretrained(args.model_dir)
    cfg._attn_implementation = "sdpa"
    cfg.dtype = dt
    H = cfg.hidden_size; E = cfg.num_experts; K = cfg.num_experts_per_tok; NL = cfg.num_hidden_layers
    I = cfg.moe_intermediate_size; Is = cfg.shared_expert_intermediate_size; eps = cfg.rms_norm_eps

    dat = np.load(args.data, allow_pickle=True)
    tokens_all = dat["tokens"].astype(np.int64)
    dom_all = dat["domain"].astype(np.int64)
    names = [str(s) for s in dat["domain_names"]]
    P = int(dat["prompt_len"])
    sel = []
    for d in range(len(names)):                       # held-out sessions (odd ones, as in analyze.py)
        ss = np.where(dom_all == d)[0][1::2]
        sel += list(ss[:args.per_domain])
    sel = np.array(sel)
    tokens = tokens_all[sel]; dom = dom_all[sel]
    S, T = tokens.shape
    N = S * T
    streams = [("base", None)]
    for c in args.configs.split(","):
        fr, pol, mode = c.split(":")
        streams.append((c, (max(int(round(float(fr) * E)), K), pol, mode)))
    log(f"cfg: layers={NL} experts={E} top{K} | sessions={S} x {T} (prompt {P}) | streams={[s[0] for s in streams]}")

    ck = Ckpt(args.model_dir, args.dl_flag_dir)
    emb = ck.get("model.embed_tokens.weight").to(dev, dt)
    tok_t = torch.from_numpy(tokens).to(dev)
    xs = [emb[tok_t].clone() for _ in streams]
    del emb
    rot = QN.Qwen3NextRotaryEmbedding(config=cfg, device=dev)
    cos, sin = rot(xs[0][:1], torch.arange(T, device=dev)[None])
    stats = {s[0]: [] for s in streams if s[1] is not None}

    qq = queue.Queue(maxsize=2)
    pin = dev.type == "cuda"

    def loader():
        try:
            ck2 = Ckpt(args.model_dir, args.dl_flag_dir)
            for L in range(NL):
                qq.put(load_layer(ck2, L, cfg, dt, pin))
        except Exception as e:
            qq.put(e)
    threading.Thread(target=loader, daemon=True).start()

    CH = args.chunk
    for L in range(NL):
        tl0 = time.time()
        W = qq.get()
        if isinstance(W, Exception):
            raise W
        mixer = build_mixer(cfg, L, W["mixer"], dev)
        in_norm = make_norm(W["in_norm"], eps, dev)
        post_norm = make_norm(ck.get(f"model.layers.{L}.post_attention_layernorm.weight").to(dt), eps, dev)
        gate_w = ck.get(f"model.layers.{L}.mlp.gate.weight").to(dev, dt)
        GU, DN = W["gu"].to(dev, non_blocking=True), W["down"].to(dev, non_blocking=True)
        SGU, SDN, SG = (W[k].to(dev) for k in ("sh_gu", "sh_down", "sh_gate"))
        t0 = time.time()
        line = []
        for si_, (name, spec) in enumerate(streams):
            x = xs[si_]
            for b0 in range(0, S, args.attn_batch):
                xb = x[b0:b0 + args.attn_batch]
                h = in_norm(xb)
                if cfg.layer_types[L] == "linear_attention":
                    o = mixer(hidden_states=h, cache_params=None, cache_position=None, attention_mask=None)
                else:
                    o, _ = mixer(hidden_states=h, position_embeddings=(cos, sin), attention_mask=None)
                xb.add_(o)
                del h, o
            xf = x.view(N, H)
            lg = torch.empty(N, E, dtype=torch.float32, device=dev)
            for c0 in range(0, N, CH):
                lg[c0:c0 + CH] = F.linear(post_norm(xf[c0:c0 + CH]), gate_w).float()
            if spec is None:
                ti, tw, st = simulate(lg.view(S, T, E), "exact", None, E, P, K, 0, 0)
            else:
                cap, pol, mode = spec
                ti, tw, st = simulate(lg.view(S, T, E), mode, pol, cap, P, K, args.tau, args.alpha)
                stats[name].append(st)
                line.append(f"{name} hit {st['wanted_hit']:.3f} sub {st['subst']:.3f}/{st['subst_mass']:.3f}")
            del lg
            ti = ti.view(N, K); tw = tw.view(N, K)
            for c0 in range(0, N, CH):
                xc = xf[c0:c0 + CH]
                n = xc.shape[0]
                h = post_norm(xc)
                tic = ti[c0:c0 + n]; twc = tw[c0:c0 + n].to(dt)
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
                    gu = F.linear(h.index_select(0, t_), GU[e])
                    y = F.linear(F.silu(gu[:, :I]) * gu[:, I:], DN[e])
                    out.index_add_(0, t_, (y * ww[:, None]).float())
                sgu = F.linear(h, SGU)
                sh = torch.sigmoid(F.linear(h, SG)) * F.linear(F.silu(sgu[:, :Is]) * sgu[:, Is:], SDN)
                xc.add_((out + sh.float()).to(dt))
                del out, h, sgu, sh
            del ti, tw
        del W, GU, DN, SGU, SDN, SG, mixer, gate_w
        if dev.type == "cuda":
            torch.cuda.synchronize()
        log(f"layer {L+1}/{NL} ({cfg.layer_types[L][:3]}) {time.time()-t0:.1f}s wait {t0-tl0:.1f}s | " + " | ".join(line))

    # ---------------------------------------------------------------- outputs on decode positions
    fnorm = make_norm(ck.get("model.norm.weight").to(dt), eps, dev)
    lm = ck.get("lm_head.weight").to(dev, dt)
    labels = torch.full((S, T), -100, dtype=torch.long, device=dev)
    labels[:, P:T - 1] = tok_t[:, P + 1:]
    labels = labels.view(N)
    ns = len(streams)
    nll = np.zeros((ns, S)); kl = np.zeros((ns, S)); agree = np.zeros((ns, S)); acc1 = np.zeros((ns, S))
    cnt = np.zeros(S)
    LC = 1024
    for c0 in range(0, N, LC):
        lb = labels[c0:c0 + LC]
        valid = lb != -100
        if not bool(valid.any()):
            continue
        sid = torch.arange(c0, c0 + lb.shape[0], device=dev) // T
        vs = sid[valid]
        lbv = lb[valid]
        np.add.at(cnt, vs.cpu().numpy(), 1)
        base_lp = None
        for si_ in range(ns):
            lp = F.log_softmax(F.linear(fnorm(xs[si_].view(N, H)[c0:c0 + LC][valid]), lm).float(), -1)
            am = lp.argmax(-1)
            if si_ == 0:
                base_lp, base_am = lp, am
            vv = vs.cpu().numpy()
            np.add.at(nll[si_], vv, (-lp.gather(1, lbv[:, None])[:, 0]).double().cpu().numpy())
            np.add.at(acc1[si_], vv, (am == lbv).double().cpu().numpy())
            np.add.at(agree[si_], vv, (am == base_am).double().cpu().numpy())
            np.add.at(kl[si_], vv, (base_lp.exp() * (base_lp - lp)).sum(-1).double().cpu().numpy())
            if si_ > 0:
                del lp
    res = {"meta": {"sessions": S, "T": T, "prompt": P, "layers": NL, "E": E, "K": K, "tau": args.tau,
                    "alpha": args.alpha, "domains": names, "t_c_ms": args.t_c, "expert_mb": args.expert_mb,
                    "seconds": time.time() - t_start}, "streams": {}}

    def dom_avg(v):
        out = {"ALL": float(v.sum() / cnt.sum())}
        for d, nm in enumerate(names):
            m = dom == d
            if m.any():
                out[nm] = float(v[m].sum() / cnt[m].sum())
        return out
    base_nll = dom_avg(nll[0])
    for si_, (name, spec) in enumerate(streams):
        e = {"ppl": {k: round(float(np.exp(v)), 4) for k, v in dom_avg(nll[si_]).items()},
             "dppl_pct": {k: round(100 * (float(np.exp(v - base_nll[k])) - 1), 3) for k, v in dom_avg(nll[si_]).items()},
             "kl_nats": round(dom_avg(kl[si_])["ALL"], 5),
             "argmax_agree": round(dom_avg(agree[si_])["ALL"], 5),
             "top1_acc": round(dom_avg(acc1[si_])["ALL"], 5)}
        if spec is not None:
            sts = stats[name]
            m = {k: float(np.mean([s[k] for s in sts])) for k in sts[0]}
            stall_mb = m["stall_per_tok"] * NL * args.expert_mb
            bg_mb = m["bg_per_tok"] * NL * args.expert_mb
            e["routing"] = {"wanted_hit": round(m["wanted_hit"], 4), "subst_rate": round(m["subst"], 4),
                            "subst_gate_mass": round(m["subst_mass"], 4),
                            "stall_mb_per_tok": round(stall_mb, 2), "bg_mb_per_tok": round(bg_mb, 2),
                            "subst_rate_per_layer": [round(s["subst"], 4) for s in sts]}
            for b in (7.0, 12.0):
                e["routing"][f"speed_{b:g}GBs"] = round(args.t_c / max(args.t_c + stall_mb / b, (stall_mb + bg_mb) / b), 4)
        res["streams"][name] = e
    compact = {}
    for name, e in res["streams"].items():
        c = dict(e)
        if "routing" in c:
            c["routing"] = {k: v for k, v in c["routing"].items() if k != "subst_rate_per_layer"}
        compact[name] = c
        log(f"{name}: " + json.dumps(c, separators=(",", ":")))
    json.dump(res, open(os.path.join(args.out, "masked_route.json"), "w"), indent=1)
    print("SECTION_MR " + json.dumps(compact, separators=(",", ":")), flush=True)
    log(f"masked routing done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
