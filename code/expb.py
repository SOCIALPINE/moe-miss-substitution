"""Experiment B (PREREG_expB.md): at equal substituted gate mass, does the composition of the substitution matter?
BF16 Qwen3-Next-80B-A3B emulation without a cache, computed layer by layer with each layer's weights loaded once.

Streams: the original routing ("base") and arm streams. At a decode layer-token of an arm stream, with that stream's own
router logits: W = the top-K experts, g = softmax over their logits (sums to 1); the candidates are the K-1 experts of W
other than the rank-1 expert. For a nonempty subset A of the candidates let s(A) = sum of g and c(A) = sum of g^2 / s(A)^2
(concentration: 1 for one expert, 1/n for n equal experts). A is feasible if |s(A) - m*| <= delta, with delta = delta_rel
* m*. Among pairs (A, B) of feasible subsets with |s(A) - s(B)| <= eps (eps = eps_rel * m*), the pair with the largest
c(A) - c(B) is chosen (ties: the first pair with rows in bitmask order). The heavy arm (H) substitutes A, the light arm (L)
substitutes B. If no pair has c(A) > c(B), the arm substitutes nothing at this layer-token. Both arms apply the same rule
to their own logits, so on the same logits their substituted masses differ by at most eps.
  fill   : the substituted experts are excluded and the selection becomes the top-K of the remaining logits, so the freed
           slots go to the highest-logit experts outside W; gates = softmax over the new selection (as in Section 3.1).
  renorm : the substituted slots are dropped and the remaining g are rescaled to sum to 1.
Prompt positions (< P) use the original routing in every stream; attention, norms and the shared expert are exact.

Outputs on decode positions P..T-2 (where the next data token exists): per stream and session the summed NLL, the summed
KL(base || stream) in nats, argmax agreement with base, top-1 accuracy and the token count; per arm and layer the
statistics of the intervention. The result document is written to <out>/expb.json and printed into the log as xz + base64
lines of at most 4000 characters, each with the md5 of its payload (tag DXB)."""
import os, sys, json, time, argparse, threading, queue, hashlib, lzma, base64
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_next import modeling_qwen3_next as QN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_next import Ckpt, load_layer, build_mixer, make_norm   # noqa: E402

WIDTH = 4000


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def put(tag, doc):
    raw = json.dumps(doc, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode()
    comp = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    enc = base64.b64encode(comp).decode()
    parts = [enc[i:i + WIDTH] for i in range(0, len(enc), WIDTH)]
    for k, p in enumerate(parts):
        print(f"{tag} {k + 1}/{len(parts)} {hashlib.md5(p.encode()).hexdigest()[:8]} {p}", flush=True)
    print(f"{tag} END n={len(parts)} bytes={len(comp)} md5={hashlib.md5(comp).hexdigest()} raw={hashlib.md5(raw).hexdigest()}"
          f" rawbytes={len(raw)}", flush=True)


def subset_matrix(k, device):
    """all nonempty subsets of k candidates as a [2^k - 1, k] 0/1 float matrix, row r = bitmask r + 1"""
    m = torch.arange(1, 2 ** k).unsqueeze(1)
    return ((m >> torch.arange(k)) & 1).to(device=device, dtype=torch.float32)


def topk_gates(lg, K):
    """lg [..., E] fp32 -> top-K indices (descending logit) and gates g = softmax over the top-K logits"""
    v, i = lg.topk(K, dim=-1)
    return i, torch.softmax(v, -1)


def pick_pair(gc, Bm, mstar, delta, eps, budget=6e7):
    """gc [M, K-1] candidate gates (ranks 2..K) of M layer-tokens -> feasible-any [M], defined [M], heavy row [M],
    light row [M], gap [M] (rows index Bm; gap = |s(A) - s(B)| of the chosen pair, 0 where undefined)."""
    M, NB = gc.shape[0], Bm.shape[0]
    dev = gc.device
    anyf = torch.zeros(M, dtype=torch.bool, device=dev)
    ok = torch.zeros(M, dtype=torch.bool, device=dev)
    ha = torch.zeros(M, dtype=torch.long, device=dev)
    la = torch.zeros(M, dtype=torch.long, device=dev)
    gap = torch.zeros(M, dtype=torch.float32, device=dev)
    rows = torch.arange(NB, device=dev)
    c0 = 0
    while c0 < M:
        n = min(4096, M - c0)
        g = gc[c0:c0 + n]
        s = g @ Bm.T
        c = ((g * g) @ Bm.T) / (s * s)
        feas = (s - mstar).abs() <= delta
        nf = feas.sum(-1)
        anyf[c0:c0 + n] = nf > 0
        fmax = int(nf.max()) if n else 0
        if fmax >= 2:
            order = torch.where(feas, rows, rows + NB).argsort(-1)[:, :fmax]     # feasible rows first, in row order
            valid = torch.arange(fmax, device=dev)[None] < nf[:, None]
            sf, cf = s.gather(1, order), c.gather(1, order)
            step = max(1, int(budget // (fmax * fmax)))
            for d0 in range(0, n, step):
                sl = slice(d0, d0 + step)
                pv = valid[sl]
                near = ((sf[sl, :, None] - sf[sl, None, :]).abs() <= eps) & pv[:, :, None] & pv[:, None, :]
                diff = torch.where(near, cf[sl, :, None] - cf[sl, None, :], torch.full_like(near, -1.0, dtype=cf.dtype))
                flat = diff.view(diff.shape[0], -1)
                j = flat.argmax(-1)                                                  # first maximum
                best = flat.gather(1, j[:, None])[:, 0]
                a_, b_ = j // fmax, j % fmax
                good = best > 0
                r = slice(c0 + d0, c0 + d0 + diff.shape[0])
                ok[r] = good
                ha[r] = torch.where(good, order[sl].gather(1, a_[:, None])[:, 0], torch.zeros_like(a_))
                la[r] = torch.where(good, order[sl].gather(1, b_[:, None])[:, 0], torch.zeros_like(b_))
                gap[r] = torch.where(good, (sf[sl].gather(1, a_[:, None]) - sf[sl].gather(1, b_[:, None])).abs()[:, 0],
                                     torch.zeros_like(best))
        c0 += n
    return anyf, ok, ha, la, gap


def arm_route(lg, P, K, Bm, mstar, delta, eps, arm, mode):
    """lg [S,T,E] fp32 router logits of one arm stream at one layer -> selection [S,T,K], gates [S,T,K], statistics"""
    S, T, _ = lg.shape
    idx, g = topk_gates(lg, K)
    sel_i, sel_w = idx.clone(), g.clone()
    gd = g[:, P:, 1:]
    anyf, ok, ha, la, gap = (t.view(S, T - P) for t in pick_pair(gd.reshape(-1, K - 1), Bm, mstar, delta, eps))
    drop = Bm[ha if arm == "H" else la].bool() & ok.unsqueeze(-1)          # [S,T',K-1]
    dropK = torch.cat([torch.zeros_like(drop[..., :1]), drop], -1)          # the rank-1 expert is never substituted
    idx_d = idx[:, P:]
    if mode == "fill":
        z = lg[:, P:].clone()
        z.scatter_(-1, idx_d, torch.where(dropK, torch.full_like(g[:, P:], float("-inf")), z.gather(-1, idx_d)))
        v, i = z.topk(K, dim=-1)
        sel_i[:, P:] = i
        sel_w[:, P:] = torch.softmax(v, -1)
        del z
    elif mode == "renorm":
        w = torch.where(dropK, torch.zeros_like(g[:, P:]), g[:, P:])
        sel_w[:, P:] = torch.where(ok.unsqueeze(-1), w / w.sum(-1, keepdim=True), g[:, P:])   # untouched if nothing dropped
    else:
        raise ValueError(mode)
    dropf = drop.float()
    m = (gd * dropf).sum(-1)
    q = (gd * gd * dropf).sum(-1)
    ranks = torch.arange(2, K + 1, device=lg.device, dtype=torch.float32)
    st = {"layer_tokens": int(ok.numel()), "feasible": int(anyf.sum()), "defined": int(ok.sum()),
          "mass_sum": float(m.double().sum()), "sq_sum": float(q.double().sum()),
          "conc_sum": float(torch.where(ok, q / (m * m).clamp_min(1e-30), torch.zeros_like(m)).double().sum()),
          "gap_sum": float(gap.double().sum()),
          "nsub_sum": float(dropf.sum(-1).double().sum()), "rank_sum": float((dropf * ranks).double().sum())}
    return sel_i, sel_w, st


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
    ap.add_argument("--idx", default="4,6,8,10,12,14,16,17", help="0-based within-domain session indices")
    ap.add_argument("--arms", default="H:0.10:fill,L:0.10:fill",
                    help="registered run: the primary pair only; e.g. H:0.05:fill,L:0.05:fill or H:0.10:renorm,... for others")
    ap.add_argument("--delta_rel", type=float, default=0.2, help="feasible: |s - m*| <= delta_rel * m*")
    ap.add_argument("--eps_rel", type=float, default=0.01, help="paired subsets: |s(A) - s(B)| <= eps_rel * m*")
    ap.add_argument("--dl_flag_dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
    want = [int(x) for x in args.idx.split(",")]
    sel, within = [], []
    for d in range(len(names)):
        ss = np.where(dom_all == d)[0]
        if len(ss) <= max(want):
            raise SystemExit(f"domain {names[d]} has {len(ss)} sessions; index {max(want)} is needed")
        sel += [int(ss[i]) for i in want]
        within += want
    sel = np.array(sel)
    tokens = tokens_all[sel]; dom = dom_all[sel]
    S, T = tokens.shape
    N = S * T
    tok_md5 = hashlib.md5(np.ascontiguousarray(tokens.astype(np.int32)).tobytes()).hexdigest()
    arms = []
    for a in args.arms.split(","):
        arm, ms, mode = a.split(":")
        assert arm in ("H", "L") and mode in ("fill", "renorm")
        arms.append((f"{arm}{float(ms):g}{mode[0]}", (arm, float(ms), mode)))
    log(f"cfg: layers={NL} experts={E} top{K} | sessions={S} x {T} (prompt {P}) | pool sessions {len(tokens_all)} | "
        f"within-domain idx {want} | tokens md5 {tok_md5} | arms {[a[0] for a in arms]}")
    log(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}")

    ck = Ckpt(args.model_dir, args.dl_flag_dir)
    tok_t = torch.from_numpy(tokens).to(dev)
    rot = QN.Qwen3NextRotaryEmbedding(config=cfg, device=dev)
    cos, sin = rot(torch.zeros(1, T, H, device=dev, dtype=dt), torch.arange(T, device=dev)[None])
    Bm = subset_matrix(K - 1, dev)
    CH = args.chunk

    def run_pass(specs):
        emb = ck.get("model.embed_tokens.weight").to(dev, dt)
        xs = [emb[tok_t].clone() for _ in specs]
        del emb
        stats = [[] for _ in specs]
        qq = queue.Queue(maxsize=2)
        pin = dev.type == "cuda"

        def loader():
            try:
                ck2 = Ckpt(args.model_dir, args.dl_flag_dir)
                for L in range(NL):
                    qq.put(load_layer(ck2, L, cfg, dt, pin))
            except Exception as e:      # surfaced in the main thread
                qq.put(e)
        threading.Thread(target=loader, daemon=True).start()
        for L in range(NL):
            tl0 = time.time()
            Wt = qq.get()
            if isinstance(Wt, Exception):
                raise Wt
            mixer = build_mixer(cfg, L, Wt["mixer"], dev)
            in_norm = make_norm(Wt["in_norm"], eps, dev)
            post_norm = make_norm(ck.get(f"model.layers.{L}.post_attention_layernorm.weight").to(dt), eps, dev)
            gate_w = ck.get(f"model.layers.{L}.mlp.gate.weight").to(dev, dt)
            GU, DN = Wt["gu"].to(dev, non_blocking=True), Wt["down"].to(dev, non_blocking=True)
            SGU, SDN, SG = (Wt[k].to(dev) for k in ("sh_gu", "sh_down", "sh_gate"))
            t0 = time.time()
            line = []
            for si, spec in enumerate(specs):
                x = xs[si]
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
                lg3 = lg.view(S, T, E)
                if spec is None:
                    ti, tw = topk_gates(lg3, K)
                else:
                    arm, ms, mode = spec
                    ti, tw, st = arm_route(lg3, P, K, Bm, ms, args.delta_rel * ms, args.eps_rel * ms, arm, mode)
                    stats[si].append(st)
                    line.append(f"{arm}{ms:g}{mode[0]} def {st['defined'] / st['layer_tokens']:.3f} "
                                f"m {st['mass_sum'] / st['layer_tokens']:.4f}")
                del lg, lg3
                ti = ti.reshape(N, K); tw = tw.reshape(N, K)
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
            del Wt, GU, DN, SGU, SDN, SG, mixer, gate_w
            if dev.type == "cuda":
                torch.cuda.synchronize()
            log(f"layer {L + 1}/{NL} ({cfg.layer_types[L][:3]}) {time.time() - t0:.1f}s wait {t0 - tl0:.1f}s"
                + (" | " + " | ".join(line) if line else ""))
        return xs, stats

    fnorm = make_norm(ck.get("model.norm.weight").to(dt), eps, dev)
    lm = ck.get("lm_head.weight").to(dev, dt)
    labels = torch.full((S, T), -100, dtype=torch.long, device=dev)
    labels[:, P:T - 1] = tok_t[:, P + 1:]
    labels = labels.view(N)

    def evaluate(xs):
        ns = len(xs)
        nll = np.zeros((ns, S)); kl = np.zeros((ns, S)); agree = np.zeros((ns, S)); acc1 = np.zeros((ns, S))
        cnt = np.zeros(S)
        LC = 1024
        for c0 in range(0, N, LC):
            lb = labels[c0:c0 + LC]
            valid = lb != -100
            if not bool(valid.any()):
                continue
            vv = (torch.arange(c0, c0 + lb.shape[0], device=dev) // T)[valid].cpu().numpy()
            lbv = lb[valid]
            np.add.at(cnt, vv, 1)
            base_lp = base_am = None
            for si in range(ns):
                lp = F.log_softmax(F.linear(fnorm(xs[si].view(N, H)[c0:c0 + LC][valid]), lm).float(), -1)
                am = lp.argmax(-1)
                if si == 0:
                    base_lp, base_am = lp, am
                np.add.at(nll[si], vv, (-lp.gather(1, lbv[:, None])[:, 0]).double().cpu().numpy())
                np.add.at(acc1[si], vv, (am == lbv).double().cpu().numpy())
                np.add.at(agree[si], vv, (am == base_am).double().cpu().numpy())
                np.add.at(kl[si], vv, (base_lp.exp() * (base_lp - lp)).sum(-1).double().cpu().numpy())
                if si > 0:
                    del lp
        return nll, kl, agree, acc1, cnt

    # ---------------------------------------------------------------- base and arms in one pass
    log("base + arms")
    specs = [None] + [a[1] for a in arms]
    xs, stats = run_pass(specs)
    nll, kl, agree, acc1, cnt = evaluate(xs)
    del xs
    streams = ["base"] + [a[0] for a in arms]
    doc = {"meta": {"S": S, "T": T, "P": P, "layers": NL, "E": E, "K": K, "domains": names,
                    "session_domain": [names[d] for d in dom], "session_pool_index": sel.tolist(),
                    "session_within_index": within, "tokens_md5": tok_md5, "delta_rel": args.delta_rel, "eps_rel": args.eps_rel,
                    "arms": {a[0]: {"arm": a[1][0], "mstar": a[1][1], "mode": a[1][2], "delta": args.delta_rel * a[1][1],
                                    "eps": args.eps_rel * a[1][1]} for a in arms},
                    "torch": torch.__version__, "seconds": round(time.time() - t_start, 1),
                    "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"},
           "cnt": cnt.tolist(), "streams": {}}
    for si, nm in enumerate(streams):
        e = {"nll": nll[si].tolist(), "kl": kl[si].tolist(), "agree": agree[si].tolist(), "acc1": acc1[si].tolist()}
        if si > 0:
            sts = stats[si]
            e["per_layer"] = {k: [s[k] for s in sts] for k in sts[0]}
            tot = {k: sum(s[k] for s in sts) for k in sts[0]}
            lt = tot["layer_tokens"]
            e["summary"] = {"defined": tot["defined"] / lt, "feasible": tot["feasible"] / lt,
                            "mass_all": tot["mass_sum"] / lt, "sq_all": tot["sq_sum"] / lt,
                            "mass_defined": tot["mass_sum"] / max(tot["defined"], 1),
                            "sq_defined": tot["sq_sum"] / max(tot["defined"], 1),
                            "conc_defined": tot["conc_sum"] / max(tot["defined"], 1),
                            "gap_defined": tot["gap_sum"] / max(tot["defined"], 1),
                            "nsub_defined": tot["nsub_sum"] / max(tot["defined"], 1),
                            "rank_mean": tot["rank_sum"] / max(tot["nsub_sum"], 1)}
        doc["streams"][nm] = e
    json.dump(doc, open(os.path.join(args.out, "expb.json"), "w"))
    tot_cnt = cnt.sum()
    for si, nm in enumerate(streams):
        d = 100 * (np.exp((nll[si].sum() - nll[0].sum()) / tot_cnt) - 1)
        s = doc["streams"][nm].get("summary", {})
        log(f"{nm}: ppl {np.exp(nll[si].sum() / tot_cnt):.4f} dppl {d:+.3f}% kl {kl[si].sum() / tot_cnt:.5f} "
            f"agree {agree[si].sum() / tot_cnt:.4f} " + " ".join(f"{k} {v:.4f}" for k, v in s.items()))
    put("DXB", doc)
    log(f"experiment B done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
