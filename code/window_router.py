"""Window router: predict the expert SET of the next N tokens from the router's own logits (no learning).

Model, per layer, at time tau, for the next N tokens:
    z_t = W~ v_t,   v_t = mu_tau + eps_t          =>  z_t,e ~ N(m_tau,e + b_e, s_e^2)
    m_tau = EMA_alpha(z_<=tau)                     linear state (closed form / parallel scan)
    p_tau,e = Phi((m_tau,e + b_e - theta_tau) / s_e),  theta_tau chosen so sum_e p = K   (smoothed top-K)
    P_tau,e(N) = 1 - (1 - p_tau,e)^N               prob. that expert e is used in the next N tokens
b_e, s_e: per-expert mean/std of z_{tau+j} - m_tau (j = 1..N), fit on train sessions.
Baselines: FE = same window formula on an EMA of actual usage (no logits), REC = recency (== LRU).

Measured per layer on held-out sessions, after the prompt:
  distinct : predicted E|A(N)| = sum_e P(N)  vs  actual number of distinct experts in the next N tokens
  coverage : keep top-c by score at tau and do not look again for N tokens; share of the calls in
             (tau, tau+N] that are covered. Compared with the oracle set of that window.
  cache sim: per-token update, target = top_c(score + lam * resident); prefetch = target minus resident,
             misses are demand-loaded. Hit rate, loads/token, speed
             rho = t_c / max(t_c + demand/beta, (prefetch + demand)/beta)
  reference: LRU and OPT (Belady) with the same per-layer capacity (numba sims from analyze.py).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def ema_scan(x, a):
    """Inclusive EMA along dim 1 of [S,T,E]."""
    out = torch.empty_like(x)
    m = x[:, 0].clone()
    out[:, 0] = m
    for t in range(1, x.shape[1]):
        m.mul_(1 - a).add_(x[:, t], alpha=a)
        out[:, t] = m
    return out


def window_prob(mb, s, K, N, iters=30):
    """mb [s,T,E] = m + b, s [E] -> P(N) [s,T,E] with a per-token threshold so that sum_e p = K."""
    smax = float(s.max())
    lo = mb.amin(-1) - 8 * smax
    hi = mb.amax(-1) + 8 * smax
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tot = torch.special.ndtr((mb - mid[..., None]) / s).sum(-1)
        gt = tot > K
        lo = torch.where(gt, mid, lo)
        hi = torch.where(gt, hi, mid)
    th = 0.5 * (lo + hi)
    p = torch.special.ndtr((mb - th[..., None]) / s).clamp_(0, 1 - 1e-6)
    return -torch.expm1(N * torch.log1p(-p))


def by_domain(values, dom_of, names):
    out = {"ALL": float(np.mean(values))}
    for d, nm in enumerate(names):
        sel = dom_of == d
        if sel.any():
            out[nm] = float(np.mean(values[sel]))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--logits_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alphas", default="0.25,0.0625,0.015625")
    ap.add_argument("--Ns", default="8,32,128")
    ap.add_argument("--lams", default="0,0.1,0.3")
    ap.add_argument("--caps", default="0.25,0.125")
    ap.add_argument("--t_c", type=float, default=21.0, help="ms/token with everything in RAM")
    ap.add_argument("--expert_mb", type=float, default=1.8, help="one 4-bit expert")
    ap.add_argument("--betas", default="7,12", help="SSD GB/s")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--tau_step", type=int, default=16)
    ap.add_argument("--skip_numba", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    t_start = time.time()

    idx = np.load(os.path.join(args.trace_dir, "trace_idx.npy"))        # [S,T,NL,K]
    dat = np.load(args.data, allow_pickle=True)
    dom = dat["domain"].astype(np.int64)
    names = [str(s) for s in dat["domain_names"]]
    P = int(dat["prompt_len"])
    S, T, NL, K = idx.shape
    z0 = np.load(os.path.join(args.logits_dir, "logits_00.npy"), mmap_mode="r")
    E = z0.shape[-1]
    assert z0.shape == (S, T, E)
    train = np.zeros(S, bool)
    for d in range(len(names)):
        ss = np.where(dom == d)[0]
        train[ss[::2]] = True                                              # same split as analyze.py
    tr = np.where(train)[0]; te = np.where(~train)[0]
    tr_t = torch.from_numpy(tr).to(dev); te_t = torch.from_numpy(te).to(dev)
    alphas = [float(a) for a in args.alphas.split(",")]
    Ns = [int(n) for n in args.Ns.split(",")]
    Nmax = max(Ns)
    lams = [float(v) for v in args.lams.split(",")]
    fracs = [float(f) for f in args.caps.split(",")]
    caps = [max(int(round(f * E)), K) for f in fracs]
    betas = [float(b) for b in args.betas.split(",")]
    log(f"S={S} T={T} NL={NL} E={E} K={K} prompt={P} train={len(tr)} test={len(te)} caps={caps}")
    assert T - Nmax - 1 > P

    sc_names = [f"WR_a{a:g}_N{n}" for a in alphas for n in Ns] + \
               [f"FE_a{a:g}_N{n}" for a in alphas for n in Ns] + ["REC"]
    n_sc, nl = len(sc_names), len(lams)
    Sc = torch.empty(T, n_sc, S, E, dtype=torch.float16, device=dev)
    # +1e-5: on exact score ties keep the resident expert (makes REC == LRU instead of random swaps)
    lam_t = (torch.tensor(lams, device=dev, dtype=torch.float32) + 1e-5).view(1, nl, 1, 1)
    taus = torch.arange(P, T - Nmax - 1, args.tau_step, device=dev)
    ntok = T - P

    acc_hits = {c: torch.zeros(n_sc, nl, S, dtype=torch.int64, device=dev) for c in caps}
    acc_pref = {c: torch.zeros(n_sc, nl, S, dtype=torch.int64, device=dev) for c in caps}
    cover = {c: {n: {} for n in Ns} for c in caps}
    distinct = {n: {"actual": 0.0, **{f"WR_a{a:g}": 0.0 for a in alphas}, **{f"FE_a{a:g}": 0.0 for a in alphas}}
                for n in Ns}
    calib = {}

    def add_cov(c, n, key, val):
        cover[c][n][key] = cover[c][n].get(key, 0.0) + val

    # ------------------------------------------------------------------ summaries
    def speed(hit_calls, pref_loads, n_sess, n_layers):
        calls = K * ntok * n_sess * n_layers
        dem = calls - hit_calls
        per_tok = lambda v: v / (ntok * n_sess) * NL / n_layers            # scale to all NL layers
        pref_mb = per_tok(pref_loads) * args.expert_mb
        dem_mb = per_tok(dem) * args.expert_mb
        out = {"hit": hit_calls / calls, "prefetch_mb": pref_mb, "demand_mb": dem_mb}
        for b in betas:
            out[f"rho_{b:g}"] = args.t_c / max(args.t_c + dem_mb / b, (pref_mb + dem_mb) / b)
        return out

    def summarize(n_layers, final):
        sim = {}
        best = {}
        for c, f in zip(caps, fracs):
            H = acc_hits[c].cpu().numpy(); Pf = acc_pref[c].cpu().numpy()
            rows = {}
            for i, sn in enumerate(sc_names):
                for j, lm in enumerate(lams):
                    if sn == "REC" and lm != 0:
                        continue
                    key = sn if sn == "REC" else f"{sn}_lam{lm:g}"
                    rows[key] = {"train": speed(H[i, j, tr].sum(), Pf[i, j, tr].sum(), len(tr), n_layers),
                                 "test": speed(H[i, j, te].sum(), Pf[i, j, te].sum(), len(te), n_layers),
                                 "_ij": (i, j)}
            sel = {}
            b0 = f"rho_{betas[0]:g}"
            for fam in ("WR", "FE"):
                cand = [k for k in rows if k.startswith(fam)]
                kb = max(cand, key=lambda k: (rows[k]["train"][b0], rows[k]["train"]["hit"]))
                kh = max(cand, key=lambda k: rows[k]["train"]["hit"])
                sel[fam + "_best_speed"] = kb
                sel[fam + "_best_hit"] = kh
            sel["LRU(REC)"] = "REC"
            best[str(f)] = {}
            for tag, k in sel.items():
                i, j = rows[k]["_ij"]
                ent = {"config": k, **{kk: round(v, 4) for kk, v in rows[k]["test"].items()}}
                if final:
                    per_s = H[i, j, te] / (K * ntok * n_layers)
                    ent["hit_by_domain"] = {kk: round(v, 4) for kk, v in by_domain(per_s, dom[te], names).items()}
                best[str(f)][tag] = ent
            sim[str(f)] = {k: {"train": {kk: round(v, 4) for kk, v in r["train"].items()},
                               "test": {kk: round(v, 4) for kk, v in r["test"].items()}} for k, r in rows.items()}
        cov = {str(f): {str(n): {k: round(v / n_layers, 4) for k, v in cover[c][n].items()} for n in Ns}
               for c, f in zip(caps, fracs)}
        dis = {str(n): {k: round(v / n_layers, 2) for k, v in distinct[n].items()} for n in Ns}
        return {"layers_done": n_layers, "summary": best, "coverage": cov, "distinct": dis, "sim_all": sim}

    # ------------------------------------------------------------------ per layer
    for L in range(NL):
        tl = time.time()
        z = torch.from_numpy(np.load(os.path.join(args.logits_dir, f"logits_{L:02d}.npy"))).to(dev).float()
        ids = torch.from_numpy(idx[:, :, L, :].astype(np.int64)).to(dev)
        U = torch.zeros(S, T, E, dtype=torch.bool, device=dev).scatter_(2, ids, True)
        del ids
        si = 0
        calib[L] = {}
        # ---- window router scores
        z_tr = z[tr_t]
        T0, T1 = 32, T - Nmax
        for a in alphas:
            m = ema_scan(z, a)
            m_tr = m[tr_t][:, T0:T1]
            s1 = torch.zeros(E, dtype=torch.float64, device=dev); s2 = torch.zeros_like(s1)
            bs = {}
            for j in range(1, Nmax + 1):
                r = z_tr[:, T0 + j:T1 + j] - m_tr
                s1 += r.sum((0, 1), dtype=torch.float64); s2 += (r * r).sum((0, 1), dtype=torch.float64)
                if j in Ns:
                    cnt = r.shape[0] * r.shape[1] * j
                    b = s1 / cnt
                    sd = (s2 / cnt - b * b).clamp_min(1e-6).sqrt()
                    bs[j] = (b.float(), sd.float().clamp_min(1e-3))
            del m_tr, r
            calib[L][f"a{a:g}"] = {str(n): {"b_mean": float(bs[n][0].mean()), "s_mean": float(bs[n][1].mean())}
                                   for n in Ns}
            for n in Ns:
                b, sd = bs[n]
                for c0 in range(0, S, args.chunk):
                    pn = window_prob(m[c0:c0 + args.chunk] + b, sd, K, n)
                    Sc[:, si, c0:c0 + args.chunk] = pn.transpose(0, 1).half()
                    del pn
                si += 1
            del m
        del z_tr
        # ---- usage-EMA scores (same window formula, no logits)
        for a in alphas:
            f = ema_scan(U.float(), a).clamp_(0, 1 - 1e-6)
            for n in Ns:
                Sc[:, si] = (-torch.expm1(n * torch.log1p(-f))).transpose(0, 1).half()
                si += 1
            del f
        # ---- recency (LRU order)
        tt = torch.arange(T, device=dev, dtype=torch.float32).view(1, T, 1)
        last = torch.where(U, tt, torch.full_like(tt, -1.0)).cummax(1).values
        Sc[:, si] = ((last + 1) / T).transpose(0, 1).half()
        del last
        assert si + 1 == n_sc
        del z

        # ---- distinct counts + coverage ("predict once at tau, don't look for N tokens")
        Cc = torch.zeros(len(te), T + 1, E, dtype=torch.int16, device=dev)
        Cc[:, 1:] = U[te_t].to(torch.int16).cumsum(1, dtype=torch.int16)
        Sct = Sc[taus][:, :, te_t]                                                 # [tau, n_sc, s, E]
        for n in Ns:
            cf = (Cc[:, taus + n + 1] - Cc[:, taus + 1]).float()                   # [s,tau,E] future counts
            cp = (Cc[:, taus + 1] - Cc[:, (taus + 1 - n).clamp_min(0)]).float()    # previous-N counts
            distinct[n]["actual"] += float((cf > 0).sum(-1).float().mean())
            scs = {}
            for i, sn in enumerate(sc_names):
                if sn == "REC" or sn.endswith(f"_N{n}"):
                    scs[sn.replace(f"_N{n}", "")] = Sct[:, i].transpose(0, 1).float()
            for k, v in scs.items():
                if k != "REC":
                    distinct[n][k] += float(v.sum(-1).mean())
            scs["prevN"] = cp + 1e-3 * scs["REC"]
            scs["oracle"] = cf
            for c in caps:
                for k, v in scs.items():
                    top = v.topk(c, dim=-1).indices
                    add_cov(c, n, k, float(cf.gather(-1, top).sum(-1).mean() / (n * K)))
            del cf, cp, scs
        del Cc, Sct

        # ---- cache simulation with prefetch
        for c in caps:
            M = torch.zeros(n_sc, nl, S, E, dtype=torch.bool, device=dev)
            hits, pref = acc_hits[c], acc_pref[c]
            for t in range(T - 1):
                key = Sc[t].float().unsqueeze(1) + lam_t * M
                top = key.topk(c, dim=-1, sorted=False).indices
                tgt = torch.zeros_like(M).scatter_(-1, top, True)
                u = U[None, None, :, t + 1]
                if t + 1 >= P:
                    pref += (tgt & ~M).sum(-1)
                    hits += (tgt & u).sum(-1)
                M = tgt | u
            del M, key, top, tgt
        del U
        if dev.type == "cuda":
            torch.cuda.synchronize()
        c0_ = caps[0]
        h = acc_hits[c0_]
        rec_i = sc_names.index("REC")
        wr_best = max(float(h[i, j, te_t].sum()) for i in range(n_sc) for j in range(nl) if sc_names[i].startswith("WR"))
        den = K * ntok * len(te) * (L + 1)
        log(f"layer {L+1}/{NL} {time.time()-tl:.1f}s | cap {c0_}: LRU hit {float(h[rec_i, 0, te_t].sum())/den:.4f}, "
            f"best WR hit {wr_best/den:.4f} | cover N=32 oracle {cover[c0_][32]['oracle']/(L+1) if 32 in Ns else 0:.4f} "
            f"WR {max(v for k, v in cover[c0_][Ns[len(Ns)//2]].items() if k.startswith('WR'))/(L+1):.4f}")
        if (L + 1) in (12, 24, 36) and L + 1 < NL:
            s = summarize(L + 1, False)
            print("SECTION_WR_PARTIAL " + json.dumps({"layers_done": L + 1, "summary": s["summary"],
                                                      "coverage": s["coverage"], "distinct": s["distinct"]},
                                                     separators=(",", ":")), flush=True)

    res = summarize(NL, True)
    res["calibration_layer_means"] = {str(L): v for L, v in calib.items()}
    res["meta"] = {"S": S, "T": T, "NL": NL, "E": E, "K": K, "prompt": P, "caps": caps, "fracs": fracs,
                   "alphas": alphas, "Ns": Ns, "lams": lams, "t_c_ms": args.t_c, "expert_mb": args.expert_mb,
                   "betas": betas, "seconds_gpu": time.time() - t_start}
    print("SECTION_WR_DISTINCT " + json.dumps(res["distinct"], separators=(",", ":")), flush=True)
    print("SECTION_WR_COVER " + json.dumps(res["coverage"], separators=(",", ":")), flush=True)
    print("SECTION_WR_SUMMARY " + json.dumps(res["summary"], separators=(",", ":")), flush=True)
    json.dump(res, open(os.path.join(args.out, "window_router.json"), "w"), indent=1)

    # ------------------------------------------------------------------ reference: numba LRU / OPT per layer
    if not args.skip_numba:
        from analyze import run_perlayer
        seqs_pl = np.ascontiguousarray(idx[te].astype(np.int32).transpose(0, 2, 1, 3).reshape(len(te), NL, T * K))
        ref = {}
        for c, f in zip(caps, fracs):
            ref[str(f)] = {}
            for kind, nm in ((0, "OPT_pl"), (1, "LRU_pl")):
                hits = np.zeros(seqs_pl.shape, np.uint8)
                run_perlayer(kind, seqs_pl, c, E, hits)
                hr = hits.reshape(len(te), NL, T, K)[:, :, P:]
                hit = float(hr.mean())
                load_mb = (1 - hit) * K * NL * args.expert_mb
                ent = {"hit": round(hit, 4), "loads_mb": round(load_mb, 2),
                       "hit_by_domain": {k: round(v, 4) for k, v in by_domain(hr.mean(axis=(1, 2, 3)), dom[te], names).items()}}
                for b in betas:
                    ent[f"rho_serial_{b:g}"] = round(args.t_c / (args.t_c + load_mb / b), 4)
                    ent[f"rho_overlap_{b:g}"] = round(args.t_c / max(args.t_c, load_mb / b), 4)
                ref[str(f)][nm] = ent
        res["reference"] = ref
        print("SECTION_WR_REF " + json.dumps(ref, separators=(",", ":")), flush=True)
        json.dump(res, open(os.path.join(args.out, "window_router.json"), "w"), indent=1)
    log(f"window router done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
