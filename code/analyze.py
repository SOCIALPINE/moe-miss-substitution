"""Offline analysis of recorded routing traces.

Cache model: an "expert slot" holds one (layer, expert) FFN. Capacity = fraction of all
NL*E experts. Requests are replayed in true decode order (token t, layer 0..NL-1, k=0..K-1).
Hit rate = fraction of expert activations whose weights were already resident.

Policies
  OPT   : Belady/MIN with bypass (knows the future) -> upper bound for ANY policy/predictor
  LRU   : least-recently-used
  LFU   : perfect-LFU with admission (full-history counts in the session, bypass if colder)
  static_*: a fixed resident set chosen from counts (global train / domain train / prompt / oracle)
  *_g = one global pool across layers, *_pl = equal slots per layer.
"""
import os, sys, json, time, argparse
import numpy as np
from numba import njit, prange

INF = np.int64(1) << np.int64(62)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------- heaps
@njit(cache=True)
def _hpush(hk, hv, sz, k, v):
    i = sz[0]
    sz[0] = i + 1
    hk[i] = k
    hv[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] < hk[i]:
            tk = hk[p]; hk[p] = hk[i]; hk[i] = tk
            tv = hv[p]; hv[p] = hv[i]; hv[i] = tv
            i = p
        else:
            break


@njit(cache=True)
def _hpop(hk, hv, sz):
    n = sz[0] - 1
    sz[0] = n
    hk[0] = hk[n]
    hv[0] = hv[n]
    i = 0
    while True:
        l = 2 * i + 1
        r = l + 1
        m = i
        if l < n and hk[l] > hk[m]:
            m = l
        if r < n and hk[r] > hk[m]:
            m = r
        if m == i:
            break
        tk = hk[m]; hk[m] = hk[i]; hk[i] = tk
        tv = hv[m]; hv[m] = hv[i]; hv[i] = tv
        i = m


# ----------------------------------------------------------------------------- policies
@njit(cache=True)
def sim_opt(seq, cap, nid, hit):
    n = seq.shape[0]
    nxt = np.empty(n, np.int64)
    last = np.full(nid, INF, np.int64)
    for i in range(n - 1, -1, -1):
        v = seq[i]
        nxt[i] = last[v]
        last[v] = i
    cached = np.zeros(nid, np.bool_)
    cur = np.zeros(nid, np.int64)
    hk = np.empty(n + cap + 2, np.int64)
    hv = np.empty(n + cap + 2, np.int32)
    sz = np.zeros(1, np.int64)
    cnt = 0
    for i in range(n):
        v = seq[i]
        nx = nxt[i]
        if cached[v]:
            hit[i] = 1
            cur[v] = nx
            _hpush(hk, hv, sz, nx, v)
        else:
            hit[i] = 0
            if cnt < cap:
                cached[v] = True
                cur[v] = nx
                _hpush(hk, hv, sz, nx, v)
                cnt += 1
            else:
                while True:
                    tv = hv[0]
                    if cached[tv] and cur[tv] == hk[0]:
                        break
                    _hpop(hk, hv, sz)
                tk = hk[0]
                if nx < tk:              # evict the resident expert needed furthest in future
                    _hpop(hk, hv, sz)
                    cached[tv] = False
                    cached[v] = True
                    cur[v] = nx
                    _hpush(hk, hv, sz, nx, v)
                # else: bypass (use it once without keeping it)


@njit(cache=True)
def sim_lru(seq, cap, nid, hit):
    n = seq.shape[0]
    prv = np.full(nid, -1, np.int32)
    nxl = np.full(nid, -1, np.int32)
    cached = np.zeros(nid, np.bool_)
    head = -1
    tail = -1
    cnt = 0
    for i in range(n):
        v = seq[i]
        if cached[v]:
            hit[i] = 1
            if v != head:
                p = prv[v]
                q = nxl[v]
                nxl[p] = q
                if q != -1:
                    prv[q] = p
                else:
                    tail = p
                prv[v] = -1
                nxl[v] = head
                prv[head] = v
                head = v
        else:
            hit[i] = 0
            if cnt == cap:
                t = tail
                tail = prv[t]
                if tail != -1:
                    nxl[tail] = -1
                else:
                    head = -1
                cached[t] = False
                cnt -= 1
            prv[v] = -1
            nxl[v] = head
            if head != -1:
                prv[head] = v
            head = v
            if tail == -1:
                tail = v
            cached[v] = True
            cnt += 1


@njit(cache=True)
def sim_lfu(seq, cap, nid, hit):
    n = seq.shape[0]
    cntv = np.zeros(nid, np.int64)
    cached = np.zeros(nid, np.bool_)
    curk = np.zeros(nid, np.int64)
    hk = np.empty(n + cap + 2, np.int64)   # max-heap of -key  => min-heap of key
    hv = np.empty(n + cap + 2, np.int32)
    sz = np.zeros(1, np.int64)
    cnt = 0
    SH = np.int64(1) << np.int64(32)
    for i in range(n):
        v = seq[i]
        cntv[v] += 1
        k = cntv[v] * SH + i
        if cached[v]:
            hit[i] = 1
            curk[v] = k
            _hpush(hk, hv, sz, -k, v)
        else:
            hit[i] = 0
            if cnt < cap:
                cached[v] = True
                curk[v] = k
                _hpush(hk, hv, sz, -k, v)
                cnt += 1
            else:
                while True:
                    tv = hv[0]
                    if cached[tv] and curk[tv] == -hk[0]:
                        break
                    _hpop(hk, hv, sz)
                if k > -hk[0]:
                    _hpop(hk, hv, sz)
                    cached[tv] = False
                    cached[v] = True
                    curk[v] = k
                    _hpush(hk, hv, sz, -k, v)


@njit(parallel=True, cache=True)
def run_global(kind, seqs, cap, nid, hits):
    for s in prange(seqs.shape[0]):
        if kind == 0:
            sim_opt(seqs[s], cap, nid, hits[s])
        elif kind == 1:
            sim_lru(seqs[s], cap, nid, hits[s])
        else:
            sim_lfu(seqs[s], cap, nid, hits[s])


@njit(parallel=True, cache=True)
def run_perlayer(kind, seqs_pl, cap, nid, hits_pl):
    S = seqs_pl.shape[0]
    NL = seqs_pl.shape[1]
    for j in prange(S * NL):
        s = j // NL
        L = j % NL
        if kind == 0:
            sim_opt(seqs_pl[s, L], cap, nid, hits_pl[s, L])
        elif kind == 1:
            sim_lru(seqs_pl[s, L], cap, nid, hits_pl[s, L])
        else:
            sim_lfu(seqs_pl[s, L], cap, nid, hits_pl[s, L])


# ----------------------------------------------------------------------------- helpers
def topc_mask(counts, C, tiebreak=None):
    c = counts.astype(np.float64)
    if tiebreak is not None:
        c = c + 1e-6 * tiebreak / max(tiebreak.max(), 1)
    m = np.zeros(counts.shape[0], bool)
    m[np.argsort(-c, kind="stable")[:C]] = True
    return m


def onehot(idx_tl, E):  # idx [N, K] -> [N, E] float32
    N = idx_tl.shape[0]
    oh = np.zeros((N, E), np.float32)
    oh[np.arange(N)[:, None], idx_tl] = 1.0
    return oh


def topm_recall(scores, target_oh, m_list, K):
    mm = max(m_list)
    part = np.argpartition(-scores, mm - 1, axis=1)[:, :mm]
    ps = np.take_along_axis(scores, part, 1)
    order = np.argsort(-ps, axis=1, kind="stable")
    top = np.take_along_axis(part, order, 1)
    hits = np.take_along_axis(target_oh, top, 1).cumsum(1)
    return [float(hits[:, m - 1].mean() / K) for m in m_list]


def by_domain(values, dom_of, names):
    out = {"ALL": float(np.mean(values))}
    for d, nm in enumerate(names):
        sel = dom_of == d
        if sel.any():
            out[nm] = float(np.mean(values[sel]))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fracs", default="0.0625,0.125,0.25,0.5")
    args = ap.parse_args()
    t0 = time.time()

    idx = np.load(os.path.join(args.trace_dir, "trace_idx.npy"))       # [S,T,NL,K] uint8
    wts = np.load(os.path.join(args.trace_dir, "trace_w.npy"))         # [S,T,NL,K] f16
    dat = np.load(args.data, allow_pickle=True)
    dom = dat["domain"].astype(np.int64)
    names = [str(s) for s in dat["domain_names"]]
    P = int(dat["prompt_len"])
    tstats = json.load(open(os.path.join(args.trace_dir, "trace_stats.json")))
    E = int(tstats["model_cfg"]["experts"])
    S, T, NL, K = idx.shape
    NID = NL * E
    per_tok = NL * K
    log(f"trace: S={S} T={T} NL={NL} E={E} K={K} prompt={P}")

    # train/test split: alternate sessions within each domain
    train = np.zeros(S, bool)
    for d in range(len(names)):
        ss = np.where(dom == d)[0]
        train[ss[::2]] = True
    test = ~train
    te = np.where(test)[0]
    tr = np.where(train)[0]
    dom_te = dom[te]
    log(f"train sessions {len(tr)}, test sessions {len(te)}")

    gid = (np.arange(NL, dtype=np.int32)[None, None, :, None] * E + idx.astype(np.int32))  # [S,T,NL,K]
    seqs = np.ascontiguousarray(gid[te].reshape(len(te), -1))                               # order t,L,k
    seqs_pl = np.ascontiguousarray(idx[te].astype(np.int32).transpose(0, 2, 1, 3).reshape(len(te), NL, T * K))
    w_te = wts[te].astype(np.float32)                                                       # [s,T,NL,K]

    res = {"meta": {"S": S, "T": T, "NL": NL, "E": E, "K": K, "prompt_len": P, "domains": names,
                    "n_train": int(len(tr)), "n_test": int(len(te)),
                    "sessions_per_domain": {nm: int((dom == d).sum()) for d, nm in enumerate(names)}},
           "trace_stats": tstats}

    # ---------------- counts for static policies
    cnt_sess = np.stack([np.bincount(gid[s].ravel(), minlength=NID) for s in range(S)])  # [S,NID]
    cnt_prompt = np.stack([np.bincount(gid[s, :P].ravel(), minlength=NID) for s in range(S)])
    cnt_tail = cnt_sess - cnt_prompt
    cnt_global_tr = cnt_sess[tr].sum(0)
    cnt_dom_tr = {d: cnt_sess[tr[dom[tr] == d]].sum(0) for d in range(len(names))}

    fracs = [float(f) for f in args.fracs.split(",")]
    cache = {}
    rank_stats = {}
    for f in fracs:
        C = max(int(round(f * NID)), 1)
        Cpl = max(int(round(f * E)), 1)
        fr = {}
        # dynamic policies
        for kind, nm in ((0, "OPT"), (1, "LRU"), (2, "LFU")):
            hits = np.zeros(seqs.shape, np.uint8)
            run_global(kind, seqs, C, NID, hits)
            h4 = hits.reshape(len(te), T, NL, K)
            fr[nm + "_g"] = {"after_prompt": by_domain(h4[:, P:].mean(axis=(1, 2, 3)), dom_te, names),
                             "all_tokens": by_domain(h4.mean(axis=(1, 2, 3)), dom_te, names),
                             "per_layer_after_prompt": h4[:, P:].mean(axis=(0, 1, 3)).round(4).tolist()}
            if nm in ("OPT", "LRU"):
                miss = 1.0 - h4[:, P:].astype(np.float32)
                ww = w_te[:, P:]
                rank_stats[f"{nm}_g@{f}"] = {
                    "miss_rate_by_weight_rank": (miss.mean(axis=(0, 1, 2))).round(4).tolist(),
                    "gate_weight_mass_on_misses": float((miss * ww).sum() / ww.sum()),
                    "miss_rate": float(miss.mean())}
            if nm != "LFU":
                hpl = np.zeros(seqs_pl.shape, np.uint8)
                run_perlayer(kind, seqs_pl, Cpl, E, hpl)
                h4p = hpl.reshape(len(te), NL, T, K)
                fr[nm + "_pl"] = {"after_prompt": by_domain(h4p[:, :, P:].mean(axis=(1, 2, 3)), dom_te, names),
                                  "all_tokens": by_domain(h4p.mean(axis=(1, 2, 3)), dom_te, names)}
        # static policies, evaluated on tokens after the prompt
        st = {"static_global": [], "static_domain": [], "prompt_fingerprint": [], "oracle_static": []}
        for j, s in enumerate(te):
            tail = cnt_tail[s]
            tot = tail.sum()
            masks = {
                "static_global": topc_mask(cnt_global_tr, C),
                "static_domain": topc_mask(cnt_dom_tr[dom[s]], C, cnt_global_tr),
                "prompt_fingerprint": topc_mask(cnt_prompt[s], C, cnt_dom_tr[dom[s]]),
                "oracle_static": topc_mask(tail, C),
            }
            for k2, m in masks.items():
                st[k2].append(tail[m].sum() / tot)
        for k2, v in st.items():
            fr[k2] = {"after_prompt": by_domain(np.array(v), dom_te, names)}
        cache[str(f)] = fr
        log(f"cache f={f}: " + ", ".join(f"{k}={v['after_prompt']['ALL']:.3f}" for k, v in fr.items()))
    res["cache"] = cache
    res["miss_structure"] = rank_stats
    print("SECTION_CACHE " + json.dumps({f: {p: {k: round(x, 4) for k, x in v["after_prompt"].items()} for p, v in fr.items()} for f, fr in cache.items()}, separators=(",", ":")), flush=True)
    print("SECTION_MISS " + json.dumps(rank_stats, separators=(",", ":"), default=float), flush=True)

    # ---------------- temporal locality
    rng = np.random.default_rng(0)
    Ws = [w for w in (8, 32, 128, 512, 2048) if w <= T]
    reuse, reuse_sh = [], []
    wd = {w: [] for w in Ws}
    wd_sh = {w: [] for w in Ws}
    for s in te:
        oh = np.zeros((T, NL, E), bool)
        oh[np.arange(T)[:, None, None], np.arange(NL)[None, :, None], idx[s]] = True
        perm = rng.permutation(T)
        ohs = oh[perm]
        reuse.append((oh[1:] & oh[:-1]).sum() / ((T - 1) * NL * K))
        reuse_sh.append((ohs[1:] & ohs[:-1]).sum() / ((T - 1) * NL * K))
        for w in Ws:
            nb = T // w
            wd[w].append(oh[:nb * w].reshape(nb, w, NL, E).any(1).sum(-1).mean())
            wd_sh[w].append(ohs[:nb * w].reshape(nb, w, NL, E).any(1).sum(-1).mean())
    res["locality"] = {
        "reuse_next_token": by_domain(np.array(reuse), dom_te, names),
        "reuse_next_token_shuffled": by_domain(np.array(reuse_sh), dom_te, names),
        "reuse_uniform_baseline": K / E,
        "distinct_experts_per_layer_in_window": {
            str(w): {"actual": by_domain(np.array(wd[w]), dom_te, names),
                     "shuffled_tokens": by_domain(np.array(wd_sh[w]), dom_te, names),
                     "uniform_random": float(E * (1 - (1 - K / E) ** w))} for w in Ws},
    }
    print("SECTION_LOCALITY " + json.dumps(res["locality"], separators=(",", ":"), default=float), flush=True)
    log("locality done")

    # ---------------- concentration (in-sample: how much of the traffic the top-f experts carry)
    conc = {}
    for f in (0.125, 0.25):
        C = int(round(f * NID))

        def cov(c):
            return float(np.sort(c)[::-1][:C].sum() / max(c.sum(), 1))
        allc = cnt_sess.sum(0)
        win = []
        for s in te:
            for b in range(0, T - 255, 256):
                win.append(cov(np.bincount(gid[s, b:b + 256].ravel(), minlength=NID)))
        conc[str(f)] = {
            "uniform": f,
            "global_pool_all_sessions": cov(allc),
            "per_domain_pool": {nm: cov(cnt_sess[dom == d].sum(0)) for d, nm in enumerate(names)},
            "per_session": by_domain(np.array([cov(cnt_sess[s]) for s in te]), dom_te, names),
            "per_256_token_window": float(np.mean(win)),
        }
    res["concentration"] = conc
    print("SECTION_CONC " + json.dumps(conc, separators=(",", ":"), default=float), flush=True)
    log("concentration done")

    # ---------------- 'map' predictors (learned on train sessions, tested on test sessions)
    m_list = [m for m in (K, 2 * K, 4 * K) if m <= E]
    alpha = 0.5
    acc = {k: np.zeros(len(m_list)) for k in ("layer_map_global", "layer_map_domain", "temporal_map",
                                               "layer+temporal_map", "prev_token_same_layer")}
    ce = {"marginal": 0.0, "given_prev_layer_expert": 0.0, "given_prev_token_expert": 0.0}
    npair = 0
    for L in range(NL - 1):
        a_tr = onehot(idx[tr, :, L].reshape(-1, K), E)
        b_tr = onehot(idx[tr, :, L + 1].reshape(-1, K), E)
        a_te = onehot(idx[te, :, L].reshape(-1, K), E)
        b_te = onehot(idx[te, :, L + 1].reshape(-1, K), E)
        Cg = a_tr.T @ b_tr
        Pg = (Cg + alpha) / (Cg.sum(1, keepdims=True) + alpha * E)
        sc_g = a_te @ Pg
        acc["layer_map_global"] += topm_recall(sc_g, b_te, m_list, K)
        # domain-specific map
        dom_rows_tr = np.repeat(dom[tr], T)
        dom_rows_te = np.repeat(dom[te], T)
        sc_d = np.zeros_like(sc_g)
        for d in range(len(names)):
            rt, re_ = dom_rows_tr == d, dom_rows_te == d
            if not re_.any():
                continue
            Cd = a_tr[rt].T @ b_tr[rt]
            Pd = (Cd + alpha) / (Cd.sum(1, keepdims=True) + alpha * E)
            sc_d[re_] = a_te[re_] @ Pd
        acc["layer_map_domain"] += topm_recall(sc_d, b_te, m_list, K)
        # temporal map: (t-1, L+1) -> (t, L+1)
        b_tr3 = b_tr.reshape(len(tr), T, E)
        b_te3 = b_te.reshape(len(te), T, E)
        Ct = b_tr3[:, :-1].reshape(-1, E).T @ b_tr3[:, 1:].reshape(-1, E)
        Pt = (Ct + alpha) / (Ct.sum(1, keepdims=True) + alpha * E)
        prev = b_te3[:, :-1].reshape(-1, E)
        tgt = b_te3[:, 1:].reshape(-1, E)
        sc_t = prev @ Pt
        acc["temporal_map"] += topm_recall(sc_t, tgt, m_list, K)
        sc_gl = sc_g.reshape(len(te), T, E)[:, 1:].reshape(-1, E)
        acc["layer+temporal_map"] += topm_recall(sc_gl + sc_t, tgt, m_list, K)
        acc["prev_token_same_layer"] += topm_recall(prev + 1e-3 * sc_t, tgt, m_list, K)
        # cross-entropy (bits) of the next-layer expert given one current-layer expert
        Mte = a_te.T @ b_te
        pm = (b_tr.sum(0) + alpha) / (b_tr.sum() + alpha * E)
        ce["marginal"] += float(-(Mte.sum(0) * np.log2(pm)).sum() / Mte.sum())
        ce["given_prev_layer_expert"] += float(-(Mte * np.log2(Pg)).sum() / Mte.sum())
        Mt = prev.T @ tgt
        ce["given_prev_token_expert"] += float(-(Mt * np.log2(Pt)).sum() / Mt.sum())
        npair += 1
    res["maps"] = {"m_list": m_list, "recall": {k: (v / npair).round(4).tolist() for k, v in acc.items()},
                   "cross_entropy_bits": {k: v / npair for k, v in ce.items()},
                   "max_entropy_bits": float(np.log2(E))}
    print("SECTION_MAPS " + json.dumps(res["maps"], separators=(",", ":"), default=float), flush=True)
    log("maps done")
    res["seconds"] = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    log(f"analysis done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
