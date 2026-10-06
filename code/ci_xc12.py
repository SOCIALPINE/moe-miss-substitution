"""Round-12 confidence intervals (Part 18): block bootstrap (blocks of 128 generated tokens inside a sequence, 2000
resamples, 95% percentile intervals) over the per-token output of xcache-bench (*.json.raw.json).
  E12: every plan against the exact plan of its run (ppl change, top-1 agreement, ms)
  C12: paired plan comparisons inside one run (ppl change, time change); caps plans against the same plan of the main run
  B12: 512-token position bins of the topic-change run (ppl change against exact)
main is also split into the six sessions of round 11 (old6: 1, 5, 9, 13, 17, 21) and the other eighteen (new18)."""
import json, math, os, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "/workspace/res"
rng = np.random.default_rng(12)
NB, L = 2000, 128
O = "aio=1+uf=1+up=1+bg=0+ram=0"
FE = "+fe=0.0078125+dk=1"
EX = "exact@0.25/nvme|" + O
ENG, C1 = "C@0.25/nvme:0.15|" + O, "C1@0.25/nvme|" + O
PAIRS = [(ENG + FE, ENG), (ENG + FE + "+rb=1", ENG + FE), (ENG + FE + "+rb=0.5", ENG + FE), (C1 + FE, C1),
         (C1 + FE + "+rb=0.5", C1 + FE), (C1 + FE + "+rb=0.5", C1), (C1, ENG), (C1 + FE + "+rb=0.5", ENG),
         (ENG + FE + "+rb=1", ENG), (ENG + FE + "+rb=0.5", ENG), (C1 + FE, ENG)]


def load(run):
    p = os.path.join(R, run + ".json.raw.json")
    if not os.path.exists(p):
        return None, 0
    return json.load(open(p)), int(json.load(open(os.path.join(R, run + ".json")))["D"])


def setup(D, n, seqs):
    idx = np.arange(n)
    keep = np.isin(idx // D, seqs)
    nbs = (D + L - 1) // L
    bid_all = (idx // D) * nbs + (idx % D) // L
    used = np.unique(bid_all[keep])
    remap = -np.ones(bid_all.max() + 1, int)
    remap[used] = np.arange(len(used))
    bid = remap[bid_all[keep]]
    nb = len(used)
    pick = rng.integers(0, nb, size=(NB, nb))
    cnt = np.bincount(bid, minlength=nb).astype(float)
    return keep, bid, nb, pick, cnt


def ci(est, bs):
    return [round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def stats(run, subset, D, a, b, ex, seqs):
    """a, b: plan dicts (b may be None); ex: exact plan dict"""
    n = len(a["nll"])
    keep, bid, nb, pick, cnt = setup(D, n, seqs)
    s = lambda x: np.bincount(bid, weights=np.asarray(x, float)[:n][keep], minlength=nb)
    out = {}
    if ex is not None:
        d = s(np.array(a["nll"]) - np.array(ex["nll"]))
        g = s((np.array(a["am"]) == np.array(ex["am"])).astype(float))
        out["dppl_exact"] = ci(100 * (math.exp(d.sum() / cnt.sum()) - 1), 100 * (np.exp(d[pick].sum(1) / cnt[pick].sum(1)) - 1))
        out["agree"] = ci(g.sum() / cnt.sum(), g[pick].sum(1) / cnt[pick].sum(1))
    if b is not None:
        d = s(np.array(a["nll"]) - np.array(b["nll"]))
        ma, mb = s(a["ms"]), s(b["ms"])
        out["dppl"] = ci(100 * (math.exp(d.sum() / cnt.sum()) - 1), 100 * (np.exp(d[pick].sum(1) / cnt[pick].sum(1)) - 1))
        out["dms"] = ci(100 * (ma.sum() / mb.sum() - 1), 100 * (ma[pick].sum(1) / mb[pick].sum(1) - 1))
    out["ms"] = round(float(s(a["ms"]).sum() / cnt.sum()), 4)
    return out


def emit(tag, **kw):
    print(tag + " " + json.dumps(kw, separators=(",", ":")), flush=True)


main, Dm = load("main")
caps, _ = load("caps")
cross, Dc = load("cross")
OLD6 = [1, 5, 9, 13, 17, 21]
if main:
    S = len(main[EX]["nll"]) // Dm
    subsets = {"all": list(range(S)), "old6": [s for s in OLD6 if s < S], "new18": [s for s in range(S) if s not in OLD6]}
    for sub, seqs in subsets.items():
        for k, v in main.items():
            emit("E12", run="main", subset=sub, key=k, **stats("main", sub, Dm, v, None, main[EX], seqs))
        for k, b in PAIRS:
            if k in main and b in main:
                emit("C12", run="main", subset=sub, key=k, base=b, **stats("main", sub, Dm, main[k], main[b], None, seqs))
    if caps:
        for k, v in caps.items():
            emit("E12", run="caps", subset="all", key=k, **stats("caps", "all", Dm, v, None, main[EX], subsets["all"]))
            for b in dict.fromkeys((k, ENG)):
                if b in main:
                    emit("C12", run="caps", subset="all", key=k, base="main:" + b, **stats("caps", "all", Dm, v, main[b], None, subsets["all"]))
if cross:
    S = len(cross[EX]["nll"]) // Dc
    seqs = list(range(S))
    for k, v in cross.items():
        emit("E12", run="cross", subset="all", key=k, **stats("cross", "all", Dc, v, None, cross[EX], seqs))
    for k, b in PAIRS:
        if k in cross and b in cross:
            emit("C12", run="cross", subset="all", key=k, base=b, **stats("cross", "all", Dc, cross[k], cross[b], None, seqs))
    ex = cross[EX]
    for k, v in cross.items():
        n = len(v["nll"])
        nb = (Dc + 511) // 512
        bn, bd = [0] * nb, [0.0] * nb
        for i in range(n):
            j = (i % Dc) // 512
            bn[j] += 1
            bd[j] += v["nll"][i] - ex["nll"][i]
        emit("B12", run="cross", key=k, D=Dc, dppl=[round(100 * (math.exp(bd[j] / bn[j]) - 1), 3) for j in range(nb)])
print("CI12 DONE", flush=True)
