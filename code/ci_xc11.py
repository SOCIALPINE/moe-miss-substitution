"""Round-11 confidence intervals (Part 17), run on the pod over the per-token output of xcache-bench (*.json.raw.json).
Block bootstrap (blocks of 128 generated tokens inside a session, 2000 resamples, 95% percentile intervals) for
  ppl change against the exact reference, top-1 agreement with the exact reference,
  ppl change and time change of every plan against the current engine plans (paired, same tokens),
  for the static-slot run (caps) and the repeat run (extra) also against the same plan of the main run ("same")."""
import json, os, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "/workspace/res"
NB = 2000
L = 128
rng = np.random.default_rng(0)
O = "+aio=1+uf=1+up=1+bg=0+ram=0"
BASES = {"eng": "C@0.25/nvme:0.15|" + O[1:], "c1": "C1@0.25/nvme|" + O[1:],
         "eng12": "C@0.125/nvme:0.15|" + O[1:], "c1_12": "C1@0.125/nvme|" + O[1:]}


def load(run):
    p = os.path.join(R, run + ".json.raw.json")
    return json.load(open(p)) if os.path.exists(p) else None


def ci(sums_a, sums_b, cnt, pick, f):
    """f(sum_a, sum_b, count) over resampled blocks"""
    est = f(sums_a.sum(), sums_b.sum(), cnt.sum())
    bs = f(sums_a[pick].sum(1), sums_b[pick].sum(1), cnt[pick].sum(1))
    return [round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def run(name, raw, D, ref, base_raw, same_raw=None):
    rk = next(k for k in ref if k.startswith("exact@"))
    en = np.array(ref[rk]["nll"])
    ea = np.array(ref[rk]["am"])
    n = len(en)
    idx = np.arange(n)
    nbs = (D + L - 1) // L
    bid = (idx // D) * nbs + (idx % D) // L
    nb = int(bid.max()) + 1
    cnt = np.bincount(bid, minlength=nb).astype(float)
    pick = rng.integers(0, nb, size=(NB, nb))
    bsum = lambda x: np.bincount(bid, weights=np.asarray(x[:n], float), minlength=nb)
    zero = np.zeros(nb)
    for k, v in raw.items():
        nl, am, ms = np.array(v["nll"][:n]), np.array(v["am"][:n]), np.array(v["ms"][:n])
        rec = {"run": name, "key": k, "n": n,
               "dppl_exact": ci(bsum(nl - en), zero, cnt, pick, lambda a, b, c: 100 * (np.exp(a / c) - 1)),
               "agree": ci(bsum((am == ea).astype(float)), zero, cnt, pick, lambda a, b, c: a / c),
               "ms": ci(bsum(ms), zero, cnt, pick, lambda a, b, c: a / c)}
        cands = dict(BASES)
        if same_raw is not None and k in same_raw:
            cands["same"] = None
        for bn, bk in cands.items():
            br = same_raw[k] if bk is None else base_raw.get(bk)
            if br is None or (bk == k and base_raw is raw):
                continue
            bnl, bms = np.array(br["nll"][:n]), np.array(br["ms"][:n])
            rec["dppl_" + bn] = ci(bsum(nl - bnl), zero, cnt, pick, lambda a, b, c: 100 * (np.exp(a / c) - 1))
            rec["dms_" + bn] = ci(bsum(ms), bsum(bms), cnt, pick, lambda a, b, c: 100 * (a / b - 1))
        print("CI " + json.dumps(rec, separators=(",", ":")), flush=True)


main = load("main")
if main:
    D = int(json.load(open(os.path.join(R, "main.json")))["D"])
    run("main", main, D, main, main)
    caps = load("caps")
    if caps:
        run("caps", caps, D, main, main, same_raw=main)
    extra = load("extra")
    if extra:
        run("extra", extra, D, main, main, same_raw=main)
cross = load("cross")
if cross:
    D = int(json.load(open(os.path.join(R, "cross.json")))["D"])
    run("cross", cross, D, cross, cross)
print("CI DONE", flush=True)
