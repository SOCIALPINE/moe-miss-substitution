"""Round-11b attribution (Part 17): rent-or-buy admission against the frequency score alone, inside one run
(same block bootstrap as ci_xc11.py)."""
import json, os, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "/workspace/res"
rng = np.random.default_rng(1)
O = "aio=1+uf=1+up=1+bg=0+ram=0"
FE = "+fe=0.0078125+dk=1"
raw = {r: json.load(open(os.path.join(R, r + ".json.raw.json"))) for r in ("main", "extra")}
D = int(json.load(open(os.path.join(R, "main.json")))["D"])


def pair(run, k, brun, bk, L=128, NB=2000):
    a, b = raw[run][k], raw[brun][bk]
    n = min(len(a["nll"]), len(b["nll"]))
    idx = np.arange(n)
    bid = (idx // D) * ((D + L - 1) // L) + (idx % D) // L
    nb = int(bid.max()) + 1
    s = lambda x: np.bincount(bid, weights=np.asarray(x[:n], float), minlength=nb)
    dn, ma, mb, c = s(np.array(a["nll"][:n]) - np.array(b["nll"][:n])), s(a["ms"]), s(b["ms"]), np.bincount(bid, minlength=nb).astype(float)
    pick = rng.integers(0, nb, size=(NB, nb))
    f1 = lambda d, cc: 100 * (np.exp(d / cc) - 1)
    f2 = lambda x, y: 100 * (x / y - 1)
    r = lambda est, bs: [round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    out = {"run": run, "key": k, "base_run": brun, "base": bk, "n": n,
           "dppl": r(f1(dn.sum(), c.sum()), f1(dn[pick].sum(1), c[pick].sum(1))),
           "dms": r(f2(ma.sum(), mb.sum()), f2(ma[pick].sum(1), mb[pick].sum(1)))}
    print("CIB " + json.dumps(out, separators=(",", ":")), flush=True)


fe = "C@0.25/nvme:0.15|" + O + FE
c1fe = "C1@0.25/nvme|" + O + FE
for b in ("1", "0.5", "0.25"):
    pair("main", fe + "+rb=" + b, "main", fe)
pair("extra", c1fe + "+rb=0.5", "extra", c1fe)
pair("extra", fe + "+rb=0.5", "main", fe)
pair("main", fe, "main", "C@0.25/nvme:0.15|" + O)
pair("extra", c1fe, "main", "C1@0.25/nvme|" + O)
print("CIB DONE", flush=True)
