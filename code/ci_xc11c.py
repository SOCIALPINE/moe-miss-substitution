"""Round-11c attribution on the topic-change sequences (Part 17): paired block bootstrap (as ci_xc11.py) of plans of the
cross2 run against plans of the cross run (same sequences and tokens), and 512-token position bins against exact."""
import json, math, os, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "/workspace/res"
rng = np.random.default_rng(2)
O = "aio=1+uf=1+up=1+bg=0+ram=0"
FE = "+fe=0.0078125+dk=1"
raw = {r: json.load(open(os.path.join(R, r + ".json.raw.json"))) for r in ("cross", "cross2")}
D = int(json.load(open(os.path.join(R, "cross.json")))["D"])
ENG, C1 = "C@0.25/nvme:0.15|" + O, "C1@0.25/nvme|" + O
EX = "exact@0.25/nvme|" + O


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
    print("CIC " + json.dumps(out, separators=(",", ":")), flush=True)


c1fe, efe = C1 + FE, ENG + FE
pair("cross2", c1fe, "cross", C1)
pair("cross2", efe, "cross", ENG)
pair("cross", c1fe + "+rb=0.5", "cross2", c1fe)
pair("cross", efe + "+rb=1", "cross2", efe)
pair("cross2", c1fe + "+rb=0.5", "cross", c1fe + "+rb=0.5")
pair("cross2", c1fe, "cross", EX)
pair("cross2", efe, "cross", EX)
ex = raw["cross"][EX]
for k, v in raw["cross2"].items():
    n = len(v["nll"])
    nb = (D + 511) // 512
    bn, bd, bms = [0] * nb, [0.0] * nb, [0.0] * nb
    for i in range(n):
        b = (i % D) // 512
        bn[b] += 1
        bd[b] += v["nll"][i] - ex["nll"][i]
        bms[b] += v["ms"][i]
    print("B11 " + json.dumps({"run": "cross2", "key": k, "D": D, "dppl": [round(100 * (math.exp(bd[b] / bn[b]) - 1), 3) for b in range(nb)],
                               "ms": [round(bms[b] / bn[b], 3) for b in range(nb)]}, separators=(",", ":")), flush=True)
print("CIC DONE", flush=True)
