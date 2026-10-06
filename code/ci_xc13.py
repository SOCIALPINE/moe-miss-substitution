"""Round-13 confidence intervals (Part 19): topic-change sequences (cross24: 24 sequences, prompt 512 + 4096 generated, the
topic changes at generated tokens 1536 and 3584), from the per-token output of xcache-bench (NAME.json.raw.json).

95% percentile intervals from 2000 resamples of two kinds:
  block: 128-token blocks inside a sequence (as in rounds 11 and 12)
  seq:   whole sequences (cluster bootstrap; also covers the differences between sequences)
Lines:
  SEQ13  the domains of every sequence
  E13    every plan against exact: ppl change, top-1 agreement, ms, 512-token bins, the 512 tokens after each topic change
         ("post") and the rest, the ppl change of every sequence
  C13    paired plan comparisons: ppl and time change over all tokens, ppl change in "post" and in the rest, the ppl change
         of every sequence and how many sequences favour the first plan"""
import json, math, os, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "/workspace/res"
NAME = sys.argv[2] if len(sys.argv) > 2 else "cross24"
rng = np.random.default_rng(13)
NB, L, W = 2000, 128, 512
SW = (1536, 3584)
O = "aio=1+uf=1+up=1+bg=0+ram=0"
FE = "+fe=0.0078125+dk=1"
EX, ENG, C1 = "exact@0.25/nvme|" + O, "C@0.25/nvme:0.15|" + O, "C1@0.25/nvme|" + O
QF, SF = ENG + FE + "+rb=0.5", C1 + FE + "+rb=0.5"
PAIRS = [(ENG + FE, ENG), (QF, ENG + FE), (QF, ENG), (C1, ENG), (C1 + FE, C1), (SF, C1 + FE), (SF, C1), (SF, ENG),
         (C1 + FE, ENG), (SF, QF)]


def emit(tag, **kw):
    print(tag + " " + json.dumps(kw, separators=(",", ":")), flush=True)


def ci(num, den, pick, f):
    """num, den: per-group sums; f(sum_num, sum_den)"""
    est = f(num.sum(), den.sum())
    bs = f(num[pick].sum(1), den[pick].sum(1))
    return [round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


dppl = lambda a, c: 100 * (np.exp(a / c) - 1)
ratio = lambda a, b: 100 * (a / b - 1)
mean = lambda a, c: a / c

p = os.path.join(R, NAME + ".json.raw.json")
if not os.path.exists(p):
    print("CI13 MISSING", p, flush=True)
    sys.exit(0)
raw = json.load(open(p))
D = int(json.load(open(os.path.join(R, NAME + ".json")))["D"])
n = len(raw[EX]["nll"])
S = n // D
meta = os.path.join(os.path.dirname(os.path.abspath(R)), "seq", NAME + ".json")
if os.path.exists(meta):
    m = json.load(open(meta))
    for s in range(S):
        emit("SEQ13", seq=s, sessions=m["sessions"][s], domains=m["domains"][s])

idx = np.arange(n)
seq, pos = idx // D, idx % D
post = np.zeros(n, bool)
for sw in SW:
    post |= (pos >= sw) & (pos < sw + W)
nbs = (D + L - 1) // L
blk = seq * nbs + pos // L
NBLK = S * nbs
pick_b = rng.integers(0, NBLK, size=(NB, NBLK))
pick_s = rng.integers(0, S, size=(NB, S))
one = np.ones(n)


def g(x, groups, ng, mask=None):
    x = np.asarray(x, float)[:n]
    return np.bincount(groups, weights=x if mask is None else x * mask, minlength=ng)


def both(num, den, f, mask=None):
    return {"block": ci(g(num, blk, NBLK, mask), g(den, blk, NBLK, mask), pick_b, f),
            "seq": ci(g(num, seq, S, mask), g(den, seq, S, mask), pick_s, f)}


ex = raw[EX]
enll, eam = np.array(ex["nll"][:n]), np.array(ex["am"][:n])
for k, v in raw.items():
    nll, am, ms = np.array(v["nll"][:n]), np.array(v["am"][:n]), np.array(v["ms"][:n])
    d = nll - enll
    bins = [round(float(dppl(d[(pos >= b) & (pos < b + W)].sum(), ((pos >= b) & (pos < b + W)).sum())), 3) for b in range(0, D, W)]
    per = g(d, seq, S) / g(one, seq, S)
    emit("E13", run=NAME, key=k, n=n, S=S, dppl_exact=both(d, one, dppl), agree=both((am == eam).astype(float), one, mean),
         ms=round(float(ms.mean()), 4), post=both(d, one, dppl, post), rest=both(d, one, dppl, ~post), bins=bins,
         per_seq=[round(float(100 * (math.exp(x) - 1)), 3) for x in per])
for a, b in PAIRS:
    if a not in raw or b not in raw:
        continue
    na, nb_ = np.array(raw[a]["nll"][:n]), np.array(raw[b]["nll"][:n])
    ma, mb = np.array(raw[a]["ms"][:n]), np.array(raw[b]["ms"][:n])
    d = na - nb_
    per = g(d, seq, S) / g(one, seq, S)
    per_post = g(d, seq, S, post) / g(one, seq, S, post)
    emit("C13", run=NAME, key=a, base=b, dppl=both(d, one, dppl), dms=both(ma, mb, ratio), post=both(d, one, dppl, post),
         rest=both(d, one, dppl, ~post), per_seq=[round(float(100 * (math.exp(x) - 1)), 3) for x in per],
         per_seq_post=[round(float(100 * (math.exp(x) - 1)), 3) for x in per_post], n_better=int((per < 0).sum()),
         n_better_post=int((per_post < 0).sum()))
print("CI13 DONE", flush=True)
