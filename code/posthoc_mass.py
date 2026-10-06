"""Post-hoc analysis (2026-10-04) of mass vs. perplexity in the engine runs of Part 18 (xc12, single topic, 24 sessions)
and Part 19 (xc13, topic change, 24 sequences). Not pre-registered: written after the results were known.

Inputs: RESULT lines (whole run) and S11 lines (per chunk) of results/xc12_3090.log and results/xc13_3090.log.
For each run, over the policies other than 'exact':
  - policy level: Pearson r with a Fisher-z 95% interval, OLS slope (dppl points per 0.01 of mass), leave-one-out slopes,
    Spearman correlation;
  - chunk level (Part 18: 4 chunks x 6 sessions, Part 19: 8 chunks x 3 sequences): within-chunk (fixed-effects) slope
    after subtracting chunk means, 95% interval from 2,000 bootstrap resamples of chunks (numpy default_rng(14)),
    per-chunk Spearman, correlation of chunk means (between inputs), and per-policy correlation across chunks.
Usage: python3 posthoc_mass.py > results/posthoc_mass_2026-10-04.out
"""
import json, math, collections
import numpy as np
from scipy import stats

R = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "results", "")


def load(fn, prefix):
    tot, ch = {}, collections.OrderedDict()
    for line in open(R + fn):
        if line.startswith("RESULT "):
            _, name, js = line.split(" ", 2)
            if name == prefix:
                for k, v in json.loads(js)["runs"].items():
                    if not k.startswith("exact"):
                        tot[k] = (v["mass"], v["dppl_pct"])
        elif line.startswith("S11 "):
            j = json.loads(line[4:])
            if j["run"].startswith(prefix + "_") and not j["key"].startswith("exact"):
                ch[(j["run"], j["key"])] = (j["mass"], j["dppl_pct"])
    return tot, ch


def main():
    for fn, prefix, label in [("xc12_3090.log", "main", "Part 18 single topic"),
                              ("xc13_3090.log", "cross24", "Part 19 topic change")]:
        tot, ch = load(fn, prefix)
        keys = sorted(tot)
        x = np.array([tot[k][0] for k in keys])
        y = np.array([tot[k][1] for k in keys])
        n = len(x)
        r = np.corrcoef(x, y)[0, 1]
        z, se = np.arctanh(r), 1 / math.sqrt(n - 3)
        slope = np.polyfit(x, y, 1)[0] * 0.01
        loo = [np.polyfit(np.delete(x, i), np.delete(y, i), 1)[0] * 0.01 for i in range(n)]
        print(f"== {label} ({fn}, RESULT {prefix}), policies n={n}")
        for k in keys:
            print(f"   {k}: mass {tot[k][0]:.6f}  dppl_pct {tot[k][1]:.6f}")
        print(f"   Pearson r={r:.4f}  Fisher 95% [{np.tanh(z - 1.96 * se):.4f}, {np.tanh(z + 1.96 * se):.4f}]")
        print(f"   OLS slope (dppl points per 0.01 mass)={slope:.4f}  leave-one-out {min(loo):.4f}..{max(loo):.4f}")
        print(f"   Spearman={stats.spearmanr(x, y).correlation:.4f}")
        chunks = sorted({c for c, _ in ch}, key=lambda s: int(s.split("_")[-1]))
        assert all((c, k) in ch for c in chunks for k in keys), "incomplete chunk grid"
        M = np.array([[ch[(c, k)][0] for k in keys] for c in chunks])
        Y = np.array([[ch[(c, k)][1] for k in keys] for c in chunks])

        def fe(idx):
            xc = (M[idx] - M[idx].mean(1, keepdims=True)).ravel()
            yc = (Y[idx] - Y[idx].mean(1, keepdims=True)).ravel()
            return (xc @ yc) / (xc @ xc) * 0.01

        rng = np.random.default_rng(14)
        boot = [fe(rng.integers(0, len(chunks), len(chunks))) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sp = [stats.spearmanr(M[i], Y[i]).correlation for i in range(len(chunks))]
        pp = [np.corrcoef(M[:, j], Y[:, j])[0, 1] for j in range(len(keys))]
        print(f"   chunks={len(chunks)}  within-chunk slope={fe(np.arange(len(chunks))):.4f}  bootstrap 95% [{lo:.4f}, {hi:.4f}]")
        print(f"   per-chunk Spearman: {', '.join(f'{v:.4f}' for v in sp)}  (min {min(sp):.4f}, max {max(sp):.4f})")
        print(f"   correlation of chunk means (between inputs)={np.corrcoef(M.mean(1), Y.mean(1))[0, 1]:.4f}")
        print(f"   per-policy correlation across chunks: {', '.join(f'{v:.4f}' for v in pp)}  (min {min(pp):.4f}, max {max(pp):.4f})")


if __name__ == "__main__":
    main()
