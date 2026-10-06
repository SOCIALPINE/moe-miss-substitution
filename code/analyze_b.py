"""Registered analysis of Experiment B (PREREG_expB.md, section 6). Reads the result document of expb.py (expb.json, or
the DXB lines of a pod log) and prints the primary judgment and the registered auxiliary results.

Delta ppl of a stream = 100 * (exp((sum of its NLL - sum of base NLL) / number of scored tokens) - 1), pooled over the
sessions used. Session bootstrap: one set of 2,000 resamples of the 48 sessions with replacement,
numpy.random.default_rng(seed).integers(0, S, size=(2000, S)), shared by every statistic; each resample pools the NLL sums
and token counts of the drawn sessions; 95% intervals are the 2.5 and 97.5 percentiles (numpy's default linear
interpolation). Primary statistic: R = Delta ppl(H, m* 0.10, fill) / Delta ppl(L, m* 0.10, fill). Auxiliary: the same
for m* 0.05 and 0.15 (fill) and m* 0.10 (renorm); R_m, the ratio of Delta ppl per unit of realised mass (mass averaged over
all decode layer-tokens of each arm, as in Section 3.1); H - L differences; KL and argmax agreement."""
import argparse, base64, hashlib, json, lzma, re, sys
import numpy as np

PAIRS = [("0.1", "f", True), ("0.05", "f", False), ("0.15", "f", False), ("0.1", "r", False)]


def load(args):
    if args.res:
        return json.load(open(args.res))
    parts, end = {}, None
    for path in args.log:
        txt = open(path, encoding="utf-8").read()
        try:
            lines = [x["line"] for x in json.loads(txt)["items"]]
        except (ValueError, KeyError, TypeError):
            lines = txt.splitlines()
        for line in lines:
            m = re.search(r"DXB (\d+)/(\d+) ([0-9a-f]{8}) (\S+)$", line)
            if m and hashlib.md5(m.group(4).encode()).hexdigest()[:8] == m.group(3):
                parts[int(m.group(1))] = (int(m.group(2)), m.group(4))
            m = re.search(r"DXB END n=(\d+) bytes=(\d+) md5=([0-9a-f]{32}) raw=([0-9a-f]{32})", line)
            if m:
                end = (int(m.group(1)), int(m.group(2)), m.group(3), m.group(4))
    if end is None or any(k not in parts for k in range(1, end[0] + 1)):
        sys.exit("DXB lines incomplete")
    comp = base64.b64decode("".join(parts[k][1] for k in range(1, end[0] + 1)))
    assert len(comp) == end[1] and hashlib.md5(comp).hexdigest() == end[2], "compressed bytes mismatch"
    raw = lzma.decompress(comp)
    assert hashlib.md5(raw).hexdigest() == end[3], "raw md5 mismatch"
    print(f"DXB OK lines {end[0]} raw md5 {end[3]}")
    return json.loads(raw)


def pct(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def judge(r_ci, l_ci):
    """section 6.2 of the registration, in this order"""
    if l_ci[0] <= 0 <= l_ci[1]:
        return "no judgment: the interval of the light arm's Delta ppl includes 0"
    lo, hi = r_ci
    if 0.8 <= lo and hi <= 1.25:
        return "small difference in composition: the interval of R lies within [0.8, 1.25]"
    if lo > 1.25:
        return "at equal mass, the heavy substitution is more harmful: lower bound of R > 1.25"
    if hi < 0.8:
        return "at equal mass, many light substitutions are more harmful: upper bound of R < 0.8"
    if lo > 1:
        return "direction only: heavy more harmful (the interval of R excludes 1); size not judged"
    if hi < 1:
        return "direction only: light more harmful (the interval of R excludes 1); size not judged"
    return "no judgment: the difference is reported only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res")
    ap.add_argument("--log", nargs="*")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    doc = load(args)
    cnt = np.array(doc["cnt"], dtype=np.float64)
    S = len(cnt)
    st = doc["streams"]
    nll = {k: np.array(v["nll"], dtype=np.float64) for k, v in st.items()}
    idx = np.random.default_rng(args.seed).integers(0, S, size=(args.B, S))
    Wb = np.stack([np.bincount(r, minlength=S) for r in idx]).astype(np.float64)       # [B, S] multiplicities

    def dppl(name, w=None):
        if w is None:
            return 100 * (np.exp((nll[name].sum() - nll["base"].sum()) / cnt.sum()) - 1)
        return 100 * (np.exp((w @ nll[name] - w @ nll["base"]) / (w @ cnt)) - 1)

    def per_token(name, key, w=None):
        v = np.array(st[name][key], dtype=np.float64)
        return v.sum() / cnt.sum() if w is None else (w @ v) / (w @ cnt)

    out = {"delta_rel": doc["meta"]["delta_rel"], "eps_rel": doc["meta"]["eps_rel"], "seed": args.seed,
           "B": args.B, "sessions": S, "scored_tokens": int(cnt.sum()), "streams": {}, "pairs": {}}
    boot = {}
    for nm in sorted(st):
        if nm == "base":
            continue
        boot[nm] = dppl(nm, Wb)
        e = {"dppl": float(dppl(nm)), "dppl_ci": pct(boot[nm]),
             "kl": float(per_token(nm, "kl")), "kl_ci": pct(per_token(nm, "kl", Wb)),
             "agree": float(per_token(nm, "agree")), "agree_ci": pct(per_token(nm, "agree", Wb)),
             "acc1": float(per_token(nm, "acc1"))}
        e.update({k: st[nm]["summary"][k] for k in st[nm]["summary"]})
        doms = doc["meta"]["session_domain"]
        e["dppl_by_domain"] = {}
        for d in doc["meta"]["domains"]:
            m = np.array([x == d for x in doms])
            e["dppl_by_domain"][d] = float(100 * (np.exp((nll[nm][m].sum() - nll["base"][m].sum()) / cnt[m].sum()) - 1))
        out["streams"][nm] = e
    out["base"] = {"ppl": float(np.exp(nll["base"].sum() / cnt.sum())), "acc1": float(per_token("base", "acc1"))}
    for ms, mode, primary in PAIRS:
        h, l = f"H{ms}{mode}", f"L{ms}{mode}"
        if h not in st or l not in st:
            continue
        r = dppl(h) / dppl(l)
        rb = boot[h] / boot[l]
        p = {"R": float(r), "R_ci": pct(rb), "light_dppl_ci": out["streams"][l]["dppl_ci"],
             "diff_dppl": float(dppl(h) - dppl(l)), "diff_dppl_ci": pct(boot[h] - boot[l]),
             "kl_ratio": float(out["streams"][h]["kl"] / out["streams"][l]["kl"]),
             "kl_ratio_ci": pct(per_token(h, "kl", Wb) / per_token(l, "kl", Wb)),
             "mass_diff_all": out["streams"][h]["mass_all"] - out["streams"][l]["mass_all"]}
        mh, ml = out["streams"][h]["mass_all"], out["streams"][l]["mass_all"]
        if mh > 0 and ml > 0:
            p["R_m"] = float((dppl(h) / mh) / (dppl(l) / ml))
            p["R_m_ci"] = pct((boot[h] / mh) / (boot[l] / ml))
        p["judgment"] = judge(p["R_ci"], p["light_dppl_ci"])
        p["role"] = "primary" if primary else "auxiliary"
        out["pairs"][f"m*={ms} {'fill' if mode == 'f' else 'renorm'}"] = p
    print("EXPB_RESULT " + json.dumps(out, separators=(",", ":"), sort_keys=True), flush=True)
    for k, p in out["pairs"].items():
        print(f"[{p['role']}] {k}: R {p['R']:.3f} [{p['R_ci'][0]:.3f}, {p['R_ci'][1]:.3f}] | light dppl CI "
              f"[{p['light_dppl_ci'][0]:+.3f}, {p['light_dppl_ci'][1]:+.3f}] | H-L {p['diff_dppl']:+.3f} "
              f"[{p['diff_dppl_ci'][0]:+.3f}, {p['diff_dppl_ci'][1]:+.3f}] | KL ratio {p['kl_ratio']:.3f} | "
              f"mass H-L {p['mass_diff_all']:+.5f} | R_m {p.get('R_m', float('nan')):.3f} -> {p['judgment']}", flush=True)
    for nm, e in out["streams"].items():
        print(f"{nm}: dppl {e['dppl']:+.3f} [{e['dppl_ci'][0]:+.3f}, {e['dppl_ci'][1]:+.3f}] kl {e['kl']:.5f} "
              f"agree {e['agree']:.4f} defined {e['defined']:.4f} mass_all {e['mass_all']:.4f} "
              f"mass_defined {e['mass_defined']:.4f} conc {e['conc_defined']:.3f} gap {e['gap_defined']:.5f} "
              f"nsub {e['nsub_defined']:.2f} rank {e['rank_mean']:.2f}", flush=True)
    print("EXPB_ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    main()
