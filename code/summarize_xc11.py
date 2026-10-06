"""Round-11 tables from xcache-bench output (Part 17): the current engine against rent-or-buy admission (rb), frequency
score (fe) and static per-layer slots, Qwen3-Next-80B-A3B Q4_K_M on an RTX 3090, experts read from NVMe (ram=0).

  summarize_xc11.py RUN.json            one line per plan (speed, quality, cache counters) + S11 JSON lines
  summarize_xc11.py --final RESDIR      cross-run numbers: static-slot run against the main run's exact reference,
                                        position bins of the topic-change run (B11 JSON lines)"""
import json, math, os, sys

BASE = "aio=1+uf=1+up=1+bg=0+ram=0"


def label(key):
    mode, _, opts = key.partition("|")
    extra = [o for o in opts.split("+") if o and o not in BASE.split("+")]
    return mode.replace("/nvme", "") + ("" if not extra else " " + "+".join(extra))


def one(path):
    d = json.load(open(path))
    run = os.path.basename(path)[:-5]
    print(f"### {run}: {d['sessions']} sessions, P {d['P']}, D {d['D']}")
    print(f"{'plan':42s} {'tok/s':>6s} {'ms':>6s} {'ppl':>7s} {'dppl%':>6s} {'agree':>6s} {'hit':>6s} {'mass':>6s} "
          f"{'MB/tok':>6s} {'bg/tok':>6s} {'urg':>5s} {'stall':>5s} {'cfg':>6s}")
    for k, v in d["runs"].items():
        g = lambda x, dflt=0.0: v.get(x, dflt)
        print(f"{label(k):42s} {g('tok_s'):6.1f} {g('ms_mean'):6.2f} {g('ppl'):7.4f} {g('dppl_pct'):+6.2f} {g('agree'):6.3f} "
              f"{g('hit'):6.3f} {g('mass'):6.4f} {g('MB_per_tok'):6.1f} {g('bg_per_tok'):6.2f} {g('urg_per_tok'):5.2f} "
              f"{g('stall_ms'):5.2f} {g('cfg_ms'):6.0f}")
        keep = {x: v[x] for x in ("ms_mean", "ms_p50", "ms_p90", "tok_s", "ppl", "dppl_pct", "agree", "stall_ms", "wait_pending_ms",
                                  "cb_ms", "hit", "subst", "urg_per_tok", "bg_per_tok", "drop_per_tok", "cancel_per_tok",
                                  "steal_per_tok", "skip_per_tok", "MB_per_tok", "cfg_ms", "mass", "q_ms", "q_ppl", "q_dppl",
                                  "q_hit", "q_MB") if x in v}
        print("S11 " + json.dumps({"run": run, "key": k, **keep}, separators=(",", ":")))


def final(rd):
    raw = {}
    for run in ("main", "caps", "cross"):
        p = os.path.join(rd, run + ".json.raw.json")
        if os.path.exists(p):
            raw[run] = json.load(open(p))
    ref_key = lambda r: next((k for k in r if k.startswith("exact@")), None)
    # static slots against the main run: same sessions and tokens, so the per-token NLL lines up
    if "main" in raw and "caps" in raw:
        rk = ref_key(raw["main"])
        ref = raw["main"][rk]["nll"]
        for k, v in raw["caps"].items():
            n = min(len(ref), len(v["nll"]))
            dp = 100 * (math.exp(sum(v["nll"][:n]) / n - sum(ref[:n]) / n) - 1)
            ag = sum(a == b for a, b in zip(v["am"][:n], raw["main"][rk]["am"][:n])) / n
            same = raw["main"].get(k)
            dsame = None
            if same:
                dsame = 100 * (math.exp(sum(v["nll"][:n]) / n - sum(same["nll"][:n]) / n) - 1)
            print("F11 " + json.dumps({"run": "caps", "key": k, "n": n, "dppl_vs_exact": round(dp, 4), "agree": round(ag, 5),
                                       "dppl_vs_uniform": None if dsame is None else round(dsame, 4),
                                       "ms_mean": sum(v["ms"][:n]) / n}, separators=(",", ":")))
    # topic changes: 512-token position bins of the generated part, per sequence order
    if "cross" in raw:
        r = raw["cross"]
        rk = ref_key(r)
        meta = json.load(open(os.path.join(rd, "cross.json")))
        D = int(meta["D"])
        for k, v in r.items():
            n = len(v["nll"])
            nb = (D + 511) // 512
            bn, bnl, brn, bms = [0] * nb, [0.0] * nb, [0.0] * nb, [0.0] * nb
            for i in range(n):
                b = (i % D) // 512
                bn[b] += 1
                bnl[b] += v["nll"][i]
                brn[b] += r[rk]["nll"][i]
                bms[b] += v["ms"][i]
            print("B11 " + json.dumps({"run": "cross", "key": k, "D": D,
                                       "dppl": [round(100 * (math.exp((bnl[b] - brn[b]) / bn[b]) - 1), 3) for b in range(nb)],
                                       "ppl": [round(math.exp(bnl[b] / bn[b]), 4) for b in range(nb)],
                                       "ms": [round(bms[b] / bn[b], 3) for b in range(nb)]}, separators=(",", ":")))


if __name__ == "__main__":
    if sys.argv[1] == "--final":
        final(sys.argv[2])
    else:
        one(sys.argv[1])
