"""Chunked xcache-bench runs (round 12, Part 18): a long run is split into chunks of whole sessions so that an interrupted
pod keeps the chunks it finished, and the chunk outputs are merged back into one run file afterwards.

  merge_runs.py split TOKENS.bin N PREFIX   write PREFIX0.bin .. PREFIX{N-1}.bin (S/N sessions each, same T and P)
  merge_runs.py ok RESDIR NAME              exit 0 when NAME.json and NAME.json.raw.json are complete
  merge_runs.py merge RESDIR NAME N         NAME_0 .. NAME_{N-1} -> NAME.json + NAME.json.raw.json

Every chunk has the same number of sessions and generated tokens, so the per-token and per-session averages in the
summary (hit, mass, MB_per_tok, ...) merge as plain means. Whatever depends on the token values (ms_mean, ms_p50/p90,
tok_s, ppl, dppl_pct, agree and the quarter values of ms, ppl and dppl) is computed again from the merged raw arrays."""
import json, math, os, sys
import numpy as np


def split(src, nc, prefix):
    S, T, P = (int(x) for x in np.fromfile(src, np.int32, 3))
    t = np.fromfile(src, np.int32, offset=12).reshape(S, T)
    per = S // nc
    assert per * nc == S, (S, nc)
    for c in range(nc):
        p = f"{prefix}{c}.bin"
        if not os.path.exists(p):
            open(p, "wb").write(np.array([per, T, P], np.int32).tobytes() + np.ascontiguousarray(t[c * per:(c + 1) * per]).tobytes())
    print("split", src, S, "->", nc, "x", per, flush=True)


def ok(rd, name):
    try:
        d = json.load(open(os.path.join(rd, name + ".json")))
        r = json.load(open(os.path.join(rd, name + ".json.raw.json")))
        return len(d["runs"]) > 0 and set(d["runs"]) == set(r) and all(len(v["nll"]) == d["sessions"] * d["D"] for v in r.values())
    except Exception:
        return False


def merge(rd, name, nc):
    parts = [f"{name}_{c}" for c in range(nc)]
    bad = [p for p in parts if not ok(rd, p)]
    if bad:
        print("MERGE MISSING", name, bad, flush=True)
        return 1
    ds = [json.load(open(os.path.join(rd, p + ".json"))) for p in parts]
    rs = [json.load(open(os.path.join(rd, p + ".json.raw.json"))) for p in parts]
    keys = list(ds[0]["runs"])
    assert all(list(d["runs"]) == keys and d["D"] == ds[0]["D"] and d["sessions"] == ds[0]["sessions"] for d in ds)
    raw = {k: {f: sum((r[k][f] for r in rs), []) for f in ("nll", "am", "ms")} for k in keys}
    out = {x: ds[0][x] for x in ds[0] if x not in ("runs", "ttft_ms", "ttft_by_p", "load_s", "sessions")}
    out["sessions"] = sum(d["sessions"] for d in ds)
    out["chunks"] = nc
    out["load_s"] = [d["load_s"] for d in ds]
    out["ttft_ms"] = sum((d["ttft_ms"] for d in ds), [])
    out["ttft_by_p"] = {p: {f: sum((d["ttft_by_p"][p][f] for d in ds), []) for f in ds[0]["ttft_by_p"][p]} for p in ds[0]["ttft_by_p"]}
    D = int(out["D"])
    ref = next((k for k in keys if k.startswith(("exact@", "stock"))), None)
    runs = {}
    for k in keys:
        v = {}
        for f in ds[0]["runs"][k]:
            vals = [d["runs"][k][f] for d in ds]
            v[f] = [float(np.mean([x[i] for x in vals])) for i in range(len(vals[0]))] if isinstance(vals[0], list) else float(np.mean(vals))
        ms, nll, am = (np.array(raw[k][f], float) for f in ("ms", "nll", "am"))
        v.update(ms_mean=float(ms.mean()), ms_p50=float(np.percentile(ms, 50)), ms_p90=float(np.percentile(ms, 90)),
                 tok_s=1000.0 / float(ms.mean()), ppl=math.exp(float(nll.mean())))
        q = np.minimum(3, (np.arange(len(ms)) % D) * 4 // D)
        v["q_ms"] = [float(ms[q == b].mean()) for b in range(4)]
        v["q_ppl"] = [math.exp(float(nll[q == b].mean())) for b in range(4)]
        if ref is not None:
            rn = np.array(raw[ref]["nll"], float)
            v["agree"] = float((am == np.array(raw[ref]["am"], float)).mean())
            v["dppl_pct"] = 100.0 * (math.exp(float(nll.mean() - rn.mean())) - 1.0)
            v["q_dppl"] = [100.0 * (math.exp(float((nll[q == b] - rn[q == b]).mean())) - 1.0) for b in range(4)]
        runs[k] = v
    out["runs"] = runs
    json.dump(out, open(os.path.join(rd, name + ".json"), "w"), separators=(",", ":"))
    json.dump(raw, open(os.path.join(rd, name + ".json.raw.json"), "w"), separators=(",", ":"))
    print("MERGED", name, nc, "chunks", out["sessions"], "sessions", flush=True)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "split":
        split(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif cmd == "ok":
        sys.exit(0 if ok(sys.argv[2], sys.argv[3]) else 1)
    elif cmd == "merge":
        sys.exit(merge(sys.argv[2], sys.argv[3], int(sys.argv[4])))
