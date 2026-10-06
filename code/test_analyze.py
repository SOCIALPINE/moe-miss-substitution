import sys, os, subprocess, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze import sim_opt, sim_lru, sim_lfu


def naive_opt(seq, cap):
    cache = set(); hits = []
    for i, v in enumerate(seq):
        def nu(x):
            for j in range(i + 1, len(seq)):
                if seq[j] == x:
                    return j
            return 1 << 62
        if v in cache:
            hits.append(1); continue
        hits.append(0)
        if len(cache) < cap:
            cache.add(v); continue
        far = max(cache, key=nu)
        if nu(v) < nu(far):
            cache.remove(far); cache.add(v)
    return np.array(hits)


def naive_lru(seq, cap):
    cache = []; hits = []
    for v in seq:
        if v in cache:
            hits.append(1); cache.remove(v); cache.insert(0, v)
        else:
            hits.append(0)
            if len(cache) == cap:
                cache.pop()
            cache.insert(0, v)
    return np.array(hits)


def naive_lfu(seq, cap):
    cnt = {}; key = {}; cache = set(); hits = []
    for i, v in enumerate(seq):
        cnt[v] = cnt.get(v, 0) + 1
        k = (cnt[v], i)
        if v in cache:
            hits.append(1); key[v] = k; continue
        hits.append(0)
        if len(cache) < cap:
            cache.add(v); key[v] = k; continue
        mn = min(cache, key=lambda x: key[x])
        if k > key[mn]:
            cache.remove(mn); cache.add(v); key[v] = k
    return np.array(hits)


rng = np.random.default_rng(1)
for trial in range(300):
    nid = int(rng.integers(3, 20)); n = int(rng.integers(1, 120)); cap = int(rng.integers(1, nid + 1))
    p = rng.dirichlet(np.ones(nid) * 0.5)
    seq = rng.choice(nid, size=n, p=p).astype(np.int32)
    for f, g in ((sim_opt, naive_opt), (sim_lru, naive_lru), (sim_lfu, naive_lfu)):
        h = np.zeros(n, np.uint8); f(seq, cap, nid, h)
        ref = g(list(seq), cap)
        assert (h == ref).all(), (f.__name__, seq, cap, h, ref)
print("policy simulators match naive references (300 random cases x 3)")

D = "/tmp/moe_test/tiny_qwen3moe"
r = subprocess.run([sys.executable, "analyze.py", "--trace_dir", D + "/out", "--data", D + "/sessions.npz",
                    "--out", D + "/res"], capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-4000:])
assert r.returncode == 0
res = json.load(open(D + "/res/results.json"))
for f, fr in res["cache"].items():
    opt = fr["OPT_g"]["after_prompt"]["ALL"]
    for k, v in fr.items():
        assert v["after_prompt"]["ALL"] <= opt + 1e-9 or k in ("oracle_static", "static_domain", "static_global", "prompt_fingerprint"), (f, k)
print(json.dumps(res["maps"], indent=0)[:1500])
print(json.dumps(res["locality"]["reuse_next_token"]))
print("ANALYZE TEST PASSED")
