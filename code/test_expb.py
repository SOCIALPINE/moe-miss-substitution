"""Local checks of expb.py and analyze_b.py on a tiny random Qwen3Next (CPU, fp32):
1. pick_pair() equals a plain-python enumeration of all pairs (defined flag, heavy and light subsets, gap), and its
   result does not depend on the chunking
2. arm_route(): fill excludes exactly the substituted experts and refills from outside W; renorm drops and rescales;
   positions where nothing is substituted are untouched
3. end to end: base NLL equals transformers; arms whose m* has no feasible pair equal base exactly; realised masses lie
   within m* +- delta and the heavy arm is more concentrated; the run is deterministic; the DXB log lines rebuild the
   same document; analyze_b.py gives the same result from the log and from the json."""
import os, sys, json, subprocess, tempfile
import numpy as np
import torch
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expb  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. pick_pair() against an enumeration of pairs
K = 10
Bm = expb.subset_matrix(K - 1, "cpu")
NB = Bm.shape[0]
M = 600
G = torch.from_numpy(np.stack([rng.dirichlet(np.full(K, a)) for a in rng.uniform(0.3, 3.0, M)])).float()
G = G.sort(-1, descending=True).values
gc = G[:, 1:].contiguous()
nbad = ndef = 0
for ms in (0.05, 0.10, 0.15):
    dl, ep = 0.2 * ms, 0.01 * ms
    anyf, ok, ha, la, gap = expb.pick_pair(gc, Bm, ms, dl, ep)
    a2 = expb.pick_pair(gc, Bm, ms, dl, ep, budget=3000)                 # tiny sub-chunks
    assert all(torch.equal(x, y) for x, y in zip((anyf, ok, ha, la, gap), a2)), "chunking changed the result"
    s32 = gc @ Bm.T
    c32 = ((gc * gc) @ Bm.T) / (s32 * s32)                 # the same fp32 values as pick_pair, so that the check is of the
    for i in range(M):                                     # selection logic, not of fp32 versus fp64 rounding
        F = [r for r in range(NB) if abs(float(s32[i, r]) - ms) <= dl]
        best, pair = 0.0, None
        for a in F:                                        # pairs in row order: the first strict maximum wins
            for b in F:
                if abs(float(s32[i, a]) - float(s32[i, b])) <= ep:
                    d = float(c32[i, a]) - float(c32[i, b])
                    if d > best:
                        best, pair = d, (a, b)
        want = (len(F) > 0, pair is not None, pair[0] if pair else 0, pair[1] if pair else 0)
        got = (bool(anyf[i]), bool(ok[i]), int(ha[i]), int(la[i]))
        if want != got:
            nbad += 1
        if pair:
            ndef += 1
            assert abs(float(gap[i]) - abs(float(s32[i, pair[0]]) - float(s32[i, pair[1]]))) < 1e-7
            sa = [float(gc[i, j]) for j in range(K - 1) if int(Bm[pair[0], j])]
            sb = [float(gc[i, j]) for j in range(K - 1) if int(Bm[pair[1], j])]
            assert abs(sum(sa) - ms) <= dl + 1e-6 and abs(sum(sb) - ms) <= dl + 1e-6 and abs(sum(sa) - sum(sb)) <= ep + 1e-6
print(f"1. pick_pair vs enumeration: {nbad} mismatches in {3 * M} (defined {ndef})")
assert nbad == 0 and ndef > 300

# ---------------------------------------------------------------- 2. arm_route on synthetic logits
S, T, E, P = 3, 40, 64, 8
lg = torch.randn(S, T, E) * 3.0
for mode in ("fill", "renorm"):
    for arm in ("H", "L"):
        ms = 0.10
        si, sw, st = expb.arm_route(lg, P, K, Bm, ms, 0.2 * ms, 0.01 * ms, arm, mode)
        idx, g = expb.topk_gates(lg, K)
        assert torch.equal(si[:, :P], idx[:, :P]) and torch.equal(sw[:, :P], g[:, :P])
        anyf, ok, ha, la, gap = expb.pick_pair(g[:, P:, 1:].reshape(-1, K - 1), Bm, ms, 0.2 * ms, 0.01 * ms)
        ch = (ha if arm == "H" else la).view(S, T - P)
        ok = ok.view(S, T - P)
        for s in range(S):
            for t in range(T - P):
                W = idx[s, P + t].tolist()
                if not ok[s, t]:
                    assert si[s, P + t].tolist() == W and torch.equal(sw[s, P + t], g[s, P + t])
                    continue
                drop = {W[1 + j] for j, b in enumerate(Bm[ch[s, t]].tolist()) if b}
                assert W[0] not in drop and 1 <= len(drop) <= K - 1
                m = sum(float(g[s, P + t, W.index(e)]) for e in drop)
                assert abs(m - ms) <= 0.2 * ms + 1e-6
                if mode == "fill":
                    z = lg[s, P + t].clone(); z[list(drop)] = float("-inf")
                    v, i = z.topk(K)
                    assert si[s, P + t].tolist() == i.tolist() and not (set(i.tolist()) & drop)
                    assert set(W) - drop <= set(i.tolist())
                    assert torch.allclose(sw[s, P + t], torch.softmax(v, -1), atol=1e-6)
                else:
                    assert si[s, P + t].tolist() == W
                    w = sw[s, P + t]
                    assert abs(float(w.sum()) - 1) < 1e-5
                    for j, e in enumerate(W):
                        if e in drop:
                            assert float(w[j]) == 0.0
                        else:
                            assert abs(float(w[j]) - float(g[s, P + t, j]) / (1 - m)) < 1e-5
        assert st["defined"] == int(ok.sum()) and st["defined"] > 0
print("2. arm_route fill/renorm: ok")

# ---------------------------------------------------------------- 3. end to end on a tiny model
D = tempfile.mkdtemp(prefix="tiny_expb_")
cfg = Qwen3NextConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                      shared_expert_intermediate_size=48, num_hidden_layers=4, full_attention_interval=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32, partial_rotary_factor=0.25,
                      linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                      linear_value_head_dim=16, linear_conv_kernel_dim=4, num_experts=48, num_experts_per_tok=10,
                      norm_topk_prob=True, rope_theta=1e7, rms_norm_eps=1e-6, tie_word_embeddings=False,
                      max_position_embeddings=512)
cfg._attn_implementation = "eager"
m = Qwen3NextForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith("mlp.gate.weight"):
            p.normal_(0, 1.0)
        elif "norm" in n and p.dim() == 1:
            p.uniform_(-0.5, 0.5)
        elif p.dim() == 2:
            p.mul_(3.0)
m.save_pretrained(D, safe_serialization=True)
ndom, per, T, P = 3, 3, 72, 16
tokens = rng.integers(0, 1000, size=(ndom * per, T)).astype(np.int32)
np.savez(os.path.join(D, "sessions.npz"), tokens=tokens, domain=np.repeat(np.arange(ndom), per),
         domain_names=np.array(["a", "b", "c"]), prompt_len=np.array(P))
arms = "H:0.10:fill,L:0.10:fill,H:0.05:fill,L:0.05:fill,H:0.15:fill,L:0.15:fill,H:0.10:renorm,L:0.10:renorm,H:9:fill,L:9:renorm"


def run(out):
    r = subprocess.run([sys.executable, os.path.join(HERE, "expb.py"), "--model_dir", D, "--data",
                        os.path.join(D, "sessions.npz"), "--out", out, "--device", "cpu", "--dtype", "fp32",
                        "--attn_batch", "2", "--chunk", "64", "--idx", "1,2", "--arms", arms],
                       capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if not l.startswith("DXB ") or " END " in l)[-2500:], r.stderr[-1500:])
    assert r.returncode == 0
    open(os.path.join(out, "stdout.log"), "w").write(r.stdout)
    return json.load(open(os.path.join(out, "expb.json")))


d1 = run(os.path.join(D, "o1"))
d2 = run(os.path.join(D, "o2"))
assert json.dumps(d1["streams"], sort_keys=True) == json.dumps(d2["streams"], sort_keys=True), "not deterministic"
meta = d1["meta"]
sel = [i for d in range(ndom) for i in np.where(np.repeat(np.arange(ndom), per) == d)[0][[1, 2]]]
assert meta["session_pool_index"] == [int(x) for x in sel]
tok = torch.from_numpy(tokens[sel].astype(np.int64))
with torch.no_grad():
    lp = torch.log_softmax(m(tok).logits.float()[:, P:-1], -1)
nll_hf = -lp.gather(2, tok[:, P + 1:, None])[..., 0]
base = np.array(d1["streams"]["base"]["nll"])
assert np.allclose(base, nll_hf.sum(1).double().numpy(), rtol=1e-4), (base, nll_hf.sum(1))
for nm in ("H9f", "L9r"):
    s = d1["streams"][nm]
    assert s["summary"]["defined"] == 0 and np.array_equal(np.array(s["nll"]), base) and max(s["kl"]) < 1e-9, nm
for nm, s in d1["streams"].items():
    if nm == "base" or nm in ("H9f", "L9r"):
        continue
    a = meta["arms"][nm]
    sm = s["summary"]
    assert sm["defined"] > 0.05, (nm, sm)
    assert abs(sm["mass_defined"] - a["mstar"]) <= a["delta"] + 1e-6 and sm["gap_defined"] <= a["eps"] + 1e-6, (nm, sm)
    print(nm, {k: round(v, 4) for k, v in sm.items()}, "kl", round(sum(s["kl"]) / sum(d1["cnt"]), 5))
for ms in ("0.1f", "0.05f", "0.15f", "0.1r"):
    hs, ls = d1["streams"]["H" + ms]["summary"], d1["streams"]["L" + ms]["summary"]
    assert hs["conc_defined"] > ls["conc_defined"] and hs["nsub_defined"] < ls["nsub_defined"], ms
    print(ms, "mass_defined H-L", round(hs["mass_defined"] - ls["mass_defined"], 5), "mass_all H-L",
          round(hs["mass_all"] - ls["mass_all"], 5), "delta", meta["arms"]["H" + ms]["delta"])

# analysis from the json and from the log lines
r1 = subprocess.run([sys.executable, os.path.join(HERE, "analyze_b.py"), "--res", os.path.join(D, "o1", "expb.json")],
                    capture_output=True, text=True)
r2 = subprocess.run([sys.executable, os.path.join(HERE, "analyze_b.py"), "--log", os.path.join(D, "o1", "stdout.log")],
                    capture_output=True, text=True)
print("\n".join(l for l in r1.stdout.splitlines() if not l.startswith("EXPB_RESULT")), r1.stderr[-2000:])
assert r1.returncode == 0 and r2.returncode == 0, r2.stderr
res1 = [l for l in r1.stdout.splitlines() if l.startswith("EXPB_RESULT ")]
res2 = [l for l in r2.stdout.splitlines() if l.startswith("EXPB_RESULT ")]
assert res1 and res1 == res2 and "DXB OK" in r2.stdout
print("EXPB TEST PASSED")
