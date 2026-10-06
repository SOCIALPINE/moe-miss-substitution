"""Local CPU check of live.py on a tiny random Qwen3-Next with an FP8-block checkpoint split over 2 shards:
convert -> engine exact decode vs HF reference (same int4 experts), all modes run, routing vs simulate()."""
import os, sys, json, shutil, argparse
import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live  # noqa: E402
from masked_route import simulate  # noqa: E402
from transformers import Qwen3NextConfig
from transformers.models.qwen3_next import modeling_qwen3_next as QN

W = "/tmp/moe_test/tlive"
torch.manual_seed(0)
shutil.rmtree(W, ignore_errors=True)
os.makedirs(W)
MD, EX = f"{W}/model", f"{W}/ex"
cfg = Qwen3NextConfig(vocab_size=1000, hidden_size=256, intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                      linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                      num_experts=32, num_experts_per_tok=4, moe_intermediate_size=128,
                      shared_expert_intermediate_size=64, max_position_embeddings=512, full_attention_interval=2,
                      norm_topk_prob=True)
cfg._attn_implementation = "sdpa"
ref = QN.Qwen3NextForCausalLM(cfg).eval()
sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
for L in range(cfg.num_hidden_layers):
    sd[f"model.layers.{L}.mlp.gate.weight"] *= 40.0          # peaky router so the cache matters
    for e in range(cfg.num_experts):
        for n in ("gate", "up", "down"):
            sd[f"model.layers.{L}.mlp.experts.{e}.{n}_proj.weight"] *= 5.0

# FP8 block checkpoint, 2 shards, some scales in the other shard, one mtp tensor to skip
out = [{}, {}]
for i, (k, v) in enumerate(sd.items()):
    sh = i % 2
    if k.endswith("proj.weight") and v.dim() == 2 and "shared_expert_gate" not in k:
        O, In = v.shape
        vb = torch.nn.functional.pad(v, (0, (-In) % 128, 0, (-O) % 128))
        blk = vb.view(vb.shape[0] // 128, 128, vb.shape[1] // 128, 128).abs().amax((1, 3)).clamp_min(1e-12) / 448.0
        sc = blk.repeat_interleave(128, 0)[:O].repeat_interleave(128, 1)[:, :In]
        out[sh][k] = (v / sc).to(torch.float8_e4m3fn)
        out[(sh + (i % 3 == 0)) % 2][k + "_scale_inv"] = blk.float()
    else:
        out[sh][k] = v.to(torch.bfloat16)
out[1]["mtp.fc.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
os.makedirs(MD)
wmap = {}
for j, d in enumerate(out):
    fn = f"model-{j + 1:05d}-of-00002.safetensors"
    save_file(d, f"{MD}/{fn}")
    wmap.update({k: fn for k in d})
json.dump({"weight_map": wmap}, open(f"{MD}/model.safetensors.index.json", "w"))
cfg.save_pretrained(MD)

a = argparse.Namespace(model_dir=MD, ex_dir=EX, flag_dir=None, device="cpu", batch=7)
live.convert(a)
live.convert(a)                                                  # idempotent

# reference model: FP8-dequantised non-experts (bf16 like the engine) + the same int4 experts
lay = live.Layout(cfg.hidden_size, cfg.moe_intermediate_size)
from safetensors.torch import load_file
ne = load_file(f"{EX}/nonexp.safetensors")
blob = np.fromfile(f"{EX}/experts.bin", dtype=np.uint8).reshape(cfg.num_hidden_layers, cfg.num_experts, lay.blob)
I = cfg.moe_intermediate_size
rsd = {k: v.float() for k, v in ne.items()}
for L in range(cfg.num_hidden_layers):
    v = lay.views(torch.from_numpy(blob[L]))
    gu = live.deq(v["guq"], v["gus"], v["gum"], torch.float32)
    dn = live.deq(v["dnq"], v["dns"], v["dnm"], torch.float32)
    for e in range(cfg.num_experts):
        rsd[f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight"] = gu[e, :I]
        rsd[f"model.layers.{L}.mlp.experts.{e}.up_proj.weight"] = gu[e, I:]
        rsd[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"] = dn[e]
    # int4 error vs the original weights (sanity)
    w0 = sd[f"model.layers.{L}.mlp.experts.3.down_proj.weight"]
    rel = float((dn[3] - w0).norm() / w0.norm())
    assert rel < 0.15, rel
m, u = ref.load_state_dict(rsd, strict=False)
assert not m and not u, (m[:4], u[:4])
print("convert ok; int4 rel err (layer 3 expert 3 down)", round(rel, 4))

# sessions
P, T, S = 24, 88, 3
rng = np.random.default_rng(0)
tokens = rng.integers(0, cfg.vocab_size, (S, T))
np.savez(f"{W}/sessions.npz", tokens=tokens, domain=np.arange(S), domain_names=np.array(["a", "b", "c"]),
         prompt_len=np.array(P))
D = T - P - 1

with torch.no_grad():
    ref_lp = torch.log_softmax(ref(input_ids=torch.from_numpy(tokens)).logits.float(), -1)
ref_nll = -ref_lp[:, P:P + D].gather(-1, torch.from_numpy(tokens[:, P + 1:P + 1 + D])[..., None])[..., 0].numpy()

# engine
dev, dt = torch.device("cpu"), torch.float32
E, K, NL = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
tier = live.Tier(f"{EX}/experts.bin", lay.blob, E)
print("O_DIRECT:", tier.direct)
print("nvme test", live.nvme_test(tier, NL, E, False, n=16))
host, hinfo = live.load_host_tier(tier, NL, E, False)
eng = live.Engine(lay, NL, E, K, 16, dev, dt, tier, host=host, n_bg=2, n_urg=3)
refl = [eng]
cfg2, model = live.build_model(MD, EX, dev, dt, refl)

res = {}
with torch.no_grad():
    for s in range(S):
        ids = torch.from_numpy(tokens[s][None])
        eng.capture_pf = {}
        snap, prec, pi = live.prefill(model, eng, cfg2, ids, P, "nvme")
        pf_lg = eng.capture_pf
        eng.capture_pf = None
        snap_h, prec_h, _ = live.prefill(model, eng, cfg2, ids, P, "host")
        assert np.array_equal(prec, prec_h)
        for mode, cap, src, tau, sync in (("exact", 8, "nvme", 0, False), ("exact", 16, "host", 0, False),
                                          ("C", 12, "nvme", 2.0, True), ("C", 12, "nvme", 0.15, True),
                                          ("B", 12, "nvme", 0, True), ("C", 8, "nvme", 2.0, False),
                                          ("B", 8, "host", 0, False), ("ideal", 16, "nvme", 0, False),
                                          ("ideal_sync", 16, "nvme", 0, False)):
            eng.reset(cap, src, mode, tau, prec)
            eng.sync_bg = sync
            eng.capture = {L: [] for L in range(NL)}
            tm, nll, am = live.decode_run(model, eng, ids, P, D, snap, dev)
            eng.q.join()
            key = f"{mode}@{cap}/{src}/tau{tau}/sync{int(sync)}"
            r = res.setdefault(key, {"nll": [], "S": [], "sim": []})
            r["nll"].append(nll)
            r["S"].append(dict(eng.S))
            if sync:
                sims = []
                for L in range(NL):
                    lg = np.concatenate([pf_lg[L], np.stack(eng.capture[L])])[None]
                    _, _, st = simulate(torch.from_numpy(lg), mode, "lru", cap, P, K, tau, 0.0)
                    sims.append(st)
                r["sim"].append(sims)
            eng.capture = None
            eng.sync_bg = False

ok = True
for key, r in res.items():
    nll = np.stack(r["nll"])
    S = {k: sum(x[k] for x in r["S"]) for k in r["S"][0]}
    line = f"{key:34s} ppl {np.exp(nll.mean()):9.3f}"
    if key.startswith("exact"):
        d = np.abs(nll - ref_nll).max()
        line += f" | max|nll-ref| {d:.2e}"
        ok &= d < 2e-3
    if S["n_want"]:
        hit = S["n_hit"] / S["n_want"]
        sub = S["n_sub"] / S["n_want"]
        line += f" | hit {hit:.3f} sub {sub:.3f} urg {S['n_urg']} bg {S['n_bg']} drop {S['n_drop']} " \
                f"cancel {S['n_cancel']} steal {S['n_steal']}"
        if r["sim"]:
            sh = np.mean([st["wanted_hit"] for ss in r["sim"] for st in ss])
            ssub = np.mean([st["subst"] for ss in r["sim"] for st in ss])
            line += f" || simulate hit {sh:.3f} sub {ssub:.3f}"
            ok &= abs(sh - hit) < 0.03 and abs(ssub - sub) < 0.03
    print(line)
print("REF ppl", np.exp(ref_nll.mean()))
print("ALL OK" if ok else "MISMATCH")
