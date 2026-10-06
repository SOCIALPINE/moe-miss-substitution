"""Local check: my layer-wise forward == transformers Qwen3MoeForCausalLM on a tiny random model."""
import os, sys, json, subprocess, shutil
import numpy as np
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

D = "/tmp/moe_test/tiny_qwen3moe"
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)
torch.manual_seed(0)
cfg = Qwen3MoeConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                     num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                     num_experts=16, num_experts_per_tok=4, norm_topk_prob=True, rope_theta=1e7,
                     rms_norm_eps=1e-6, tie_word_embeddings=False, max_position_embeddings=512,
                     attn_implementation="eager")
m = Qwen3MoeForCausalLM(cfg).eval()
with torch.no_grad():
    for L in range(cfg.num_hidden_layers):
        m.model.layers[L].mlp.gate.weight.mul_(40.0)  # make routing decisive
        m.model.layers[L].post_attention_layernorm.weight.uniform_(0.5, 1.5)
        m.model.layers[L].self_attn.q_norm.weight.uniform_(0.5, 1.5)
    for p in m.parameters():
        if p.dim() == 2:
            p.mul_(3.0)
m.save_pretrained(D, safe_serialization=True)
print("saved files:", os.listdir(D))

S, T = 6, 64
rng = np.random.default_rng(0)
tokens = rng.integers(0, 1000, size=(S, T)).astype(np.int32)
dom = np.array([0, 0, 1, 1, 2, 2])
np.savez(os.path.join(D, "sessions.npz"), tokens=tokens, domain=dom,
         domain_names=np.array(["a", "b", "c"]), prompt_len=np.array(16))

out = os.path.join(D, "out")
r = subprocess.run([sys.executable, "trace_routes.py", "--model_dir", D, "--data", os.path.join(D, "sessions.npz"),
                    "--out", out, "--device", "cpu", "--dtype", "fp32", "--attn_batch", "4", "--chunk", "100"],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
assert r.returncode == 0

idx = np.load(os.path.join(out, "trace_idx.npy"))     # [S,T,NL,K]
w = np.load(os.path.join(out, "trace_w.npy"))
nll_mine = np.load(os.path.join(out, "nll_sess.npy"))
with torch.no_grad():
    o = m(torch.from_numpy(tokens).long(), output_router_logits=True, labels=None)
    logits = o.logits.float()
    nll_hf = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 1000), torch.from_numpy(tokens[:, 1:]).long().reshape(-1),
                                               reduction="none").view(S, T - 1).mean(1).numpy()
    mism = 0; tot = 0; wdiff = 0
    for L, rl in enumerate(o.router_logits):
        pr = torch.softmax(rl.float(), -1)
        tw, ti = pr.topk(4, -1)
        tw = tw / tw.sum(-1, keepdim=True)
        ti = ti.view(S, T, 4).numpy(); tw = tw.view(S, T, 4).numpy()
        mine = idx[:, :, L, :]
        mism += (np.sort(mine, -1) != np.sort(ti, -1)).any(-1).sum(); tot += S * T
        wdiff = max(wdiff, np.abs(w[:, :, L, :].astype(np.float32) - tw).max())
print("routing mismatched tokens:", mism, "/", tot, " max weight diff:", wdiff)
print("nll hf  :", np.round(nll_hf, 5))
print("nll mine:", np.round(nll_mine, 5))
st = json.load(open(os.path.join(out, "trace_stats.json")))
for Dd, v in st["lookahead"].items():
    print("D", Dd, [round(a, 4) for a in v["recall_mean"]], v["m_list"])
assert mism == 0, "routing mismatch"
assert np.abs(nll_hf - nll_mine).max() < 1e-3, "loss mismatch"
assert abs(st["lookahead"]["0"]["recall_mean"][0] - 1.0) < 1e-9, "D=0 must be exact"
print("TRACE TEST PASSED")
