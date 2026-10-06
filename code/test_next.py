"""Local check: trace_next.py layer-wise forward == transformers Qwen3NextForCausalLM (tiny random model)."""
import os, sys, json, subprocess, shutil
import numpy as np
import torch
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

D = "/tmp/moe_test/tiny_qwen3next"
shutil.rmtree(D, ignore_errors=True)
os.makedirs(D)
torch.manual_seed(0)
cfg = Qwen3NextConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
                      shared_expert_intermediate_size=48, num_hidden_layers=4, full_attention_interval=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32, partial_rotary_factor=0.25,
                      linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                      linear_value_head_dim=16, linear_conv_kernel_dim=4, num_experts=24, num_experts_per_tok=5,
                      norm_topk_prob=True, rope_theta=1e7, rms_norm_eps=1e-6, tie_word_embeddings=False,
                      max_position_embeddings=512)
cfg._attn_implementation = "eager"
m = Qwen3NextForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if n.endswith("mlp.gate.weight"):
            p.normal_(0, 1.0)
        elif "norm" in n and p.dim() == 1:
            p.uniform_(-0.5, 0.5)       # (1+w) style norms: exercise non-trivial weights
        elif p.dim() == 2:
            p.mul_(3.0)
m.save_pretrained(D, safe_serialization=True)
print("saved:", os.listdir(D))

S, T = 6, 80
rng = np.random.default_rng(0)
tokens = rng.integers(0, 1000, size=(S, T)).astype(np.int32)
np.savez(os.path.join(D, "sessions.npz"), tokens=tokens, domain=np.array([0, 0, 1, 1, 2, 2]),
         domain_names=np.array(["a", "b", "c"]), prompt_len=np.array(16))
out = os.path.join(D, "out")
r = subprocess.run([sys.executable, "trace_next.py", "--model_dir", D, "--data", os.path.join(D, "sessions.npz"),
                    "--out", out, "--device", "cpu", "--dtype", "fp32", "--attn_batch", "4", "--chunk", "100",
                    "--save_logits", os.path.join(out, "logits")],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
assert r.returncode == 0

idx = np.load(os.path.join(out, "trace_idx.npy")).astype(np.int64)
nll_mine = np.load(os.path.join(out, "nll_sess.npy"))
with torch.no_grad():
    o = m(torch.from_numpy(tokens).long(), output_router_logits=True)
    lg = o.logits.float()
    nll_hf = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, 1000), torch.from_numpy(tokens[:, 1:]).long().reshape(-1),
                                               reduction="none").view(S, T - 1).mean(1).numpy()
    mism = tot = 0
    lg_err = 0.0
    for L, rl in enumerate(o.router_logits):
        ti = torch.softmax(rl.float(), -1).topk(cfg.num_experts_per_tok, -1).indices.view(S, T, -1).numpy()
        mism += (np.sort(idx[:, :, L], -1) != np.sort(ti, -1)).any(-1).sum(); tot += S * T
        mine = np.load(os.path.join(out, "logits", f"logits_{L:02d}.npy")).astype(np.float32)
        ref = rl.float().view(S, T, -1).numpy()
        lg_err = max(lg_err, float(np.abs(mine - ref).max() / (np.abs(ref).max() + 1e-9)))
    print("saved router logits: max rel err vs HF", lg_err)
    assert lg_err < 2e-3
print("routing mismatched tokens:", mism, "/", tot)
print("nll hf  :", np.round(nll_hf, 5))
print("nll mine:", np.round(nll_mine, 5))
st = json.load(open(os.path.join(out, "trace_stats.json")))
for Dd, v in st["lookahead"].items():
    print("D", Dd, [round(a, 4) for a in v["recall_mean"]], v["m_list"])
assert mism == 0
assert np.abs(nll_hf - nll_mine).max() < 1e-3
assert abs(st["lookahead"]["0"]["recall_mean"][0] - 1.0) < 1e-9
print("NEXT TRACE TEST PASSED")
