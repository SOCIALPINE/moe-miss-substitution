"""Local check of mr_run.py's gpt-oss adapter against transformers on a tiny random GptOss model.
Two checkpoints: plain bf16/fp32 expert tensors, and MXFP4 blocks/scales (dequantized by the adapter)."""
import os, sys, json, shutil, subprocess
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

D = "/tmp/moe_test/tiny_gptoss"
shutil.rmtree(D, ignore_errors=True); os.makedirs(D)
torch.manual_seed(0)
cfg = GptOssConfig(vocab_size=1000, hidden_size=64, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=16, num_local_experts=16, num_experts_per_tok=4, sliding_window=8,
                   layer_types=["sliding_attention", "full_attention"] * 2, rope_theta=150000,
                   rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                                 "original_max_position_embeddings": 4096, "truncate": False},
                   max_position_embeddings=131072, tie_word_embeddings=False, attention_bias=True)
cfg._attn_implementation = "eager"

# MXFP4 blocks/scales for the experts -> the reference model uses exactly their dequantization
E, H, I = 16, 64, 64
g = torch.Generator().manual_seed(1)
blk = {"gate_up_proj": torch.randint(0, 256, (E, 2 * I, H // 32, 16), dtype=torch.uint8, generator=g),
       "down_proj": torch.randint(0, 256, (E, H, I // 32, 16), dtype=torch.uint8, generator=g)}
scl = {k: torch.randint(120, 125, v.shape[:-1], dtype=torch.uint8, generator=g) for k, v in blk.items()}

m = GptOssForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if "sinks" in n or "router" in n:
            p.normal_(0, 1.0)
        elif p.dim() >= 2:
            p.normal_(0, 0.3)
        elif "bias" in n:
            p.normal_(0, 0.1)
    for L in range(cfg.num_hidden_layers):
        ex = m.model.layers[L].mlp.experts
        ex.gate_up_proj.copy_(convert_moe_packed_tensors(blk["gate_up_proj"], scl["gate_up_proj"], dtype=torch.float32))
        ex.down_proj.copy_(convert_moe_packed_tensors(blk["down_proj"], scl["down_proj"], dtype=torch.float32))
m.save_pretrained(D, safe_serialization=True)

# second checkpoint: same weights, experts stored as MXFP4 blocks/scales (like openai/gpt-oss-120b)
D2 = D + "_mx"
shutil.rmtree(D2, ignore_errors=True); shutil.copytree(D, D2)
sd = {}
for f in os.listdir(D2):
    if f.endswith(".safetensors"):
        sd.update(load_file(os.path.join(D2, f)))
        os.remove(os.path.join(D2, f))
for L in range(cfg.num_hidden_layers):
    for proj in ("gate_up_proj", "down_proj"):
        del sd[f"model.layers.{L}.mlp.experts.{proj}"]
        sd[f"model.layers.{L}.mlp.experts.{proj}_blocks"] = blk[proj].clone()
        sd[f"model.layers.{L}.mlp.experts.{proj}_scales"] = scl[proj].clone()
save_file(sd, os.path.join(D2, "model.safetensors"))
for f in os.listdir(D2):
    if f.endswith(".index.json"):
        os.remove(os.path.join(D2, f))

S, T, P = 6, 64, 16
tokens = np.random.default_rng(0).integers(0, 1000, size=(S, T)).astype(np.int32)
np.savez(os.path.join(D, "sessions.npz"), tokens=tokens, domain=np.array([0, 0, 1, 1, 2, 2]),
         domain_names=np.array(["a", "b", "c"]), prompt_len=np.array(P))

with torch.no_grad():
    lg = m(torch.from_numpy(tokens).long()).logits.float()
sel = [1, 3, 5]
tok = torch.from_numpy(tokens[sel]).long()
lp = torch.log_softmax(lg[sel][:, P:-1], -1)
ppl_hf = float(torch.exp(-lp.gather(2, tok[:, P + 1:, None])[..., 0].mean()))

for d in (D, D2):
    out = d + "/mr"
    r = subprocess.run([sys.executable, "mr_run.py", "--adapter", "gpt_oss", "--model_dir", d, "--data",
                        os.path.join(D, "sessions.npz"), "--out", out, "--device", "cpu", "--dtype", "fp32",
                        "--attn_batch", "2", "--chunk", "64", "--per_domain", "1",
                        "--configs", "1.0:lru:B,1.0:lru:C,0.5:lru:B,0.5:lru:C", "--tau", "0.3"],
                       capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-2500:])
    assert r.returncode == 0
    st = json.load(open(out + "/mr.json"))["streams"]
    print(d, "ppl hf", ppl_hf, "base", st["base"]["ppl"]["ALL"])
    assert abs(ppl_hf - st["base"]["ppl"]["ALL"]) / ppl_hf < 1e-4
    for k in ("1.0:lru:B", "1.0:lru:C"):
        assert st[k]["kl_nats"] < 1e-6 and st[k]["routing"]["subst_rate"] == 0
    assert st["0.5:lru:B"]["kl_nats"] > 0 and st["0.5:lru:B"]["routing"]["stall_mb_per_tok"] == 0
print("GPT-OSS ADAPTER TEST PASSED")
