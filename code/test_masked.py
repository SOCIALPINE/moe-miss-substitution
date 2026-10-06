"""Local check of masked_route.py on the tiny random Qwen3Next saved by test_next.py:
- base stream NLL (decode positions) == transformers
- cap=1.0 streams (every expert resident) are identical to base (KL 0, same ppl)
- small caps change the output and report sane routing stats."""
import os, sys, json, subprocess
import numpy as np
import torch
from transformers import Qwen3NextForCausalLM

D = "/tmp/moe_test/tiny_qwen3next"
assert os.path.exists(os.path.join(D, "config.json")), "run test_next.py first"
out = os.path.join(D, "mr")
cfgs = "1.0:lru:B,1.0:fe:C,0.25:lru:B,0.25:lru:C,0.5:fe:B"
r = subprocess.run([sys.executable, "masked_route.py", "--model_dir", D, "--data", os.path.join(D, "sessions.npz"),
                    "--out", out, "--device", "cpu", "--dtype", "fp32", "--attn_batch", "2", "--chunk", "64",
                    "--per_domain", "1", "--configs", cfgs, "--tau", "0.3"], capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-3000:])
assert r.returncode == 0
res = json.load(open(os.path.join(out, "masked_route.json")))
st = res["streams"]

dat = np.load(os.path.join(D, "sessions.npz"), allow_pickle=True)
dom = dat["domain"]; P = int(dat["prompt_len"])
sel = []
for d in range(len(dat["domain_names"])):
    sel += list(np.where(dom == d)[0][1::2][:1])
tok = torch.from_numpy(dat["tokens"][sel].astype(np.int64))
m = Qwen3NextForCausalLM.from_pretrained(D, torch_dtype=torch.float32).eval()
with torch.no_grad():
    lg = m(tok).logits.float()
lp = torch.log_softmax(lg[:, P:-1], -1)
nll = -lp.gather(2, tok[:, P + 1:, None])[..., 0]
ppl_hf = float(torch.exp(nll.mean()))
print("ppl hf", ppl_hf, "ppl base", st["base"]["ppl"]["ALL"])
assert abs(ppl_hf - st["base"]["ppl"]["ALL"]) / ppl_hf < 1e-4
for k in ("1.0:lru:B", "1.0:fe:C"):
    assert st[k]["kl_nats"] < 1e-6 and abs(st[k]["dppl_pct"]["ALL"]) < 1e-3, st[k]
    assert st[k]["routing"]["subst_rate"] == 0 and st[k]["routing"]["wanted_hit"] == 1.0
for k in ("0.25:lru:B", "0.25:lru:C", "0.5:fe:B"):
    rt = st[k]["routing"]
    print(k, st[k]["dppl_pct"]["ALL"], st[k]["kl_nats"], rt)
    assert st[k]["kl_nats"] > 0 and 0 < rt["wanted_hit"] < 1
assert st["0.25:lru:B"]["routing"]["stall_mb_per_tok"] == 0
assert st["0.25:lru:C"]["routing"]["subst_rate"] < st["0.25:lru:B"]["routing"]["subst_rate"] + 0.05
print("MASKED ROUTE TEST PASSED")
