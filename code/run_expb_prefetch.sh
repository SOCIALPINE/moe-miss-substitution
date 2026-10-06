#!/bin/bash
# Experiment B, stage 1 (no model computation): fetch the BF16 checkpoint and build the session pool on the pod's volume,
# so that stage 2 (run_expb.sh, after PREREG_expB.md is fixed) only computes. Prints the model and dataset revisions.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME=/workspace/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
W=/workspace; MD=$W/model; MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
echo "=== PREFETCH START $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus=$(nproc)"; free -g | sed -n 2p; df -h $W | tail -1; df -h / | tail -1
python3 -c "import torch; print('TORCH', torch.__version__, torch.version.cuda); torch.zeros(1).cuda(); print('CUDA OK')"
python3 -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer "transformers==4.57.6" pyarrow 2>&1 | grep -vi warn | tail -2
python3 - <<PY
from huggingface_hub import HfApi
api = HfApi()
print("MODEL_REV", api.model_info("$MODEL").sha, flush=True)
for r in ["wikimedia/wikipedia", "codeparrot/codeparrot-clean-valid", "HuggingFaceH4/ultrachat_200k", "maywell/koVast", "AI-MO/NuminaMath-CoT"]:
    try:
        print("DATA_REV", r, api.dataset_info(r).sha, flush=True)
    except Exception as e:
        print("DATA_REV", r, "error", type(e).__name__, flush=True)
PY
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='$MD', allow_patterns=['*.json','*.txt','tokenizer*','vocab*','merges*']); print('tokenizer/config ready')"
( t0=$(date +%s); for i in 1 2 3; do python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='$MD', max_workers=16)" && touch $W/download.done && break; echo "download retry $i"; sleep 10; done; echo "MODEL DOWNLOAD END $(( $(date +%s) - t0 ))s" ) > $W/download.log 2>&1 &
DL=$!
( while kill -0 $DL 2>/dev/null; do sleep 60; echo "[dl $(date -u +%T)] $(du -sh $MD 2>/dev/null | cut -f1)"; done ) &
if [ ! -f $W/data/sessions.npz ]; then
  for i in 1 2 3; do timeout 1800 python3 prep_data.py --model_dir $MD --out $W/data > $W/prep.log 2>&1 && break; echo "prep retry $i"; tail -3 $W/prep.log; done
  grep -E "saved|sessions|packed|!!|Error" $W/prep.log | tail -12
fi
python3 -c "import numpy as np, hashlib; d=np.load('$W/data/sessions.npz', allow_pickle=True); print('SESSIONS', d['tokens'].shape, [str(x) for x in d['domain_names']], [int((d['domain']==k).sum()) for k in range(len(d['domain_names']))], hashlib.md5(d['tokens'].astype('int32').tobytes()).hexdigest(), flush=True)"
wait $DL; cat $W/download.log | tail -4
du -sh $MD
python3 - <<'PY'
import json, os
d = "/workspace/model"
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
files = sorted(set(idx["weight_map"].values()))
miss = [f for f in files if not os.path.exists(os.path.join(d, f))]
print("SHARDS", len(files), "missing", len(miss), miss[:5], flush=True)
PY
[ -f $W/download.done ] && echo "=== PREFETCH DONE $(date -u) ===" || echo "=== PREFETCH FAILED $(date -u) ==="
