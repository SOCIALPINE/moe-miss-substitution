#!/bin/bash
# gpt-oss-120b cache-constrained routing (B/C) + dump DeepSeek-V4-Flash reference inference code to the log.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
PY=python3; for c in python python3; do if $c -c "import torch" 2>/dev/null; then PY=$c; break; fi; done
W=/workspace; J=$W/job; MD=$W/model_go; MODEL=openai/gpt-oss-120b
cd $J
echo "=== JOB START $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus=$(nproc)"; free -g | sed -n 2p; df -h $W | tail -1
$PY -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer 2>&1 | grep -vi warn | tail -2
$PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$MD', allow_patterns=['*.json','*.txt','*.jinja','tokenizer*','vocab*','merges*'])
print('tokenizer/config ready')"
dl() {
$PY -c "
import time; t=time.time()
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$MD', allow_patterns=['*.json','*.txt','*.jinja','tokenizer*','*.safetensors'], max_workers=16)
print(f'MODEL DOWNLOAD DONE in {time.time()-t:.0f}s', flush=True)"
}
rm -f $W/download.done $W/download.failed
( if dl > $W/download.log 2>&1 || dl >> $W/download.log 2>&1; then touch $W/download.done; else touch $W/download.failed; fi ) &
DLPID=$!
( while [ ! -f $W/download.done ] && [ ! -f $W/download.failed ]; do sleep 60; echo "[dl] $(du -sh $MD 2>/dev/null | cut -f1)"; done ) &

# ---- DeepSeek-V4-Flash reference code -> log (xz+base64 in 3000-char chunks)
$PY - <<'EOF'
import os, io, tarfile, lzma, base64
from huggingface_hub import hf_hub_download
fs = ["README.md", "inference/README.md", "inference/requirements.txt", "inference/config.json",
      "inference/model.py", "inference/kernel.py", "inference/convert.py", "inference/generate.py"]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tf:
    for f in fs:
        try:
            p = hf_hub_download("deepseek-ai/DeepSeek-V4-Flash", f, local_dir="/workspace/dsv4")
            tf.add(p, arcname=f)
        except Exception as e:
            print("DSV4 missing", f, e)
b = base64.b64encode(lzma.compress(buf.getvalue(), preset=9)).decode()
n = (len(b) + 2999) // 3000
for i in range(n):
    print(f"DSV4_B64 {i+1}/{n} {b[i*3000:(i+1)*3000]}", flush=True)
EOF

$PY -m pip install -q "transformers==4.57.6" pyarrow 2>&1 | grep -vi warn | tail -2
if [ ! -f $W/data_go/sessions.npz ]; then
  $PY prep_data.py --model_dir $MD --out $W/data_go || { echo "PREP FAILED"; exit 1; }
fi
ARGS="--adapter gpt_oss --model_dir $MD --data $W/data_go/sessions.npz --out $W/mr_go --dl_flag_dir $W --per_domain 8"
$PY mr_run.py $ARGS 2>&1 | tee $W/mr_go.log
if [ ! -f $W/mr_go/mr.json ]; then
  echo "RUN FAILED during download; waiting for download and retrying"
  wait $DLPID; tail -3 $W/download.log
  $PY mr_run.py $ARGS 2>&1 | tee $W/mr_go.log
fi
[ -f $W/mr_go/mr.json ] || { echo "GPT-OSS RUN FAILED"; exit 1; }
echo "=== ALL DONE $(date -u) ==="
