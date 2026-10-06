#!/bin/bash
# Real end-to-end measurement: Qwen3-Next-80B-A3B (FP8 checkpoint -> int4 expert blobs on local NVMe),
# VRAM expert cache at 25% / 12.5%, misses read with O_DIRECT. Conversion starts while shards download.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3; for c in python python3; do if $c -c "import torch" 2>/dev/null; then PY=$c; break; fi; done
W=/workspace; J=$W/job; MD=$W/model; EX=$W/ex; MODEL=${LIVE_MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct-FP8}
cd $J
echo "=== JOB START $(date -u) ==="
if [ -f $W/live/live.json ]; then echo "SECTION_LIVE $(cat $W/live/live.json | $PY -c 'import json,sys;print(json.dumps(json.load(sys.stdin),separators=(",",":")))')"; echo "=== ALL DONE (cached) ==="; exit 0; fi
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader
echo "cpus=$(nproc) $(grep -m1 'model name' /proc/cpuinfo)"; free -g | sed -n 2p; df -hT $W | tail -1
cat /sys/fs/cgroup/memory.max 2>/dev/null; lsblk -d -o NAME,ROTA,SIZE,MODEL 2>/dev/null | head -8
$PY -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
$PY -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer 2>&1 | grep -vi warn | tail -2
$PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$MD', allow_patterns=['*.json','*.txt','*.jinja','tokenizer*','vocab*','merges*'])
print('tokenizer/config ready')" || { echo "CONFIG DOWNLOAD FAILED"; exit 1; }
mkdir -p $W/flag; rm -f $W/flag/download.done $W/flag/download.failed
( for a in 1 2 3; do $PY -c "
import time; t=time.time()
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$MD', max_workers=16)
print(f'MODEL DOWNLOAD DONE in {time.time()-t:.0f}s', flush=True)" >> $W/flag/download.log 2>&1 && { touch $W/flag/download.done; exit 0; }; sleep 5; done
  touch $W/flag/download.failed ) &
( while [ ! -f $W/flag/download.done ] && [ ! -f $W/flag/download.failed ]; do sleep 60; echo "[dl] $(du -sh $MD 2>/dev/null | cut -f1) | free $(df -h $W | tail -1 | awk '{print $4}')"; done; tail -2 $W/flag/download.log ) &
$PY -m pip install -q "transformers==4.57.6" flash-linear-attention pyarrow 2>&1 | grep -vi warn | tail -3
$PY -c "import transformers, fla; print('transformers', transformers.__version__, 'fla', fla.__version__)" 2>&1 | tail -1
if [ ! -f $W/data/sessions.npz ]; then
  $PY prep_data.py --model_dir $MD --out $W/data --per_domain 4 || { echo "PREP FAILED"; exit 1; }
fi
$PY live.py convert --model_dir $MD --ex_dir $EX --flag_dir $W/flag || { echo "CONVERT FAILED"; exit 1; }
rm -f $MD/*.safetensors; sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null; free -g | sed -n 2p; df -h $W | tail -1
$PY live.py bench --model_dir $MD --ex_dir $EX --data $W/data/sessions.npz --out $W/live --per_domain 1 --decode 256 \
  2>&1 | grep -v "Warning\|warn(" || true
[ -f $W/live/live.json ] || { echo "BENCH FAILED"; exit 1; }
echo "=== ALL DONE $(date -u) ==="
