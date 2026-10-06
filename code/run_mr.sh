#!/bin/bash
# Cache-constrained routing (B/C) on Qwen3-Next-80B-A3B. Runs while the checkpoint downloads (waits per shard).
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
PY=python3; for c in python python3; do if $c -c "import torch" 2>/dev/null; then PY=$c; break; fi; done
W=/workspace; J=$W/job; MD=$W/model; MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
cd $J
echo "=== JOB START $(date -u) ==="
if [ -f $W/mr/masked_route.json ]; then grep -h "^SECTION_MR" $W/mr.log; echo "=== ALL DONE (cached) ==="; exit 0; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus=$(nproc)"; free -g | sed -n 2p; df -h $W | tail -1
$PY -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer 2>&1 | grep -vi warn | tail -2
$PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$MD', allow_patterns=['*.json','*.txt','tokenizer*','vocab*','merges*'])
print('tokenizer/config ready')"
dl() {
$PY -c "
import time; t=time.time()
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$MD', max_workers=16)
print(f'MODEL DOWNLOAD DONE in {time.time()-t:.0f}s', flush=True)"
}
rm -f $W/download.done $W/download.failed
( if dl > $W/download.log 2>&1 || dl >> $W/download.log 2>&1; then touch $W/download.done; else touch $W/download.failed; fi ) &
DLPID=$!
( while [ ! -f $W/download.done ] && [ ! -f $W/download.failed ]; do sleep 60; echo "[dl] $(du -sh $MD 2>/dev/null | cut -f1)"; done ) &
$PY -m pip install -q "transformers==4.57.6" pyarrow 2>&1 | grep -vi warn | tail -2
if [ ! -f $W/data/sessions.npz ]; then
  $PY prep_data.py --model_dir $MD --out $W/data || { echo "PREP FAILED"; exit 1; }
fi
ARGS="--model_dir $MD --data $W/data/sessions.npz --out $W/mr --dl_flag_dir $W --per_domain 8"
if ! $PY masked_route.py $ARGS 2>&1 | tee $W/mr.log; then :; fi
if [ ! -f $W/mr/masked_route.json ]; then
  echo "MASKED ROUTE FAILED during download; waiting for download and retrying"
  wait $DLPID; tail -3 $W/download.log
  $PY masked_route.py $ARGS 2>&1 | tee $W/mr.log
fi
[ -f $W/mr/masked_route.json ] || { echo "MASKED ROUTE FAILED"; exit 1; }
echo "=== ALL DONE $(date -u) ==="
