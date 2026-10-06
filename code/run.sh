#!/bin/bash
# Entry point on the RunPod pod. Idempotent: every stage caches its output on /workspace,
# so a container restart resumes instead of redoing work.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore NUMBA_NUM_THREADS=12
PY=python3; for c in python python3; do if $c -c "import torch" 2>/dev/null; then PY=$c; break; fi; done
W=/workspace; J=$W/job; MD=$W/model; MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
cd $J
echo "=== JOB START $(date -u) ==="
if [ -f $W/out/results.json ]; then $PY report.py $W/out/results.json; echo "=== ALL DONE (cached) ==="; exit 0; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus=$(nproc)"; free -g | sed -n 2p; df -h $W | tail -1
$PY -c "import torch,numpy,sys;print('torch',torch.__version__,'numpy',numpy.__version__,'py',sys.version.split()[0],'cuda',torch.cuda.is_available())"

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
print(f'MODEL DOWNLOAD DONE in {time.time()-t:.0f}s')"
}
( dl > $W/download.log 2>&1 ) &
DLPID=$!

$PY -m pip install -q "transformers==4.56.2" numba pyarrow 2>&1 | grep -vi warn | tail -2
if [ ! -f $W/data/sessions.npz ]; then
  $PY prep_data.py --model_dir $MD --out $W/data || { echo "PREP FAILED"; exit 1; }
fi

while kill -0 $DLPID 2>/dev/null; do sleep 30; echo "[dl] $(du -sh $MD 2>/dev/null | cut -f1) downloaded"; done
tail -3 $W/download.log
if ! grep -q "MODEL DOWNLOAD DONE" $W/download.log; then
  echo "download failed once, retrying"; dl || { echo "DOWNLOAD FAILED"; exit 1; }
fi

if [ ! -f $W/trace/trace_stats.json ]; then
  $PY trace_routes.py --model_dir $MD --data $W/data/sessions.npz --out $W/trace || { echo "TRACE FAILED"; exit 1; }
fi
$PY analyze.py --trace_dir $W/trace --data $W/data/sessions.npz --out $W/out || { echo "ANALYZE FAILED"; exit 1; }
$PY report.py $W/out/results.json
echo "=== ALL DONE $(date -u) ==="
