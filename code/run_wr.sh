#!/bin/bash
# Window-router experiment on Qwen3-Next-80B-A3B: trace (saving router logits) while the checkpoint
# downloads, then window_router.py on the GPU. Idempotent per stage.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore NUMBA_NUM_THREADS=8
PY=python3; for c in python python3; do if $c -c "import torch" 2>/dev/null; then PY=$c; break; fi; done
W=/workspace; J=$W/job; MD=$W/model; MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct
cd $J
echo "=== JOB START $(date -u) ==="
if [ -f $W/wr/window_router.json ]; then grep -h . $W/wr_sections.log 2>/dev/null; echo "=== ALL DONE (cached) ==="; exit 0; fi
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
print(f'MODEL DOWNLOAD DONE in {time.time()-t:.0f}s', flush=True)"
}
rm -f $W/download.done $W/download.failed
( if dl > $W/download.log 2>&1 || dl >> $W/download.log 2>&1; then touch $W/download.done; else touch $W/download.failed; fi ) &
DLPID=$!
( while [ ! -f $W/download.done ] && [ ! -f $W/download.failed ]; do sleep 60; echo "[dl] $(du -sh $MD 2>/dev/null | cut -f1)"; done ) &

$PY -m pip install -q "transformers==4.57.6" numba pyarrow 2>&1 | grep -vi warn | tail -2
if [ ! -f $W/data/sessions.npz ]; then
  $PY prep_data.py --model_dir $MD --out $W/data || { echo "PREP FAILED"; exit 1; }
fi

if [ ! -f $W/trace/trace_stats.json ]; then
  if ! $PY trace_next.py --model_dir $MD --data $W/data/sessions.npz --out $W/trace --dl_flag_dir $W --save_logits $W/logits; then
    echo "TRACE FAILED during download; waiting for download to finish and retrying"
    wait $DLPID; tail -3 $W/download.log
    $PY trace_next.py --model_dir $MD --data $W/data/sessions.npz --out $W/trace --dl_flag_dir $W --save_logits $W/logits || { echo "TRACE FAILED"; exit 1; }
  fi
fi
du -sh $W/logits
$PY window_router.py --trace_dir $W/trace --logits_dir $W/logits --data $W/data/sessions.npz --out $W/wr \
  --lams 0,0.03,0.1,0.3 2>&1 | tee $W/wr.log
grep -h "^SECTION_" $W/wr.log > $W/wr_sections.log
[ -f $W/wr/window_router.json ] || { echo "WINDOW ROUTER FAILED"; exit 1; }
echo "=== ALL DONE $(date -u) ==="
