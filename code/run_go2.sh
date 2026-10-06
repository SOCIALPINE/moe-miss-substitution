#!/bin/bash
# 1) gpt-oss-120b rerun: per-domain fidelity, tau sweep, 50% cache.  2) DeepSeek-V4-Flash cache-constrained B/C.
# Both checkpoints download in parallel from the start; each run waits per shard.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3; for c in python python3; do if $c -c "import torch" 2>/dev/null; then PY=$c; break; fi; done
W=/workspace; J=$W/job
MG=openai/gpt-oss-120b; DG=$W/model_go
M4=deepseek-ai/DeepSeek-V4-Flash; D4=$W/model_ds
cd $J
echo "=== JOB START $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus=$(nproc)"; free -g | sed -n 2p; df -h $W | tail -1
$PY -c "import torch; print('torch', torch.__version__)"
$PY -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer 2>&1 | grep -vi warn | tail -2
$PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$MG', local_dir='$DG', allow_patterns=['*.json','*.txt','*.jinja','tokenizer*'])
snapshot_download('$M4', local_dir='$D4', allow_patterns=['*.json','*.txt','*.jinja','*.model','tokenizer*','inference/*','encoding/*'])
print('tokenizer/config ready')"
ls $D4 $D4/encoding 2>/dev/null | head -20
dl() {   # repo dir flagdir
mkdir -p $3; rm -f $3/download.done $3/download.failed
( for a in 1 2; do $PY -c "
import time; t=time.time()
from huggingface_hub import snapshot_download
snapshot_download('$1', local_dir='$2', allow_patterns=['*.json','*.txt','*.jinja','*.model','tokenizer*','*.safetensors','inference/*','encoding/*'], ignore_patterns=['original/*','metal/*'], max_workers=16)
print('DOWNLOAD DONE $1 in %.0fs' % (time.time()-t), flush=True)" >> $3/download.log 2>&1 && { touch $3/download.done; exit 0; }; done
  touch $3/download.failed ) &
}
dl $MG $DG $W/f_go
dl $M4 $D4 $W/f_ds
( while [ ! -f $W/f_ds/download.done ] && [ ! -f $W/f_ds/download.failed ]; do sleep 90; echo "[dl] go $(du -sh $DG 2>/dev/null | cut -f1) ds $(du -sh $D4 2>/dev/null | cut -f1) | $(df -h $W | tail -1 | awk '{print $4}') free"; done; tail -2 $W/f_ds/download.log ) &
$PY -m pip install -q "transformers==4.57.6" pyarrow 2>&1 | grep -vi warn | tail -2

echo "=== gpt-oss-120b rerun ==="
if [ ! -f $W/data_go/sessions.npz ]; then $PY prep_data.py --model_dir $DG --out $W/data_go || echo "PREP GO FAILED"; fi
GA="--adapter gpt_oss --model_dir $DG --data $W/data_go/sessions.npz --out $W/mr_go2 --dl_flag_dir $W/f_go --per_domain 8"
GC="0.25:lru:C,0.25:lru:C@0.3,0.25:lru:C@1.0,0.5:lru:B,0.5:lru:C,0.5:lru:C@1.0,0.125:lru:C@1.0"
$PY mr_run.py $GA --configs $GC 2>&1 | tee $W/mr_go2.log
if [ ! -f $W/mr_go2/mr.json ] && [ -f $W/data_go/sessions.npz ]; then
  echo "GO RUN FAILED; waiting for its download and retrying"
  while [ ! -f $W/f_go/download.done ] && [ ! -f $W/f_go/download.failed ]; do sleep 20; done
  $PY mr_run.py $GA --configs $GC 2>&1 | tee $W/mr_go2.log
fi

echo "=== DeepSeek-V4-Flash ==="
if [ ! -f $W/data_ds/sessions.npz ]; then $PY prep_data.py --model_dir $D4 --out $W/data_ds || { echo "PREP DS FAILED"; exit 1; }; fi
DA="--adapter deepseek_v4 --model_dir $D4 --data $W/data_ds/sessions.npz --out $W/mr_ds --dl_flag_dir $W/f_ds --per_domain 8 --chunk 16384 --attn_batch 4"
$PY mr_run.py $DA 2>&1 | tee $W/mr_ds.log
if [ ! -f $W/mr_ds/mr.json ]; then
  echo "DS RUN FAILED; waiting for its download and retrying"
  while [ ! -f $W/f_ds/download.done ] && [ ! -f $W/f_ds/download.failed ]; do sleep 20; done
  $PY mr_run.py $DA 2>&1 | tee $W/mr_ds.log
fi
[ -f $W/mr_ds/mr.json ] || { echo "DS RUN FAILED"; exit 1; }
echo "=== ALL DONE $(date -u) ==="
