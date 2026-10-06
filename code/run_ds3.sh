#!/bin/bash
# DeepSeek-V4-Flash: shared-expert fallback vs substitution, rank-1-only waits. Fresh container disk: re-download.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/workspace; D4=$W/model_ds; M4=deepseek-ai/DeepSeek-V4-Flash; cd $W/job
echo "=== DS3 START $(date -u) ==="; df -h $W | tail -1; md5sum mr_run.py masked_route.py
python3 -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer "transformers==4.57.6" pyarrow 2>&1 | grep -vi warn | tail -2
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$M4', local_dir='$D4', allow_patterns=['*.json','*.txt','*.jinja','*.model','tokenizer*','inference/*','encoding/*'])
print('tokenizer/config ready')"
mkdir -p $W/f_ds; rm -f $W/f_ds/download.*
( for a in 1 2; do python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$M4', local_dir='$D4', allow_patterns=['*.json','*.txt','*.jinja','*.model','tokenizer*','*.safetensors','inference/*','encoding/*'], ignore_patterns=['original/*','metal/*'], max_workers=16)
print('DOWNLOAD DONE')" >> $W/f_ds/download.log 2>&1 && { touch $W/f_ds/download.done; exit 0; }; done; touch $W/f_ds/download.failed ) &
( while [ ! -f $W/f_ds/download.done ] && [ ! -f $W/f_ds/download.failed ]; do sleep 90; echo "[dl] $(du -sh $D4 | cut -f1)"; done ) &
[ -f $W/data_ds/sessions.npz ] || python3 prep_data.py --model_dir $D4 --out $W/data_ds || { echo "PREP DS FAILED"; exit 1; }
A="--adapter deepseek_v4 --model_dir $D4 --data $W/data_ds/sessions.npz --out $W/mr_ds3 --dl_flag_dir $W/f_ds --per_domain 8 --chunk 16384 --attn_batch 4"
C="0.25:lru:C@1.0,0.25:lru:B/shared,0.25:lru:B/drop,0.25:lru:C@1.0/shared,0.125:lru:C@1.0/shared"
python3 mr_run.py $A --configs $C 2>&1 | tee $W/mr_ds3.log
if [ ! -f $W/mr_ds3/mr.json ]; then
  echo "DS3 RUN FAILED; waiting for download and retrying"
  while [ ! -f $W/f_ds/download.done ] && [ ! -f $W/f_ds/download.failed ]; do sleep 20; done
  python3 mr_run.py $A --configs $C 2>&1 | tee $W/mr_ds3.log
fi
[ -f $W/mr_ds3/mr.json ] && echo "=== ALL DONE DS3 $(date -u) ===" || echo "DS3 RUN FAILED"
