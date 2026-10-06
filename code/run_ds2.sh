#!/bin/bash
# DeepSeek-V4-Flash follow-up: lighter stall rules (tau 0.3 / rank-1 only) and FE policy. Reuses download + data if present.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/workspace; D4=$W/model_ds; M4=deepseek-ai/DeepSeek-V4-Flash; cd $W/job
echo "=== DS2 START $(date -u) ==="; df -h $W | tail -1; du -sh $D4 2>/dev/null
python3 -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer "transformers==4.57.6" pyarrow 2>&1 | grep -vi warn | tail -2
mkdir -p $W/f_ds; rm -f $W/f_ds/download.*
( python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$M4', local_dir='$D4', allow_patterns=['*.json','*.txt','*.jinja','*.model','tokenizer*','*.safetensors','inference/*','encoding/*'], ignore_patterns=['original/*','metal/*'], max_workers=16)
print('DOWNLOAD DONE')" > $W/f_ds/download.log 2>&1 && touch $W/f_ds/download.done || touch $W/f_ds/download.failed ) &
[ -f $W/data_ds/sessions.npz ] || python3 prep_data.py --model_dir $D4 --out $W/data_ds || { echo "PREP DS FAILED"; exit 1; }
python3 mr_run.py --adapter deepseek_v4 --model_dir $D4 --data $W/data_ds/sessions.npz --out $W/mr_ds2 --dl_flag_dir $W/f_ds --per_domain 8 --chunk 16384 --attn_batch 4 --configs "0.25:lru:C@0.3,0.25:lru:C@1.0,0.25:fe:C@0.3,0.125:lru:C@0.3,0.125:lru:C@1.0" 2>&1 | tee $W/mr_ds2.log
[ -f $W/mr_ds2/mr.json ] && echo "=== ALL DONE DS2 $(date -u) ===" || echo "DS2 RUN FAILED"
