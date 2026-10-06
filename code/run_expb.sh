#!/bin/bash
# Experiment B, stage 2 (PREREG_expB.md): run expb.py on the checkpoint and sessions prepared by run_expb_prefetch.sh on this
# pod's volume, then the registered analysis (analyze_b.py). Prints the md5 of the registration and the code first.
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
W=/workspace; MD=$W/model
echo "=== EXPB JOB START $(date -u) ==="
md5sum PREREG_expB.md expb.py analyze_b.py trace_next.py run.sh $W/job/prep_data.py
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus=$(nproc)"; free -g | sed -n 2p; df -h $W | tail -1
[ -f $W/download.done ] && [ -f $W/data/sessions.npz ] || { echo "PREFETCH MISSING"; exit 1; }
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q "transformers==4.57.6" 2>&1 | grep -vi warn | tail -2
python3 -c "import torch, transformers; print('TORCH', torch.__version__, 'TRANSFORMERS', transformers.__version__); torch.zeros(1).cuda(); print('CUDA OK')"
python3 -c "import numpy as np, hashlib; d=np.load('$W/data/sessions.npz', allow_pickle=True); print('SESSIONS', d['tokens'].shape, hashlib.md5(d['tokens'].astype('int32').tobytes()).hexdigest())"
python3 expb.py --model_dir $MD --data $W/data/sessions.npz --out $W/expb --dl_flag_dir $W --arms "H:0.10:fill,L:0.10:fill" \
  2>&1 | tee $W/expb.log | grep -v "^DXB [0-9]"
grep "^DXB " $W/expb.log
[ -f $W/expb/expb.json ] || { echo "EXPB FAILED"; exit 1; }
python3 analyze_b.py --res $W/expb/expb.json
echo "=== ALL DONE $(date -u) ==="
