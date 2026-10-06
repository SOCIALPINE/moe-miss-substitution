#!/bin/bash
# Experiment B, the auxiliary arms of the draft (section 3.5): m* 0.15 and 0.05 (fill) and m* 0.10 (renorm). Run on the same
# pod with the same code after the registered primary result was known, so these results are exploratory.
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
W=/workspace; MD=$W/model; J=$W/job2
echo "=== EXPB AUX START $(date -u) ==="
md5sum $J/expb.py $J/analyze_b.py $J/trace_next.py $J/PREREG_expB.md run.sh
python3 -c "import transformers" 2>/dev/null || python3 -m pip install -q "transformers==4.57.6" 2>&1 | grep -vi warn | tail -2
python3 -c "import torch, transformers; print('TORCH', torch.__version__, 'TRANSFORMERS', transformers.__version__)"
cd $J
python3 expb.py --model_dir $MD --data $W/data/sessions.npz --out $W/expb_aux --dl_flag_dir $W \
  --arms "H:0.15:fill,L:0.15:fill,H:0.10:renorm,L:0.10:renorm,H:0.05:fill,L:0.05:fill" 2>&1 | tee $W/expb_aux.log | grep -v "^DXB [0-9]"
grep "^DXB " $W/expb_aux.log
[ -f $W/expb_aux/expb.json ] && python3 analyze_b.py --res $W/expb_aux/expb.json
echo "=== AUX DONE $(date -u) ==="
