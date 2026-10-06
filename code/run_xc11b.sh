#!/bin/bash
# llama.cpp expert-cache benchmark, round 11b (Part 17): what the frequency score alone does with C1, and repeats of two
# rent-or-buy plans of round 11 (same sessions) to see the run-to-run spread. Then confidence intervals over all runs.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
export PATH=/usr/local/cuda/bin:$PATH
W=/workspace; J=$W/job; R=$W/res; L=/root/local; mkdir -p $R $L
cd $J
echo "=== XC JOB 11b START $(date -u) ==="
timeout 900 python3 -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer 2>&1 | grep -vi warn | tail -2
MG=$L/gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
( if [ ! -f $MG ]; then python3 -c "
import time; t=time.time()
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF', 'Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf', local_dir='$L/gguf')
print(f'GGUF DOWNLOAD DONE {time.time()-t:.0f}s', flush=True)"; fi; ls -la $L/gguf ) > $W/gguf.log 2>&1 &
DLP=$!
if [ ! -x $L/llama.cpp/build/bin/llama-xcache-bench ]; then
  rm -rf $L/llama.cpp; mkdir -p $L/llama.cpp; cd $L/llama.cpp
  SHA=05af0d2b1398394cfa67e1918fee7feabccaa9bc; OK=0
  git init -q; for i in 1 2 3; do timeout 150 git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp $SHA && git checkout -q FETCH_HEAD && OK=1 && break; echo "fetch retry $i"; sleep 5; done
  [ $OK = 1 ] && git apply $J/xcache.patch && echo "PATCH OK $(date -u +%T)" || { echo "PATCH FAILED"; exit 1; }
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release > $W/cmake.log 2>&1 || { tail -30 $W/cmake.log; echo "CMAKE FAILED"; exit 1; }
  NJ=$(nproc); [ $NJ -gt 24 ] && NJ=24; t0=$(date +%s)
  cmake --build build -j $NJ --target llama-xcache-bench > $W/build.log 2>&1 || { grep -E "error" $W/build.log | head -20; echo "BUILD FAILED"; exit 1; }
  echo "BUILD OK in $(( $(date +%s) - t0 ))s $(date -u +%T)"
  cd $J
fi
while kill -0 $DLP 2>/dev/null; do sleep 20; done
wait $DLP; tail -2 $W/gguf.log
[ -f $MG ] || { echo "GGUF MISSING"; exit 1; }
B=$L/llama.cpp/build/bin/llama-xcache-bench
Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '$1!="max"{print int($1/$2)}'); T=${Q:-$(nproc)}; [ $T -gt 16 ] && T=16; [ $T -lt 1 ] && T=$(nproc); echo "threads $T"
run() {
  name=$1; shift
  echo "=== RUN $name $(date -u +%T) ==="
  "$@" --out $R/$name.json 2> $R/$name.err | grep --line-buffered SECTION_XC | cut -c1-200
  grep -E "llama_xcache:|error|failed|what\(\)|GGML_ASSERT|out of memory" $R/$name.err | head -5
  python3 $J/summarize_xc11.py $R/$name.json 2>&1 | head -40
}
P0="aio=1+uf=1+up=1+bg=0+ram=0"
FE="fe=0.0078125+dk=1"
XCENV="LLAMA_XCACHE=1 LLAMA_XCACHE_GGUF=$MG LLAMA_XCACHE_FRAC_MAX=0.25"
COMMON="-m $MG -ngl 99 -t $T --n-cpu-moe 48 --prompts 512 --prefill-opts ps=1+ram=0"
PLAN="C:0.25:nvme:2:$P0+$FE,C:0.25:nvme:2:$P0+$FE+rb=0.5,C:0.25:nvme:0.15:$P0+$FE+rb=0.5"
run extra env $XCENV $B $COMMON --tokens $W/tok.bin --decode 1536 --sessions 6 --plan "$PLAN"
echo "=== CI $(date -u +%T) ==="
python3 $J/ci_xc11.py $R
echo "=== ALL DONE $(date -u) ==="
