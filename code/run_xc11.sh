#!/bin/bash
# llama.cpp expert-cache benchmark, round 11 (Part 17): rent-or-buy admission (rb), frequency score (fe), static per-layer
# slots (LLAMA_XCACHE_CAPS) on the real engine, against the current engine (C tau 0.15 / C1, background loads off).
# Experts read from the NVMe file (ram=0), as in the speed model of the simulation.
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
export PATH=/usr/local/cuda/bin:$PATH
W=/workspace; J=$W/job; R=$W/res; L=/root/local; mkdir -p $R $L
cd $J
echo "=== XC JOB START $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus=$(nproc)"; free -g | sed -n 2p; df -h $W | tail -1; df -h $L | tail -1; mount | grep -E " / | /root" | head -3 | cut -c1-120
echo "memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null) v1 $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)"
lsblk -d -o NAME,SIZE,MODEL 2>/dev/null | head -4
which nvcc cmake git gcc
if ! which nvcc > /dev/null; then
  apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 > /dev/null 2>&1; which nvcc || { echo "NO NVCC"; exit 1; }
fi
which cmake > /dev/null || python3 -m pip install -q cmake
nvcc --version | tail -2
timeout 900 python3 -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer "transformers==4.57.6" pyarrow 2>&1 | grep -vi warn | tail -2
echo "PIP DONE $(date -u +%T)"

MG=$L/gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
( if [ ! -f $MG ]; then python3 -c "
import time; t=time.time()
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF', 'Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf', local_dir='$L/gguf')
print(f'GGUF DOWNLOAD DONE {time.time()-t:.0f}s', flush=True)"; fi; ls -la $L/gguf ) > $W/gguf.log 2>&1 &
DLP=$!

if [ ! -f $W/tok.bin ]; then
  for i in 1 2 3; do timeout 300 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Next-80B-A3B-Instruct', local_dir='$W/hftok', allow_patterns=['*.json','tokenizer*','vocab*','merges*'])" > $W/tok.log 2>&1 && break; echo "tokenizer retry $i"; tail -2 $W/tok.log; done
  echo "TOKENIZER DONE $(date -u +%T)"
  for i in 1 2 3; do timeout 900 python3 prep_data.py --model_dir $W/hftok --out $W/data --per_domain 4 > $W/prep.log 2>&1 && break; echo "prep retry $i"; tail -3 $W/prep.log; done
  grep -E "saved|!!|Error" $W/prep.log; python3 make_tokbin.py $W/data/sessions.npz $W/tok.bin || { echo "DATA FAILED"; exit 1; }
fi
mkdir -p $W/seq; [ -f $W/seq/cross8k.bin ] || python3 make_seq.py $W/data/sessions.npz $W/seq || { echo "SEQ FAILED"; exit 1; }
echo "DATA DONE $(date -u +%T)"

if [ ! -x $L/llama.cpp/build/bin/llama-xcache-bench ]; then
  rm -rf $L/llama.cpp; mkdir -p $L/llama.cpp; cd $L/llama.cpp
  SHA=05af0d2b1398394cfa67e1918fee7feabccaa9bc; OK=0
  git init -q; for i in 1 2 3; do timeout 150 git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp $SHA && git checkout -q FETCH_HEAD && OK=1 && break; echo "fetch retry $i $(date -u +%T)"; sleep 5; done
  if [ $OK = 0 ]; then # fall back to the source tarball
    for i in 1 2 3; do timeout 300 python3 -c "import urllib.request,io,tarfile; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen('https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$SHA', timeout=60).read())).extractall('.')" && cp -a llama.cpp-$SHA/. . && rm -rf llama.cpp-$SHA && OK=1 && break; echo "tarball retry $i $(date -u +%T)"; sleep 5; done
  fi
  [ $OK = 1 ] && git apply $J/xcache.patch && echo "PATCH OK $(date -u +%T)" || { echo "PATCH FAILED"; exit 1; }
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release > $W/cmake.log 2>&1 || { tail -30 $W/cmake.log; echo "CMAKE FAILED"; exit 1; }
  t0=$(date +%s)
  NJ=$(nproc); [ $NJ -gt 24 ] && NJ=24; echo "build -j $NJ"; free -g | sed -n 2p
  cmake --build build -j $NJ --target llama-xcache-bench > $W/build.log 2>&1 || { grep -E "error" $W/build.log | head -20; tail -20 $W/build.log; echo "BUILD FAILED"; exit 1; }
  echo "BUILD OK in $(( $(date +%s) - t0 ))s $(date -u +%T)"
  cd $J
fi

while kill -0 $DLP 2>/dev/null; do echo "gguf download $(du -sb $L/gguf 2>/dev/null | cut -f1) bytes $(date -u +%T)"; sleep 120; done
wait $DLP; tail -3 $W/gguf.log
[ -f $MG ] || { echo "GGUF MISSING"; exit 1; }
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
echo "disk read (direct, 4GB): $(dd if=$MG of=/dev/null bs=16M count=256 skip=64 iflag=direct 2>&1 | tail -1)"
B=$L/llama.cpp/build/bin/llama-xcache-bench
Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '$1!="max"{print int($1/$2)}'); T=${Q:-$(nproc)}; [ $T -gt 16 ] && T=16; [ $T -lt 1 ] && T=$(nproc); echo "threads $T"
free -g | sed -n 2p

run() {
  name=$1; shift
  echo "=== RUN $name $(date -u +%T) ==="
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > $R/$name.vram 2>/dev/null & VP=$!
  "$@" --out $R/$name.json 2> $R/$name.err | grep --line-buffered SECTION_XC | cut -c1-300
  kill $VP 2>/dev/null
  echo "VRAM_MAX $name $(sort -n $R/$name.vram | tail -1) MiB"
  grep -E "llama_xcache:|error|failed|what\(\)|GGML_ASSERT|out of memory|alloc_stage" $R/$name.err | head -5
  grep -E "^session" $R/$name.err | tail -3
  python3 $J/summarize_xc11.py $R/$name.json 2>&1 | head -40
  sleep 1
}

P0="aio=1+uf=1+up=1+bg=0+ram=0"
FE="fe=0.0078125+dk=1"
XCENV="LLAMA_XCACHE=1 LLAMA_XCACHE_GGUF=$MG LLAMA_XCACHE_FRAC_MAX=0.25"
COMMON="-m $MG -ngl 99 -t $T --n-cpu-moe 48 --prompts 512 --prefill-opts ps=1+ram=0"
# per-layer slot shares: mean Denning allocation of the all24 prompts at 128 slots (round 10, ALLOC pw)
CAPS="260.2,199.5,232.2,125.7,128.7,139.0,175.8,164.5,122.8,101.2,92.0,103.8,86.5,91.0,100.0,136.7,112.0,135.2,175.3,165.8,107.3,97.2,93.7,102.5,87.3,92.7,99.3,139.5,122.5,139.7,182.5,174.0,120.8,104.7,93.3,105.8,88.7,95.2,102.2,138.2,126.5,135.2,127.3,119.0,135.5,113.3,115.8,136.5"

# 1. six sessions (one per domain) x (prompt 512 + 1535 generated), uniform slots, 25% and 12.5%
PLAN1="exact:0.25:nvme:2:$P0,C:0.25:nvme:0.15:$P0,C:0.25:nvme:2:$P0,C:0.25:nvme:0.15:$P0+$FE"
PLAN1="$PLAN1,C:0.25:nvme:0.15:$P0+$FE+rb=1,C:0.25:nvme:0.15:$P0+$FE+rb=0.5,C:0.25:nvme:0.15:$P0+$FE+rb=0.25"
PLAN1="$PLAN1,C:0.25:nvme:2:$P0+$FE+rb=1,C:0.25:nvme:2:$P0+$FE+rb=0.5"
PLAN1="$PLAN1,C:0.125:nvme:0.15:$P0,C:0.125:nvme:2:$P0,C:0.125:nvme:2:$P0+$FE+rb=1"
run main env $XCENV $B $COMMON --tokens $W/tok.bin --decode 1536 --sessions 6 --plan "$PLAN1"

# 2. same sessions, static per-layer slots at the same device memory
PLAN2="C:0.25:nvme:0.15:$P0,C:0.25:nvme:2:$P0,C:0.25:nvme:0.15:$P0+$FE+rb=0.5,C:0.25:nvme:2:$P0+$FE+rb=0.5,C:0.25:nvme:2:$P0+$FE+rb=1"
run caps env $XCENV LLAMA_XCACHE_CAPS=$CAPS $B $COMMON --tokens $W/tok.bin --decode 1536 --sessions 6 --plan "$PLAN2"

# 3. topic changes: three sequences of four sessions from different domains, prompt 512 + 4096 generated (two changes)
PLAN3="exact:0.25:nvme:2:$P0,C:0.25:nvme:0.15:$P0,C:0.25:nvme:2:$P0,C:0.25:nvme:0.15:$P0+$FE+rb=1"
PLAN3="$PLAN3,C:0.25:nvme:2:$P0+$FE+rb=0.5,C:0.25:nvme:2:$P0+$FE+rb=1"
run cross env $XCENV $B $COMMON --tokens $W/seq/cross8k.bin --decode 4096 --sessions 3 --plan "$PLAN3"

echo "=== FINAL $(date -u +%T) ==="
python3 $J/summarize_xc11.py --final $R
echo "=== RESULTS ==="
for f in $R/*.json; do case $f in *.raw.json) ;; *) echo "RESULT $(basename $f .json) $(cat $f)";; esac; done
echo "=== ALL DONE $(date -u) ==="
