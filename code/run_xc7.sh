#!/bin/bash
# llama.cpp expert-cache benchmark, round 7: the page cache source decided automatically (ram=2: on when at least half of
# the expert pages are resident), against ram=0 and ram=1, at RAM plenty / 32 GB / 16 GB, with stock llama.cpp
# (all experts on the CPU, and the same VRAM as ours with --n-cpu-moe 34) on the same machine.
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
echo "DATA DONE $(date -u +%T)"

if [ ! -x $L/llama.cpp/build/bin/llama-xcache-bench ]; then
  rm -rf $L/llama.cpp; mkdir -p $L/llama.cpp; cd $L/llama.cpp
  git init -q; for i in 1 2 3 4 5; do git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp 05af0d2b1398394cfa67e1918fee7feabccaa9bc && break; echo "fetch retry $i"; sleep 10; done
  git checkout -q FETCH_HEAD && git apply $J/xcache.patch && echo "PATCH OK" || { echo "PATCH FAILED"; exit 1; }
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release > $W/cmake.log 2>&1 || { tail -30 $W/cmake.log; echo "CMAKE FAILED"; exit 1; }
  t0=$(date +%s)
  NJ=$(nproc); [ $NJ -gt 24 ] && NJ=24; echo "build -j $NJ"; free -g | sed -n 2p
  cmake --build build -j $NJ --target llama-xcache-bench > $W/build.log 2>&1 || { grep -E "error" $W/build.log | head -20; tail -20 $W/build.log; echo "BUILD FAILED"; exit 1; }
  echo "BUILD OK in $(( $(date +%s) - t0 ))s"
  cd $J
fi

wait $DLP; tail -3 $W/gguf.log
[ -f $MG ] || { echo "GGUF MISSING"; exit 1; }
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
echo "disk read (direct, 4GB): $(dd if=$MG of=/dev/null bs=16M count=256 skip=64 iflag=direct 2>&1 | tail -1)"
B=$L/llama.cpp/build/bin/llama-xcache-bench
Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '$1!="max"{print int($1/$2)}'); T=${Q:-$(nproc)}; [ $T -gt 16 ] && T=16; [ $T -lt 1 ] && T=$(nproc); echo "threads $T"
cat $MG > /dev/null
free -g | sed -n 2p

run() {
  name=$1; shift
  echo "=== RUN $name $(date -u +%T) ==="
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > $R/$name.vram 2>/dev/null & VP=$!
  "$@" --out $R/$name.json 2> $R/$name.err | grep --line-buffered SECTION_XC | cut -c1-300
  kill $VP 2>/dev/null
  echo "VRAM_MAX $name $(sort -n $R/$name.vram | tail -1) MiB"
  grep -E "error|failed|what\(\)|GGML_ASSERT|out of memory|alloc_stage" $R/$name.err | head -5
  sleep 1
}

hog_start() { # leave about $1 GiB of the container limit
  rm -f $W/hog.ready
  LIM=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); case "$LIM" in ''|max) LIM=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null);; esac
  MT=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) * 1024 )); case "$LIM" in ''|max) LIM=$MT;; esac; [ "$LIM" -gt "$MT" ] && LIM=$MT
  HOGG=$(python3 -c "print(max(0, int($LIM/2**30) - $1))"); echo "mem limit $(( LIM >> 30 )) GiB -> hog $HOGG GiB"
  python3 hog.py $HOGG $W/hog.ready & HOG=$!
  for i in $(seq 150); do [ -f $W/hog.ready ] && break; kill -0 $HOG 2>/dev/null || break; sleep 2; done
  free -g | sed -n 2p
  [ -f $W/hog.ready ]
}
hog_stop() { kill $HOG 2>/dev/null; wait $HOG 2>/dev/null; rm -f $W/hog.ready; sleep 2; }

P0="C:0.25:nvme:0.15:aio=1+uf=1+up=1+bg=0"
COMMON="--tokens $W/tok.bin -ngl 99 -t $T --decode 256 --prompts 128,512,1536"
XCENV="LLAMA_XCACHE=1 LLAMA_XCACHE_GGUF=$MG LLAMA_XCACHE_FRAC_MAX=0.25"
ours() { # name ram [extra bench args]
  n=$1; r=$2; shift 2
  run $n env $XCENV $B -m $MG $COMMON --n-cpu-moe 48 --sessions 6 --plan $P0+ram=$r --prefill-opts ps=1+ram=$r "$@"
  grep "ram auto" $R/$n.err | awk -v n=$n '{p=$4+0; c[$NF]++; if (min == "" || p < min) min = p; if (p > max) max = p} END {printf "RAMDEC %s", n; for (k in c) printf " %s=%d", k, c[k]; printf " min=%.1f%% max=%.1f%%\n", min, max}'
}
# RAM plenty (page cache warm)
run stock_warm   $B -m $MG $COMMON --n-cpu-moe 48 --sessions 6 --plan stock
run stock34_warm $B -m $MG $COMMON --n-cpu-moe 34 --sessions 6 --plan stock
ours warm_auto 2
ours warm_r0 0
# RAM about 32 GB
if hog_start 32; then
run stock_hog32   $B -m $MG $COMMON --n-cpu-moe 48 --sessions 3 --warm 2 --plan stock
run stock34_hog32 $B -m $MG $COMMON --n-cpu-moe 34 --sessions 3 --warm 2 --plan stock
ours hog32_auto 2
ours hog32_r0 0
ours hog32_r1 1
else echo "HOG32 FAILED"; fi
hog_stop
# RAM about 16 GB
if hog_start 16; then
run stock_hog     $B -m $MG $COMMON --n-cpu-moe 48 --sessions 3 --warm 2 --plan stock
run stock34_hog   $B -m $MG $COMMON --n-cpu-moe 34 --sessions 3 --warm 2 --plan stock
ours hog_auto 2
ours hog_r0 0
ours hog_r1 1
run hog_auto_cold env $XCENV $B -m $MG $COMMON --n-cpu-moe 48 --sessions 6 --plan $P0+ram=2 --cold --prefill-opts ps=1+ram=2+fresh=1
else echo "HOG16 FAILED"; fi
hog_stop
echo "=== RESULTS ==="
for f in $R/*.json; do case $f in *.raw.json) ;; *) echo "RESULT $(basename $f .json) $(cat $f)";; esac; done
echo "=== ALL DONE $(date -u) ==="
