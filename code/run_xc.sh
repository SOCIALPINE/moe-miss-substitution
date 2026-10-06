#!/bin/bash
# llama.cpp expert-cache benchmark on one GPU pod: build patched llama.cpp (CUDA), download the Q4_K_M GGUF,
# run the expert cache modes and stock llama.cpp baselines (experts on CPU, with full RAM and with RAM limited to ~16 GB).
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
export PATH=/usr/local/cuda/bin:$PATH
W=/workspace; J=$W/job; R=$W/res; L=/root/local; mkdir -p $R $L
cd $J
echo "=== XC JOB START $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus=$(nproc)"; free -g | sed -n 2p; df -h $W | tail -1; df -h $L | tail -1; mount | grep -E " / | /root" | head -3; cat /sys/fs/cgroup/memory.max 2>/dev/null
lsblk -d -o NAME,SIZE,MODEL 2>/dev/null | head -4
which nvcc cmake git gcc
if ! which nvcc > /dev/null; then
  apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 > /dev/null 2>&1; which nvcc || { echo "NO NVCC"; exit 1; }
fi
which cmake > /dev/null || python3 -m pip install -q cmake
nvcc --version | tail -2
python3 -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer "transformers==4.57.6" pyarrow 2>&1 | grep -vi warn | tail -2

MG=$L/gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
( if [ ! -f $MG ]; then python3 -c "
import time; t=time.time()
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF', 'Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf', local_dir='$L/gguf')
print(f'GGUF DOWNLOAD DONE {time.time()-t:.0f}s', flush=True)"; fi; ls -la $L/gguf ) > $W/gguf.log 2>&1 &
DLP=$!

if [ ! -f $W/tok.bin ]; then
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Next-80B-A3B-Instruct', local_dir='$W/hftok', allow_patterns=['*.json','tokenizer*','vocab*','merges*'])" > /dev/null 2>&1
  python3 prep_data.py --model_dir $W/hftok --out $W/data --per_domain 4 2>&1 | grep -E "saved|!!|Error" ; python3 make_tokbin.py $W/data/sessions.npz $W/tok.bin || { echo "DATA FAILED"; exit 1; }
fi

if [ ! -x $L/llama.cpp/build/bin/llama-xcache-bench ]; then
  rm -rf $L/llama.cpp; mkdir -p $L/llama.cpp; cd $L/llama.cpp
  git init -q && git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp 05af0d2b1398394cfa67e1918fee7feabccaa9bc && git checkout -q FETCH_HEAD \
    && git apply $J/xcache.patch && echo "PATCH OK" || { echo "PATCH FAILED"; exit 1; }
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release > $W/cmake.log 2>&1 || { tail -30 $W/cmake.log; echo "CMAKE FAILED"; exit 1; }
  t0=$(date +%s)
  cmake --build build -j $(nproc) --target llama-xcache-bench > $W/build.log 2>&1 || { grep -E "error" $W/build.log | head -20; tail -20 $W/build.log; echo "BUILD FAILED"; exit 1; }
  echo "BUILD OK in $(( $(date +%s) - t0 ))s"
  cd $J
fi

wait $DLP; tail -3 $W/gguf.log
[ -f $MG ] || { echo "GGUF MISSING"; exit 1; }
B=$L/llama.cpp/build/bin/llama-xcache-bench
Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '$1!="max"{print int($1/$2)}'); T=${Q:-$(nproc)}; [ $T -gt 16 ] && T=16; [ $T -lt 1 ] && T=$(nproc); echo "threads $T"
cat $MG > /dev/null
free -g | sed -n 2p

run() {
  name=$1; shift
  echo "=== RUN $name $(date -u +%T) ==="
  "$@" --out $R/$name.json 2> >(tee $R/$name.err | grep --line-buffered -E "session [0-9]|llama_xcache:|xcache-bench:|error|failed|what\(\)|GGML_ASSERT|out of memory") | grep --line-buffered SECTION_XC
  sleep 1
}

XCPLAN="ideal:0.25:nvme,ideal_sync:0.25:nvme,exact:0.25:nvme,C:0.25:nvme,C:0.25:nvme:0.15,B:0.25:nvme,exact:0.125:nvme,C:0.125:nvme,C:0.125:nvme:0.15,B:0.125:nvme,exact:0.25:ram,C:0.25:ram,exact:0.125:ram,C:0.125:ram"
XCENV="LLAMA_XCACHE=1 LLAMA_XCACHE_GGUF=$MG LLAMA_XCACHE_FRAC_MAX=0.25"
run xc env $XCENV $B -m $MG --tokens $W/tok.bin -ngl 99 --n-cpu-moe 48 -t $T --decode 256 --plan $XCPLAN
run stock48 $B -m $MG --tokens $W/tok.bin -ngl 99 --n-cpu-moe 48 -t $T --decode 256 --plan stock
run stock36 $B -m $MG --tokens $W/tok.bin -ngl 99 --n-cpu-moe 36 -t $T --decode 256 --plan stock

# RAM limited to about 16 GB: hold the rest of the container memory with anonymous pages
rm -f $W/hog.ready
LIM=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); case "$LIM" in ''|max) LIM=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) * 1024 ));; esac
HOGG=$(python3 -c "print(max(0, int($LIM/2**30) - 16))"); echo "mem limit $(( LIM >> 30 )) GiB -> hog $HOGG GiB"
python3 hog.py $HOGG $W/hog.ready & HOG=$!
while [ ! -f $W/hog.ready ]; do sleep 2; done
free -g | sed -n 2p
run stock48_hog $B -m $MG --tokens $W/tok.bin -ngl 99 --n-cpu-moe 48 -t $T --sessions 2 --decode 32 --warm 2 --plan stock
run xc_hog env $XCENV $B -m $MG --tokens $W/tok.bin -ngl 99 --n-cpu-moe 48 -t $T --sessions 2 --decode 256 --plan exact:0.25:nvme,C:0.25:nvme,C:0.125:nvme
kill $HOG; rm -f $W/hog.ready

echo "=== RESULTS ==="
for f in $R/*.json; do case $f in *.raw.json) ;; *) echo "RESULT $(basename $f .json) $(cat $f)";; esac; done
echo "=== ALL DONE $(date -u) ==="
