#!/bin/bash
# llama.cpp expert-cache study, round 10: same router-logit dumps as round 9 (Qwen3-Next-80B-A3B, stock llama.cpp at the
# base commit, xdump), replayed with xsim: rent-or-buy admission of substituted experts (regret), gate-weighted
# frequency, SwapMoE-style masking (never wait) with periodic refill, and per-layer slot allocation from miss curves
# (prompt LRU stack distances, prompt working-set / Denning curve, oracle) at the same total VRAM.
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

if [ ! -f $L/llama.cpp/build/bin/libllama.so ]; then
  rm -rf $L/llama.cpp; mkdir -p $L/llama.cpp; cd $L/llama.cpp
  SHA=05af0d2b1398394cfa67e1918fee7feabccaa9bc; OK=0
  git init -q; for i in 1 2 3; do timeout 150 git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp $SHA && git checkout -q FETCH_HEAD && OK=1 && break; echo "fetch retry $i $(date -u +%T)"; sleep 5; done
  if [ $OK = 0 ]; then # fall back to the source tarball
    for i in 1 2 3; do timeout 300 python3 -c "import urllib.request,io,tarfile; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen('https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$SHA', timeout=60).read())).extractall('.')" && cp -a llama.cpp-$SHA/. . && rm -rf llama.cpp-$SHA && OK=1 && break; echo "tarball retry $i $(date -u +%T)"; sleep 5; done
  fi
  [ $OK = 1 ] && echo "SOURCE OK $(date -u +%T)" || { echo "SOURCE FAILED"; exit 1; }
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release > $W/cmake.log 2>&1 || { tail -30 $W/cmake.log; echo "CMAKE FAILED"; exit 1; }
  t0=$(date +%s)
  NJ=$(nproc); [ $NJ -gt 24 ] && NJ=24; echo "build -j $NJ"; free -g | sed -n 2p
  cmake --build build -j $NJ --target llama > $W/build.log 2>&1 || { grep -E "error" $W/build.log | head -20; tail -20 $W/build.log; echo "BUILD FAILED"; exit 1; }
  echo "BUILD OK in $(( $(date +%s) - t0 ))s $(date -u +%T)"
  cd $J
fi

while kill -0 $DLP 2>/dev/null; do echo "gguf download $(du -sb $L/gguf 2>/dev/null | cut -f1) bytes $(date -u +%T)"; sleep 120; done
wait $DLP; tail -3 $W/gguf.log
[ -f $MG ] || { echo "GGUF MISSING"; exit 1; }
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
echo "disk read (direct, 4GB): $(dd if=$MG of=/dev/null bs=16M count=256 skip=64 iflag=direct 2>&1 | tail -1)"
Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '$1!="max"{print int($1/$2)}'); T=${Q:-$(nproc)}; [ $T -gt 16 ] && T=16; [ $T -lt 1 ] && T=$(nproc); echo "threads $T"
LC=$L/llama.cpp; XD=$L/xdump; X=$L/xsim
g++ -O2 -std=c++17 $J/xdump.cpp -I$LC/include -I$LC/ggml/include -L$LC/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LC/build/bin -o $XD || { echo "XDUMP BUILD FAILED"; exit 1; }
g++ -O3 -march=native -std=c++17 -pthread -o $X $J/xsim.cpp || { echo "XSIM BUILD FAILED"; exit 1; }
python3 make_seq.py $W/data/sessions.npz $W || { echo "SEQ FAILED"; exit 1; }

DD=$L/dump; mkdir -p $DD
dump() { # name tokens
  echo "=== DUMP $1 $(date -u +%T) ==="
  $XD -m $MG --tokens $2 -ngl 99 -t $T --n-cpu-moe 48 --out $DD/$1 2> $R/dump_$1.err | grep DUMP_DONE
  grep -E "^xdump:" $R/dump_$1.err | tail -2; grep -iE "error|failed|out of memory" $R/dump_$1.err | head -3
}
dump all24 $W/all24.bin
dump same8k $W/same8k.bin
dump cross8k $W/cross8k.bin
du -sh $DD; df -h $L | tail -1

mkcfg() { # capacity file
  C=$1
  {
    echo "eng lru s none tau bins"; echo "eng_c1 lru s none c1"; echo "fe128_d fe:0.0078125 d none tau"
    echo "l256_fe128 fe:0.0078125 d lazy:256 tau bins"; echo "l256_fe256_c1 fe:0.00390625 d lazy:256 c1"; echo "l256_fe64_c1 fe:0.015625 d lazy:256 c1"
    echo "bg4_fe64 fe:0.015625 d bg:4 tau"
    echo "few128_d few:0.0078125 d none tau"; echo "l256_few128 few:0.0078125 d lazy:256 tau"; echo "l256_few256_c1 few:0.00390625 d lazy:256 c1"
    echo "mask_lru lru s none none"; echo "sw16 few:0.03125 d lazy:16 none"; echo "sw64 few:0.015625 d lazy:64 none"; echo "sw256 few:0.0078125 d lazy:256 none"
    for B in 0.1 0.25 0.5 1; do for H in 128 512 2048; do echo "rg${B}_$H fe:0.0078125 d regret:$B:$H tau$([ $B = 0.5 ] && [ $H = 512 ] && echo ' bins')"; done; done
    for B in 0.25 0.5; do echo "rg${B}_512g fe:0.0078125 d regret:$B:512:g tau"; done
    for B in 0.25 0.5 1; do echo "rg${B}_512_c1 fe:0.0078125 d regret:$B:512 c1"; done
    echo "rg0.5_512_few few:0.0078125 d regret:0.5:512 tau"
    for A in pm pw om; do
      echo "eng_$A lru s none tau alloc:$A"; echo "eng_c1_$A lru s none c1 alloc:$A"
      echo "l256_fe128_$A fe:0.0078125 d lazy:256 tau alloc:$A"; echo "l256_fe256_c1_$A fe:0.00390625 d lazy:256 c1 alloc:$A"
      echo "rg0.5_512_$A fe:0.0078125 d regret:0.5:512 tau alloc:$A"; echo "rg0.5_512_c1_$A fe:0.0078125 d regret:0.5:512 c1 alloc:$A"
    done
  } > $2
}
mkcfg 128 $W/cfg_128.txt; mkcfg 64 $W/cfg_64.txt; wc -l $W/cfg_128.txt; cat $W/cfg_128.txt | tr '\n' ';' | cut -c1-400; echo
printf "eng lru s none tau\nrg0.25_512 fe:0.0078125 d regret:0.25:512 tau\nrg0.5_512 fe:0.0078125 d regret:0.5:512 tau\nl256_fe128 fe:0.0078125 d lazy:256 tau\neng_pm lru s none tau alloc:pm\nrg0.5_512_pm fe:0.0078125 d regret:0.5:512 tau alloc:pm\nl256_fe128_pm fe:0.0078125 d lazy:256 tau alloc:pm\n" > $W/cfg_tau.txt
printf "rg0.25_512 fe:0.0078125 d regret:0.25:512 tau\nrg0.5_512 fe:0.0078125 d regret:0.5:512 tau\nrg0.5_512_c1 fe:0.0078125 d regret:0.5:512 c1\nl256_fe128 fe:0.0078125 d lazy:256 tau\n" > $W/cfg_br.txt
OUT=$R/sim.log; : > $OUT
run() { # set capacity tau cfgfile [env]
  local S=$1 C=$2 TAU=$3 F=$4; shift 4
  if [ $S = all24 ]; then P=512; D=1536; BW=256; else P=512; D=7680; BW=512; fi
  env "$@" $X policy $S $P $D $C $TAU 10 $BW $F $DD/${S}_s*.f16 >> $OUT || echo "SIM FAILED $S $C $TAU $F $*"
}
echo "=== SIM $(date -u +%T) ==="
for S in all24 same8k cross8k; do for C in 128 64; do run $S $C 0.15 $W/cfg_$C.txt; echo "main $S $C $(date -u +%T)"; done; done
for S in all24 same8k cross8k; do for C in 128 64; do for TAU in 0.08 0.11 0.2 0.5; do run $S $C $TAU $W/cfg_tau.txt; done; done; echo "tau $S $(date -u +%T)"; done
for S in all24 same8k cross8k; do
  for BR in 0.25 1; do run $S 128 0.15 $W/cfg_br.txt XSIM_BRATE=$BR; done
  for BD in 4 16; do run $S 128 0.15 $W/cfg_br.txt XSIM_BDELAY=$BD; done
  echo "sens $S $(date -u +%T)"
done
grep -c "^SIM" $OUT; grep -c "^ALLOC" $OUT
echo "=== SUMMARY $(date -u +%T) ==="
python3 summarize_xc10.py $OUT
echo "=== RAW ==="
python3 -c "
import base64, zlib
b = base64.b64encode(zlib.compress(open('$OUT', 'rb').read(), 9)).decode()
n = (len(b) + 2999) // 3000
for i in range(n):
    print(f'B64 {i + 1}/{n} ' + b[i * 3000:(i + 1) * 3000], flush=True)
"
echo "=== ALL DONE $(date -u) ==="
