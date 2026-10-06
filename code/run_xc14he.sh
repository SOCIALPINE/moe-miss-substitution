#!/bin/bash
# Part 20 addendum (PREREG_xc14_addendum.md): regenerate HumanEval (164 problems x six settings) under the conditions of
# round-14 pod C, keep the outputs, and score them with the original and the corrected code-block extraction.
# Identical to run_xc14.sh except: default chunks are the two HumanEval chunks, no repeats, XC_THREADS can pin the thread
# count (pod C ran 16), the start prints the toolchain and model-file identity, and the end adds score_fix.py (rules A and
# B) and dump14.py (all generations into the log, xz + base64, md5 per line).
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1 HF_HOME=/root/hfcache PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
export PATH=/usr/local/cuda/bin:$PATH
W=/workspace; J=$W/job; R=$W/res; L=/root/local; mkdir -p $R $L
cd $J
echo "=== XC JOB START $(date -u) ==="
CHUNKS=${XC_CHUNKS:-"500:582 582:664"}; REPS=${XC_REPS:-0}
echo "POD ${XC_POD:-single} chunks [$CHUNKS] repeats $REPS"
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
python3 -c "import torch; print('TORCH', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null
ls /usr/local/cuda/lib64/libcublas.so.* /usr/local/cuda/lib64/libcudart.so.* 2>/dev/null | tr '\n' ' '; echo
grep -E "^(PRETTY_NAME|VERSION_ID)=" /etc/os-release | tr '\n' ' '; echo
BP=""
if [ ! -x $L/llama.cpp/build/bin/llama-xcache-bench ]; then ( # fetch, patch and build while pip and the data run
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
  ) &
  BP=$!
fi
timeout 900 python3 -m pip install -q "huggingface_hub[hf_xet]>=0.34,<1.0" hf_transfer "transformers==4.57.6" pyarrow 2>&1 | grep -vi warn | tail -2
echo "PIP DONE $(date -u +%T)"

MG=$L/gguf/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
( if [ ! -f $MG ]; then python3 -c "
import time; t=time.time()
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF', 'Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf', local_dir='$L/gguf')
print(f'GGUF DOWNLOAD DONE {time.time()-t:.0f}s', flush=True)"; fi; ls -la $L/gguf ) > $W/gguf.log 2>&1 &
DLP=$!

if [ ! -f $W/hftok/tokenizer.json ]; then
  for i in 1 2 3; do timeout 300 python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Next-80B-A3B-Instruct', local_dir='$W/hftok', allow_patterns=['*.json','tokenizer*','vocab*','merges*'])" > $W/tok.log 2>&1 && break; echo "tokenizer retry $i"; tail -2 $W/tok.log; done
  echo "TOKENIZER DONE $(date -u +%T)"
fi
if [ ! -f $W/tasks/tasks.json ]; then
  for i in 1 2 3; do timeout 600 python3 prep_tasks.py --model_dir $W/hftok --out $W/tasks --gsm 500 --seed 14 > $W/tasks.log 2>&1 && break; echo "tasks retry $i"; tail -3 $W/tasks.log; done
fi
grep -E "^TASKS" $W/tasks.log; [ -f $W/tasks/tasks.json ] || { echo "TASKS FAILED"; exit 1; }
echo "tasks.bin md5 $(md5sum $W/tasks/tasks.bin | cut -c1-32)"
echo "DATA DONE $(date -u +%T)"


[ -n "$BP" ] && wait $BP
[ -x $L/llama.cpp/build/bin/llama-xcache-bench ] || { echo "ENGINE MISSING"; exit 1; }
while kill -0 $DLP 2>/dev/null; do echo "gguf download $(du -sb $L/gguf 2>/dev/null | cut -f1) bytes $(date -u +%T)"; sleep 120; done
wait $DLP; tail -3 $W/gguf.log
[ -f $MG ] || { echo "GGUF MISSING"; exit 1; }
python3 -c "
from huggingface_hub import HfApi
for repo, fn in (('unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF', 'Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf'), ('Qwen/Qwen3-Next-80B-A3B-Instruct', None)):
    try:
        mi = HfApi().model_info(repo, files_metadata=fn is not None)
        f = [x for x in mi.siblings if x.rfilename == fn] if fn else []
        print('MODELFILE', repo, 'commit', mi.sha, fn or '', (f[0].lfs.sha256 if f and f[0].lfs else ''), (f[0].size if f else ''), flush=True)
    except Exception as e:
        print('MODELFILE', repo, 'lookup failed', repr(e)[:120], flush=True)
" 2>&1 | grep MODELFILE
ls -l $MG | awk '{print "GGUF bytes", $5}'
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
echo "disk read (direct, 4GB): $(dd if=$MG of=/dev/null bs=16M count=256 skip=64 iflag=direct 2>&1 | tail -1)"
B=$L/llama.cpp/build/bin/llama-xcache-bench
Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '$1!="max"{print int($1/$2)}'); T=${Q:-$(nproc)}; [ $T -gt 16 ] && T=16; [ $T -lt 1 ] && T=$(nproc); echo "threads (computed) $T"
T=${XC_THREADS:-$T}; echo "threads $T"
free -g | sed -n 2p

P0="aio=1+uf=1+up=1+bg=0+ram=0"
FE="fe=0.0078125+dk=1"
declare -A PL
PL[E]="exact:0.25:nvme:2:aio=1+uf=1+up=1+bg=0+ram=1"
PL[Q]="C:0.25:nvme:0.15:$P0+$FE+rb=0.5"
PL[S]="C:0.25:nvme:2:$P0+$FE+rb=0.5"
PL[T]="C:0.25:nvme:0.15:$P0"
PL[C]="C:0.25:nvme:2:$P0"
PL[H]="C:0.125:nvme:2:$P0"
PL[Qr]=${PL[Q]}
PL[Sr]=${PL[S]}
XCENV="LLAMA_XCACHE=1 LLAMA_XCACHE_GGUF=$MG LLAMA_XCACHE_FRAC_MAX=0.25"
COMMON="-m $MG -ngl 99 -t $T --n-cpu-moe 48 --prefill-opts ps=1+ram=1+fresh=1 --max-new 1024 --gen $W/tasks/tasks.bin"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done ) > $R/vram.log 2>/dev/null &
VMON=$!

# gen LABEL A:B: one plan entry on prompts A..B-1, kept only when every prompt finished
gen() {
  local lab=$1 rg=$2 ga gb gf ge gn=0 gp
  ga=${rg%:*}; gb=${rg#*:}; gf=$R/gen_${ga}-${gb}_${lab}.jsonl; ge=$R/gen_${ga}-${gb}_${lab}.err
  if [ -f $gf ] && [ "$(wc -l < $gf)" = "$((gb - ga))" ]; then echo "SKIP $lab $rg"; return; fi
  echo "=== GEN $lab $rg $(date -u +%T) ==="
  env $XCENV $B $COMMON --gen-range $rg --plan "${PL[$lab]}" --out $gf.tmp 2> $ge > /dev/null &
  gp=$!
  while kill -0 $gp 2>/dev/null; do sleep 10; gn=$((gn + 1)); [ $((gn % 30)) = 0 ] && grep -E "^gen " $ge | tail -1 | cut -c1-130; done
  wait $gp
  if [ -f $gf.tmp ] && [ "$(wc -l < $gf.tmp)" = "$((gb - ga))" ]; then
    mv $gf.tmp $gf
  else
    echo "GEN FAILED $lab $rg"; grep -E "error|failed|GGML_ASSERT|what\(\)|out of memory" $ge | head -5; tail -3 $ge; return
  fi
  python3 -c "
import json
d = [json.loads(l) for l in open('$gf')]
nd = sum(max(0, x['n'] - 1) for x in d)
print('DONE14 $lab $rg items', len(d), 'tokens', sum(x['n'] for x in d), 'ms/tok %.2f' % (sum(x['ms'] for x in d) / max(1, nd)),
      'mass %.4f' % (sum(x['mass'] for x in d) / max(1, sum(x['ltok'] for x in d))), 'trunc', sum(1 - x['eog'] for x in d),
      'prefill_s %.2f' % (sum(x['pf_ms'] for x in d) / len(d) / 1000), flush=True)"
}

for rg in $CHUNKS; do
  for lab in E Q S T C H; do gen $lab $rg; done
  echo "=== PARTIAL $(date -u +%T) ==="
  python3 $J/score_tasks.py --tasks $W/tasks --tok $W/hftok --res $R --brief 2>&1 | grep -E "^(G14|P14)|Error" | cut -c1-400
done
[ "$REPS" = 1 ] && for lab in Qr Sr; do gen $lab 0:100; done
kill $VMON 2>/dev/null
echo "VRAM_MAX $(sort -n $R/vram.log | tail -1) MiB"
echo "=== SCORE $(date -u +%T) ==="
python3 $J/score_tasks.py --tasks $W/tasks --tok $W/hftok --res $R 2>&1
(cd $R && md5sum gen_*.jsonl | sed "s/^/GENMD5 /")
echo "=== SCORE FIX $(date -u +%T) ==="
python3 $J/score_fix.py --tasks $W/tasks --tok $W/hftok --res $R 2>&1
echo "=== DUMP $(date -u +%T) ==="
python3 $J/dump14.py --tasks $W/tasks --tok $W/hftok --res $R 2>&1 | tee $R/dump14.txt
echo "=== ALL DONE $(date -u) ==="
