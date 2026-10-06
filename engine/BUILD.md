# Building the engine

The engine is a patch to [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds an expert cache for
mixture-of-experts models whose experts do not fit in GPU memory. In the code and file names it is called `xcache`.

- Base commit: `05af0d2b1398394cfa67e1918fee7feabccaa9bc` (ggml-org/llama.cpp)
- Patch: `llama_xcache.patch` (md5 `48c17a90a44cca01047e5a2a5db32185`), 11 files, +2,759 lines. It adds `src/llama-xcache.{h,cpp}`,
  changes the context and graph code, and adds the benchmark tool `tools/xcache-bench` (`llama-xcache-bench`).
  This is the final version, used in R20 and R20s. Earlier runs used earlier versions that were not kept; according to our
  notes, the version used in R18 and R19 differs from this one only by the greedy-generation mode.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 05af0d2b1398394cfa67e1918fee7feabccaa9bc
git apply ../engine/llama_xcache.patch
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-xcache-bench
```

The cache is switched on with environment variables, for example
`LLAMA_XCACHE=1 LLAMA_XCACHE_GGUF=<model.gguf> LLAMA_XCACHE_FRAC_MAX=0.25`. The exact settings of every
configuration in the paper are in the run scripts (`code/run_xc*.sh`), and Appendix A of the paper describes the implementation.
The model file used was the Q4_K_M GGUF of Qwen3-Next-80B-A3B-Instruct published by unsloth.
