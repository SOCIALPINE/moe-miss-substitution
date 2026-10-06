// xdump: router logits of every MoE layer for every token of whole sequences (one batched forward per sequence, stock
// llama.cpp), written as fp16. Teacher forcing gives the same routing as decoding the sequence token by token.
//
//   xdump -m MODEL --tokens FILE --out PREFIX [-ngl 99] [-t 16] [--n-cpu-moe 48] [--ubatch 2048]
//   tokens file: int32 S, T, P, then S*T tokens.   output: PREFIX_s<i>.f16 = int32 T, L, N, then per layer T x N fp16
#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct dump_state {
    std::map<int, std::vector<uint16_t>> buf; // layer -> [tokens][n_expert]
    std::vector<float> tmp;
};

static bool dump_cb(ggml_tensor * t, bool ask, void * ud) {
    const char * nm = ggml_get_name(t);
    const bool is = strncmp(nm, "ffn_moe_logits-", 15) == 0;
    if (ask) {
        return is;
    }
    if (!is) {
        return true;
    }
    auto * d = (dump_state *) ud;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "xdump: unexpected tensor %s\n", nm);
        exit(1);
    }
    const int il = atoi(nm + 15);
    const int64_t n = ggml_nelements(t);
    d->tmp.resize(n);
    ggml_backend_tensor_get(t, d->tmp.data(), 0, n * sizeof(float));
    auto & b = d->buf[il];
    const size_t o = b.size();
    b.resize(o + n);
    ggml_fp32_to_fp16_row(d->tmp.data(), (ggml_fp16_t *) b.data() + o, n);
    return true;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    std::string model_path, tokens_path, out;
    int ngl = 99, n_threads = 8, n_cpu_moe = 0, n_ubatch = 2048;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "-m")          { model_path = next(); }
        else if (a == "--tokens")    { tokens_path = next(); }
        else if (a == "--out")       { out = next(); }
        else if (a == "-ngl")        { ngl = atoi(next().c_str()); }
        else if (a == "-t")          { n_threads = atoi(next().c_str()); }
        else if (a == "--n-cpu-moe") { n_cpu_moe = atoi(next().c_str()); }
        else if (a == "--ubatch")    { n_ubatch = atoi(next().c_str()); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    std::ifstream fin(tokens_path, std::ios::binary);
    int32_t hdr[3];
    fin.read((char *) hdr, sizeof(hdr));
    const int S = hdr[0], T = hdr[1];
    std::vector<int32_t> toks((size_t) S * T);
    fin.read((char *) toks.data(), sizeof(int32_t) * toks.size());
    if (!fin) {
        fprintf(stderr, "xdump: cannot read %s\n", tokens_path.c_str());
        return 1;
    }

    llama_backend_init();
    std::vector<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> ov;
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    for (int il = 0; il < n_cpu_moe; ++il) {
        patterns.push_back("blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
    }
    for (auto & p : patterns) {
        ov.push_back({ p.c_str(), cpu_buft });
    }
    ov.push_back({ nullptr, nullptr });
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.tensor_buft_overrides = ov.data();
    const double t_load0 = now_ms();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "xdump: failed to load model\n");
        return 1;
    }
    fprintf(stderr, "xdump: load %.1fs, %d sequences x %d tokens\n", (now_ms() - t_load0) / 1000.0, S, T);

    dump_state dst;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = T + 64;
    cp.n_batch   = T;
    cp.n_ubatch  = std::min(T, n_ubatch);
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf   = true;
    cp.cb_eval   = dump_cb;
    cp.cb_eval_user_data = &dst;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "xdump: failed to create context\n");
        return 1;
    }
    for (int s = 0; s < S; ++s) {
        llama_memory_clear(llama_get_memory(ctx), true);
        dst.buf.clear();
        const double t0 = now_ms();
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + (size_t) s * T, T)) != 0) {
            fprintf(stderr, "xdump: decode failed\n");
            return 1;
        }
        llama_synchronize(ctx);
        const int L = (int) dst.buf.size();
        const int N = L > 0 ? (int) (dst.buf.begin()->second.size() / T) : 0;
        const std::string fn = out + "_s" + std::to_string(s) + ".f16";
        FILE * f = fopen(fn.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "xdump: cannot write %s\n", fn.c_str());
            return 1;
        }
        const int32_t h2[3] = { T, L, N };
        fwrite(h2, sizeof(h2), 1, f);
        for (auto & kv : dst.buf) {
            if (kv.second.size() != (size_t) T * N) {
                fprintf(stderr, "xdump: layer %d has %zu values, expected %d x %d\n", kv.first, kv.second.size(), T, N);
                return 1;
            }
            fwrite(kv.second.data(), sizeof(uint16_t), kv.second.size(), f);
        }
        fclose(f);
        fprintf(stderr, "xdump: %s  T %d  layers %d  experts %d  %.1fs\n", fn.c_str(), T, L, N, (now_ms() - t0) / 1000.0);
    }
    printf("DUMP_DONE %s %d\n", out.c_str(), S);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
