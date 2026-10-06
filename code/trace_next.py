"""Layer-wise exact forward of Qwen3-Next (hybrid Gated-DeltaNet / gated full attention + MoE with
512 experts, top-10 + shared expert) that records the true routing of every token at every layer.

Token mixers and norms are transformers' own Qwen3-Next modules (weights assigned per layer), so the
tricky linear-attention math is the reference implementation. The MoE (routing + experts + shared
expert) is re-implemented with a sort-by-expert kernel and checked against transformers in
test_next.py. Weights are read layer by layer, so a 160GB model runs on one GPU; the loader waits for
shards that are still downloading, so compute overlaps the download.

Outputs match trace_routes.py (trace_idx is uint16 because E > 255).
"""
import os, sys, json, time, argparse, threading, queue
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_next import modeling_qwen3_next as QN


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class Ckpt:
    """safetensors reader that waits for shards still being downloaded."""
    def __init__(self, d, dl_flag_dir=None):
        self.d = d
        self.flag = dl_flag_dir
        p = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(p):
            self.wmap = json.load(open(p))["weight_map"]
        else:
            fn = "model.safetensors"
            f = safe_open(os.path.join(d, fn), framework="pt", device="cpu")
            self.wmap = {k: fn for k in f.keys()}
        self.h = {}

    def _ready(self, path):
        if not os.path.exists(path):
            return False
        if self.flag and os.path.exists(os.path.join(self.flag, "download.done")):
            return True
        # still downloading: require no *.incomplete temp for this file and a stable size
        dl_tmp = os.path.join(self.d, ".cache", "huggingface", "download")
        base = os.path.basename(path)
        if os.path.isdir(dl_tmp) and any(f.startswith(base) and f.endswith(".incomplete")
                                         for f in os.listdir(dl_tmp)):
            return False
        s1 = os.path.getsize(path)
        time.sleep(1.5)
        return s1 == os.path.getsize(path) and s1 > 0

    def _wait(self, path):
        t0 = time.time()
        while not self._ready(path):
            if self.flag and os.path.exists(os.path.join(self.flag, "download.failed")):
                raise RuntimeError(f"download failed, missing {path}")
            if self.flag and os.path.exists(os.path.join(self.flag, "download.done")) and not os.path.exists(path):
                raise RuntimeError(f"shard missing after download finished: {path}")
            if time.time() - t0 > 5400:
                raise RuntimeError(f"timeout waiting for {path}")
            time.sleep(2)

    def get(self, name):
        fn = self.wmap[name]
        if fn not in self.h:
            path = os.path.join(self.d, fn)
            self._wait(path)
            self.h[fn] = safe_open(path, framework="pt", device="cpu")
        return self.h[fn].get_tensor(name)

    def prefix(self, pre):
        return {k[len(pre):]: self.get(k) for k in self.wmap if k.startswith(pre)}


def load_layer(ck, L, cfg, dtype, pin):
    p = f"model.layers.{L}."
    E = cfg.num_experts
    mix = "linear_attn." if cfg.layer_types[L] == "linear_attention" else "self_attn."
    d = {"mixer": {k: v.to(dtype) if v.is_floating_point() else v for k, v in ck.prefix(p + mix).items()}}
    d["in_norm"] = ck.get(p + "input_layernorm.weight").to(dtype)
    g = torch.stack([ck.get(f"{p}mlp.experts.{e}.gate_proj.weight") for e in range(E)])
    u = torch.stack([ck.get(f"{p}mlp.experts.{e}.up_proj.weight") for e in range(E)])
    d["gu"] = torch.cat([g, u], dim=1).to(dtype).contiguous()
    del g, u
    d["down"] = torch.stack([ck.get(f"{p}mlp.experts.{e}.down_proj.weight") for e in range(E)]).to(dtype).contiguous()
    d["sh_gu"] = torch.cat([ck.get(p + "mlp.shared_expert.gate_proj.weight"),
                            ck.get(p + "mlp.shared_expert.up_proj.weight")], dim=0).to(dtype)
    d["sh_down"] = ck.get(p + "mlp.shared_expert.down_proj.weight").to(dtype)
    d["sh_gate"] = ck.get(p + "mlp.shared_expert_gate.weight").to(dtype)
    if pin:
        d = {k: ({kk: vv.pin_memory() for kk, vv in v.items()} if isinstance(v, dict) else v.pin_memory())
             for k, v in d.items()}
    return d


def build_mixer(cfg, L, sd, dev):
    with torch.device("meta"):
        m = QN.Qwen3NextGatedDeltaNet(cfg, L) if cfg.layer_types[L] == "linear_attention" \
            else QN.Qwen3NextAttention(cfg, L)
    m.load_state_dict({k: v.to(dev, non_blocking=True) for k, v in sd.items()}, strict=True, assign=True)
    for n, t in list(m.named_parameters()) + list(m.named_buffers()):
        assert not t.is_meta, f"meta tensor left in mixer: {n}"
    return m.eval()


def make_norm(w, eps, dev):
    with torch.device("meta"):
        n = QN.Qwen3NextRMSNorm(w.shape[0], eps=eps)
    n.load_state_dict({"weight": w.to(dev)}, strict=True, assign=True)
    return n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--attn_batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--dists", default="0,1,2,3,4,6,8")
    ap.add_argument("--dl_flag_dir", default=None)
    ap.add_argument("--save_logits", default=None, help="dir for per-layer router logits")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.save_logits:
        os.makedirs(args.save_logits, exist_ok=True)
    dev = torch.device(args.device)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    t_start = time.time()

    cfg = AutoConfig.from_pretrained(args.model_dir)
    cfg._attn_implementation = "sdpa"
    cfg.dtype = dt
    H = cfg.hidden_size; E = cfg.num_experts; K = cfg.num_experts_per_tok; NL = cfg.num_hidden_layers
    I = cfg.moe_intermediate_size; Is = cfg.shared_expert_intermediate_size; eps = cfg.rms_norm_eps
    assert not cfg.mlp_only_layers and cfg.decoder_sparse_step == 1
    log(f"cfg: layers={NL} experts={E} top{K} shared_I={Is} hidden={H} types={''.join('L' if t[0]=='l' else 'F' for t in cfg.layer_types)}")

    dat = np.load(args.data, allow_pickle=True)
    tokens = dat["tokens"].astype(np.int64)
    dom = dat["domain"].astype(np.int64)
    names = [str(s) for s in dat["domain_names"]]
    S, T = tokens.shape
    N = S * T
    ndom = len(names)
    log(f"data: {S} sessions x {T} tokens = {N} tokens")

    ck = Ckpt(args.model_dir, args.dl_flag_dir)
    post_norm, gate_w = {}, {}

    def get_route_params(L):
        if L not in gate_w:
            post_norm[L] = make_norm(ck.get(f"model.layers.{L}.post_attention_layernorm.weight").to(dt), eps, dev)
            gate_w[L] = ck.get(f"model.layers.{L}.mlp.gate.weight").to(dev, dt)
        return post_norm[L], gate_w[L]

    emb = ck.get("model.embed_tokens.weight").to(dev, dt)
    tok_t = torch.from_numpy(tokens).to(dev)
    x = emb[tok_t]
    del emb
    xf = x.view(N, H)
    dom_tok = torch.from_numpy(np.repeat(dom, T)).to(dev)
    ntok_dom = np.bincount(dom, minlength=ndom) * T

    rot = QN.Qwen3NextRotaryEmbedding(config=cfg, device=dev)
    pos = torch.arange(T, device=dev)[None]
    cos, sin = rot(x[:1], pos)

    dists = [int(v) for v in args.dists.split(",")]
    M_LIST = [K, 2 * K, 4 * K]
    TOPM = max(M_LIST)
    rec = {D: np.zeros((NL, len(M_LIST), ndom)) for D in dists}
    pending = {}
    CH = args.chunk
    trace_idx = np.zeros((NL, N, K), np.uint16)
    trace_w = np.zeros((NL, N, K), np.float16)

    def route(xc, L, want_logits=False):
        nrm, gw = get_route_params(L)
        h = nrm(xc)
        lg = F.linear(h, gw)
        pr = F.softmax(lg, dim=-1, dtype=torch.float)
        return (h, pr, lg) if want_logits else (h, pr)

    def stash(p):
        for D in dists:
            if (p + D - 1) % 2:
                continue
            Tl = (p + D - 1) // 2
            if Tl < 0 or Tl >= NL:
                continue
            out = torch.empty(N, TOPM, dtype=torch.int16, device=dev)
            for c0 in range(0, N, CH):
                _, pr = route(xf[c0:c0 + CH], Tl)
                out[c0:c0 + CH] = pr.topk(TOPM, dim=-1).indices.to(torch.int16)
            pending[(Tl, D)] = out

    def eval_pending(L, ti_all):
        for key in [k for k in pending if k[0] == L]:
            P = pending.pop(key)
            D = key[1]
            for c0 in range(0, N, CH):
                A = torch.zeros(min(CH, N - c0), E, dtype=torch.bool, device=dev)
                A.scatter_(1, ti_all[c0:c0 + CH], True)
                hitc = A.gather(1, P[c0:c0 + CH].long()).to(torch.int32).cumsum(1)
                for mi, m in enumerate(M_LIST):
                    s = torch.zeros(ndom, device=dev, dtype=torch.float64)
                    s.index_add_(0, dom_tok[c0:c0 + CH], hitc[:, m - 1].double())
                    rec[D][L, mi] += s.cpu().numpy()

    qq = queue.Queue(maxsize=2)
    pin = dev.type == "cuda"

    def loader():
        try:
            ck2 = Ckpt(args.model_dir, args.dl_flag_dir)
            for L in range(NL):
                qq.put(load_layer(ck2, L, cfg, dt, pin))
        except Exception as e:
            qq.put(e)
    threading.Thread(target=loader, daemon=True).start()

    t_wait = 0.0
    for L in range(NL):
        tl0 = time.time()
        W = qq.get()
        if isinstance(W, Exception):
            raise W
        mixer = build_mixer(cfg, L, W["mixer"], dev)
        in_norm = make_norm(W["in_norm"], eps, dev)
        GU, DN = W["gu"].to(dev, non_blocking=True), W["down"].to(dev, non_blocking=True)
        SGU, SDN, SG = (W[k].to(dev) for k in ("sh_gu", "sh_down", "sh_gate"))
        t0 = time.time()
        t_wait += t0 - tl0

        stash(2 * L)
        for b0 in range(0, S, args.attn_batch):
            xb = x[b0:b0 + args.attn_batch]
            h = in_norm(xb)
            if cfg.layer_types[L] == "linear_attention":
                o = mixer(hidden_states=h, cache_params=None, cache_position=None, attention_mask=None)
            else:
                o, _ = mixer(hidden_states=h, position_embeddings=(cos, sin), attention_mask=None)
            xb.add_(o)
            del h, o
        stash(2 * L + 1)

        ti_all = torch.empty(N, K, dtype=torch.long, device=dev)
        if args.save_logits:
            lg_all = torch.empty(N, E, dtype=torch.float16, device=dev)
        for c0 in range(0, N, CH):
            xc = xf[c0:c0 + CH]
            n = xc.shape[0]
            h, pr, lg = route(xc, L, True)
            if args.save_logits:
                lg_all[c0:c0 + n] = lg.to(torch.float16)
            del lg
            tw, ti = pr.topk(K, dim=-1)
            if cfg.norm_topk_prob:
                tw = tw / tw.sum(dim=-1, keepdim=True)
            ti_all[c0:c0 + n] = ti
            trace_idx[L, c0:c0 + n] = ti.to(torch.int32).cpu().numpy().astype(np.uint16)
            trace_w[L, c0:c0 + n] = tw.to(torch.float16).cpu().numpy()
            tw = tw.to(dt)
            flat = ti.reshape(-1)
            order = torch.argsort(flat, stable=True)
            cnts = torch.bincount(flat, minlength=E).tolist()
            tok_s = order // K
            w_s = tw.reshape(-1)[order]
            out = torch.zeros(n, H, dtype=torch.float32, device=dev)
            off = 0
            for e in range(E):
                c = cnts[e]
                if c == 0:
                    continue
                t_ = tok_s[off:off + c]
                ww = w_s[off:off + c]
                off += c
                hh = h.index_select(0, t_)
                gu = F.linear(hh, GU[e])
                y = F.linear(F.silu(gu[:, :I]) * gu[:, I:], DN[e])
                out.index_add_(0, t_, (y * ww[:, None]).float())
            sgu = F.linear(h, SGU)
            sh = F.linear(F.silu(sgu[:, :Is]) * sgu[:, Is:], SDN)
            sh = torch.sigmoid(F.linear(h, SG)) * sh
            xc.add_((out + sh.float()).to(dt))
            del out, h, pr, sgu, sh
        eval_pending(L, ti_all)
        if args.save_logits:   # router logits of this layer, [S,T,E] fp16 (input to window_router.py)
            np.save(os.path.join(args.save_logits, f"logits_{L:02d}.npy"), lg_all.view(S, T, E).cpu().numpy())
            del lg_all
        del W, GU, DN, SGU, SDN, SG, mixer
        post_norm.pop(L - 1, None); gate_w.pop(L - 1, None)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        log(f"layer {L+1}/{NL} ({cfg.layer_types[L][:3]}): compute {time.time()-t0:.1f}s, waited {t0-tl0:.1f}s")

    fnorm = make_norm(ck.get("model.norm.weight").to(dt), eps, dev)
    lm = ck.get("lm_head.weight").to(dev, dt)
    labels = torch.full((S, T), -100, dtype=torch.long, device=dev)
    labels[:, :-1] = tok_t[:, 1:]
    labels = labels.view(N)
    nll_dom = np.zeros(ndom); cnt_dom = np.zeros(ndom); nll_sess = np.zeros(S)
    LC = 2048 if dev.type == "cuda" else 256
    for c0 in range(0, N, LC):
        lg = F.linear(fnorm(xf[c0:c0 + LC]), lm).float()
        lb = labels[c0:c0 + LC]
        l = F.cross_entropy(lg, lb, reduction="none", ignore_index=-100)
        valid = (lb != -100).double()
        ln = (l.double() * valid).cpu().numpy(); vv = valid.cpu().numpy()
        sid = np.arange(c0, c0 + len(ln)) // T
        np.add.at(nll_sess, sid, ln); np.add.at(nll_dom, dom[sid], ln); np.add.at(cnt_dom, dom[sid], vv)
    nll_sess /= (T - 1)

    np.save(os.path.join(args.out, "trace_idx.npy"),
            np.ascontiguousarray(trace_idx.reshape(NL, S, T, K).transpose(1, 2, 0, 3)))
    np.save(os.path.join(args.out, "trace_w.npy"),
            np.ascontiguousarray(trace_w.reshape(NL, S, T, K).transpose(1, 2, 0, 3)))
    np.save(os.path.join(args.out, "nll_sess.npy"), nll_sess)
    look = {}
    for D in dists:
        valid_layers = [L for L in range(NL) if (2 * L + 1 - D) >= 0]
        tot = rec[D][valid_layers].sum(axis=(0, 2)) / (len(valid_layers) * N * K)
        look[str(D)] = {"m_list": M_LIST, "recall_mean": tot.tolist(),
                        "recall_per_layer": (rec[D].sum(axis=2) / (N * K)).tolist(),
                        "recall_per_domain": {names[d]: (rec[D][valid_layers, :, d].sum(0) /
                                              (len(valid_layers) * max(ntok_dom[d], 1) * K)).tolist()
                                              for d in range(ndom)},
                        "n_layers": len(valid_layers)}
    stats = {"model_cfg": {"layers": NL, "experts": E, "top_k": K, "hidden": H, "shared_expert": True},
             "S": S, "T": T, "domains": names,
             "ppl_per_domain": {names[d]: float(np.exp(nll_dom[d] / max(cnt_dom[d], 1))) for d in range(ndom)},
             "lookahead": look, "seconds": time.time() - t_start, "seconds_waiting_for_weights": t_wait}
    json.dump(stats, open(os.path.join(args.out, "trace_stats.json"), "w"), indent=1)
    log("ppl:", {k: round(v, 3) for k, v in stats["ppl_per_domain"].items()})
    for D in dists:
        log(f"lookahead D={D}: recall@{M_LIST} = {[round(v, 4) for v in look[str(D)]['recall_mean']]}")
    log(f"trace done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
