"""Real end-to-end decode benchmark for cache-constrained expert routing (Qwen3-Next-80B-A3B).

Fast tier : a per-layer VRAM slot pool holding `cap` experts (int4, g128 asymmetric, 1.67 MB each).
Slow tier : one file holding every expert blob on local NVMe, read with O_DIRECT (page cache bypassed),
            or (optional) the whole file in pinned host RAM, copied over PCIe.
Non-expert weights (attention, linear attention, shared experts, routers, embeddings) stay resident in bf16.

Decode modes (per token and layer; the router's own top-K is the wanted set):
  ideal      : no I/O and no host round trip; experts come from arbitrary resident slots (= full-memory speed)
  ideal_sync : same compute, but routing goes through the host like the real modes (sync/routing overhead)
  exact      : every miss is loaded before the layer runs (naive offloading, exact outputs)
  C1         : wait only for a missing rank-1 expert; other misses are replaced by the router's best resident
               experts and loaded in the background (final configuration)
  C          : like C1, plus wait for misses whose normalised gate weight >= tau
  B          : never wait
Every missed wanted expert is (re)loaded so the cache follows the router; LRU eviction; wanted never evicted.

  python live.py convert --model_dir M --ex_dir X          (FP8 or bf16 checkpoint -> int4 blobs + bf16 rest)
  python live.py bench   --model_dir M --ex_dir X --data sessions.npz --out O
"""
import os, re, sys, gc, json, time, copy, queue, argparse, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BLK = 4096
U8, BF = torch.uint8, torch.bfloat16
QUEUED, RUNNING, DONE, CANCELLED = range(4)
EXP_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def gsz(n, g=128):
    while n % g:
        g //= 2
    return g


class Layout:
    """Byte layout of one expert blob: packed int4 gate|up and down weights, bf16 group scales and minima."""

    def __init__(self, H, I, gs=128):
        self.H, self.I = H, I
        gh, gi = gsz(H, gs), gsz(I, gs)
        self.gh, self.gi = gh, gi
        spec = [("guq", (2 * I, H // 2), U8), ("dnq", (H, I // 2), U8),
                ("gus", (2 * I, H // gh), BF), ("gum", (2 * I, H // gh), BF),
                ("dns", (H, I // gi), BF), ("dnm", (H, I // gi), BF)]
        self.parts, off = [], 0
        for n, shp, dt in spec:
            nb = int(np.prod(shp)) * (2 if dt == BF else 1)
            self.parts.append((n, shp, dt, off, nb))
            off += nb
        self.raw = off
        self.blob = -(-off // BLK) * BLK

    def views(self, b):
        """b [K, blob] uint8 rows -> dict of typed [K, ...] views (no copy)."""
        K = b.shape[0]
        out = {}
        for n, shp, dt, off, nb in self.parts:
            v = b[:, off:off + nb]
            if dt == BF:
                v = v.view(BF)
            out[n] = v.view(K, *shp)
        return out

    def pack(self, parts):
        buf = np.zeros(self.blob, np.uint8)
        for n, shp, dt, off, nb in self.parts:
            t = parts[n].contiguous()
            assert tuple(t.shape) == shp, (n, tuple(t.shape), shp)
            if dt == BF:
                t = t.view(U8)
            buf[off:off + nb] = t.reshape(-1).numpy()
        return buf


def quant_int4(w, gs):
    """w [B, O, In] -> packed uint8 [B, O, In/2] (low nibble = even column), scale and min [B, O, In/gs] bf16."""
    B, O, In = w.shape
    wg = w.float().view(B, O, In // gs, gs)
    mn, mx = wg.amin(-1), wg.amax(-1)
    s = ((mx - mn) / 15).clamp_min(1e-8).to(BF)
    m = mn.to(BF)
    q = torch.round((wg - m.float()[..., None]) / s.float()[..., None]).clamp_(0, 15).to(U8).view(B, O, In)
    return q[..., 0::2] | (q[..., 1::2] << 4), s, m


def deq(q, s, m, dt):
    """packed [K, O, In/2] + scale/min [K, O, G] -> weights [K, O, In] in dt."""
    K, O, h = q.shape
    G = s.shape[-1]
    w = torch.stack((q & 15, q >> 4), -1).view(K, O, G, (2 * h) // G).to(dt)
    return torch.addcmul(m.to(dt)[..., None], w, s.to(dt)[..., None]).view(K, O, 2 * h)


def deq_fp8(w, sinv, blk=128):
    s = sinv.float()
    if w.dim() != 2 or s.numel() == 1:
        return w.float() * s.reshape(-1)[0]
    O, In = w.shape
    if s.dim() == 2:
        s = s.repeat_interleave(blk, 0)[:O].repeat_interleave(blk, 1)[:, :In]
    return w.float() * s


def experts_decode(v, x, w, dt):
    """one token: v views [K, ...], x [H], w [K] -> sum_k w_k * expert_k(x)  [H]"""
    gu = torch.matmul(deq(v["guq"], v["gus"], v["gum"], dt), x)
    I = gu.shape[-1] // 2
    a = F.silu(gu[:, :I]) * gu[:, I:]
    y = torch.matmul(deq(v["dnq"], v["dns"], v["dnm"], dt), a[:, :, None])[:, :, 0]
    return w @ y


def experts_prefill(lay, stage, h, idx, w, E, dt, chunk=64):
    """many tokens, all experts of the layer in `stage` [E, blob]: padded per-expert batches, `chunk` experts at a time."""
    N, K = idx.shape
    H = h.shape[1]
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    es = flat[order]
    tok = order // K
    ws = w.reshape(-1)[order]
    cnt = torch.bincount(flat, minlength=E)
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(N * K, device=h.device) - start[es]
    cc, sc = cnt.tolist(), start.tolist()
    out = torch.zeros(N, H, dtype=torch.float32, device=h.device)
    for e0 in range(0, E, chunk):
        e1 = min(E, e0 + chunk)
        a, b = sc[e0], sc[e1 - 1] + cc[e1 - 1]
        if b == a:
            continue
        C = max(cc[e0:e1])
        v = lay.views(stage[e0:e1])
        ee, pp, tt = es[a:b] - e0, pos[a:b], tok[a:b]
        X = h.new_zeros(e1 - e0, C, H)
        X[ee, pp] = h[tt]
        gu = torch.bmm(X, deq(v["guq"], v["gus"], v["gum"], dt).transpose(1, 2))
        I = gu.shape[-1] // 2
        act = F.silu(gu[..., :I]) * gu[..., I:]
        y = torch.bmm(act, deq(v["dnq"], v["dns"], v["dnm"], dt).transpose(1, 2))
        out.index_add_(0, tt, (y[ee, pp] * ws[a:b, None]).float())
    return out.to(dt)


# ----------------------------------------------------------------------------------------------- conversion
class Shards:
    """safetensors reader that waits for shards still being downloaded."""

    def __init__(self, d, flag=None):
        from safetensors import safe_open
        self.so = safe_open
        self.d, self.flag = d, flag
        p = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(p):
            self.wmap = json.load(open(p))["weight_map"]
        else:
            with safe_open(os.path.join(d, "model.safetensors"), "pt") as f:
                self.wmap = {k: "model.safetensors" for k in f.keys()}
        self.h = {}

    def _flag(self, n):
        return self.flag is not None and os.path.exists(os.path.join(self.flag, n))

    def ready(self, fn):
        path = os.path.join(self.d, fn)
        if not os.path.exists(path):
            return False
        if self.flag is None or self._flag("download.done"):
            return True
        tmp = os.path.join(self.d, ".cache", "huggingface", "download")
        base = os.path.basename(fn)
        if os.path.isdir(tmp) and any(f.startswith(base) and f.endswith(".incomplete") for f in os.listdir(tmp)):
            return False
        s1 = os.path.getsize(path)
        time.sleep(1.0)
        return s1 == os.path.getsize(path) and s1 > 0

    def check_failed(self):
        if self._flag("download.failed"):
            raise RuntimeError("download failed")

    def get(self, name):
        fn = self.wmap[name]
        if fn not in self.h:
            t0 = time.time()
            while not self.ready(fn):
                self.check_failed()
                if time.time() - t0 > 5400:
                    raise RuntimeError(f"timeout waiting for {fn}")
                time.sleep(2)
            self.h[fn] = self.so(os.path.join(self.d, fn), "pt", device="cpu")
        return self.h[fn].get_tensor(name)

    def f32(self, name, dev):
        t = self.get(name)
        if t.dtype == torch.float8_e4m3fn:
            return deq_fp8(t.to(dev), self.get(name + "_scale_inv").to(dev))
        return t.to(dev).float()


@torch.no_grad()
def convert(a):
    from transformers import AutoConfig
    from safetensors.torch import save_file
    dev = torch.device(a.device)
    cfg = AutoConfig.from_pretrained(a.model_dir)
    H, I, E, NL = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts, cfg.num_hidden_layers
    lay = Layout(H, I)
    os.makedirs(a.ex_dir, exist_ok=True)
    flag = os.path.join(a.ex_dir, "convert.done")
    if os.path.exists(flag):
        log("already converted")
        return
    path = os.path.join(a.ex_dir, "experts.bin")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, NL * E * lay.blob)
    sh = Shards(a.model_dir, a.flag_dir)
    by = {}
    for k, fn in sh.wmap.items():
        by.setdefault(fn, []).append(k)
    todo = sorted(by)
    part, nonexp, written, t0 = {}, {}, 0, time.time()
    log(f"convert: {len(todo)} shards, {NL}x{E} experts, blob {lay.blob} B ({lay.raw} used), total "
        f"{NL * E * lay.blob / 1e9:.2f} GB, group {lay.gh}/{lay.gi}")
    while todo:
        fn = next((f for f in todo if sh.ready(f)), None)
        if fn is None:
            sh.check_failed()
            time.sleep(2)
            continue
        todo.remove(fn)
        ts = time.time()
        ex = {"gate": [], "up": [], "down": []}
        for k in by[fn]:
            m = EXP_RE.match(k)
            if m:
                ex[m.group(3)].append((int(m.group(1)), int(m.group(2)), k))
            elif k.endswith("_scale_inv") or k.startswith("mtp."):
                continue
            else:
                t = sh.get(k)
                if t.dtype == torch.float8_e4m3fn:
                    t = sh.f32(k, dev)
                nonexp[k] = (t.to(BF) if t.is_floating_point() else t).cpu().contiguous()
        for kind, lst in ex.items():
            g = lay.gi if kind == "down" else lay.gh
            for b0 in range(0, len(lst), a.batch):
                bl = lst[b0:b0 + a.batch]
                W = torch.stack([sh.f32(k, dev) for _, _, k in bl])
                q, s, mn = (x.cpu() for x in quant_int4(W, g))
                del W
                for j, (L, e, _) in enumerate(bl):
                    d = part.setdefault((L, e), {})
                    d[kind] = (q[j], s[j], mn[j])
                    if len(d) == 3:
                        (gq, gs_, gm), (uq, us, um), (dq, ds, dm) = d["gate"], d["up"], d["down"]
                        buf = lay.pack({"guq": torch.cat([gq, uq]), "gus": torch.cat([gs_, us]),
                                        "gum": torch.cat([gm, um]), "dnq": dq, "dns": ds, "dnm": dm})
                        os.pwrite(fd, buf, (L * E + e) * lay.blob)
                        written += 1
                        del part[(L, e)]
        log(f"converted {fn} in {time.time() - ts:.1f}s | experts {written}/{NL * E} | {len(todo)} shards left "
            f"| {time.time() - t0:.0f}s")
    assert written == NL * E and not part, (written, len(part))
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    save_file(nonexp, os.path.join(a.ex_dir, "nonexp.safetensors"))
    json.dump({"blob": lay.blob, "raw": lay.raw, "H": H, "I": I, "E": E, "NL": NL, "gh": lay.gh, "gi": lay.gi},
              open(os.path.join(a.ex_dir, "layout.json"), "w"))
    open(flag, "w").write("ok")
    log(f"convert done in {time.time() - t0:.0f}s: {len(nonexp)} non-expert tensors "
        f"({sum(t.numel() * t.element_size() for t in nonexp.values()) / 1e9:.2f} GB)")


# ----------------------------------------------------------------------------------------------- slow tier
def abuf(nbytes, pin):
    """4096-aligned uint8 host buffer (pinned on GPU hosts) + a writable memoryview of it."""
    raw = torch.empty(nbytes + BLK, dtype=U8, pin_memory=pin)
    o = (-raw.data_ptr()) % BLK
    t = raw[o:o + nbytes]
    return t, memoryview(t.numpy())


class Tier:
    def __init__(self, path, blob, E, direct=True):
        self.blob, self.E, self.path = blob, E, path
        self.direct = False
        self.lock = threading.Lock()
        self.nread = 0
        if direct:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                _, mv = abuf(4 * BLK, False)
                os.preadv(fd, [mv], 0)
                self.fd, self.direct = fd, True
            except OSError as ex:
                log(f"O_DIRECT unavailable ({ex}); buffered reads + FADV_DONTNEED")
        if not self.direct:
            self.fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def read(self, off, mv):
        n, got = len(mv), 0
        while got < n:
            r = os.preadv(self.fd, [mv[got:]], off + got)
            if r <= 0:
                raise IOError(f"short read at {off + got}")
            got += r
        if not self.direct:
            os.posix_fadvise(self.fd, off, n, os.POSIX_FADV_DONTNEED)
        with self.lock:
            self.nread += n

    def read_expert(self, L, e, mv):
        self.read((L * self.E + e) * self.blob, mv)


def nvme_test(tier, NL, E, pin, n=240):
    blob = tier.blob
    rng = np.random.default_rng(0)
    ids = rng.integers(0, NL * E, 4 * n)
    res = {"o_direct": tier.direct}
    b = abuf(blob, pin)
    t0 = time.perf_counter()
    for x in ids[:n]:
        tier.read(int(x) * blob, b[1])
    dt = time.perf_counter() - t0
    res["qd1_ms_per_expert"] = round(dt / n * 1e3, 3)
    res["qd1_GBs"] = round(n * blob / dt / 1e9, 3)
    t0 = time.perf_counter()
    tier.read(int(ids[0]) * blob, b[1])
    res["reread_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
    for qd in (4, 8, 16):
        bufs = [abuf(blob, pin) for _ in range(qd)]
        pool = ThreadPoolExecutor(qd)
        t0 = time.perf_counter()
        fs = [pool.submit(tier.read, int(x) * blob, bufs[i % qd][1]) for i, x in enumerate(ids[n:3 * n])]
        for f in fs:
            f.result()
        dt = time.perf_counter() - t0
        res[f"qd{qd}_GBs"] = round(2 * n * blob / dt / 1e9, 3)
        pool.shutdown()
    return res


def mem_free_bytes():
    avail = None
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) * 1024
    lim = None
    for p, u, st, fk in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat",
                          "file"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                          "/sys/fs/cgroup/memory/memory.stat", "total_cache")):
        try:
            v = open(p).read().strip()
            if v != "max" and int(v) < 1 << 60:
                cache = 0
                for line in open(st):
                    k_, v_ = line.split()[:2]
                    if k_ == fk:
                        cache = int(v_)
                lim = int(v) - (int(open(u).read().strip()) - cache)     # page cache is reclaimable
            break
        except (OSError, ValueError):
            continue
    return min(x for x in (avail, lim) if x is not None) if (avail or lim) else 0


def load_host_tier(tier, NL, E, cuda):
    blob = tier.blob
    n = NL * E * blob
    t0 = time.perf_counter()
    t, mv = abuf(n, False)
    pool = ThreadPoolExecutor(8)
    CH = 16 << 20
    for f in [pool.submit(tier.read, o, mv[o:min(n, o + CH)]) for o in range(0, n, CH)]:
        f.result()
    pool.shutdown()
    t1 = time.perf_counter()
    pinned = False
    if cuda:
        r = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), n, 0)
        pinned = "success" in str(r)          # torch's is_pinned() does not see host-registered memory
    log(f"host tier: {n / 1e9:.1f} GB read in {t1 - t0:.1f}s, pin {time.perf_counter() - t1:.1f}s pinned={pinned}")
    if cuda and not pinned:
        return None, {"error": f"cudaHostRegister failed ({r})"}
    return [t[L * E * blob:(L + 1) * E * blob].view(E, blob) for L in range(NL)], \
        {"load_s": round(t1 - t0, 1), "GB": round(n / 1e9, 2)}


# ----------------------------------------------------------------------------------------------- engine
class Job:
    __slots__ = ("L", "e", "slot", "st", "done", "ev")

    def __init__(self, L, e, slot):
        self.L, self.e, self.slot, self.st, self.done, self.ev = L, e, slot, QUEUED, threading.Event(), None


class Engine:
    def __init__(self, lay, NL, E, K, cap_max, dev, dt, tier, host=None, n_bg=4, n_urg=8):
        self.lay, self.NL, self.E, self.K, self.dev, self.dt, self.tier, self.host = lay, NL, E, K, dev, dt, tier, host
        self.cuda = dev.type == "cuda"
        self.cap_max = self.cap = cap_max
        self.pool = [torch.empty(cap_max, lay.blob, dtype=U8, device=dev) for _ in range(NL)]
        self.state = np.zeros((NL, E), np.int8)          # 0 absent, 1 resident, 2 loading in background
        self.slot_of = np.full((NL, E), -1, np.int64)
        self.slot_e = np.full((NL, cap_max), -1, np.int64)
        self.rec = np.full((NL, E), -1.0)
        self.pend = [dict() for _ in range(NL)]
        self.lock = threading.Lock()
        self.cs = torch.cuda.Stream(dev) if self.cuda else None
        self.ubufs = [abuf(lay.blob, self.cuda) for _ in range(2 * K)]
        self.ui = 0
        self.upool = ThreadPoolExecutor(n_urg)
        self.q = queue.Queue()
        self.err = None
        for _ in range(n_bg):
            threading.Thread(target=self._worker, daemon=True).start()
        self.lbuf = torch.empty(E, dtype=torch.float32, pin_memory=self.cuda)
        self.obuf = torch.empty(2 * K, dtype=torch.float32, pin_memory=self.cuda)
        self.lev = torch.cuda.Event() if self.cuda else None
        self.src, self.mode, self.tau, self.t, self.decoding = "nvme", "exact", 0.15, 0, False
        self.capture = None
        self.capture_pf = None
        self.sync_bg = False
        self.bk = None
        self.pf_gpu = None
        self.reset_stats()

    def reset_stats(self):
        self.S = dict(stall_s=0.0, wait_pending_s=0.0, n_urg=0, n_bg=0, n_cancel=0, n_drop=0, n_steal=0,
                      n_want=0, n_hit=0, n_sub=0, bytes=0)

    # ---------------------------------------------------------------- background loader
    def _worker(self):
        bufs = [abuf(self.lay.blob, self.cuda) for _ in range(2)]
        evs, i = [None, None], 0
        while True:
            job = self.q.get()
            try:
                if job is None:
                    return
                with self.lock:
                    if job.st != QUEUED:
                        continue
                    job.st = RUNNING
                ev = None
                if self.src == "host":
                    src_t = self.host[job.L][job.e]
                else:
                    if evs[i] is not None:
                        evs[i].synchronize()
                    self.tier.read_expert(job.L, job.e, bufs[i][1])
                    src_t = bufs[i][0]
                if self.cuda:
                    with torch.cuda.stream(self.cs):
                        self.pool[job.L][job.slot].copy_(src_t, non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(self.cs)
                    if self.src != "host":
                        evs[i] = ev
                        i ^= 1
                else:
                    self.pool[job.L][job.slot].copy_(src_t)
                job.ev = ev
                with self.lock:
                    job.st = DONE
                    self.S["n_bg"] += 1
                    self.S["bytes"] += self.lay.blob
                job.done.set()
            except Exception as ex:          # surfaced in the main thread
                self.err = ex
                job.st = DONE
                job.done.set()
            finally:
                self.q.task_done()

    def _poll(self, L):
        pd = self.pend[L]
        if pd:
            for e in [e for e, j in pd.items() if j.st == DONE and (j.ev is None or j.ev.query())]:
                self.state[L, e] = 1
                del pd[e]

    def _victim(self, L, prot):
        se = self.slot_e[L, :self.cap]
        c = se[(self.state[L, se] == 1) & ~prot[se]]
        if c.size == 0:
            return -1
        return int(c[np.argmin(self.rec[L, c])])

    def _assign(self, L, v, e):
        s = self.slot_of[L, v]
        self.slot_of[L, v] = -1
        self.state[L, v] = 0
        self.slot_of[L, e] = s
        self.slot_e[L, s] = e
        return int(s)

    def _wait_job(self, job):
        t0 = time.perf_counter()
        job.done.wait()
        d = time.perf_counter() - t0
        self.S["stall_s"] += d
        self.S["wait_pending_s"] += d
        if job.ev is not None:
            torch.cuda.current_stream().wait_event(job.ev)

    def _claim(self, L, e):
        """a background load of e is pending: take it over (queued) or wait for it (running)."""
        job = self.pend[L].pop(e)
        with self.lock:
            if job.st == QUEUED:
                job.st = CANCELLED
                self.S["n_cancel"] += 1
                return job.slot
        self._wait_job(job)
        return -1

    def _slot_for_must(self, L, e, prot, psel):
        v = self._victim(L, prot)
        if v >= 0:
            return self._assign(L, v, e)
        self.S["n_steal"] += 1
        if not self.pend[L]:                      # tiny caches only: every resident expert is protected
            v = self._victim(L, psel)             # then only the experts this token computes with are kept
            if v < 0:
                raise RuntimeError(f"no evictable slot in layer {L} (cap {self.cap})")
            return self._assign(L, v, e)
        pe, job = next(iter(self.pend[L].items()))      # every unprotected slot is loading: take one over
        del self.pend[L][pe]
        with self.lock:
            queued = job.st == QUEUED
            if queued:
                job.st = CANCELLED
        if not queued:
            self._wait_job(job)
        self.state[L, pe] = 1
        return self._assign(L, pe, e)

    def _urgent(self, L, pairs):
        t0 = time.perf_counter()
        if self.src == "host":
            for e, s in pairs:
                self.pool[L][s].copy_(self.host[L][e], non_blocking=True)
        else:
            bufs = []
            for _ in pairs:
                bufs.append(self.ubufs[self.ui])
                self.ui = (self.ui + 1) % len(self.ubufs)
            if len(pairs) == 1:
                self.tier.read_expert(L, pairs[0][0], bufs[0][1])
            else:
                for f in [self.upool.submit(self.tier.read_expert, L, e, b[1]) for (e, _), b in zip(pairs, bufs)]:
                    f.result()
            for (e, s), b in zip(pairs, bufs):
                self.pool[L][s].copy_(b[0], non_blocking=True)
        self.S["stall_s"] += time.perf_counter() - t0
        self.S["n_urg"] += len(pairs)
        self.S["bytes"] += len(pairs) * self.lay.blob

    # ---------------------------------------------------------------- routing on the host
    def route_cpu(self, L, z):
        K, E = self.K, self.E
        wi = np.argpartition(-z, K)[:K]
        wi = wi[np.argsort(-z[wi], kind="stable")]
        if self.mode == "ideal_sync":
            sel = wi
            slots = sel % self.cap
        else:
            if self.err is not None:
                raise self.err
            self._poll(L)
            st = self.state[L]
            rw = st[wi] == 1
            nh = int(rw.sum())
            self.S["n_want"] += K
            self.S["n_hit"] += nh
            must = bgm = wi[:0]
            sel = wi
            if nh < K:
                miss = wi[~rw]
                if self.mode == "exact":
                    must = miss
                elif self.mode != "B":
                    zw = z[wi].astype(np.float64)
                    p = np.exp(zw - zw[0])
                    p /= p.sum()
                    must = wi[~rw & ((np.arange(K) == 0) | (p >= self.tau))]
                bgm = miss[~np.isin(miss, must)]
                if bgm.size:
                    zz = np.where(st == 1, z, -np.inf)
                    zz[must] = z[must]
                    sel = np.argpartition(-zz, K)[:K]
                    sel = sel[np.argsort(-zz[sel], kind="stable")]
            self.rec[L, wi] = self.t
            self.rec[L, sel] = self.t
            if must.size or bgm.size:
                psel = np.zeros(E, bool)
                psel[sel] = True
                psel[must] = True
                prot = psel.copy()
                prot[wi] = True
                urg = []
                for e in must.tolist():
                    if e in self.pend[L]:
                        s = self._claim(L, e)
                        if s >= 0:
                            urg.append((e, s))
                    else:
                        urg.append((e, self._slot_for_must(L, e, prot, psel)))
                    self.state[L, e] = 1
                if urg:
                    self._urgent(L, urg)
                lim = max(self.cap - 3 * K, self.cap // 4)
                new_jobs = []
                for e in bgm.tolist():
                    if e in self.pend[L]:
                        continue
                    v = self._victim(L, prot) if len(self.pend[L]) < lim else -1
                    if v < 0:
                        self.S["n_drop"] += 1
                        continue
                    s = self._assign(L, v, e)
                    self.state[L, e] = 2
                    j = Job(L, e, s)
                    self.pend[L][e] = j
                    self.q.put(j)
                    new_jobs.append(j)
                if self.sync_bg:                      # test hook: background loads finish before the next token
                    for j in new_jobs:
                        j.done.wait()
                self.S["n_sub"] += K - int(np.isin(sel, wi).sum())
            slots = self.slot_of[L, sel]
        zs = z[sel].astype(np.float64)
        w = np.exp(zs - zs.max())
        return slots, w / w.sum()

    def start_route(self, lg):
        if self.cuda:
            self.lbuf.copy_(lg[0].float(), non_blocking=True)
            self.lev.record()

    def decode_moe(self, L, h, lg):
        K = self.K
        if self.mode == "ideal":
            w, idx = torch.softmax(lg.float(), -1).topk(K, -1)
            w = (w / w.sum(-1, keepdim=True)).to(self.dt)[0]
            slots = idx[0] % self.cap
        else:
            if self.cuda:
                self.lev.synchronize()
            else:
                self.lbuf.copy_(lg[0].float())
            z = self.lbuf.numpy()
            if self.capture is not None:
                self.capture[L].append(z.copy())
            s_np, w_np = self.route_cpu(L, z)
            ob = self.obuf.numpy()
            ob[:K] = s_np
            ob[K:] = w_np
            g = self.obuf.to(self.dev, non_blocking=True)
            slots = g[:K].long()
            w = g[K:].to(self.dt)
        v = self.lay.views(self.pool[L].index_select(0, slots))
        return experts_decode(v, h[0], w, self.dt)[None]

    # ---------------------------------------------------------------- prefill: stream every expert of each layer
    def prefill_begin(self, src):
        NL, E, blob = self.NL, self.E, self.lay.blob
        if self.pf_gpu is None:
            self.pf_gpu = [torch.empty(E, blob, dtype=U8, device=self.dev) for _ in range(2)]
            self.pf_host = [abuf(E * blob, self.cuda) for _ in range(2)]
            self.pf_pool = ThreadPoolExecutor(8)
        self.pf_ready = [threading.Event() for _ in range(NL)]
        self.pf_used = [threading.Event() for _ in range(NL)]
        self.pf_ev, self.pf_uev = [None] * NL, [None] * NL
        self.pf_wait, self.pf_src, self.pf_err = 0.0, src, None
        self.prompt_rec = np.full((NL, E), -1.0)
        self.pf_thread = threading.Thread(target=self._prefetch, daemon=True)
        self.pf_thread.start()

    def _prefetch(self):
        try:
            NL, E, blob = self.NL, self.E, self.lay.blob
            hev = [None, None]
            for L in range(NL):
                i = L % 2
                if L >= 2:
                    self.pf_used[L - 2].wait()
                if self.pf_src == "host":
                    src_t = self.host[L]
                else:
                    if hev[i] is not None:
                        hev[i].synchronize()
                    ht, mv = self.pf_host[i]
                    base, n, CH = L * E * blob, E * blob, 16 << 20
                    for f in [self.pf_pool.submit(self.tier.read, base + o, mv[o:min(n, o + CH)])
                              for o in range(0, n, CH)]:
                        f.result()
                    src_t = ht.view(E, blob)
                if self.cuda:
                    with torch.cuda.stream(self.cs):
                        if L >= 2:
                            self.cs.wait_event(self.pf_uev[L - 2])
                        self.pf_gpu[i].copy_(src_t, non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(self.cs)
                    self.pf_ev[L] = ev
                    if self.pf_src != "host":
                        hev[i] = ev
                else:
                    self.pf_gpu[i].copy_(src_t)
                self.pf_ready[L].set()
        except Exception as ex:
            self.pf_err = ex
            for e in self.pf_ready:
                e.set()

    def prefill_moe(self, L, h, lg):
        K, E = self.K, self.E
        w, idx = torch.softmax(lg.float(), -1).topk(K, -1)
        w = w / w.sum(-1, keepdim=True)
        N = h.shape[0]
        rec = torch.full((E,), -1.0, device=h.device)
        rec.scatter_reduce_(0, idx.reshape(-1), torch.arange(N, device=h.device, dtype=torch.float32)
                            .repeat_interleave(K), "amax")
        self.prompt_rec[L] = rec.cpu().numpy()
        if self.capture_pf is not None:
            self.capture_pf[L] = lg.float().cpu().numpy()
        t0 = time.perf_counter()
        self.pf_ready[L].wait()
        self.pf_wait += time.perf_counter() - t0
        if self.pf_err is not None:
            raise self.pf_err
        if self.cuda:
            torch.cuda.current_stream().wait_event(self.pf_ev[L])
        out = experts_prefill(self.lay, self.pf_gpu[L % 2], h, idx, w.to(self.dt), E, self.dt)
        if self.cuda:
            ev = torch.cuda.Event()
            ev.record()
            self.pf_uev[L] = ev
        self.pf_used[L].set()
        return out

    # ---------------------------------------------------------------- per-run reset: cache = most recent prompt experts
    def reset(self, cap, src, mode, tau, prompt_rec):
        self.q.join()
        if self.err is not None:
            raise self.err
        if self.cuda:
            torch.cuda.synchronize()                  # background copies still in flight on the copy stream
        NL, E, blob = self.NL, self.E, self.lay.blob
        self.cap, self.src, self.mode, self.tau = cap, src, mode, tau
        self.state[:] = 0
        self.slot_of[:] = -1
        self.slot_e[:] = -1
        for p in self.pend:
            p.clear()
        self.rec[:] = prompt_rec
        init = np.argsort(-prompt_rec, axis=1, kind="stable")[:, :cap]
        if self.bk is None:
            self.bk = [abuf(self.cap_max * blob, self.cuda) for _ in range(2)]
            self.bkev = [None, None]
        for L in range(NL):
            ids = init[L]
            self.state[L, ids] = 1
            self.slot_of[L, ids] = np.arange(cap)
            self.slot_e[L, :cap] = ids
            i = L % 2
            if self.bkev[i] is not None:
                self.bkev[i].synchronize()
            bt, mv = self.bk[i]
            dst = bt[:cap * blob].view(cap, blob)
            if self.host is not None:                 # same bytes either way; RAM is faster than re-reading NVMe
                torch.index_select(self.host[L], 0, torch.from_numpy(ids), out=dst)
            else:
                for f in [self.upool.submit(self.tier.read_expert, L, int(e), mv[j * blob:(j + 1) * blob])
                          for j, e in enumerate(ids)]:
                    f.result()
            self.pool[L][:cap].copy_(dst, non_blocking=True)
            if self.cuda:
                self.bkev[i] = torch.cuda.Event()
                self.bkev[i].record()
        if self.cuda:
            torch.cuda.synchronize()
        self.reset_stats()


class CachedMoE(nn.Module):
    def __init__(self, cfg, L, ref):
        super().__init__()
        from transformers.models.qwen3_next import modeling_qwen3_next as QN
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.shared_expert = QN.Qwen3NextMLP(cfg, intermediate_size=cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False)
        self.L, self.ref = L, ref

    def forward(self, x):
        B, T, H = x.shape
        h = x.reshape(-1, H)
        eng = self.ref[0]
        lg = self.gate(h)
        dec = T == 1 and eng.decoding
        if dec and eng.mode != "ideal":
            eng.start_route(lg)
        sh = self.shared_expert(h) * torch.sigmoid(self.shared_expert_gate(h))
        out = eng.decode_moe(self.L, h, lg) if dec else eng.prefill_moe(self.L, h, lg)
        return (out + sh).view(B, T, H)


def setup_fla(dev, want):
    from transformers.models.qwen3_next import modeling_qwen3_next as QN
    ok, err = False, None
    if want and dev.type == "cuda" and QN.fused_recurrent_gated_delta_rule is not None:
        try:
            g0 = torch.Generator(device=dev).manual_seed(0)

            def r(*s, dt=BF):
                return torch.randn(*s, device=dev, generator=g0).to(dt)
            q, k, v = r(1, 64, 4, 128), r(1, 64, 4, 128), r(1, 64, 4, 128)
            g = -torch.rand(1, 64, 4, device=dev, generator=g0)
            beta = torch.rand(1, 64, 4, device=dev, generator=g0).to(BF)
            s0 = r(1, 4, 128, 128, dt=torch.float32) * 0.05
            errs = []
            for fa, fb, n, init in ((QN.fused_recurrent_gated_delta_rule, QN.torch_recurrent_gated_delta_rule, 1, s0),
                                    (QN.chunk_gated_delta_rule, QN.torch_chunk_gated_delta_rule, 64, None)):
                o1, s1 = fa(q[:, :n], k[:, :n], v[:, :n], g=g[:, :n], beta=beta[:, :n], initial_state=init,
                            output_final_state=True, use_qk_l2norm_in_kernel=True)
                o2, s2 = fb(q[:, :n], k[:, :n], v[:, :n], g=g[:, :n], beta=beta[:, :n], initial_state=init,
                            output_final_state=True, use_qk_l2norm_in_kernel=True)
                errs.append(float((o1.float() - o2.float()).abs().max() / o2.float().abs().max().clamp_min(1e-6)))
                errs.append(float((s1.float() - s2.float()).abs().max() / s2.float().abs().max().clamp_min(1e-6)))
            ok = max(errs) < 3e-2
            err = errs
        except Exception as ex:
            err = repr(ex)[:300]
    if not ok:
        QN.fused_recurrent_gated_delta_rule = None
        QN.chunk_gated_delta_rule = None
        QN.FusedRMSNormGated = None
    log(f"fla kernels: {'on' if ok else 'off'} ({err})")
    return ok


def build_model(model_dir, ex_dir, dev, dt, ref):
    from transformers import AutoConfig
    from transformers.models.qwen3_next import modeling_qwen3_next as QN
    from safetensors.torch import load_file
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg._attn_implementation = "sdpa"
    cfg.dtype = dt
    small = copy.deepcopy(cfg)
    small.num_experts = 1                      # skip building 512 x 48 expert modules on meta
    with torch.device("meta"):
        model = QN.Qwen3NextForCausalLM(small)
        for L, layer in enumerate(model.model.layers):
            layer.mlp = CachedMoE(cfg, L, ref)
    model.config = cfg
    model.model.config = cfg
    sd = load_file(os.path.join(ex_dir, "nonexp.safetensors"), device=str(dev))
    sd = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.model.rotary_emb = QN.Qwen3NextRotaryEmbedding(config=cfg, device=dev)
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        assert not t.is_meta, f"meta tensor left: {n}"
    return cfg, model.eval()


# ----------------------------------------------------------------------------------------------- benchmark
def prefill(model, eng, cfg, ids, P, src):
    from transformers.models.qwen3_next import modeling_qwen3_next as QN
    cache = QN.Qwen3NextDynamicCache(cfg)
    eng.decoding = False
    eng.tier.nread = 0
    eng.prefill_begin(src)
    t0 = time.perf_counter()
    out = model.model(input_ids=ids[:, :P], past_key_values=cache, use_cache=True)
    model.lm_head(out.last_hidden_state[:, -1])
    if eng.cuda:
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    eng.pf_thread.join()
    return cache, eng.prompt_rec.copy(), {"ttft_s": ttft, "io_wait_s": eng.pf_wait, "GB": eng.tier.nread / 1e9}


def decode_run(model, eng, ids, P, D, snap, dev):
    cache = copy.deepcopy(snap)
    eng.decoding = True
    cps = torch.arange(P + D + 1, device=dev)
    nll = torch.zeros(D, device=dev)
    am = torch.zeros(D, dtype=torch.long, device=dev)
    times = np.zeros(D)
    for i in range(D):
        pos = P + i
        eng.t = pos
        t0 = time.perf_counter()
        out = model.model(input_ids=ids[:, pos:pos + 1], past_key_values=cache, use_cache=True,
                          cache_position=cps[pos:pos + 1])
        lg = model.lm_head(out.last_hidden_state[:, -1]).float()[0]
        nll[i] = torch.logsumexp(lg, -1) - lg[ids[0, pos + 1]]
        am[i] = lg.argmax()
        if eng.cuda:
            torch.cuda.synchronize()
        times[i] = time.perf_counter() - t0
    eng.decoding = False
    return times, nll.cpu().numpy(), am.cpu().numpy()


def pick_sessions(dat, per_domain):
    dom = dat["domain"].astype(np.int64)
    names = [str(s) for s in dat["domain_names"]]
    sel = []
    for d in range(len(names)):
        idx = np.where(dom == d)[0]
        pk = idx[1::2] if len(idx) > 1 else idx
        sel += [int(x) for x in pk[:per_domain]]
    return sel, names


@torch.no_grad()
def bench(a):
    gc.disable()
    dev = torch.device(a.device)
    dt = BF if a.dtype == "bf16" else torch.float32
    cuda = dev.type == "cuda"
    fla = setup_fla(dev, not a.no_fla)
    from transformers import AutoConfig
    cfg0 = AutoConfig.from_pretrained(a.model_dir)
    H, I, E, K, NL = (cfg0.hidden_size, cfg0.moe_intermediate_size, cfg0.num_experts, cfg0.num_experts_per_tok,
                      cfg0.num_hidden_layers)
    lay = Layout(H, I)
    tier = Tier(os.path.join(a.ex_dir, "experts.bin"), lay.blob, E, direct=not a.no_direct)
    caps = [float(c) for c in a.caps.split(",")]
    capn = {c: max(K + 1, int(round(c * E))) for c in caps}
    meta = {"model": a.model_dir, "E": E, "K": K, "layers": NL, "blob_MB": lay.blob / 1e6,
            "caps": {str(c): n for c, n in capn.items()}, "fla": fla, "o_direct": tier.direct, "dtype": a.dtype,
            "torch": torch.__version__, "cpus": os.cpu_count()}
    if cuda:
        p = torch.cuda.get_device_properties(dev)
        meta.update(gpu=p.name, gpu_GB=round(p.total_memory / 1e9, 1))
    meta["ram_free_GB"] = round(mem_free_bytes() / 1e9, 1)
    log("meta", json.dumps(meta))
    if not a.no_nvme_test:
        meta["nvme"] = nvme_test(tier, NL, E, cuda)
        log("nvme", json.dumps(meta["nvme"]))

    host = None
    need = NL * E * lay.blob
    if a.host == "on" or (a.host == "auto" and mem_free_bytes() > need + a.host_margin_gb * 1e9):
        try:
            host, hinfo = load_host_tier(tier, NL, E, cuda)
        except Exception as ex:
            host, hinfo = None, {"error": repr(ex)[:200]}
        meta["host_tier"] = hinfo
    else:
        meta["host_tier"] = {"skipped": f"free {mem_free_bytes() / 1e9:.1f} GB < need {need / 1e9:.1f} + margin"}
    log("host tier", json.dumps(meta["host_tier"]))

    eng = Engine(lay, NL, E, K, max(capn.values()), dev, dt, tier, host=host)
    ref = [eng]
    cfg, model = build_model(a.model_dir, a.ex_dir, dev, dt, ref)
    if cuda:
        meta["vram_alloc_GB"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    dat = np.load(a.data, allow_pickle=True)
    tokens = dat["tokens"].astype(np.int64)
    P = int(dat["prompt_len"])
    sel, names = pick_sessions(dat, a.per_domain)
    if a.max_sessions:
        sel = sel[:a.max_sessions]
    D = min(a.decode, tokens.shape[1] - P - 1)
    dom = dat["domain"].astype(np.int64)
    meta.update(P=P, D=D, sessions=len(sel), domains=[names[dom[s]] for s in sel])
    cmax = capn[max(caps)]
    modes = a.modes.split(",")
    plan = [("ideal", cmax, "nvme"), ("ideal_sync", cmax, "nvme")]
    for c in caps:
        plan += [(m, capn[c], "nvme") for m in modes]
    if host is not None:
        for c in caps:
            plan += [(m, capn[c], "host") for m in a.host_modes.split(",")]
    tau_of = {"C1": 2.0, "C": a.tau, "exact": 0.0, "B": 0.0, "ideal": 0.0, "ideal_sync": 0.0}
    mode_of = {"C1": "C"}
    log(f"sessions {len(sel)} x (prompt {P} + decode {D}); plan: {[f'{m}@{c}/{s}' for m, c, s in plan]}")

    # warm-up (kernel compilation, allocator) on the first session
    ids0 = torch.from_numpy(tokens[sel[0]][None]).to(dev)
    snap, prec, _ = prefill(model, eng, cfg, ids0, P, "nvme")
    for m, n in (("C", 12), ("ideal", 6), ("exact", 6)):
        eng.reset(cmax, "nvme", m, 2.0, prec)
        decode_run(model, eng, ids0, P, n, snap, dev)
    del snap
    gc.collect()

    runs = {}
    ttft = []
    errors = []
    t_all = time.time()
    for si, s in enumerate(sel):
        ids = torch.from_numpy(tokens[s][None]).to(dev)
        snap, prec, pi = prefill(model, eng, cfg, ids, P, "nvme")
        ttft.append({"src": "nvme", **pi})
        if host is not None and si == 0:
            _, _, pi2 = prefill(model, eng, cfg, ids, P, "host")
            ttft.append({"src": "host", **pi2})
        for (m, cap, src) in plan:
            key = f"{m}@{cap}/{src}"
            try:
                eng.reset(cap, src, mode_of.get(m, m), tau_of[m], prec)
                gc.collect()
                tm, nll, am = decode_run(model, eng, ids, P, D, snap, dev)
                eng.q.join()
            except Exception as ex:
                import traceback
                traceback.print_exc()
                log(f"RUN FAILED {key}: {ex!r}")
                eng.err = None
                errors.append(f"{key}: {ex!r}"[:300])
                continue
            r = runs.setdefault(key, {"mode": m, "cap": cap, "src": src, "times": [], "nll": [], "am": [],
                                      "S": {k: 0.0 for k in eng.S}})
            r["times"].append(tm)
            r["nll"].append(nll)
            r["am"].append(am)
            for k, v in eng.S.items():
                r["S"][k] += v
            log(f"session {si + 1}/{len(sel)} {key}: {tm.mean() * 1e3:.1f} ms/tok (p50 {np.median(tm) * 1e3:.1f}) "
                f"stall {eng.S['stall_s'] / D * 1e3:.2f} ms/tok hit {eng.S['n_hit'] / max(1, eng.S['n_want']):.3f} "
                f"urg {eng.S['n_urg'] / D:.2f} bg {eng.S['n_bg'] / D:.2f} drop {eng.S['n_drop'] / D:.2f} "
                f"ppl {np.exp(nll.mean()):.3f}")
        del snap
        gc.collect()
    log(f"all runs done in {time.time() - t_all:.0f}s")

    # ---------------------------------------------------------------- summary
    ref_key = f"exact@{capn[caps[0]]}/nvme"
    ref_nll = np.concatenate(runs[ref_key]["nll"]) if ref_key in runs else None
    ref_am = np.concatenate(runs[ref_key]["am"]) if ref_key in runs else None
    ideal_ms = np.concatenate(runs[f"ideal@{cmax}/nvme"]["times"]).mean() * 1e3
    qd = (meta.get("nvme") or {}).get("qd8_GBs") or 3.0
    summ = {}
    for key, r in runs.items():
        tm = np.concatenate(r["times"]) * 1e3
        n = tm.size
        S = r["S"]
        d = {"ms_mean": round(float(tm.mean()), 2), "ms_p50": round(float(np.median(tm)), 2),
             "ms_p90": round(float(np.percentile(tm, 90)), 2), "tok_s": round(1e3 / float(tm.mean()), 2),
             "rel_ideal": round(ideal_ms / float(tm.mean()), 4)}
        if r["mode"] not in ("ideal", "ideal_sync"):
            nll = np.concatenate(r["nll"])
            stall = S["stall_s"] * 1e3 / n
            mb = S["bytes"] / n / 1e6
            d.update(stall_ms=round(stall, 3), wait_pending_ms=round(S["wait_pending_s"] * 1e3 / n, 3),
                     urg_per_tok=round(S["n_urg"] / n, 2), bg_per_tok=round(S["n_bg"] / n, 2),
                     drop_per_tok=round(S["n_drop"] / n, 2), cancel_per_tok=round(S["n_cancel"] / n, 3),
                     steal_per_tok=round(S["n_steal"] / n, 3), MB_per_tok=round(mb, 2),
                     hit=round(S["n_hit"] / max(1, S["n_want"]), 4), subst=round(S["n_sub"] / max(1, S["n_want"]), 4),
                     ppl=round(float(np.exp(nll.mean())), 4),
                     other_ms=round(float(tm.mean()) - ideal_ms - stall, 2))
            if ref_nll is not None:
                d["dppl_pct"] = round(100 * (float(np.exp(nll.mean() - ref_nll.mean())) - 1), 3)
                d["agree"] = round(float((np.concatenate(r["am"]) == ref_am).mean()), 4)
            for tc in (21.0, ideal_ms):
                io = mb / qd if r["src"] == "nvme" else 0.0     # ms of NVMe time per token at QD8 bandwidth
                d[f"proj_tc{tc:.0f}"] = round(tc / max(tc + stall, io), 4)
        summ[key] = d
    res = {"meta": meta, "ideal_ms": round(float(ideal_ms), 2), "ttft": ttft, "runs": summ, "errors": errors}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "live.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "live_raw.npz"),
                        **{k.replace("/", "_"): np.concatenate(r["times"]) for k, r in runs.items()})
    print("SECTION_LIVE " + json.dumps(res, separators=(",", ":")), flush=True)
    for k, d in summ.items():
        log(f"{k:24s} " + " ".join(f"{kk}={vv}" for kk, vv in d.items()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["convert", "bench"])
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--ex_dir", required=True)
    ap.add_argument("--flag_dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--per_domain", type=int, default=1)
    ap.add_argument("--max_sessions", type=int, default=0)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--caps", default="0.25,0.125")
    ap.add_argument("--modes", default="exact,C1,C,B")
    ap.add_argument("--host_modes", default="exact,C1")
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--host", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--host_margin_gb", type=float, default=8.0)
    ap.add_argument("--no_direct", action="store_true")
    ap.add_argument("--no_fla", action="store_true")
    ap.add_argument("--no_nvme_test", action="store_true")
    a = ap.parse_args()
    if a.cmd == "convert":
        convert(a)
    else:
        bench(a)


if __name__ == "__main__":
    main()
