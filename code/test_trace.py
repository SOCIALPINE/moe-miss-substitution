"""Replay an engine decision trace (LLAMA_XCACHE_TRACE, run with sbg=1 so background reads land before the next token)
through an independent implementation of the policy and report every disagreement.
Covers: C rule (tau / C1 / exact), substitution selection, LRU or frequency score with supply/demand keying,
victim choice, rent-or-buy admission (rb, rh, rq), per-layer caps, frequency-score init from the prompt."""
import ctypes, math, sys

libm = ctypes.CDLL("libm.so.6")
libm.exp2.restype = ctypes.c_double
libm.exp2.argtypes = [ctypes.c_double]
exp2 = libm.exp2


class Layer:
    pass


def main(path, E, K):
    st = {}
    cur = None
    n = dict(D=0, H=0, G=0, mis=0, cfg=0, init_fe=0)
    first = []
    mass = 0.0

    def mismatch(msg):
        n["mis"] += 1
        if len(first) < 15:
            first.append(msg)

    def victim(s, prot):
        best, br = -1, 0.0
        for sl in range(s.cap):
            e = s.slot_e[sl]
            if e < 0 or s.state[e] != 1 or e in prot:
                continue
            if best < 0 or s.rec[e] < br:
                best, br = e, s.rec[e]
        return best

    def assign(s, v, e):
        sl = s.slot_of[v]
        s.slot_of[v] = -1
        s.state[v] = 0
        s.slot_of[e] = sl
        s.slot_e[sl] = e

    def touch(s, e, t):
        if s.lam <= 0:
            s.rec[e] = t
            return
        k = s.rec[e]
        sc = 0.0 if k < -1e29 else exp2(k - s.lam * t)
        s.rec[e] = math.log2(sc + 1.0) + s.lam * t

    for line in open(path):
        f = line.split()
        if not f:
            continue
        tag = f[0]
        if tag == "CFG":
            s = Layer()
            s.il, s.cap, t0, s.mode = int(f[1]), int(f[2]), float(f[3]), int(f[4])
            s.tau, s.lam, s.dk, s.rb, s.rh, s.rq = float(f[5]), float(f[6]), int(f[7]), float(f[8]), float(f[9]), int(f[10])
            st[s.il] = s
            cur = s
            n["cfg"] += 1
        elif tag == "S":
            cur.slot_e = [int(x) for x in f[1:]]
            assert len(cur.slot_e) == cur.cap
            cur.state = [0] * E
            cur.slot_of = [-1] * E
            for sl, e in enumerate(cur.slot_e):
                if e >= 0:
                    cur.state[e] = 1
                    cur.slot_of[e] = sl
            cur.rv = [0.0] * E
            cur.rt = [0.0] * E
            cur.pend = []
        elif tag == "R":
            cur.rec = [float(x) for x in f[1:]]
        elif tag == "P":
            pids = [int(x) for x in f[1:]]
            if cur.lam > 0 and pids:
                # frequency key from the prompt: log2(sum 2^(lam*pos)); check it when the engine used it (option fe)
                key = [-1e30] * E
                for i, e in enumerate(pids):
                    if e < 0:
                        continue
                    x = cur.lam * (i // K)
                    k = key[e]
                    key[e] = x if k < -1e29 else max(k, x) + math.log2(1.0 + exp2(-abs(k - x)))
                lru_like = all((cur.rec[e] < -1e29) == (key[e] < -1e29) for e in range(E))
                if lru_like and max(abs(cur.rec[e] - key[e]) for e in range(E) if key[e] > -1e29) < 1e-9:
                    n["init_fe"] += 1
                    order = sorted(range(E), key=lambda e: (-key[e], e))
                    if set(order[: cur.cap]) != {e for e in cur.slot_e if e >= 0}:
                        mismatch(f"init set layer {cur.il}")
        elif tag in ("H", "D"):
            il, t = int(f[1]), float(f[2])
            s = st[il]
            for e in s.pend:
                s.state[e] = 1
            s.pend = []
            if tag == "H":
                n["H"] += 1
                W = [int(x) for x in f[3:]]
                if not all(s.state[e] == 1 for e in W):
                    mismatch(f"H not all resident layer {il} t {t}")
                for e in W:
                    touch(s, e, t)
                continue
            n["D"] += 1
            parts, key = {}, None
            for x in f[3:]:
                if x in ("W", "M", "B", "G", "L", "Z"):
                    key = x
                    parts[key] = []
                else:
                    parts[key].append(x)
            W = [int(x) for x in parts["W"]]
            M = [int(x) for x in parts["M"]]
            B = [int(x) for x in parts["B"]]
            G = [int(x) for x in parts["G"]]
            Lsel = [int(x) for x in parts["L"]]
            z = [float(x) for x in parts["Z"]]
            hit = [s.state[e] == 1 for e in W]
            if all(hit):
                mismatch(f"D line with all hits layer {il} t {t}")
            p = [math.exp(z[W[k]] - z[W[0]]) for k in range(K)]
            sm = sum(p)
            must, bgm, bgp = [], [], []
            for k in range(K):
                if hit[k]:
                    continue
                if s.mode == 2:
                    wait = True
                elif s.mode == 3:
                    wait = k == 0 or p[k] / sm >= s.tau
                else:
                    wait = False
                if wait:
                    must.append(W[k])
                else:
                    bgm.append(W[k])
                    bgp.append(p[k] / sm)
                    mass += p[k] / sm
            if must != M or bgm != B:
                mismatch(f"wait split layer {il} t {t}: ref M{must} B{bgm} eng M{M} B{B}")
            sel = list(W)
            if bgm:
                cand = [e for e in range(E) if s.state[e] == 1] + must
                cand.sort(key=lambda e: (-z[e], e))
                sel = cand[:K]
                while len(sel) < K:
                    sel.append(sel[-1])
            if sel != Lsel:
                mismatch(f"selection layer {il} t {t}: ref {sel} eng {Lsel}")
            for e in W:
                touch(s, e, t)
            if not s.dk:
                for e in sel:
                    if e not in W:
                        touch(s, e, t)
            psel = set(sel) | set(must)
            prot = psel | set(W)
            for e in must:
                v = victim(s, prot)
                if v < 0:
                    v = victim(s, psel)
                    assert v >= 0
                assign(s, v, e)
                s.state[e] = 1
            lim = max(s.cap - 3 * K, s.cap // 4)
            Gr = []
            if s.rb > 0:
                lam_r = math.log(2.0) / s.rh
                for i, e in enumerate(bgm):
                    if e in s.pend:
                        continue
                    r = s.rv[e] * math.exp(-lam_r * (t - s.rt[e])) + bgp[i]
                    s.rt[e] = t
                    s.rv[e] = r
                    if r < s.rb:
                        continue
                    v = victim(s, prot) if len(s.pend) < min(lim, s.rq) else -1
                    if v < 0:
                        continue
                    s.rv[e] = 0.0
                    Gr += [e, v]
                    assign(s, v, e)
                    s.state[e] = 2
                    s.pend.append(e)
            n["G"] += len(Gr) // 2
            if Gr != G:
                mismatch(f"admission layer {il} t {t}: ref {Gr} eng {G}")
                # follow the engine so that one difference does not cascade
    print(f"cfg {n['cfg']} (fe init checked {n['init_fe']})  D {n['D']}  H {n['H']}  admissions {n['G']}  mismatches {n['mis']}  "
          f"mass/layer-token {mass / max(1, n['D'] + n['H']):.5f}")
    for m in first:
        print("  ", m)
    return n["mis"]


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])) else 0)
