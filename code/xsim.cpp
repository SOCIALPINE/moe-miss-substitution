// xsim: replay dumped router logits (xdump) through expert-cache policies, per MoE layer.
//
//   xsim policy SET P D CAP TAU K BINW CFGFILE files...   one JSON line per config: hit rate, waited loads, substitution
//   xsim curves SET P TAU K files...                      working-set curves s(w) and back-distance histograms
//
// Decode model (one layer, token t): wanted R = top-K by logit, normalized gates p (softmax over R). J = max(1, #{p_k >= tau})
// (or 1 / K). Must set U = top-J experts not resident: loaded synchronously and admitted (victim chosen by the policy score).
// Other misses are substituted: the final selection is the top-K by logit among resident experts and U. If fewer than K
// experts are resident or in U, the best substituted misses become waited loads too.
//
// Background reads go through one channel per layer at BRATE reads per token (env XSIM_BRATE, default 0.5) and arrive
// BDELAY tokens after their read ends (env XSIM_BDELAY, default 1).
//   every:D        every D tokens the resident set becomes the top-CAP experts by score: experts outside it are dropped at once,
//                  the new ones stream in best first into reserved slots (round 9)
//   lazy:D         same target, but an old expert stays usable until a new one arrives: each arrival replaces the
//                  lowest-scored resident outside the target (read into a staging buffer, slot overwritten on arrival)
//   drift:W:N / ldrift:W:N   refresh (eager / lazy) when the last W tokens had N or more waited loads, at most every W tokens
//   bg:Q           per-miss streaming: every substituted miss is read in the background (dropped if the channel is more than Q
//                  tokens behind) and admitted on arrival in place of the lowest-scored resident
//
// Config line: name score key batch j [bins]
//   score: lru | fe:<alpha> | dmax:<kappa> | tier:<cA>    key: d (wanted only) | s (wanted and substitutes)
//   batch: none | every:<D> | lazy:<D> | drift:<W>:<N> | ldrift:<W>:<N> | bg:<Q>      j: tau | c1 | all
//   bins: also print hit / waited loads / mass / batch reads per position bin
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
    if (e == 0) {
        if (m == 0) {
            x = s;
        } else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) {
                m <<= 1;
                e--;
            }
            m &= 0x3ff;
            x = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        x = s | 0x7f800000 | (m << 13);
    } else {
        x = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

struct seqfile {
    std::string path;
    int T = 0, L = 0, N = 0;
};

static seqfile open_seq(const std::string & p) {
    seqfile s;
    s.path = p;
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", p.c_str());
        exit(1);
    }
    int32_t h[3];
    if (fread(h, sizeof(h), 1, f) != 1) {
        fprintf(stderr, "bad header %s\n", p.c_str());
        exit(1);
    }
    fclose(f);
    s.T = h[0];
    s.L = h[1];
    s.N = h[2];
    return s;
}

static std::vector<float> load_layer(const seqfile & s, int l) {
    std::vector<uint16_t> b((size_t) s.T * s.N);
    FILE * f = fopen(s.path.c_str(), "rb");
    fseek(f, 12 + (long) l * s.T * s.N * 2, SEEK_SET);
    if (fread(b.data(), 2, b.size(), f) != b.size()) {
        fprintf(stderr, "short read %s layer %d\n", s.path.c_str(), l);
        exit(1);
    }
    fclose(f);
    std::vector<float> z(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
        z[i] = h2f(b[i]);
    }
    return z;
}

// per token: experts sorted by logit (first M), normalized gates of the top K, J by tau
struct unit {
    int T = 0, N = 0, M = 0, K = 0;
    std::vector<float> z;
    std::vector<int32_t> ord; // T x M
    std::vector<float> p;     // T x K
    std::vector<int8_t> jt;   // tau rule J per token
};

static void prep(unit & u, std::vector<float> && z, int T, int N, int K, double tau) {
    u.T = T;
    u.N = N;
    u.K = K;
    u.M = std::min(N, 128);
    u.z = std::move(z);
    u.ord.resize((size_t) T * u.M);
    u.p.resize((size_t) T * K);
    u.jt.resize(T);
    std::vector<int32_t> idx(N);
    for (int t = 0; t < T; ++t) {
        const float * zt = &u.z[(size_t) t * N];
        for (int e = 0; e < N; ++e) {
            idx[e] = e;
        }
        std::partial_sort(idx.begin(), idx.begin() + u.M, idx.end(), [&](int a, int b) { return zt[a] > zt[b] || (zt[a] == zt[b] && a < b); });
        std::copy(idx.begin(), idx.begin() + u.M, u.ord.begin() + (size_t) t * u.M);
        double sum = 0;
        double ex[64];
        for (int k = 0; k < K; ++k) {
            ex[k] = std::exp((double) zt[idx[k]] - zt[idx[0]]);
            sum += ex[k];
        }
        int j = 0;
        for (int k = 0; k < K; ++k) {
            u.p[(size_t) t * K + k] = (float) (ex[k] / sum);
            if (ex[k] / sum >= tau) {
                j = k + 1;
            }
        }
        u.jt[t] = (int8_t) std::max(1, j);
    }
}

struct cfg {
    std::string name, score, key, batch, j, alloc = "none";
    double par = 0;     // alpha / kappa / cA
    int delta = 0, dw = 0, dthr = 0, qmax = 0;
    bool lazy = false, bins = false;
    double rbeta = 0, rhalf = 0; // regret admission: buy cost (gate mass), memory (tokens)
    bool rguard = false;
};

static cfg parse_cfg(const std::string & line) {
    std::istringstream is(line);
    cfg c;
    is >> c.name >> c.score >> c.key >> c.batch >> c.j;
    for (std::string o; is >> o;) {
        if (o == "bins") {
            c.bins = true;
        } else if (o.rfind("alloc:", 0) == 0) {
            c.alloc = o.substr(6);
            if (c.alloc != "pm" && c.alloc != "pw" && c.alloc != "om") {
                fprintf(stderr, "bad alloc in: %s\n", line.c_str());
                exit(1);
            }
        } else {
            fprintf(stderr, "bad option %s in: %s\n", o.c_str(), line.c_str());
            exit(1);
        }
    }
    if (c.j != "tau" && c.j != "c1" && c.j != "all" && c.j != "none") {
        fprintf(stderr, "bad j in: %s\n", line.c_str());
        exit(1);
    }
    auto colon = c.score.find(':');
    if (colon != std::string::npos) {
        c.par = atof(c.score.substr(colon + 1).c_str());
        c.score = c.score.substr(0, colon);
    }
    if (c.batch.rfind("every:", 0) == 0) {
        c.delta = atoi(c.batch.c_str() + 6);
        c.batch = "every";
    } else if (c.batch.rfind("lazy:", 0) == 0) {
        c.delta = atoi(c.batch.c_str() + 5);
        c.batch = "every";
        c.lazy = true;
    } else if (c.batch.rfind("drift:", 0) == 0) {
        sscanf(c.batch.c_str() + 6, "%d:%d", &c.dw, &c.dthr);
        c.batch = "drift";
    } else if (c.batch.rfind("ldrift:", 0) == 0) {
        sscanf(c.batch.c_str() + 7, "%d:%d", &c.dw, &c.dthr);
        c.batch = "drift";
        c.lazy = true;
    } else if (c.batch.rfind("bg:", 0) == 0) {
        c.qmax = atoi(c.batch.c_str() + 3);
        c.batch = "bg";
        c.lazy = true;
    } else if (c.batch.rfind("regret:", 0) == 0) {
        // regret:BETA:HALF[:g] -- rent-or-buy admission of substituted experts
        char g[8] = { 0 };
        const int nf = sscanf(c.batch.c_str() + 7, "%lf:%lf:%7s", &c.rbeta, &c.rhalf, g);
        if (nf < 2 || c.rbeta <= 0 || c.rhalf <= 0) {
            fprintf(stderr, "bad regret spec in: %s\n", line.c_str());
            exit(1);
        }
        c.rguard = nf == 3 && g[0] == 'g';
        c.qmax = 16;
        c.batch = "regret";
        c.lazy = true;
    } else if (c.batch != "none") {
        fprintf(stderr, "bad batch spec in: %s\n", line.c_str());
        exit(1);
    }
    if ((c.batch == "every" && c.delta <= 0) || (c.batch == "drift" && c.dw <= 0)) {
        fprintf(stderr, "bad batch period in: %s\n", line.c_str());
        exit(1);
    }
    return c;
}

struct acc {
    double n = 0, hit = 0, must = 0, must_pend = 0, sub = 0, mass = 0, batch = 0, refresh = 0;
    std::vector<double> bn, bhit, bmust, bmass, bbatch;
    void bins(int nb) {
        bn.assign(nb, 0);
        bhit.assign(nb, 0);
        bmust.assign(nb, 0);
        bmass.assign(nb, 0);
        bbatch.assign(nb, 0);
    }
    void add(const acc & o) {
        n += o.n;
        hit += o.hit;
        must += o.must;
        must_pend += o.must_pend;
        sub += o.sub;
        mass += o.mass;
        batch += o.batch;
        refresh += o.refresh;
        for (size_t i = 0; i < bn.size(); ++i) {
            bn[i] += o.bn[i];
            bhit[i] += o.bhit[i];
            bmust[i] += o.bmust[i];
            bmass[i] += o.bmass[i];
            bbatch[i] += o.bbatch[i];
        }
    }
};

static constexpr double NEG = -1e300;

static double logaddexp(double a, double b) {
    if (a < b) {
        std::swap(a, b);
    }
    if (b <= NEG) {
        return a;
    }
    return a + std::log1p(std::exp(b - a));
}

struct sim {
    const unit & u;
    const cfg & c;
    int cap, P, D, binw, bdelay;
    double brate = 0.5; // background reads per token for this layer (spare SSD bandwidth share)
    std::vector<char> res, keep;
    std::vector<int> pend; // arrival token of a background read in flight, -1 none
    std::vector<double> k1, kc;
    double beta = 0;
    int nres = 0, nresv = 0; // resident experts, slots reserved for eager batch reads
    std::vector<double> rv;  // regret: substituted gate mass, decayed with half-life rhalf
    std::vector<int> rt;     // token of the last regret update

    sim(const unit & u_, const cfg & c_, int cap_, int P_, int D_, int binw_, int bdelay_)
        : u(u_), c(c_), cap(cap_), P(P_), D(D_), binw(binw_), bdelay(bdelay_) {
        res.assign(u.N, 0);
        keep.assign(u.N, 0);
        rv.assign(u.N, 0.0);
        rt.assign(u.N, 0);
        pend.assign(u.N, -1);
        k1.assign(u.N, NEG);
        kc.assign(u.N, NEG);
        if (c.score == "fe" || c.score == "few") {
            beta = -std::log(1.0 - c.par);
        }
        if (c.score != "lru" && c.score != "tier" && c.score != "dmax" && c.score != "fe" && c.score != "few") {
            fprintf(stderr, "bad score %s\n", c.score.c_str());
            exit(1);
        }
        if (getenv("XSIM_BRATE")) {
            brate = atof(getenv("XSIM_BRATE"));
        }
    }

    void put(int e) {
        if (!res[e]) {
            res[e] = 1;
            nres++;
        }
    }

    void drop(int e) {
        if (res[e]) {
            res[e] = 0;
            nres--;
        }
    }

    void cancel(int e) {
        if (pend[e] >= 0) {
            pend[e] = -1;
            if (!c.lazy) {
                nresv--;
            }
        }
    }

    void upd(int e, double q, int t, bool core) {
        if (c.score == "lru" || c.score == "tier") {
            k1[e] = t;
        } else if (c.score == "dmax") {
            k1[e] = std::max(k1[e], std::log(std::max(q, 1e-12)) + c.par * t);
        } else if (c.score == "fe") {
            k1[e] = logaddexp(k1[e], beta * t);
        } else if (c.score == "few") {
            k1[e] = logaddexp(k1[e], beta * t + std::log(std::max(q, 1e-12)));
        }
        if (core) {
            kc[e] = t;
        }
    }

    // experts ordered by score, best first (ties: lower id first)
    std::vector<int> ranked(const std::vector<double> & k) const {
        std::vector<int> o(u.N);
        for (int e = 0; e < u.N; ++e) {
            o[e] = e;
        }
        std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return k[a] > k[b]; });
        return o;
    }

    // the CAP best experts by score. The initial set fills up with never-used experts like the engine does (ties by id);
    // a batch refresh leaves them out (no point reading them).
    std::vector<char> target(std::vector<int> * order = nullptr, bool skip_unused = false) const {
        std::vector<char> tg(u.N, 0);
        int n = 0;
        if (c.score == "tier") {
            const int ca = std::min(cap, (int) c.par);
            for (int e : ranked(kc)) {
                if (n >= ca || kc[e] <= NEG) {
                    break;
                }
                tg[e] = 1;
                n++;
                if (order) {
                    order->push_back(e);
                }
            }
        }
        for (int e : ranked(k1)) {
            if (n >= cap || (skip_unused && k1[e] <= NEG)) {
                break;
            }
            if (!tg[e]) {
                tg[e] = 1;
                n++;
                if (order) {
                    order->push_back(e);
                }
            }
        }
        return tg;
    }

    int victim(const std::vector<char> & prot) const {
        std::vector<char> inA;
        if (c.score == "tier") {
            const int ca = (int) c.par;
            std::vector<int> r;
            for (int e = 0; e < u.N; ++e) {
                if (res[e] && kc[e] > NEG) {
                    r.push_back(e);
                }
            }
            std::stable_sort(r.begin(), r.end(), [&](int a, int b) { return kc[a] > kc[b]; });
            inA.assign(u.N, 0);
            for (int i = 0; i < (int) r.size() && i < ca; ++i) {
                inA[r[i]] = 1;
            }
        }
        int best = -1;
        for (int e = 0; e < u.N; ++e) {
            if (!res[e] || prot[e] || (!inA.empty() && inA[e])) {
                continue;
            }
            if (best < 0 || k1[e] < k1[best]) {
                best = e;
            }
        }
        if (best < 0 && !inA.empty()) {
            for (int e = 0; e < u.N; ++e) {
                if (!res[e] || prot[e]) {
                    continue;
                }
                if (best < 0 || kc[e] < kc[best]) {
                    best = e;
                }
            }
        }
        return best;
    }

    void run(acc & a) {
        const int K = u.K, N = u.N, M = u.M;
        // prompt: wanted experts only, then the initial set
        for (int t = 0; t < P; ++t) {
            const int J = jcount(t);
            for (int k = 0; k < K; ++k) {
                upd(u.ord[(size_t) t * M + k], u.p[(size_t) t * K + k], t, k < J);
            }
        }
        {
            auto tg = target();
            for (int e = 0; e < N; ++e) {
                if (tg[e]) {
                    put(e);
                }
            }
        }
        std::vector<char> prot(N, 0), inU(N, 0);
        std::vector<int> R(K), Rp(K), U;
        std::vector<int> ring(std::max(1, c.dw), 0);
        int ring_sum = 0, last_refresh = P;
        double chan = 0;        // background channel busy until (token units)
        std::vector<int> plist; // experts with a background read in flight (may repeat; pend[] is the truth)
        auto schedule = [&](int e, int t) {
            chan = std::max(chan, (double) (t + 1)) + 1.0 / brate;
            pend[e] = (int) std::ceil(chan) + bdelay;
            plist.push_back(e);
            if (!c.lazy) {
                nresv++;
            }
        };
        for (int t = P; t < P + D; ++t) {
            // arrivals
            if (!plist.empty()) {
                size_t w = 0;
                for (size_t i = 0; i < plist.size(); ++i) {
                    const int e = plist[i];
                    if (pend[e] < 0) {
                        continue;
                    }
                    if (pend[e] > t) {
                        plist[w++] = e;
                        continue;
                    }
                    pend[e] = -1;
                    if (!c.lazy) {
                        nresv--;
                        put(e); // into its reserved slot
                    } else if (!res[e]) {
                        if (nres < cap) {
                            put(e);
                        } else {
                            const int v = victim(keep);
                            if (v >= 0 && (!c.rguard || k1[e] > k1[v])) {
                                drop(v);
                                put(e);
                            }
                        }
                    }
                }
                plist.resize(w);
            }
            const int32_t * ord = &u.ord[(size_t) t * M];
            const float * p = &u.p[(size_t) t * K];
            const float * zt = &u.z[(size_t) t * N];
            const int J = jcount(t);
            int h = 0;
            U.clear();
            double mass = 0;
            int nsub = 0;
            for (int k = 0; k < K; ++k) {
                R[k] = ord[k];
                if (res[R[k]]) {
                    h++;
                } else if (k < J) {
                    U.push_back(R[k]);
                } else {
                    mass += p[k];
                    nsub++;
                }
            }
            // a batch can leave fewer than K usable experts: then the best substituted misses are waited for too
            for (int k = J; k < K && nres + (int) U.size() < K; ++k) {
                if (!res[R[k]] && std::find(U.begin(), U.end(), R[k]) == U.end()) {
                    U.push_back(R[k]);
                    mass -= p[k];
                    nsub--;
                }
            }
            for (int e : U) {
                inU[e] = 1;
            }
            if (nsub > 0) {
                int n = 0;
                for (int i = 0; i < M && n < K; ++i) {
                    if (res[ord[i]] || inU[ord[i]]) {
                        Rp[n++] = ord[i];
                    }
                }
                if (n < K) {
                    std::vector<int> cand;
                    for (int e = 0; e < N; ++e) {
                        if (res[e] || inU[e]) {
                            cand.push_back(e);
                        }
                    }
                    if ((int) cand.size() < K) {
                        fprintf(stderr, "xsim: %zu usable experts < K\n", cand.size());
                        exit(1);
                    }
                    std::partial_sort(cand.begin(), cand.begin() + K, cand.end(), [&](int x, int y) { return zt[x] > zt[y] || (zt[x] == zt[y] && x < y); });
                    for (n = 0; n < K; ++n) {
                        Rp[n] = cand[n];
                    }
                }
            } else {
                Rp = R;
            }
            for (int k = 0; k < K; ++k) {
                upd(R[k], p[k], t, k < J);
            }
            if (c.key == "s" && nsub > 0) {
                double sum = 0, ex[64];
                for (int k = 0; k < K; ++k) {
                    ex[k] = std::exp((double) zt[Rp[k]] - zt[Rp[0]]);
                    sum += ex[k];
                }
                for (int k = 0; k < K; ++k) {
                    if (std::find(R.begin(), R.end(), Rp[k]) == R.end()) {
                        upd(Rp[k], ex[k] / sum, t, false);
                    }
                }
            }
            for (int k = 0; k < K; ++k) {
                prot[R[k]] = 1;
                prot[Rp[k]] = 1;
            }
            int npend = 0;
            for (int e : U) {
                if (pend[e] >= 0) {
                    npend++;
                    if (!c.lazy) {
                        // in flight from an eager batch: its slot is reserved, wait for it
                        pend[e] = -1;
                        nresv--;
                        put(e);
                        continue;
                    }
                    pend[e] = -1; // wait for the read in flight, then admit as usual
                }
                if (nres + nresv < cap) {
                    put(e);
                    continue;
                }
                const int v = victim(prot);
                if (v < 0) {
                    continue; // every resident expert is in use this token (tiny caches only): read without admitting
                }
                drop(v);
                put(e);
            }
            for (int k = 0; k < K; ++k) {
                prot[R[k]] = 0;
                prot[Rp[k]] = 0;
            }
            for (int e : U) {
                inU[e] = 0;
            }
            const int b = std::min((int) a.bn.size() - 1, (t - P) / binw);
            a.n += 1;
            a.hit += h;
            a.must += U.size();
            a.must_pend += npend;
            a.sub += nsub;
            a.mass += mass;
            a.bn[b] += 1;
            a.bhit[b] += h;
            a.bmust[b] += U.size();
            a.bmass[b] += mass;
            // background reads at the end of the token
            int nb = 0;
            bool fire = false;
            if (c.batch == "every") {
                fire = (t - P + 1) % c.delta == 0;
            } else if (c.batch == "drift") {
                const int slot = (t - P) % c.dw;
                ring_sum += (int) U.size() - ring[slot];
                ring[slot] = (int) U.size();
                fire = ring_sum >= c.dthr && t + 1 - last_refresh >= c.dw;
            } else if (c.batch == "bg") {
                for (int k = J; k < K; ++k) {
                    const int e = R[k];
                    if (!res[e] && pend[e] < 0 && chan - (t + 1) <= c.qmax) {
                        schedule(e, t);
                        nb++;
                    }
                }
            } else if (c.batch == "regret") {
                // rent-or-buy: each substitution "rents" (loses its gate mass); once the decayed loss reaches the
                // buy cost, read the expert in the background
                const double lam = std::log(2.0) / c.rhalf;
                for (int k = J; k < K; ++k) {
                    const int e = R[k];
                    if (res[e] || pend[e] >= 0) {
                        continue;
                    }
                    const double r = rv[e] * std::exp(-lam * (t - rt[e])) + p[k];
                    rt[e] = t;
                    rv[e] = r;
                    if (r >= c.rbeta && chan - (t + 1) <= c.qmax) {
                        schedule(e, t);
                        rv[e] = 0;
                        nb++;
                    }
                }
            }
            if (fire) {
                last_refresh = t + 1;
                std::vector<int> order;
                auto tg = target(&order, true);
                for (int e = 0; e < N; ++e) {
                    if (!tg[e]) {
                        cancel(e);
                        if (!c.lazy) {
                            drop(e);
                        }
                    }
                }
                if (c.lazy) {
                    keep = tg;
                }
                // new experts stream in best first at the batch rate, after a fixed latency
                for (int e : order) {
                    if (!res[e] && pend[e] < 0) {
                        schedule(e, t);
                        nb++;
                    }
                }
                a.refresh += 1;
            }
            a.batch += nb;
            a.bbatch[b] += nb;
        }
    }

    int jcount(int t) const {
        if (c.j == "c1") {
            return 1;
        }
        if (c.j == "all") {
            return u.K;
        }
        if (c.j == "none") {
            return 0; // never wait: every miss is substituted (SwapMoE-style masking)
        }
        return u.jt[t];
    }
};


// ---- per-layer capacity allocation (same total slots across the MoE layers of a sequence) ----
// Miss curves f(c) = misses per token with c slots, then greedy: give 4 slots at a time to the layer with the largest drop.
//   pm: LRU stack distances (Mattson) of the wanted experts in the prompt, weighted by gate
//   pw: working-set (Denning) curve from the prompt: miss(c) = m(w), s(w) = c, s(w) = sum_{v<w} m(v), counts
//   om: oracle, gate-weighted LRU stack distances of the generated part (stack warmed by the prompt)
static std::vector<double> mattson_curve(const unit & u, int t1, int tstat) {
    const int N = u.N, K = u.K, M = u.M;
    const int S = t1 * K + 2;
    std::vector<int> bit(S + 1, 0);
    auto add = [&](int i, int v) {
        for (++i; i <= S; i += i & -i) {
            bit[i] += v;
        }
    };
    auto pre = [&](int i) { // sum of [0, i]
        int r = 0;
        for (++i; i > 0; i -= i & -i) {
            r += bit[i];
        }
        return r;
    };
    std::vector<int> last(N, -1);
    std::vector<double> hist(N + 1, 0.0);
    int stamp = 0, active = 0;
    for (int t = 0; t < t1; ++t) {
        for (int k = 0; k < K; ++k) {
            const int e = u.ord[(size_t) t * M + k];
            int sd = N; // first reference: always a miss
            if (last[e] >= 0) {
                sd = active - pre(last[e]); // distinct experts used after e's last use
                add(last[e], -1);
                active--;
            }
            if (t >= tstat) {
                hist[std::min(sd, N)] += u.p[(size_t) t * K + k];
            }
            last[e] = stamp;
            add(stamp, 1);
            active++;
            stamp++;
        }
    }
    std::vector<double> f(N + 1, 0.0);
    double acc = 0;
    const double nt = std::max(1, t1 - tstat);
    for (int c = N; c >= 0; --c) {
        acc += hist[c];
        f[c] = acc / nt;
    }
    return f;
}

static std::vector<double> denning_curve(const unit & u, int P) {
    const int N = u.N, K = u.K, M = u.M;
    std::vector<double> bh(P + 2, 0.0);
    std::vector<int> lastT(N, -1);
    for (int t = 0; t < P; ++t) {
        for (int k = 0; k < K; ++k) {
            const int e = u.ord[(size_t) t * M + k];
            bh[lastT[e] < 0 ? P + 1 : t - lastT[e]] += 1;
            lastT[e] = t;
        }
    }
    std::vector<double> m(P + 1, 0.0), sw(P + 2, 0.0);
    double tail = bh[P + 1];
    for (int w = P; w >= 0; --w) { // m(w) = references per token with back distance > w
        m[w] = tail / P;
        tail += bh[w];
    }
    for (int w = 0; w <= P; ++w) {
        sw[w + 1] = sw[w] + m[w];
    }
    std::vector<double> f(N + 1, 0.0);
    for (int c = 0; c <= N; ++c) {
        if (c >= sw[P + 1]) {
            f[c] = m[P];
            continue;
        }
        int w = 0;
        while (sw[w + 1] <= c) {
            w++;
        }
        const double fr = (c - sw[w]) / std::max(1e-12, sw[w + 1] - sw[w]);
        f[c] = m[w] + fr * ((w + 1 <= P ? m[w + 1] : m[P]) - m[w]);
    }
    return f;
}

static std::vector<int> allocate(const std::vector<std::vector<double>> & f, int cap, int N) {
    const int L = (int) f.size(), step = 4;
    const int cmin = std::max(12, cap / 4), cmax = std::min(N, 3 * cap);
    std::vector<int> c(L, cmin);
    int budget = L * cap - L * cmin;
    while (budget >= step) {
        int best = -1;
        double bg = -1e300;
        for (int l = 0; l < L; ++l) {
            if (c[l] + step > cmax) {
                continue;
            }
            const double g = f[l][c[l]] - f[l][c[l] + step];
            if (g > bg) {
                bg = g;
                best = l;
            }
        }
        if (best < 0) {
            break;
        }
        c[best] += step;
        budget -= step;
    }
    for (int l = 0; budget > 0; l = (l + 1) % L) {
        if (c[l] < cmax) {
            c[l]++;
            budget--;
        }
    }
    return c;
}

static std::vector<std::string> read_lines(const std::string & p) {
    std::ifstream f(p);
    std::vector<std::string> v;
    std::string l;
    while (std::getline(f, l)) {
        if (!l.empty() && l[0] != '#') {
            v.push_back(l);
        }
    }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: see header\n");
        return 1;
    }
    const std::string mode = argv[1];
    const int nthr = std::max(1u, std::thread::hardware_concurrency());
    if (mode == "policy") {
        if (argc < 11) {
            fprintf(stderr, "xsim policy SET P D CAP TAU K BINW CFGFILE files...\n");
            return 1;
        }
        const std::string set = argv[2];
        const int P = atoi(argv[3]), D0 = atoi(argv[4]), cap = atoi(argv[5]);
        const double tau = atof(argv[6]);
        const int K = atoi(argv[7]), binw = atoi(argv[8]);
        const int bdelay = getenv("XSIM_BDELAY") ? atoi(getenv("XSIM_BDELAY")) : 1;
        const double brate = getenv("XSIM_BRATE") ? atof(getenv("XSIM_BRATE")) : 0.5;
        std::vector<cfg> cfgs;
        for (auto & l : read_lines(argv[9])) {
            cfgs.push_back(parse_cfg(l));
        }
        std::vector<seqfile> seqs;
        for (int i = 10; i < argc; ++i) {
            seqs.push_back(open_seq(argv[i]));
        }
        int D = D0;
        for (auto & s : seqs) {
            D = std::min(D, s.T - P);
        }
        const int nb = (D + binw - 1) / binw;
        std::vector<acc> tot(cfgs.size());
        for (auto & a : tot) {
            a.bins(nb);
        }
        std::vector<std::pair<int, int>> units;
        for (int s = 0; s < (int) seqs.size(); ++s) {
            for (int l = 0; l < seqs[s].L; ++l) {
                units.push_back({ s, l });
            }
        }
        // per-layer capacities for configs with alloc:* (phase A: miss curves, then greedy per sequence)
        const std::vector<std::string> atypes = { "pm", "pw", "om" };
        std::map<std::string, std::vector<std::vector<int>>> caps; // type -> seq -> layer -> capacity
        bool need_alloc = false;
        for (auto & c : cfgs) {
            need_alloc = need_alloc || c.alloc != "none";
        }
        if (need_alloc) {
            std::vector<std::vector<std::vector<double>>> fc(units.size());
            std::atomic<int> na(0);
            std::vector<std::thread> tha;
            for (int w = 0; w < nthr; ++w) {
                tha.emplace_back([&]() {
                    for (;;) {
                        const int i = na++;
                        if (i >= (int) units.size()) {
                            break;
                        }
                        const auto & sq = seqs[units[i].first];
                        unit u;
                        prep(u, load_layer(sq, units[i].second), sq.T, sq.N, K, tau);
                        fc[i] = { mattson_curve(u, P, 0), denning_curve(u, P), mattson_curve(u, P + D, P) };
                    }
                });
            }
            for (auto & t : tha) {
                t.join();
            }
            for (size_t a = 0; a < atypes.size(); ++a) {
                auto & cs = caps[atypes[a]];
                cs.assign(seqs.size(), {});
                for (int s = 0; s < (int) seqs.size(); ++s) {
                    std::vector<std::vector<double>> f;
                    for (size_t i = 0; i < units.size(); ++i) {
                        if (units[i].first == s) {
                            f.push_back(fc[i][a]);
                        }
                    }
                    cs[s] = allocate(f, cap, seqs[s].N);
                }
                // mean capacity per layer over sequences, and the predicted gain (f(cap) - f(c_l) summed, oracle curve)
                const int L = seqs[0].L;
                printf("ALLOC {\"set\":\"%s\",\"cap\":%d,\"tau\":%g,\"type\":\"%s\",\"mean\":[", set.c_str(), cap, tau, atypes[a].c_str());
                for (int l = 0; l < L; ++l) {
                    double m = 0;
                    for (auto & v : cs) {
                        m += v[l];
                    }
                    printf("%s%.1f", l ? "," : "", m / cs.size());
                }
                int mn = 1 << 30, mx = 0;
                double pred_u = 0, pred_a = 0;
                for (int s = 0; s < (int) seqs.size(); ++s) {
                    for (int l = 0; l < L; ++l) {
                        mn = std::min(mn, cs[s][l]);
                        mx = std::max(mx, cs[s][l]);
                    }
                }
                for (size_t i = 0; i < units.size(); ++i) {
                    pred_u += fc[i][2][cap];
                    pred_a += fc[i][2][cs[units[i].first][units[i].second]];
                }
                printf("],\"min\":%d,\"max\":%d,\"lru_mass_uniform\":%.5f,\"lru_mass_alloc\":%.5f}\n", mn, mx, pred_u / units.size(),
                       pred_a / units.size());
            }
            fflush(stdout);
            if (getenv("XSIM_DUMPCAPS")) {
                for (auto & kv : caps) {
                    for (int s = 0; s < (int) seqs.size(); ++s) {
                        fprintf(stderr, "CAPS %s %d", kv.first.c_str(), s);
                        for (int v : kv.second[s]) {
                            fprintf(stderr, " %d", v);
                        }
                        fprintf(stderr, "\n");
                    }
                }
            }
        }
        std::atomic<int> next(0);
        std::mutex mtx;
        std::vector<std::thread> th;
        for (int w = 0; w < nthr; ++w) {
            th.emplace_back([&]() {
                std::vector<acc> loc(cfgs.size());
                for (auto & a : loc) {
                    a.bins(nb);
                }
                for (;;) {
                    const int i = next++;
                    if (i >= (int) units.size()) {
                        break;
                    }
                    const auto & sq = seqs[units[i].first];
                    unit u;
                    prep(u, load_layer(sq, units[i].second), sq.T, sq.N, K, tau);
                    for (size_t ci = 0; ci < cfgs.size(); ++ci) {
                        const int capl = cfgs[ci].alloc == "none" ? cap : caps.at(cfgs[ci].alloc)[units[i].first][units[i].second];
                        sim s(u, cfgs[ci], capl, P, D, binw, bdelay);
                        s.run(loc[ci]);
                    }
                }
                std::lock_guard<std::mutex> lk(mtx);
                for (size_t ci = 0; ci < cfgs.size(); ++ci) {
                    tot[ci].add(loc[ci]);
                }
            });
        }
        for (auto & t : th) {
            t.join();
        }
        for (size_t ci = 0; ci < cfgs.size(); ++ci) {
            const acc & a = tot[ci];
            const double n = std::max(1.0, a.n);
            printf("SIM {\"set\":\"%s\",\"P\":%d,\"D\":%d,\"cap\":%d,\"tau\":%g,\"brate\":%g,\"bdelay\":%d,\"cfg\":\"%s\",\"n\":%.0f,\"hit\":%.5f,"
                   "\"must\":%.5f,\"must_pend\":%.5f,\"sub\":%.5f,\"mass\":%.5f,\"batch\":%.5f,\"refresh\":%.5f",
                   set.c_str(), P, D, cap, tau, brate, bdelay, cfgs[ci].name.c_str(), a.n, a.hit / (K * n), a.must / n, a.must_pend / n,
                   a.sub / (K * n), a.mass / n, a.batch / n, a.refresh / n);
            if (!cfgs[ci].bins) {
                printf("}\n");
                fflush(stdout);
                continue;
            }
            printf(",\"binw\":%d,\"bins\":{", binw);
            auto arr = [&](const char * nm, const std::vector<double> & v, double scale, bool last) {
                printf("\"%s\":[", nm);
                for (int b = 0; b < nb; ++b) {
                    printf("%s%.4f", b ? "," : "", a.bn[b] > 0 ? v[b] / (a.bn[b] * scale) : 0.0);
                }
                printf("]%s", last ? "" : ",");
            };
            arr("hit", a.bhit, K, false);
            arr("must", a.bmust, 1, false);
            arr("mass", a.bmass, 1, false);
            arr("batch", a.bbatch, 1, true);
            printf("}}\n");
            fflush(stdout);
        }
        return 0;
    }
    if (mode == "curves") {
        if (argc < 7) {
            fprintf(stderr, "xsim curves SET P TAU K files...\n");
            return 1;
        }
        const std::string set = argv[2];
        const int P = atoi(argv[3]);
        const double tau = atof(argv[4]);
        const int K = atoi(argv[5]);
        std::vector<seqfile> seqs;
        for (int i = 6; i < argc; ++i) {
            seqs.push_back(open_seq(argv[i]));
        }
        const std::vector<int> ws = { 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512 };
        const int HMAX = 8192;
        // per stream (0: wanted R, 1: core C): back-distance histogram of references at t >= P, direct s(w) sums
        struct cacc {
            std::vector<double> hist[2];
            std::vector<double> sw[2], swn[2];
            double nref[2] = { 0, 0 }, ntok = 0;
        };
        std::vector<std::pair<int, int>> units;
        for (int s = 0; s < (int) seqs.size(); ++s) {
            for (int l = 0; l < seqs[s].L; ++l) {
                units.push_back({ s, l });
            }
        }
        cacc tot;
        for (int st = 0; st < 2; ++st) {
            tot.hist[st].assign(HMAX + 2, 0);
            tot.sw[st].assign(ws.size(), 0);
            tot.swn[st].assign(ws.size(), 0);
        }
        std::atomic<int> next(0);
        std::mutex mtx;
        std::vector<std::thread> th;
        for (int w = 0; w < nthr; ++w) {
            th.emplace_back([&]() {
                cacc loc;
                for (int st = 0; st < 2; ++st) {
                    loc.hist[st].assign(HMAX + 2, 0);
                    loc.sw[st].assign(ws.size(), 0);
                    loc.swn[st].assign(ws.size(), 0);
                }
                for (;;) {
                    const int i = next++;
                    if (i >= (int) units.size()) {
                        break;
                    }
                    const auto & sq = seqs[units[i].first];
                    unit u;
                    prep(u, load_layer(sq, units[i].second), sq.T, sq.N, K, tau);
                    for (int st = 0; st < 2; ++st) {
                        auto refs = [&](int t, std::vector<int> & out) {
                            out.clear();
                            const int J = st == 0 ? K : u.jt[t];
                            for (int k = 0; k < J; ++k) {
                                out.push_back(u.ord[(size_t) t * u.M + k]);
                            }
                        };
                        std::vector<int> last(u.N, -1), r;
                        for (int t = 0; t < u.T; ++t) {
                            refs(t, r);
                            for (int e : r) {
                                if (t >= P) {
                                    const int d = last[e] < 0 ? HMAX + 1 : std::min(HMAX, t - last[e]);
                                    loc.hist[st][d] += 1;
                                    loc.nref[st] += 1;
                                }
                                last[e] = t;
                            }
                            if (st == 0 && t >= P) {
                                loc.ntok += 1;
                            }
                        }
                        // direct working-set sizes |W(t, w)| for t >= P (window reaching into the prompt)
                        for (size_t wi = 0; wi < ws.size(); ++wi) {
                            const int w = ws[wi];
                            std::vector<int> cnt(u.N, 0);
                            int distinct = 0;
                            for (int t = 0; t < u.T; ++t) {
                                refs(t, r);
                                for (int e : r) {
                                    if (cnt[e]++ == 0) {
                                        distinct++;
                                    }
                                }
                                if (t - w >= 0) {
                                    refs(t - w, r);
                                    for (int e : r) {
                                        if (--cnt[e] == 0) {
                                            distinct--;
                                        }
                                    }
                                }
                                if (t >= P && t + 1 >= w) {
                                    loc.sw[st][wi] += distinct;
                                    loc.swn[st][wi] += 1;
                                }
                            }
                        }
                    }
                }
                std::lock_guard<std::mutex> lk(mtx);
                for (int st = 0; st < 2; ++st) {
                    for (size_t j = 0; j < tot.hist[st].size(); ++j) {
                        tot.hist[st][j] += loc.hist[st][j];
                    }
                    for (size_t j = 0; j < ws.size(); ++j) {
                        tot.sw[st][j] += loc.sw[st][j];
                        tot.swn[st][j] += loc.swn[st][j];
                    }
                    tot.nref[st] += loc.nref[st];
                }
                tot.ntok += loc.ntok;
            });
        }
        for (auto & t : th) {
            t.join();
        }
        // m(w) = expected references per token-layer with back-distance > w (Denning: s(w+1) - s(w) = m(w))
        for (int st = 0; st < 2; ++st) {
            printf("CURVE {\"set\":\"%s\",\"P\":%d,\"tau\":%g,\"stream\":\"%s\",\"refs_per_tok\":%.5f,\"w\":[", set.c_str(), P, tau,
                   st == 0 ? "R" : "C", tot.nref[st] / tot.ntok);
            for (size_t j = 0; j < ws.size(); ++j) {
                printf("%s%d", j ? "," : "", ws[j]);
            }
            printf("],\"s\":[");
            for (size_t j = 0; j < ws.size(); ++j) {
                printf("%s%.4f", j ? "," : "", tot.sw[st][j] / std::max(1.0, tot.swn[st][j]));
            }
            // m(w) for w = 0..HMAX via the histogram tail
            std::vector<double> tail(HMAX + 3, 0);
            for (int d = HMAX + 1; d >= 0; --d) {
                tail[d] = tail[d + 1] + tot.hist[st][d];
            }
            printf("],\"m\":[");
            for (int w = 0; w <= 2048; ++w) {
                printf("%s%.5f", w ? "," : "", tail[w + 1] / tot.ntok);
            }
            printf("]}\n");
        }
        return 0;
    }
    fprintf(stderr, "unknown mode\n");
    return 1;
}
