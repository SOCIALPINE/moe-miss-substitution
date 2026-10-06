"""Round-9 tables from xsim output (SIM / CURVE lines): cache policies replayed on dumped router logits of
Qwen3-Next-80B-A3B (48 MoE layers, 512 experts, top-10).

Speed estimate: per token, 8.195 ms of work plus 0.557 ms per waited (synchronous) expert read, from the round-8 engine run
with every expert read from the SSD (ram=0, 16 GB: 11.22 ms per token, of which 3.02 ms waiting for 5.42 reads).
Quality proxy: gate weight that was substituted (sum of the normalized gates of wanted experts replaced by resident ones),
per layer and token."""
import json, re, sys
from collections import defaultdict

T0, ELL, NL, MB = 8.195, 0.557, 48, 1.908  # ms, ms per waited read, MoE layers, MB per expert
ENGINE = {"hit": 0.8481, "sub": 0.1406, "must": 5.4236 / 48}  # round 8, RAM 16 GB, ram=1 (same routing as ram=0)


def load(path):
    sims, curves = [], []
    for line in open(path, errors="replace"):
        m = re.search(r"\b(SIM|CURVE) (\{.*\})\s*$", line)
        if not m:
            continue
        d = json.loads(m.group(2))
        (sims if m.group(1) == "SIM" else curves).append(d)
    return sims, curves


def est(d):
    ms = T0 + NL * d["must"] * ELL
    return ms, 1000.0 / ms


def pareto(rows):
    """rows: (name, x=ms, y=mass) -> names on the lower-left front"""
    front = set()
    for n, x, y in rows:
        if not any((x2 <= x and y2 <= y) and (x2 < x or y2 < y) for n2, x2, y2 in rows):
            front.add(n)
    return front


def table(rows, title):
    print(f"### {title}\n")
    print("| 정책 | 적중률 | 기다린 읽기 (레이어·토큰당) | 토큰당 기다린 읽기 | 대체된 게이트 질량 | 대체 비율 | 일괄 읽기 (레이어·토큰당) | 일괄 대역폭 @추정 속도 | 추정 ms/토큰 | 추정 tok/s | 앞선 집합 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    pts = [(d["cfg"], est(d)[0], d["mass"]) for d in rows]
    front = pareto(pts)
    for d in sorted(rows, key=lambda d: (est(d)[0], d["mass"])):
        ms, ts = est(d)
        bw = d["batch"] * NL * MB * ts / 1000.0
        print(f"| {d['cfg']} | {100 * d['hit']:.1f}% | {d['must']:.4f} | {NL * d['must']:.2f} | {100 * d['mass']:.2f}% | "
              f"{100 * d['sub']:.1f}% | {d['batch']:.3f} | {bw:.2f} GB/s | {ms:.2f} | {ts:.1f} | {'●' if d['cfg'] in front else ''} |")
    print()
    base = next((d for d in rows if d["cfg"] == "eng"), None)
    if base:
        bm, bms = base["mass"], est(base)[0]
        better = [d for d in rows if est(d)[0] <= bms + 1e-9 and d["mass"] <= bm + 1e-12 and d["cfg"] != "eng"]
        print(f"지금 엔진(eng): 추정 {bms:.2f} ms/토큰, 대체 질량 {100 * bm:.2f}%. 둘 다 같거나 나은 정책 {len(better)}개:")
        for d in sorted(better, key=lambda d: (d["mass"], est(d)[0]))[:12]:
            print(f"- {d['cfg']}: {est(d)[0]:.2f} ms ({100 * (est(d)[0] / bms - 1):+.1f}%), 질량 {100 * d['mass']:.2f}% "
                  f"({100 * (d['mass'] / bm - 1):+.0f}%), 일괄 {d['batch']:.3f}")
        print()


def main(path):
    sims, curves = load(path)
    by = defaultdict(list)
    for d in sims:
        by[(d["set"], d["cap"], d["P"], d["D"])].append(d)
    # validation against the engine (round 8): 6 sessions, prompts 128/512/1536, 256 decoded tokens, 25%
    val = [d for d in sims if d["set"] == "val"]
    if val:
        print("### 보정: 엔진 실측과 비교 (6세션, 프롬프트 128/512/1536, 256토큰, 25%)\n")
        print("| 정책 | 적중률 | 대체 비율 | 기다린 읽기 (레이어·토큰당) |")
        print("|---|---|---|---|")
        agg = defaultdict(lambda: defaultdict(float))
        for d in val:
            a = agg[d["cfg"]]
            for k in ("hit", "sub", "must", "mass"):
                a[k] += d[k] * d["n"]
            a["n"] += d["n"]
        for name, a in agg.items():
            print(f"| 시뮬 {name} | {100 * a['hit'] / a['n']:.1f}% | {100 * a['sub'] / a['n']:.1f}% | {a['must'] / a['n']:.4f} |")
        print(f"| 엔진 실측 (Part 14) | {100 * ENGINE['hit']:.1f}% | {100 * ENGINE['sub']:.1f}% | {ENGINE['must']:.4f} |")
        print()
    for (st, cap, P, D), rows in sorted(by.items()):
        if st == "val":
            continue
        table(rows, f"{st}, 용량 {cap} ({100 * cap / 512:.1f}%), 프롬프트 {P}, 생성 {D}토큰")
        # by position for a few policies
        names = ["eng", "lru_d", "full"] + sorted({d["cfg"] for d in rows if d["cfg"].startswith(("b", "drift"))})[:0]
        best = sorted(rows, key=lambda d: d["mass"] + 0.02 * est(d)[0])[:3]
        names += [d["cfg"] for d in best if d["cfg"] not in names]
        bw = rows[0]["binw"]
        print(f"위치별 적중률 / 기다린 읽기 ({bw}토큰 구간)\n")
        nb = len(rows[0]["bins"]["hit"])
        print("| 정책 | " + " | ".join(f"{i * bw}~" for i in range(nb)) + " |")
        print("|---|" + "---|" * nb)
        for n in names:
            d = next((x for x in rows if x["cfg"] == n), None)
            if d:
                print(f"| {n} | " + " | ".join(f"{100 * h:.1f}% / {m:.3f}" for h, m in zip(d["bins"]["hit"], d["bins"]["must"])) + " |")
        print()
    for c in curves:
        w, s, m = c["w"], c["s"], c["m"]
        K = round(c["refs_per_tok"]) if c["stream"] == "R" else None
        print(f"### 작업 집합 곡선: {c['set']}, 흐름 {c['stream']} (토큰당 참조 {c['refs_per_tok']:.3f}개)\n")
        print("| w | " + " | ".join(str(x) for x in w) + " |")
        print("|---|" + "---|" * len(w))
        print("| s(w) 직접 | " + " | ".join(f"{x:.1f}" for x in s) + " |")
        # Denning: s(w2) - s(w1) = sum_{v=w1}^{w2-1} m(v)
        pred = [s[0]]
        for i in range(1, len(w)):
            pred.append(pred[-1] + sum(m[v] for v in range(w[i - 1], w[i])))
        print("| s(w) = s(1) + Σm | " + " | ".join(f"{x:.1f}" for x in pred) + " |")
        print("| m(w) | " + " | ".join(f"{m[x]:.3f}" for x in w if x < len(m)) + " |")
        print()
        if c["stream"] == "R":
            fulls = {d["cap"]: d for d in sims if d["set"] == c["set"] and d["cfg"] == "full"}
            for cap, d in sorted(fulls.items()):
                # w_c: s(w_c) = cap, linear interpolation on the direct curve, then h = 1 - m(w_c)/K
                wc = None
                for i in range(1, len(w)):
                    if s[i - 1] <= cap <= s[i]:
                        wc = w[i - 1] + (cap - s[i - 1]) / (s[i] - s[i - 1]) * (w[i] - w[i - 1])
                        break
                if wc is None:
                    continue
                mw = m[int(wc)] + (wc - int(wc)) * (m[int(wc) + 1] - m[int(wc)])
                print(f"- 용량 {cap}: s(w)={cap}인 w ≈ {wc:.0f}토큰, 식 1 − m(w)/K = {100 * (1 - mw / K):.1f}%, "
                      f"시뮬 LRU(미스 전부 읽기) {100 * d['hit']:.1f}%")
            print()


if __name__ == "__main__":
    main(sys.argv[1])
