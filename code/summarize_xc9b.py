"""Round-9b tables from xsim output (SIM lines): cache policies replayed on dumped router logits of Qwen3-Next-80B-A3B
(48 MoE layers, 512 experts, top-10), capacities 128 and 64.

Speed estimate (as round 9): per token, 8.195 ms of work plus 0.557 ms per waited expert read, from the round-8 engine run
with every expert read from the SSD (ram=0, 16 GB). Background reads are assumed to overlap with compute. The
"contention" column adds a rough penalty: waited reads take (1 + u) times longer, u = share of the token time the SSD
spends on background reads at 5.5 GB/s (the Part 14 machine).
Quality proxy: gate weight that was substituted (sum of the normalized gates of wanted experts replaced by resident ones),
per layer and token."""
import json, re, sys
from collections import defaultdict

T0, ELL, NL, MB, SSD = 8.195, 0.557, 48, 1.908, 5.5  # ms, ms per waited read, MoE layers, MB per expert, MB per ms
R9 = {"hit": 0.8612, "must": 0.0731, "mass": 0.0972}  # round 9, all24, capacity 128, eng


def load(path):
    sims = []
    for line in open(path, errors="replace"):
        m = re.search(r"\bSIM (\{.*\})\s*$", line)
        if m:
            sims.append(json.loads(m.group(1)))
    return sims


def est(d):
    return T0 + NL * d["must"] * ELL


def est_c(d):
    ms = est(d)
    for _ in range(20):
        u = min(0.9, NL * d["batch"] * MB / SSD / ms)
        ms = T0 + NL * d["must"] * ELL * (1 + u)
    return ms


def ssd_mb(d):
    return NL * (d["must"] + d["batch"]) * MB


def pareto(rows):
    pts = [(d["cfg"], est(d), d["mass"]) for d in rows]
    return {n for n, x, y in pts if not any((x2 <= x and y2 <= y) and (x2 < x or y2 < y) for _, x2, y2 in pts)}


def pct(a, b):
    return 100 * (a / b - 1)


def main_table(rows, title):
    print(f"### {title}\n")
    print("| 정책 | 적중률 | 토큰당 기다린 읽기 | 대체 질량 | 레이어·토큰당 배경 읽기 | 토큰당 SSD 읽기 | 추정 ms/토큰 | 경합 반영 ms | 앞선 집합 |")
    print("|---|---|---|---|---|---|---|---|---|")
    front = pareto(rows)
    for d in sorted(rows, key=lambda d: (est(d), d["mass"])):
        print(f"| {d['cfg']} | {100 * d['hit']:.1f}% | {NL * d['must']:.2f} | {100 * d['mass']:.2f}% | {d['batch']:.3f} | "
              f"{ssd_mb(d):.1f} MB | {est(d):.2f} | {est_c(d):.2f} | {'●' if d['cfg'] in front else ''} |")
    print()
    eng = next(d for d in rows if d["cfg"] == "eng")
    em, ems = eng["mass"], est(eng)
    print(f"지금 엔진(eng): 추정 {ems:.2f} ms, 대체 질량 {100 * em:.2f}%, 적중률 {100 * eng['hit']:.1f}%\n")
    print("| 속도 예산 (경합 반영, eng 대비) | 질량이 가장 낮은 정책 | 질량 | eng 대비 | 경합 반영 ms | 배경 읽기 (토큰당 MB) |")
    print("|---|---|---|---|---|---|")
    for lim in (0.0, 0.02, 0.05, 0.10):
        ok = [d for d in rows if est_c(d) <= ems * (1 + lim) + 1e-9 and d["cfg"] != "full"]
        b = min(ok, key=lambda d: (d["mass"], est_c(d)))
        print(f"| +{100 * lim:.0f}% | {b['cfg']} | {100 * b['mass']:.2f}% | {pct(b['mass'], em):+.0f}% | {est_c(b):.2f} ({pct(est_c(b), ems):+.1f}%) | "
              f"{NL * b['batch'] * MB:.1f} |")
    print()
    # lazy grid: score x period
    lz = {}
    for d in rows:
        m = re.match(r"l(\d+)_(\w+?)$", d["cfg"])
        if m and not d["cfg"].endswith("_c1"):
            lz[(m.group(2), int(m.group(1)))] = d
    if lz:
        scores = list(dict.fromkeys(k[0] for k in lz))  # config order
        dls = sorted({k[1] for k in lz})
        print("일괄 갱신(lazy) 격자: 칸마다 대체 질량 / 경합 반영 ms\n")
        print("| 점수 \\ 주기 | " + " | ".join(f"{x}토큰" for x in dls) + " |")
        print("|---|" + "---|" * len(dls))
        for s in scores:
            print(f"| {s} | " + " | ".join(f"{100 * lz[(s, x)]['mass']:.2f}% / {est_c(lz[(s, x)]):.2f}" if (s, x) in lz else "" for x in dls) + " |")
        print()


def interp(pts, x):
    """pts: (x, y) sorted by x; linear interpolation, None outside"""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return None


def fronts(sims, st, cap):
    rows = [d for d in sims if d["set"] == st and d["cap"] == cap and d["brate"] == 0.5 and d["bdelay"] == 1]
    names = sorted({d["cfg"] for d in rows if d["tau"] != 0.15})
    if not names:
        return
    eng = next(d for d in rows if d["cfg"] == "eng" and d["tau"] == 0.15)
    em, ems = eng["mass"], est_c(eng)
    print(f"### τ를 바꾼 속도-품질 곡선: {st}, 용량 {cap}\n")
    print("같은 정책에서 τ만 0.05~0.5로 바꾼 점들을 잇고, 지금 엔진(eng, τ 0.15)과 같은 속도에서의 질량, 같은 질량에서의 속도를 읽었어요 (경합 반영 ms).\n")
    print("| 정책 | τ별 (경합 반영 ms, 질량) | eng 속도에서 질량 | eng 질량에서 ms |")
    print("|---|---|---|---|")
    for n in names:
        p = sorted([d for d in rows if d["cfg"] == n], key=lambda d: -d["tau"])
        pts = sorted((est_c(d), d["mass"]) for d in p)
        m_at = interp(pts, ems)
        inv = sorted((d["mass"], est_c(d)) for d in p)
        t_at = interp(inv, em)
        cur = " ".join(f"{d['tau']:g}:({est_c(d):.2f}, {100 * d['mass']:.2f}%)" for d in p)
        print(f"| {n} | {cur} | {'' if m_at is None else f'{100 * m_at:.2f}% ({pct(m_at, em):+.0f}%)'} | "
              f"{'' if t_at is None else f'{t_at:.2f} ({pct(t_at, ems):+.1f}%)'} |")
    print()


def sens(sims, st):
    rows = [d for d in sims if d["set"] == st and d["cap"] == 128 and d["tau"] == 0.15]
    combos = [(0.25, 1), (0.5, 0), (0.5, 1), (0.5, 4), (0.5, 16), (1, 1)]
    names = sorted({d["cfg"] for d in rows if d["brate"] != 0.5 or d["bdelay"] != 1})
    if not names:
        return
    print(f"### 배경 읽기 속도·지연 민감도: {st}, 용량 128 (칸마다 대체 질량 / 경합 반영 ms / 토큰당 배경 MB)\n")
    print("| 정책 | " + " | ".join(f"속도 {b:g}/토큰, 지연 {l}" for b, l in combos) + " |")
    print("|---|" + "---|" * len(combos))
    for n in names:
        cells = []
        for b, l in combos:
            d = next((x for x in rows if x["cfg"] == n and x["brate"] == b and x["bdelay"] == l), None)
            cells.append("" if d is None else f"{100 * d['mass']:.2f}% / {est_c(d):.2f} / {NL * d['batch'] * MB:.1f}")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print()


def bins(rows, title):
    rows = [d for d in rows if "bins" in d]
    if not rows:
        return
    bw = rows[0]["binw"]
    nb = len(rows[0]["bins"]["hit"])
    print(f"{title}: 위치별 적중률 / 대체 질량 ({bw}토큰 구간)\n")
    print("| 정책 | " + " | ".join(f"{i * bw}~" for i in range(nb)) + " |")
    print("|---|" + "---|" * nb)
    for d in rows:
        print(f"| {d['cfg']} | " + " | ".join(f"{100 * h:.1f}% / {100 * m:.1f}%" for h, m in zip(d["bins"]["hit"], d["bins"]["mass"])) + " |")
    print()


def main(path):
    sims = load(path)
    base = [d for d in sims if d["tau"] == 0.15 and d["brate"] == 0.5 and d["bdelay"] == 1]
    e = next((d for d in base if d["set"] == "all24" and d["cap"] == 128 and d["cfg"] == "eng"), None)
    if e:
        print(f"재현 확인: all24 용량 128 eng — 적중률 {100 * e['hit']:.2f}% (9차 {100 * R9['hit']:.2f}%), 기다린 읽기 {e['must']:.4f} "
              f"(9차 {R9['must']:.4f}), 질량 {100 * e['mass']:.2f}% (9차 {100 * R9['mass']:.2f}%)\n")
    for st in ("all24", "same8k", "cross8k"):
        for cap in (128, 64):
            rows = [d for d in base if d["set"] == st and d["cap"] == cap]
            if not rows:
                continue
            D = rows[0]["D"]
            main_table(rows, f"{st}, 용량 {cap} ({100 * cap / 512:.1f}%), 프롬프트 512, 생성 {D}토큰, τ 0.15")
            fronts(sims, st, cap)
            if st != "all24":
                bins(rows, f"{st}, 용량 {cap}")
    for st in ("all24", "same8k", "cross8k"):
        sens(sims, st)


if __name__ == "__main__":
    main(sys.argv[1])
