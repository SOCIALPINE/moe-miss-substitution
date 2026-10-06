"""Round-10 tables from xsim output (SIM / ALLOC lines): rent-or-buy admission (regret), gate-weighted frequency,
SwapMoE-style masking with periodic refill, per-layer slot allocation. Qwen3-Next-80B-A3B (48 MoE layers, 512 experts, top-10).

Speed estimate as round 9: 8.195 ms + 0.557 ms per waited read (round-8 engine, ram=0). "Contention" adds a rough penalty:
waited reads take (1 + u) times longer, u = share of the token time the SSD (5.5 GB/s) spends on background reads.
Quality proxy: substituted gate mass per layer and token."""
import json, re, sys
from collections import defaultdict

T0, ELL, NL, MB, SSD = 8.195, 0.557, 48, 1.908, 5.5


def load(path):
    sims, allocs = [], []
    for line in open(path, errors="replace"):
        m = re.search(r"\b(SIM|ALLOC) (\{.*\})\s*$", line)
        if m:
            (sims if m.group(1) == "SIM" else allocs).append(json.loads(m.group(2)))
    return sims, allocs


def est(d):
    return T0 + NL * d["must"] * ELL


def est_c(d):
    ms = est(d)
    for _ in range(20):
        u = min(0.9, NL * d["batch"] * MB / SSD / ms)
        ms = T0 + NL * d["must"] * ELL * (1 + u)
    return ms


def bgmb(d):
    return NL * d["batch"] * MB


def pct(a, b):
    return 100 * (a / b - 1) if b else 0.0


def pareto(rows):
    pts = [(d["cfg"], est_c(d), d["mass"]) for d in rows]
    return {n for n, x, y in pts if not any((x2 <= x and y2 <= y) and (x2 < x or y2 < y) for _, x2, y2 in pts)}


def interp(pts, x):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return None


def cell(d, e):
    return f"{100 * d['mass']:.2f}% ({pct(d['mass'], e['mass']):+.0f}%) / {est_c(d):.2f} ({pct(est_c(d), est_c(e)):+.1f}%) / {bgmb(d):.1f}"


def main_table(rows, title):
    print(f"### {title}\n")
    print("| 정책 | 적중률 | 토큰당 기다린 읽기 | 대체 질량 | 토큰당 배경 MB | 추정 ms | 경합 반영 ms | 앞선 집합 |")
    print("|---|---|---|---|---|---|---|---|")
    front = pareto(rows)
    for d in sorted(rows, key=lambda d: (est_c(d), d["mass"])):
        print(f"| {d['cfg']} | {100 * d['hit']:.1f}% | {NL * d['must']:.2f} | {100 * d['mass']:.2f}% | {bgmb(d):.1f} | {est(d):.2f} | "
              f"{est_c(d):.2f} | {'●' if d['cfg'] in front else ''} |")
    print()
    eng = next(d for d in rows if d["cfg"] == "eng")
    em, ems = eng["mass"], est_c(eng)
    print(f"지금 엔진(eng): {ems:.2f} ms, 질량 {100 * em:.2f}%, 적중률 {100 * eng['hit']:.1f}%\n")
    print("| 시간 예산 (경합 반영, eng 대비) | 질량이 가장 낮은 정책 | 질량 (eng 대비) / ms / 배경 MB |")
    print("|---|---|---|")
    for lim in (-0.02, 0.0, 0.02, 0.05, 0.10):
        ok = [d for d in rows if est_c(d) <= ems * (1 + lim) + 1e-9 and d["cfg"] != "full"]
        if ok:
            b = min(ok, key=lambda d: (d["mass"], est_c(d)))
            print(f"| {100 * lim:+.0f}% | {b['cfg']} | {cell(b, eng)} |")
    print()


def regret_grid(rows):
    by = {d["cfg"]: d for d in rows}
    eng = by["eng"]
    Bs = sorted({float(m.group(1)) for n in by for m in [re.match(r"rg([\d.]+)_(\d+)$", n)] if m})
    Hs = sorted({int(m.group(2)) for n in by for m in [re.match(r"rg([\d.]+)_(\d+)$", n)] if m})
    if not Bs:
        return
    print("대체 손실 들이기 격자 (τ 0.15): 칸마다 질량 (eng 대비) / 경합 반영 ms (eng 대비) / 토큰당 배경 MB\n")
    print("| 구매 비용 β \\ 반감기 H | " + " | ".join(f"{h}토큰" for h in Hs) + " |")
    print("|---|" + "---|" * len(Hs))
    for b in Bs:
        nm = lambda h: f"rg{b:g}_{h}"
        print(f"| {b:g} | " + " | ".join(cell(by[nm(h)], eng) if nm(h) in by else "" for h in Hs) + " |")
    print()


HEAD = [("eng", "지금 엔진 (C τ0.15, LRU)"), ("eng_c1", "지금 엔진 C1"), ("fe128_d", "빈도 1/128 (갱신 없음)"),
        ("few128_d", "게이트 가중 빈도 1/128 (갱신 없음)"), ("l256_fe128", "256토큰 빈도 갱신 1/128"),
        ("l256_few128", "256토큰 게이트 가중 빈도 갱신 1/128"), ("l256_fe256_c1", "C1 + 256토큰 빈도 갱신 1/256"),
        ("l256_fe64_c1", "C1 + 256토큰 빈도 갱신 1/64"), ("l256_few256_c1", "C1 + 256토큰 게이트 가중 갱신 1/256"),
        ("bg4_fe64", "미스마다 배경 읽기"), ("mask_lru", "SwapMoE식: 기다림 없음, 갱신 없음"),
        ("sw16", "SwapMoE식: 기다림 없음 + 16토큰 갱신"), ("sw64", "SwapMoE식: 기다림 없음 + 64토큰 갱신"),
        ("sw256", "SwapMoE식: 기다림 없음 + 256토큰 갱신"), ("rg0.25_512", "대체 손실 들이기 β0.25 H512"), ("rg0.5_512", "대체 손실 들이기 β0.5 H512"),
        ("rg1_512", "대체 손실 들이기 β1 H512"), ("rg0.5_512g", "대체 손실 들이기 β0.5 + 들이기 필터"),
        ("rg0.5_512_few", "대체 손실 들이기 β0.5, 게이트 가중 퇴출"), ("rg0.25_512_c1", "C1 + 대체 손실 들이기 β0.25"),
        ("rg0.5_512_c1", "C1 + 대체 손실 들이기 β0.5"), ("rg1_512_c1", "C1 + 대체 손실 들이기 β1")]


def headline(base, sets, cap):
    print(f"### 비교: 용량 {cap}, τ 0.15 (칸마다 질량 (eng 대비) / 경합 반영 ms (eng 대비) / 토큰당 배경 MB)\n")
    print("| 정책 | " + " | ".join(sets) + " |")
    print("|---|" + "---|" * len(sets))
    for n, lab in HEAD:
        cells = []
        for st in sets:
            rows = {d["cfg"]: d for d in base if d["set"] == st and d["cap"] == cap}
            cells.append(cell(rows[n], rows["eng"]) if n in rows else "")
        print(f"| {lab} ({n}) | " + " | ".join(cells) + " |")
    print()


def alloc_table(base, sets, cap):
    bases = ["eng", "eng_c1", "l256_fe128", "l256_fe256_c1", "rg0.5_512", "rg0.5_512_c1"]
    print(f"### 레이어별 슬롯 나누기: 용량 {cap} (칸마다 질량 / 경합 반영 ms, 괄호는 같은 정책의 균등 배분 대비)\n")
    print("| 정책 | 나누기 | " + " | ".join(sets) + " |")
    print("|---|---|" + "---|" * len(sets))
    for b in bases:
        for a, lab in (("", "균등"), ("_pm", "프롬프트 LRU 곡선 (질량)"), ("_pw", "프롬프트 작업 집합 곡선 (Denning)"), ("_om", "오라클 (생성 구간 곡선)")):
            cells = []
            for st in sets:
                rows = {d["cfg"]: d for d in base if d["set"] == st and d["cap"] == cap}
                d, u = rows.get(b + a), rows.get(b)
                if d is None or u is None:
                    cells.append("")
                    continue
                cells.append(f"{100 * d['mass']:.2f}% ({pct(d['mass'], u['mass']):+.1f}%) / {est_c(d):.2f} ({pct(est_c(d), est_c(u)):+.1f}%)")
            print(f"| {b} | {lab} | " + " | ".join(cells) + " |")
    print()


def alloc_lines(allocs):
    seen = set()
    print("### 나눈 슬롯 (레이어별 평균, 시퀀스 평균)\n")
    for a in allocs:
        key = (a["set"], a["cap"], a["type"])
        if key in seen or a["tau"] != 0.15:
            continue
        seen.add(key)
        m = a["mean"]
        print(f"- {a['set']} 용량 {a['cap']} {a['type']}: 최소 {a['min']}, 최대 {a['max']}, 생성 구간 LRU 질량 예측 균등 {100 * a['lru_mass_uniform']:.2f}% → "
              f"나눔 {100 * a['lru_mass_alloc']:.2f}% ({pct(a['lru_mass_alloc'], a['lru_mass_uniform']):+.1f}%)")
        print("  - 레이어 0~47: " + " ".join(f"{x:.0f}" for x in m))
    print()


def fronts(sims, st, cap):
    rows = [d for d in sims if d["set"] == st and d["cap"] == cap and d["brate"] == 0.5 and d["bdelay"] == 1]
    names = sorted({d["cfg"] for d in rows if d["tau"] != 0.15})
    if not names:
        return
    eng = next(d for d in rows if d["cfg"] == "eng" and d["tau"] == 0.15)
    em, ems = eng["mass"], est_c(eng)
    print(f"### τ 곡선 (0.08~0.5): {st}, 용량 {cap}\n")
    print("| 정책 | τ별 (경합 반영 ms, 질량) | eng 속도에서 질량 | eng 질량에서 ms |")
    print("|---|---|---|---|")
    for n in names:
        p = sorted([d for d in rows if d["cfg"] == n], key=lambda d: -d["tau"])
        pts = sorted((est_c(d), d["mass"]) for d in p)
        m_at = interp(pts, ems)
        t_at = interp(sorted((d["mass"], est_c(d)) for d in p), em)
        cur = " ".join(f"{d['tau']:g}:({est_c(d):.2f}, {100 * d['mass']:.2f}%)" for d in p)
        print(f"| {n} | {cur} | {'' if m_at is None else f'{100 * m_at:.2f}% ({pct(m_at, em):+.0f}%)'} | "
              f"{'' if t_at is None else f'{t_at:.2f} ({pct(t_at, ems):+.1f}%)'} |")
    print()


def sens(sims, st):
    rows = [d for d in sims if d["set"] == st and d["cap"] == 128 and d["tau"] == 0.15]
    combos = [(0.25, 1), (0.5, 1), (1, 1), (0.5, 4), (0.5, 16)]
    names = sorted({d["cfg"] for d in rows if d["brate"] != 0.5 or d["bdelay"] != 1})
    if not names:
        return
    print(f"### 배경 읽기 속도·지연 민감도: {st}, 용량 128 (질량 / 경합 반영 ms / 토큰당 배경 MB)\n")
    print("| 정책 | " + " | ".join(f"속도 {b:g}, 지연 {l}" for b, l in combos) + " |")
    print("|---|" + "---|" * len(combos))
    for n in names:
        cs = []
        for b, l in combos:
            d = next((x for x in rows if x["cfg"] == n and x["brate"] == b and x["bdelay"] == l), None)
            cs.append("" if d is None else f"{100 * d['mass']:.2f}% / {est_c(d):.2f} / {bgmb(d):.1f}")
        print(f"| {n} | " + " | ".join(cs) + " |")
    print()


def bins(base, st, cap):
    rows = [d for d in base if d["set"] == st and d["cap"] == cap and "bins" in d]
    if not rows:
        return
    bw, nb = rows[0]["binw"], len(rows[0]["bins"]["hit"])
    print(f"{st}, 용량 {cap}: 위치별 적중률 / 질량 ({bw}토큰 구간)\n")
    print("| 정책 | " + " | ".join(f"{i * bw}~" for i in range(nb)) + " |")
    print("|---|" + "---|" * nb)
    for d in rows:
        print(f"| {d['cfg']} | " + " | ".join(f"{100 * h:.0f} / {100 * m:.1f}" for h, m in zip(d["bins"]["hit"], d["bins"]["mass"])) + " |")
    print()


def main(path):
    sims, allocs = load(path)
    base = [d for d in sims if d["tau"] == 0.15 and d["brate"] == 0.5 and d["bdelay"] == 1]
    sets = [s for s in ("all24", "same8k", "cross8k") if any(d["set"] == s for d in base)]
    e = next((d for d in base if d["set"] == "all24" and d["cap"] == 128 and d["cfg"] == "eng"), None)
    if e:
        print(f"재현 확인: all24 용량 128 eng 적중률 {100 * e['hit']:.2f}% (9차 86.12%), 기다린 읽기 {e['must']:.4f} (9차 0.0731), "
              f"질량 {100 * e['mass']:.2f}% (9차 9.72%)\n")
    for cap in (128, 64):
        headline(base, sets, cap)
        for st in sets:
            rows = [d for d in base if d["set"] == st and d["cap"] == cap]
            print(f"#### {st}, 용량 {cap}\n")
            regret_grid(rows)
        alloc_table(base, sets, cap)
    alloc_lines(allocs)
    for st in sets:
        for cap in (128, 64):
            fronts(sims, st, cap)
    for st in sets:
        sens(sims, st)
    for st in sets:
        bins(base, st, 128)
    for st in sets:
        for cap in (128, 64):
            rows = [d for d in base if d["set"] == st and d["cap"] == cap]
            if rows:
                main_table(rows, f"{st}, 용량 {cap}, τ 0.15 (전체)")


if __name__ == "__main__":
    main(sys.argv[1])
