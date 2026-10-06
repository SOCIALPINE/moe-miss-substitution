"""Round-6 tables: whole answers (first token + 256 generated tokens), stock llama.cpp with all experts on the CPU and with
the same VRAM as ours (--n-cpu-moe 34), and ours with the final settings, RAM plenty / 32 GB / 16 GB."""
import json, re, statistics, sys

SCEN = [  # (result name, scenario, label)
    ("stock_warm", "warm", "기본 llama.cpp (전문가 전부 CPU)"),
    ("stock34_warm", "warm", "기본 llama.cpp, 같은 VRAM (전문가 레이어 14개 GPU)"),
    ("warm_auto", "warm", "우리 (최종 설정)"),
    ("stock_hog32", "hog32", "기본 llama.cpp (전문가 전부 CPU)"),
    ("stock34_hog32", "hog32", "기본 llama.cpp, 같은 VRAM"),
    ("hog32_auto", "hog32", "우리 (최종 설정)"),
    ("stock_hog", "hog16", "기본 llama.cpp (전문가 전부 CPU)"),
    ("stock34_hog", "hog16", "기본 llama.cpp, 같은 VRAM"),
    ("hog_auto", "hog16", "우리 (최종 설정)"),
    ("hog_auto_cold", "hog16", "우리, 빈 VRAM 캐시 + 빈 페이지 캐시 (최악)"),
]
SCEN_NAME = {"warm": "RAM 넉넉 (모델이 페이지 캐시에 있음)", "hog32": "RAM 약 32GB", "hog16": "RAM 약 16GB"}


def load(path):
    res, vram = {}, {}
    for line in open(path, errors="replace"):
        m = re.search(r"RESULT (\S+) (\{.*\})\s*$", line)
        if m:
            res[m.group(1)] = json.loads(m.group(2))
        m = re.search(r"VRAM_MAX (\S+) (\d+) MiB", line)
        if m:
            vram[m.group(1)] = int(m.group(2))
    return res, vram


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def per_session(t):
    """per session: ttft s, decode s, answer s"""
    n = len(t["prefill"])
    fill = t["fill"] if t["fill"] else [0.0] * n
    dec = t.get("dec_ms", [])
    out = []
    for i in range(n):
        tt = (t["prefill"][i] + fill[i]) / 1000
        dd = dec[i] / 1000 if i < len(dec) else float("nan")
        out.append((tt, dd, tt + dd))
    return out


def main(path):
    R, V = load(path)
    ref = next(iter(R["stock_warm"]["runs"].values()))["ppl"] if "stock_warm" in R else None
    Ps = sorted({int(p) for d in R.values() for p in d.get("ttft_by_p", {})})
    for sc in ("warm", "hog32", "hog16"):
        rows = [(n, lab) for n, s, lab in SCEN if s == sc and n in R]
        if not rows:
            continue
        print(f"### {SCEN_NAME[sc]}\n")
        print("답변 전체 시간 = 첫 토큰 + 256토큰 생성 (세션 중앙값, 초). 괄호: 첫 토큰 / 생성\n")
        print("| 설정 | 세션 | " + " | ".join(f"P={p}" for p in Ps) + " | 생성 tok/s | ppl 변화 | 적중률 | VRAM 최대 |")
        print("|---|---|" + "---|" * len(Ps) + "---|---|---|---|")
        for n, lab in rows:
            d = R[n]
            cells = []
            for p in Ps:
                t = d["ttft_by_p"].get(str(p))
                if not t:
                    cells.append("-")
                    continue
                ps = per_session(t)
                cells.append(f"**{med([x[2] for x in ps]):.1f}** ({med([x[0] for x in ps]):.1f} / {med([x[1] for x in ps]):.1f})")
            runs = list(d["runs"].values())
            run = runs[0]
            tok = " / ".join(f"{r['tok_s']:.1f}" for r in runs)
            same = d["sessions"] == R["stock_warm"]["sessions"] if "stock_warm" in R else False
            dp = "기준" if n == "stock_warm" else (f"{100 * (run['ppl'] / ref - 1):+.2f}%" if ref and same else "-")
            hit = " / ".join(f"{100 * r['hit']:.1f}%" for r in runs if "hit" in r) or "-"
            vr = f"{V[n] / 1024:.1f}GiB" if n in V else "-"
            print(f"| {lab} | {d['sessions']} | " + " | ".join(cells) + f" | {tok} | {dp} | {hit} | {vr} |")
        print()
        # same 3 sessions as the stock runs
        if any(R[n]["sessions"] != R[rows[0][0]]["sessions"] for n, _ in rows):
            print("앞 3세션만 (기본 llama.cpp와 같은 세션): 답변 / 첫 토큰\n")
            print("| 설정 | " + " | ".join(f"P={p}" for p in Ps) + " |")
            print("|---|" + "---|" * len(Ps))
            for n, lab in rows:
                cells = []
                for p in Ps:
                    ps = per_session(R[n]["ttft_by_p"][str(p)])[:3]
                    cells.append(f"{med([x[2] for x in ps]):.1f} / {med([x[0] for x in ps]):.1f}")
                print(f"| {lab} | " + " | ".join(cells) + " |")
            print()
        srows = [(n, lab) for n, lab in rows if any(med(t.get("pf_used", [0])) > 0 for t in R[n]["ttft_by_p"].values())]
        if srows:
            print("프롬프트 스트리밍 내역 (중앙값): SSD GB / RAM GB / VRAM 안 복사 GB / 레이어당 사용 전문가 / 읽기 시간 초\n")
            print("| 설정 | " + " | ".join(f"P={p}" for p in Ps) + " |")
            print("|---|" + "---|" * len(Ps))
            for n, lab in srows:
                cells = []
                for p in Ps:
                    t = R[n]["ttft_by_p"][str(p)]
                    cells.append(f"{med(t['pf_read_GB']):.1f} / {med(t.get('pf_ram_GB', [0])):.1f} / {med(t['pf_d2d_GB']):.1f} / "
                                 f"{med(t['pf_used']) / 48:.0f} / {med(t['pf_ms']) / 1000:.1f}")
                print(f"| {lab} | " + " | ".join(cells) + " |")
            print()
    print("### 배율 (기본 ÷ 우리, 답변 전체 / 첫 토큰)\n")
    print("| RAM | 비교 | " + " | ".join(f"P={p}" for p in Ps) + " |")
    print("|---|---|" + "---|" * len(Ps))
    pairs = [("warm", "stock_warm", "warm_auto"), ("warm", "stock34_warm", "warm_auto"),
             ("hog32", "stock_hog32", "hog32_auto"), ("hog32", "stock34_hog32", "hog32_auto"),
             ("hog16", "stock_hog", "hog_auto"), ("hog16", "stock34_hog", "hog_auto"),
             ("hog16", "stock_hog", "hog_auto_cold"), ("hog16", "stock34_hog", "hog_auto_cold")]
    for sc, a, b in pairs:
        if a not in R or b not in R:
            continue
        cells = []
        for p in Ps:
            pa, pb = per_session(R[a]["ttft_by_p"][str(p)]), per_session(R[b]["ttft_by_p"][str(p)])
            cells.append(f"{med([x[2] for x in pa]) / med([x[2] for x in pb]):.1f} / {med([x[0] for x in pa]) / med([x[0] for x in pb]):.1f}")
        print(f"| {SCEN_NAME[sc]} | {a} ÷ {b} | " + " | ".join(cells) + " |")
    print()
    print("원자료: 이름 / P / 첫 토큰 s / 생성 s (세션별)\n")
    for n, d in R.items():
        for p, t in d.get("ttft_by_p", {}).items():
            ps = per_session(t)
            print(f"- {n} P={p}: ttft {[round(x[0], 2) for x in ps]}, gen {[round(x[1], 2) for x in ps]}")
    print()


if __name__ == "__main__":
    main(sys.argv[1])
