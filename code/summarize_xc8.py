"""Round-8 tables: ram=1 (now the default) against ram=0 on one machine, with RAM plenty, with about 16 GB where only our
own process fills the page cache (the model file is evicted before each run), and fully cold (VRAM cache and page cache
emptied before every prompt). Also prints the page cache share of the model file around the runs (PAGECACHE lines)."""
import json, re, statistics, sys

SCEN = [  # (result name, scenario, label)
    ("warm_r1", "warm", "ram=1 (기본값)"),
    ("warm_r0", "warm", "ram=0 (항상 SSD)"),
    ("hog_r1", "hog16", "ram=1 (기본값)"),
    ("hog_r0", "hog16", "ram=0 (항상 SSD)"),
    ("hog_r1b", "hog16", "ram=1 (기본값), 다시"),
    ("hog_r1_cold", "cold", "ram=1 (기본값)"),
    ("hog_r0_cold", "cold", "ram=0 (항상 SSD)"),
]
SCEN_NAME = {"warm": "RAM 넉넉 (모델이 페이지 캐시에 있음)",
             "hog16": "RAM 약 16GB, 실행 전 모델 파일을 페이지 캐시에서 비움 (우리 프로세스만 채움)",
             "cold": "RAM 약 16GB, 프롬프트마다 VRAM 캐시와 페이지 캐시를 비움 (최악)"}
PAIRS = [("warm", "warm_r0", "warm_r1"), ("hog16", "hog_r0", "hog_r1"), ("hog16", "hog_r0", "hog_r1b"),
         ("cold", "hog_r0_cold", "hog_r1_cold")]


def load(path):
    res, vram, pc = {}, {}, []
    for line in open(path, errors="replace"):
        m = re.search(r"RESULT (\S+) (\{.*\})\s*$", line)
        if m:
            res[m.group(1)] = json.loads(m.group(2))
        m = re.search(r"VRAM_MAX (\S+) (\d+) MiB", line)
        if m:
            vram[m.group(1)] = int(m.group(2))
        m = re.search(r"(PAGECACHE .*)$", line)
        if m:
            pc.append(m.group(1).strip())
    return res, vram, pc


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
    R, V, PC = load(path)
    Ps = sorted({int(p) for d in R.values() for p in d.get("ttft_by_p", {})})
    ref = next(iter(R["warm_r1"]["runs"].values()))["ppl"] if "warm_r1" in R else None
    for sc in ("warm", "hog16", "cold"):
        rows = [(n, lab) for n, s, lab in SCEN if s == sc and n in R]
        if not rows:
            continue
        print(f"### {SCEN_NAME[sc]}\n")
        print("답변 전체 시간 = 첫 토큰 + 256토큰 생성 (세션 6개 중앙값, 초). 괄호: 첫 토큰 / 생성\n")
        print("| 설정 | " + " | ".join(f"P={p}" for p in Ps) + " | 생성 tok/s | ppl | 적중률 | 토큰당 SSD MB | 로드 s | VRAM 최대 |")
        print("|---|" + "---|" * len(Ps) + "---|---|---|---|---|---|")
        for n, lab in rows:
            d = R[n]
            cells = []
            for p in Ps:
                t = d["ttft_by_p"].get(str(p))
                if not t:
                    cells.append("-")
                    continue
                ps = per_session(t)
                cells.append(f"**{med([x[2] for x in ps]):.2f}** ({med([x[0] for x in ps]):.2f} / {med([x[1] for x in ps]):.2f})")
            run = next(iter(d["runs"].values()))
            dp = f"{run['ppl']:.4f}" + ("" if not ref or n == "warm_r1" else f" ({100 * (run['ppl'] / ref - 1):+.2f}%)")
            vr = f"{V[n] / 1024:.1f}GiB" if n in V else "-"
            print(f"| {lab} | " + " | ".join(cells) +
                  f" | {run['tok_s']:.1f} | {dp} | {100 * run.get('hit', 0):.1f}% | {run.get('MB_per_tok', 0):.1f} | {d.get('load_s', 0):.1f} | {vr} |")
        print()
        print("프롬프트 스트리밍 내역 (중앙값): SSD GB / 페이지 캐시 GB / VRAM 안 복사 GB / 읽기 시간 초\n")
        print("| 설정 | " + " | ".join(f"P={p}" for p in Ps) + " |")
        print("|---|" + "---|" * len(Ps))
        for n, lab in rows:
            cells = []
            for p in Ps:
                t = R[n]["ttft_by_p"][str(p)]
                cells.append(f"{med(t['pf_read_GB']):.1f} / {med(t.get('pf_ram_GB', [0])):.1f} / {med(t['pf_d2d_GB']):.1f} / "
                             f"{med(t['pf_ms']) / 1000:.2f}")
            print(f"| {lab} | " + " | ".join(cells) + " |")
        print()
    print("### ram=0 대비 ram=1 차이 (ram=1 − ram=0, 초; 음수면 ram=1이 빠름). 답변 전체 / 첫 토큰 / 생성\n")
    print("| 상황 | 비교 | " + " | ".join(f"P={p}" for p in Ps) + " | 생성 tok/s |")
    print("|---|---|" + "---|" * len(Ps) + "---|")
    for sc, a, b in PAIRS:
        if a not in R or b not in R:
            continue
        cells = []
        for p in Ps:
            pa, pb = per_session(R[a]["ttft_by_p"][str(p)]), per_session(R[b]["ttft_by_p"][str(p)])
            da = [med([x[k] for x in pb]) - med([x[k] for x in pa]) for k in (2, 0, 1)]
            pct = 100 * da[0] / med([x[2] for x in pa])
            cells.append(f"{da[0]:+.2f} ({pct:+.0f}%) / {da[1]:+.2f} / {da[2]:+.2f}")
        ta, tb = next(iter(R[a]["runs"].values()))["tok_s"], next(iter(R[b]["runs"].values()))["tok_s"]
        print(f"| {sc} | {b} − {a} | " + " | ".join(cells) + f" | {ta:.1f} → {tb:.1f} |")
    print()
    if PC:
        print("### 모델 파일의 페이지 캐시 상주 비율\n")
        for line in PC:
            print(f"- {line}")
        print()
    print("원자료: 이름 / P / 첫 토큰 s / 생성 s (세션별)\n")
    for n, d in R.items():
        for p, t in d.get("ttft_by_p", {}).items():
            ps = per_session(t)
            print(f"- {n} P={p}: ttft {[round(x[0], 2) for x in ps]}, gen {[round(x[1], 2) for x in ps]}, "
                  f"ram GB {[round(x, 2) for x in t.get('pf_ram_GB', [])]}")
    print()


if __name__ == "__main__":
    main(sys.argv[1])
