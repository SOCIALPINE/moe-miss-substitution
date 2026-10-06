"""Round-4 tables: time to first token with prompt streaming (RESULT lines of run_xc4.sh)."""
import json, re, statistics, sys

SCEN = [  # (result name, scenario, label)
    ("stock_warm", "warm", "기본 llama.cpp (전문가 전부 RAM)"),
    ("warm_ef", "warm", "Part 9 최선 (조기 채우기)"),
    ("warm_ps", "warm", "프롬프트 스트리밍"),
    ("warm_pso", "warm", "프롬프트 스트리밍 + 다음 레이어 미리 읽기"),
    ("stock_hog", "hog", "기본 llama.cpp"),
    ("hog_ef", "hog", "Part 9 최선 (조기 채우기)"),
    ("hog_ps", "hog", "프롬프트 스트리밍"),
    ("hog_ps_fresh", "hog", "프롬프트 스트리밍, 빈 캐시에서 시작"),
    ("hog_pso", "hog", "스트리밍 + 미리 읽기"),
    ("hog_pso_fresh", "hog", "스트리밍 + 미리 읽기, 빈 캐시에서 시작"),
]
SCEN_NAME = {"warm": "RAM 넉넉 (모델이 페이지 캐시에 있음)", "hog": "RAM 약 16GB"}


def load(path):
    res = {}
    for line in open(path, errors="replace"):
        m = re.search(r"RESULT (\S+) (\{.*\})\s*$", line)
        if m:
            res[m.group(1)] = json.loads(m.group(2))
    return res


def med(xs):
    return statistics.median(xs) if xs else 0.0


def main(path):
    R = load(path)
    ref = next(iter(R["stock_warm"]["runs"].values()))["ppl"] if "stock_warm" in R else None
    Ps = sorted({int(p) for d in R.values() for p in d.get("ttft_by_p", {})})
    for sc in ("warm", "hog"):
        rows = [(n, lab) for n, s, lab in SCEN if s == sc and n in R]
        if not rows:
            continue
        print(f"### {SCEN_NAME[sc]}\n")
        print("첫 토큰까지 = 프롬프트 처리 + 집합 채우기 (3세션 중앙값, 초)\n")
        print("| 설정 | " + " | ".join(f"P={p}" for p in Ps) + " | 생성 tok/s | ppl 변화 | 적중률 |")
        print("|---|" + "---|" * len(Ps) + "---|---|---|")
        for n, lab in rows:
            d = R[n]
            cells = []
            for p in Ps:
                t = d["ttft_by_p"].get(str(p))
                if not t:
                    cells.append("-")
                    continue
                pre, fill = med(t["prefill"]) / 1000, med(t["fill"]) / 1000
                cells.append(f"**{pre + fill:.1f}** ({pre:.1f}+{fill:.1f})" if t["fill"] else f"**{pre:.1f}**")
            run = next(iter(d["runs"].values()))
            dp = "기준" if n == "stock_warm" else (f"{100 * (run['ppl'] / ref - 1):+.2f}%" if ref and "hit" in run else "-")
            hit = f"{100 * run['hit']:.1f}%" if "hit" in run else "-"
            print(f"| {lab} | " + " | ".join(cells) + f" | {run['tok_s']:.1f} | {dp} | {hit} |")
        print()
        srows = [(n, lab) for n, lab in rows if any(med(t.get("pf_used", [])) > 0 for t in R[n]["ttft_by_p"].values())]
        if srows:
            print("프롬프트 스트리밍 내역 (중앙값): SSD 읽기 GB / VRAM 안 복사 GB / 레이어당 사용 전문가 / 읽기 시간 초 / 미리 읽은 것 중 쓴 비율\n")
            print("| 설정 | " + " | ".join(f"P={p}" for p in Ps) + " |")
            print("|---|" + "---|" * len(Ps))
            for n, lab in srows:
                cells = []
                for p in Ps:
                    t = R[n]["ttft_by_p"].get(str(p), {})
                    pre = med(t.get("pf_pre", []))
                    hitr = f" / {100 * med(t.get('pf_pre_hit', [])) / pre:.0f}%" if pre > 0 else ""
                    cells.append(f"{med(t.get('pf_read_GB', [])):.1f} / {med(t.get('pf_d2d_GB', [])):.1f} / "
                                 f"{med(t.get('pf_used', [])) / 48:.0f} / {med(t.get('pf_ms', [])) / 1000:.1f}{hitr}")
                print(f"| {lab} | " + " | ".join(cells) + " |")
            print()
    print("원자료 (ms): 이름 / P / prefill 3회 / fill 3회 / SSD 읽기 GB 3회\n")
    for n, d in R.items():
        for p, t in d.get("ttft_by_p", {}).items():
            print(f"- {n} P={p}: prefill {[round(x) for x in t['prefill']]}, fill {[round(x) for x in t['fill']]}, "
                  f"read {[round(x, 1) for x in t.get('pf_read_GB', [])]}")
    print()


if __name__ == "__main__":
    main(sys.argv[1])
