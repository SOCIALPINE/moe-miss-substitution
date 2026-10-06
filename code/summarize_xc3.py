"""Round-3 tables: time to first token (RESULT lines of run_xc3.sh)."""
import json, re, statistics, sys

SCEN = [  # (result name, scenario, option label)
    ("stock_warm", "warm", "llama.cpp 기본 (전문가 전부 RAM)"),
    ("warm_base", "warm", "우리 방식"),
    ("warm_ef", "warm", "우리 방식 + 조기 채우기(ef)"),
    ("cold_base", "cold", "우리 방식"),
    ("cold_pf", "cold", "우리 방식 + 미리읽기(pf=3)"),
    ("cold_pf_ef", "cold", "우리 방식 + pf + ef"),
    ("stock_hog", "hog", "llama.cpp 기본"),
    ("hog_base", "hog", "우리 방식"),
    ("hog_pf", "hog", "우리 방식 + 미리읽기(pf=2)"),
    ("hog_pf_ef", "hog", "우리 방식 + pf + ef"),
]
SCEN_NAME = {"warm": "따뜻함 (모델이 페이지 캐시에 있음)", "cold": "차가움 (매 프롬프트 전에 전문가 페이지 비움)",
             "hog": "RAM 약 16GB 제한 + 차가움"}


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
    for sc in ("warm", "cold", "hog"):
        rows = [(n, lab) for n, s, lab in SCEN if s == sc and n in R]
        if not rows:
            continue
        print(f"### {SCEN_NAME[sc]}\n")
        print("첫 토큰까지 시간 = 프롬프트 처리 + 전문가 집합 채우기 (3세션 중앙값, 초)\n")
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
    print("원자료 (ms): 이름 / P / prefill 3회 / fill 3회\n")
    for n, d in R.items():
        for p, t in d.get("ttft_by_p", {}).items():
            print(f"- {n} P={p}: prefill {[round(x) for x in t['prefill']]}, fill {[round(x) for x in t['fill']]}")
    print()


if __name__ == "__main__":
    main(sys.argv[1])
