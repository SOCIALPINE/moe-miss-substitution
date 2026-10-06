"""Round-2 tables: engine fixes and cache policies (RESULT lines of run_xc2.sh)."""
import json, re, sys


def load(path):
    res = {}
    for line in open(path, errors="replace"):
        m = re.search(r"RESULT (\S+) (\{.*\})\s*$", line)
        if m:
            res[m.group(1)] = json.loads(m.group(2))
    return res


OPT_NAMES = {"aio": "비동기복사", "uf": "급한읽기우선", "up": "조각병렬", "adm": "들이기필터", "sbg": "토큰끝대기(진단)"}


def label(key):
    base, _, opts = key.partition("|")
    m = re.match(r"([^@]+)@([0-9.]+)/([a-z]+)(?::([0-9.]+))?", base)
    mode, frac, tau = m.group(1), float(m.group(2)), m.group(4)
    if mode == "C" and tau:
        mode = f"C(τ{tau})"
    kv = dict(x.split("=") for x in opts.split("+")) if opts else {}
    parts = []
    if kv.get("aio") == "1" and kv.get("uf") == "1" and kv.get("up") == "1":
        parts.append("엔진개선")
        kv = {k: v for k, v in kv.items() if k not in ("aio", "uf", "up")}
    for k, v in kv.items():
        if k == "bg" and v == "0":
            parts.append("**배경읽기 끔**")
        elif k == "lam":
            parts.append(f"LRFU λ{v}")
        elif v == "1":
            parts.append(OPT_NAMES.get(k, k))
    return f"{frac * 100:g}% {mode}" + (" + " + " + ".join(parts) if parts else "")


def main(path):
    R = load(path)
    ideal_any = None
    for d in R.values():
        ideal_any = ideal_any or next((v["tok_s"] for k, v in d["runs"].items() if k.startswith("ideal@")), None)
    for name in ("xc2", "xc2long"):
        if name not in R:
            continue
        d = R[name]
        runs = d["runs"]
        ideal = next((v["tok_s"] for k, v in runs.items() if k.startswith("ideal@")), None) or ideal_any
        print(f"### {name}: {d['sessions']}세션 × {d['D']}토큰, threads {d['threads']}\n")
        print("| 설정 | tok/s | ideal 대비 | 대기 ms | 콜백 ms | 적중률 | 대체율 | 읽기 MB/토큰 | 안 읽은 미스/토큰 | ppl 변화 | 정답 일치 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, v in runs.items():
            fake = k.startswith("ideal")
            pct = f"{100 * v['tok_s'] / ideal:.0f}%" if ideal else "-"
            if fake:
                print(f"| {k} | {v['tok_s']:.1f} | {pct} | - | {v.get('cb_ms', 0):.2f} | - | - | - | - | - | - |")
                continue
            print(f"| {label(k)} | {v['tok_s']:.1f} | {pct} | {v['stall_ms']:.2f} | {v['cb_ms']:.2f} | {100 * v['hit']:.1f}% | "
                  f"{100 * v['subst']:.1f}% | {v['MB_per_tok']:.0f} | {v.get('skip_per_tok', 0):.1f} | {v['dppl_pct']:+.2f}% | {100 * v['agree']:.1f}% |")
        print()
        print("구간별 (생성 위치를 4등분): 적중률 / ppl 변화 / tok/s\n")
        print("| 설정 | 1구간 | 2구간 | 3구간 | 4구간 |")
        print("|---|---|---|---|---|")
        for k, v in runs.items():
            if k.startswith("ideal") or "q_hit" not in v:
                continue
            cells = [f"{100 * v['q_hit'][i]:.1f}% / {v.get('q_dppl', [0] * 4)[i]:+.1f}% / {1000 / v['q_ms'][i]:.0f}" for i in range(4)]
            print(f"| {label(k)} | " + " | ".join(cells) + " |")
        print()
        print(f"TTFT ms: {', '.join(f'{x:.0f}' for x in d['ttft_ms'])}\n")


if __name__ == "__main__":
    main(sys.argv[1])
