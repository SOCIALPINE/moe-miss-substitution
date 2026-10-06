"""Summarize the llama.cpp expert-cache benchmark (RESULT lines from run_xc.sh) into markdown tables."""
import json, sys, re

def load(path):
    res = {}
    for line in open(path, errors="replace"):
        m = re.search(r"RESULT (\S+) (\{.*\})\s*$", line)
        if m:
            res[m.group(1)] = json.loads(m.group(2))
    return res

def label(key):
    # keys look like "C1@0.25/nvme" or "C@0.25/nvme:0.15"
    m = re.match(r"([^@]+)@([0-9.]+)/([a-z]+)(?::([0-9.]+))?", key)
    if not m:
        return key
    mode, frac, src, tau = m.group(1), float(m.group(2)), m.group(3), m.group(4)
    if mode == "C" and tau:
        mode = f"C(τ={tau})"
    return f"{mode} {frac*100:g}% {'RAM' if src == 'ram' else 'NVMe'}"

def main(path):
    R = load(path)
    out = []
    xc = R.get("xc")
    ideal = None
    if xc:
        ideal = next((v["tok_s"] for k, v in xc["runs"].items() if k.startswith("ideal@") or k.startswith("ideal:")), None)
    def row(name, run, key, base):
        r = run["runs"][key]
        pct = f"{100*r['tok_s']/ideal:.0f}%" if ideal else "-"
        stall = f"{r['stall_ms']:.1f}" if "stall_ms" in r else "-"
        fk = key.startswith("ideal")
        hit = f"{100*r['hit']:.1f}%" if "hit" in r and not fk else "-"
        mb = f"{r['MB_per_tok']:.0f}" if "MB_per_tok" in r and not fk else "-"
        fake = key.startswith("ideal")  # fixed fake slot map: outputs are not meaningful
        dp = f"{r['dppl_pct']:+.2f}%" if base and not fake else "-"
        ag = f"{100*r['agree']:.1f}%" if base and not fake and r.get("agree", -1) >= 0 else "-"
        ppl = "-" if fake else f"{r['ppl']:.3f}"
        return f"| {name} | {r['tok_s']:.2f} | {pct} | {r['ms_p50']:.1f} / {r['ms_p90']:.1f} | {stall} | {hit} | {mb} | {ppl} | {dp} | {ag} |"
    hdr = ("| 설정 | tok/s | ideal 대비 | ms p50/p90 | stall ms/tok | hit | MB/tok | ppl | Δppl vs exact | top-1 일치 |\n"
           "|---|---|---|---|---|---|---|---|---|---|")
    out.append("### 디코드 (teacher-forced, 256 tok × 세션)\n")
    out.append(hdr)
    if xc:
        for k in xc["runs"]:
            out.append(row("xcache " + label(k), xc, k, True))
    for nm, desc in [("stock36", "stock --n-cpu-moe 36 (12층 전문가 VRAM ≈ 25%)"),
                     ("stock48", "stock --n-cpu-moe 48 (전문가 전부 CPU)")]:
        if nm in R:
            out.append(row(desc, R[nm], "stock", False))
    out.append("")
    if "stock48_hog" in R or "xc_hog" in R:
        out.append("### RAM 16 GB 제한 (나머지 메모리를 hog로 점유)\n")
        out.append(hdr)
        if "xc_hog" in R:
            for k in R["xc_hog"]["runs"]:
                out.append(row("xcache " + label(k), R["xc_hog"], k, True))
        if "stock48_hog" in R:
            out.append(row("stock --n-cpu-moe 48", R["stock48_hog"], "stock", False))
        out.append("")
    out.append("### TTFT (prefill, ms)\n")
    out.append("| run | P | TTFT 세션별 (ms) | 평균 |\n|---|---|---|---|")
    for nm, r in R.items():
        t = r.get("ttft_ms", [])
        if t:
            out.append(f"| {nm} | {r['P']} | {', '.join(f'{x:.0f}' for x in t)} | {sum(t)/len(t):.0f} |")
    out.append("")
    out.append("load_s: " + ", ".join(f"{nm}={r['load_s']:.1f}s" for nm, r in R.items()))
    print("\n".join(out))

if __name__ == "__main__":
    main(sys.argv[1])
