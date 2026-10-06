"""SECTION_LIVE json -> Korean-friendly tables + projections to other per-token compute times."""
import json, sys


def proj(tc, stall, mb, bw):
    io = mb / bw if bw else 0.0
    return tc / max(tc + stall, io)


def main(path):
    txt = open(path).read()
    i = txt.index("SECTION_LIVE ")
    res = json.loads(txt[i + len("SECTION_LIVE "):].splitlines()[0])
    m = res["meta"]
    bw = (m.get("nvme") or {}).get("qd8_GBs") or 3.0
    print(f"GPU {m.get('gpu')} | NVMe qd1 {m['nvme']['qd1_ms_per_expert']} ms/expert, qd8 {bw} GB/s | "
          f"ideal {res['ideal_ms']} ms/tok | sessions {m['sessions']} x {m['D']} tokens")
    for t in res["ttft"]:
        print(f"  TTFT {t['src']}: {t['ttft_s']:.2f}s (I/O wait {t['io_wait_s']:.2f}s, {t['GB']:.1f} GB)")
    print(f"{'config':22s} {'ms/tok':>7s} {'tok/s':>6s} {'rel':>5s} {'stall':>6s} {'other':>6s} {'hit':>5s} "
          f"{'sub':>5s} {'urg':>5s} {'bg':>5s} {'MB':>5s} {'dppl%':>6s} {'agree':>6s} | proj tc=10/21/40 ms")
    for k, d in res["runs"].items():
        line = f"{k:22s} {d['ms_mean']:7.1f} {d['tok_s']:6.2f} {d['rel_ideal']:5.3f}"
        if "stall_ms" in d:
            src_bw = bw if k.endswith("/nvme") else 0.0
            pj = [proj(tc, d["stall_ms"], d["MB_per_tok"], src_bw) for tc in (10, 21, 40)]
            line += (f" {d['stall_ms']:6.2f} {d['other_ms']:6.1f} {d['hit']:5.3f} {d['subst']:5.3f} "
                     f"{d['urg_per_tok']:5.2f} {d['bg_per_tok']:5.2f} {d['MB_per_tok']:5.1f} "
                     f"{d.get('dppl_pct', 0):6.2f} {d.get('agree', 0):6.3f} | " + " / ".join(f"{p:.3f}" for p in pj))
        print(line)
    if res.get("errors"):
        print("errors:", res["errors"])


if __name__ == "__main__":
    main(sys.argv[1])
