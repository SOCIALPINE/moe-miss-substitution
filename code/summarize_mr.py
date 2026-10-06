"""Turn SECTION_MR json lines into a comparison table with the speed model.
speed = t_c / max(t_c + stall/beta, (stall + bg)/beta)  (stall loads block, background loads overlap compute)
t_c = active bytes per token at ~0.56 B/param (4-bit weights incl. scales + higher-precision attention) / 80 GB/s."""
import json, sys

ACTIVE_B = {"qwen3_next": 3.0e9, "gpt_oss": 5.13e9, "deepseek_v4": 13.0e9}
BW_RAM = 80e9
B_PER_PARAM = 0.56


def speed(tc_ms, stall_mb, bg_mb, beta_gbs):
    st = stall_mb / beta_gbs            # MB / (GB/s) = ms
    return tc_ms / max(tc_ms + st, (stall_mb + bg_mb) / beta_gbs)


def rows(sec):
    meta, streams = sec["meta"], sec["streams"]
    tc = ACTIVE_B[meta["adapter"]] * B_PER_PARAM / BW_RAM * 1e3
    out = []
    for name, s in streams.items():
        r = s.get("routing")
        d = {"stream": name, "dppl_all": s["dppl_pct"]["ALL"], "kl": s["kl_nats"], "agree": s["argmax_agree"],
             "top1": s["top1_acc"]}
        for dom in ("python_code", "en_wiki", "chat_en"):
            if "kl_dom" in s and dom in s["kl_dom"]:
                d["kl_" + dom] = s["kl_dom"][dom]
                d["agree_" + dom] = s["agree_dom"][dom]
        if r:
            d.update(hit=r["wanted_hit"], subst=r["subst_rate"], stall_mb=r["stall_mb_per_tok"], bg_mb=r["bg_mb_per_tok"],
                     v7=speed(tc, r["stall_mb_per_tok"], r["bg_mb_per_tok"], 7.0),
                     v12=speed(tc, r["stall_mb_per_tok"], r["bg_mb_per_tok"], 12.0))
        out.append(d)
    return tc, out


if __name__ == "__main__":
    for line in open(sys.argv[1]):
        if not line.startswith("SECTION_MR "):
            continue
        sec = json.loads(line[len("SECTION_MR "):])
        tc, rs = rows(sec)
        print(f"== {sec['meta']['adapter']}  t_c={tc:.1f} ms/token  expert={sec['meta']['expert_mb_4bit']:.1f} MB")
        for d in rs:
            print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()))
