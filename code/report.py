import json, sys, copy


def fmt(v):
    return f"{v*100:5.1f}" if isinstance(v, float) else str(v)


r = json.load(open(sys.argv[1]))
ts = r["trace_stats"]
m = r["meta"]
print("=" * 80)
print(f"sessions={m['S']} x {m['T']} tok | layers={m['NL']} experts={m['E']} top{m['K']} | prompt={m['prompt_len']} | per-domain={m['sessions_per_domain']}")
print("ppl per domain (sanity):", {k: round(v, 2) for k, v in ts["ppl_per_domain"].items()})
print(f"trace seconds={ts['seconds']:.0f} (waiting for weights {ts['seconds_waiting_for_weights']:.0f})")
print("\n[lookahead gating] recall (%) of real top-k by applying future router to current residual")
print("  D(half-layers)  m=" + "  ".join(str(x) for x in ts["lookahead"]["0"]["m_list"]))
for D, v in ts["lookahead"].items():
    print(f"  D={D:>2}            " + "  ".join(fmt(x) for x in v["recall_mean"]))
fr = list(r["cache"].keys())
print("\n[cache hit rate %, tokens after prompt, ALL domains]  capacity = fraction of all experts")
print("  policy               " + "  ".join(f"{float(f)*100:5.1f}%" for f in fr))
for pol in r["cache"][fr[0]].keys():
    print(f"  {pol:20s} " + "  ".join(" " + fmt(r["cache"][f][pol]["after_prompt"]["ALL"]) for f in fr))
for f in ("0.125", "0.25"):
    if f not in r["cache"]:
        continue
    print(f"\n[per-domain hit rate % at {float(f)*100:.1f}% capacity]")
    doms = [d for d in r["cache"][f]["OPT_g"]["after_prompt"] if d != "ALL"]
    print("  policy               " + " ".join(f"{d[:10]:>10s}" for d in doms))
    for pol in r["cache"][f]:
        print(f"  {pol:20s} " + " ".join(f"{fmt(r['cache'][f][pol]['after_prompt'].get(d, float('nan'))):>10s}" for d in doms))
print("\n[miss structure]")
for k, v in r["miss_structure"].items():
    print(f"  {k:12s} miss={fmt(v['miss_rate'])}%  gate-mass-on-miss={fmt(v['gate_weight_mass_on_misses'])}%  miss by rank(1..K)={[round(x*100,1) for x in v['miss_rate_by_weight_rank']]}")
loc = r["locality"]
print("\n[temporal locality]")
print(f"  reuse next token (same layer): actual={fmt(loc['reuse_next_token']['ALL'])}%  shuffled={fmt(loc['reuse_next_token_shuffled']['ALL'])}%  uniform={fmt(loc['reuse_uniform_baseline'])}%")
print("  reuse by domain:", {k: round(v * 100, 1) for k, v in loc["reuse_next_token"].items()})
for w, v in loc["distinct_experts_per_layer_in_window"].items():
    print(f"  distinct experts/layer in {w:>4} tokens: actual={v['actual']['ALL']:.1f}  shuffled={v['shuffled_tokens']['ALL']:.1f}  uniform={v['uniform_random']:.1f}")
print("\n[concentration: share of traffic carried by top-f experts (in-sample)]")
for f, v in r["concentration"].items():
    print(f"  f={f}: uniform={fmt(v['uniform'])} global={fmt(v['global_pool_all_sessions'])} session={fmt(v['per_session']['ALL'])} 256-window={fmt(v['per_256_token_window'])}")
    print(f"         per-domain pool: " + ", ".join(f"{k}={fmt(x)}" for k, x in v["per_domain_pool"].items()))
mp = r["maps"]
print(f"\n[map predictors] recall % at m={mp['m_list']} (held-out sessions)")
for k, v in mp["recall"].items():
    print(f"  {k:24s} " + "  ".join(fmt(x) for x in v))
print("  cross-entropy bits:", {k: round(v, 3) for k, v in mp["cross_entropy_bits"].items()}, "max", mp["max_entropy_bits"])
print("=" * 80)
slim = copy.deepcopy(r)
for f in slim["cache"].values():
    for p in f.values():
        p.pop("per_layer_after_prompt", None)
for v in slim["trace_stats"]["lookahead"].values():
    v["recall_per_layer"] = [round(x[0], 3) for x in v["recall_per_layer"]]
print("RESULTS_JSON_BEGIN")
print(json.dumps(slim, separators=(",", ":"), default=lambda o: round(o, 4)))
print("RESULTS_JSON_END")
