"""Part 20 addendum, local side: (1) compare the regenerated HumanEval records with the original pooled I14 arrays
(PREREG_xc14_addendum.md, reproduction criterion), (2) rescore the regenerated texts locally with the original rule and rules
A and B (independent of the pod's scoring), (3) rerun the pooled pre-registered analysis (score_tasks.analyze, same order as
the original pooled run) with GSM8K from the original records and HumanEval under each rule.

Usage: python3 verify14he.py --orig-i14 i14.json --pod-log LOG [LOG ...] --texts d14t.json --out DIR"""
import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_tasks as st
import score_fix as sf

KEYS = ("i", "ok", "same", "ident", "n", "eog", "mass_e4")


def log_lines(paths):
    out = []
    for p in paths:
        txt = open(p, encoding="utf-8").read()
        try:
            out += [x["line"] for x in json.loads(txt)["items"]]
        except (ValueError, KeyError, TypeError):
            out += txt.splitlines()
    return out


def tagged(lines, tag):
    res = []
    for l in lines:
        m = re.search(r"(?:^|\s)" + re.escape(tag) + r" (\{.*\})\s*$", l)
        if m:
            res.append(json.loads(m.group(1)))
    return res


def records(i14, task, ok_override=None):
    """score_tasks-style records from I14 arrays; ok_override[(lab, i)] replaces ok (and same is recomputed vs E)."""
    rec = {}
    for (lab, t), d in i14.items():
        if t != task:
            continue
        for k, i in enumerate(d["i"]):
            n = d["n"][k]
            ltok = 48 * max(0, n - 1)
            mass = d["mass_e4"][k] / 1e4
            ok = d["ok"][k] if ok_override is None else ok_override.get((lab, i), d["ok"][k])
            rec[(lab, i)] = {"n": n, "eog": d["eog"][k], "ok": ok, "ms": None, "ndec": max(0, n - 1), "ltok": ltok,
                             "mass": mass, "mass_sum": mass * ltok, "ans": ok if task == "humaneval" else None}
            if lab != "E" and d["same"][k] >= 0:
                rec[(lab, i)].update(same=d["same"][k], ident=d["ident"][k], fdiv=-1)  # first divergence not in I14
    if ok_override is not None:  # same = final outcome equal to the reference's under this rule
        for (lab, i), r in rec.items():
            if "same" in r:
                ref = st.REPS.get(lab, "E")
                r["same"] = int(r["ok"] == rec[(ref, i)]["ok"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-i14", required=True)
    ap.add_argument("--pod-log", nargs="+", required=True)
    ap.add_argument("--texts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    orig = {tuple(k.split("|")): v for k, v in json.load(open(a.orig_i14)).items()}
    lines = log_lines(a.pod_log)

    # (1) reproduction: the pod's original-rule I14 lines (plain tag, not I14A/I14B) against the original pooled arrays
    pod = {(d["label"], d["task"]): d for d in tagged(lines, "I14") if d["task"] == "humaneval"}
    mism, total = {}, 0
    for lab in st.LABELS:
        o, p = orig[(lab, "humaneval")], pod.get((lab, "humaneval"))
        if p is None:
            print("REPRO missing pod I14 for", lab); mism[lab] = "missing"; continue
        bad = []
        for k, i in enumerate(o["i"]):
            total += 1
            kp = p["i"].index(i) if i in p["i"] else None
            diff = [key for key in KEYS[1:] if kp is None or o[key][k] != p[key][kp]]
            if diff:
                bad.append((i, diff))
        mism[lab] = bad
        print(f"REPRO {lab}: {len(o['i'])} problems, mismatched {len(bad)}" + (f" e.g. {bad[:5]}" if bad else ""))
    nbad = sum(len(v) for v in mism.values() if isinstance(v, list)) + sum(1 for v in mism.values() if v == "missing") * 164
    print(f"REPRO TOTAL {total} records, mismatched {nbad} ->", "REPRODUCED" if nbad == 0 else "NOT REPRODUCED")

    # pod-side rule results
    fixok = {(d["rule"], d["label"]): d for d in tagged(lines, "FIXOK")}
    he_ids = orig[("E", "humaneval")]["i"]
    pod_ok = {r: {(lab, he_ids[0] + k): int(c) for lab in st.LABELS for k, c in enumerate(fixok[(r, lab)]["ok"])}
              for r in ("orig", "A", "B") if all((r, lab) in fixok for lab in st.LABELS)}

    # (2) local rescoring from the dumped texts
    loc_ok = {}
    if a.texts:
        doc = json.load(open(a.texts))
        he = {int(k): v for k, v in doc["humaneval"].items()}
        recs = [r for r in doc["records"] if r["lab"] in st.LABELS]
        progs = {"orig": st.he_program, "A": sf.RULES["A"], "B": sf.RULES["B"]}
        jobs = {}
        for r in recs:
            for rule, fn in progs.items():
                jobs.setdefault(fn(r["text"], he[r["i"]]), []).append((rule, r["lab"], r["i"]))
        srcs = list(jobs)
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(st.run_prog, srcs))
        for src, ok in zip(srcs, res):
            for rule, lab, i in jobs[src]:
                loc_ok.setdefault(rule, {})[(lab, i)] = ok
        for rule in ("orig", "A", "B"):
            agree = sum(loc_ok[rule][k] == pod_ok[rule][k] for k in loc_ok[rule]) if rule in pod_ok else None
            print(f"LOCAL {rule}: {len(loc_ok[rule])} scored, agrees with pod on {agree}")
        json.dump({r: {f"{k[0]}|{k[1]}": v for k, v in d.items()} for r, d in loc_ok.items()}, open(os.path.join(a.out, "local_ok.json"), "w"))

    # (3) pooled analysis per rule (original order: GSM8K then HumanEval, labels E Q S T C H then Qr Sr)
    items = [{"task": "gsm8k"}] * 500 + [{"task": "humaneval"}] * 164
    use = loc_ok if loc_ok else pod_ok
    for rule in ("orig", "A", "B"):
        if rule not in use:
            continue
        rec = records(orig, "gsm8k")
        rec.update(records(orig, "humaneval", None if rule == "orig" and not loc_ok else use[rule]))
        print(f"=== POOLED rule={rule} ===", flush=True)
        st.analyze(rec, items, False)


if __name__ == "__main__":
    main()
