"""Round 14 (Part 20): take one pod's log as saved from stream-pod-logs, keep it under results/, and check that its J14 lines
are complete and intact before the pod is deleted (a deleted pod's log cannot be read again).

Usage: python3 collect14.py --pod C --src <saved tool-results file or plain log> [...]
Checks: every J14 line's md5, no (label, item) twice, the (label, task) item counts and item ranges this pod was assigned,
"SCORE14 DONE" and "=== ALL DONE" present. Prints SAFE_TO_DELETE yes/no."""
import argparse, collections, hashlib, json, os, sys

RANGES = {  # pod -> label -> list of (task, first item, last item + 1)
    "A2": {**{l: [("gsm8k", 0, 200)] for l in "EQSTCH"}, "Qr": [("gsm8k", 0, 100)], "Sr": [("gsm8k", 0, 100)]},
    "B": {l: [("gsm8k", 200, 400)] for l in "EQSTCH"},
    "C": {l: [("gsm8k", 400, 500), ("humaneval", 500, 664)] for l in "EQSTCH"},
}


def read_lines(path):
    txt = open(path, encoding="utf-8").read()
    try:
        d = json.loads(txt)
        return [(x.get("ts", ""), x["line"]) for x in d["items"]], bool(d.get("truncated"))
    except (ValueError, KeyError, TypeError):
        return [("", l) for l in txt.splitlines()], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True, choices=sorted(RANGES))
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    seen, rows, trunc = set(), [], False
    for p in a.src:
        ls, t = read_lines(p)
        trunc |= t
        for r in ls:
            if r not in seen or r[0] == "":
                seen.add(r)
                rows.append(r)
    if all(ts for ts, _ in rows):
        rows.sort(key=lambda r: r[0])
    lines = [l for _, l in rows]
    out = os.path.join(a.out, f"xc14_{a.pod}.log")
    with open(out, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    bad, recs, dup = 0, collections.Counter(), 0
    have = collections.defaultdict(set)
    for l in lines:
        k = l.find("J14 {")
        if k < 0:
            continue
        s, h = l[k + 4:].rstrip().rsplit(" ", 1)
        if hashlib.md5(s.encode()).hexdigest()[:8] != h:
            bad += 1
            continue
        d = json.loads(s)
        for j in range(d["k"]):
            i = d["i0"] + j
            if i in have[(d["lab"], d["task"])]:
                dup += 1
            have[(d["lab"], d["task"])].add(i)
    want = {(lab, t): set(range(i0, i1)) for lab, v in RANGES[a.pod].items() for t, i0, i1 in v}
    miss = {f"{lab}/{t}": len(want[(lab, t)] - have.get((lab, t), set())) for (lab, t) in want}
    extra = {f"{lab}/{t}": len(s - want.get((lab, t), set())) for (lab, t), s in have.items() if s - want.get((lab, t), set())}
    done = any(l.startswith("=== ALL DONE") for l in lines)
    scored = any(l.startswith("SCORE14 DONE") for l in lines)
    failed = [l for l in lines if l.startswith(("GEN FAILED", "ENGINE MISSING", "Traceback"))]
    ok = bad == 0 and dup == 0 and not extra and all(v == 0 for v in miss.values()) and done and scored and not trunc
    print(json.dumps({"pod": a.pod, "saved": out, "lines": len(lines), "j14_lines": sum("J14 {" in l for l in lines),
                      "md5_bad": bad, "dup_items": dup, "missing": {k: v for k, v in miss.items() if v}, "extra": extra,
                      "items": {f"{lab}/{t}": len(s) for (lab, t), s in sorted(have.items())},
                      "all_done": done, "score14_done": scored, "truncated": trunc, "fail_lines": failed[:5]},
                     ensure_ascii=False))
    print("SAFE_TO_DELETE", "yes" if ok else "no")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
