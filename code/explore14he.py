"""Part 20 addendum, EXPLORATORY (not pre-registered): look inside the regenerated HumanEval responses.
  (1) fences: how many responses have more than one fenced block, a non-python tag, the defect pattern (a closed block with a
      tag outside the original regex's set that comes before the chosen block), and whether the extracted code differs
      between the original rule and rules A/B (program level, not only outcome level)
  (2) failures: every record that fails under the original rule, run again with stderr captured and classified
      (truncated = no end-of-generation token; exception type of the last traceback line; where the failing line is:
      the official test's check() or the model's own code)
Usage: python3 explore14he.py --texts results/xc14he_texts.json [--out results/xc14he_explore.json]"""
import argparse, json, os, re, subprocess, sys, tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_tasks as st
import score_fix as sf

LABELS = ["E", "Q", "S", "T", "C", "H"]
ORIG_TAGS = {"python", "py", "Python", ""}


def run_err(src):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prog.py")
        open(p, "w").write(src)
        try:
            r = subprocess.run([sys.executable, p], cwd=d, capture_output=True, timeout=10)
            return r.returncode, r.stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            return "timeout", ""


def classify(rc, err, src):
    if rc == 0:
        return "pass", "", None
    if rc == "timeout":
        return "timeout", "", None
    lines = [l for l in err.strip().splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    m = re.match(r"^(\w+(?:Error|Exception|Exit|Interrupt)?)\b", last)
    kind = m.group(1) if m else (last[:40] or "nonzero")
    # line number of the innermost frame inside prog.py
    lns = [int(x) for x in re.findall(r'File ".*?prog\.py", line (\d+)', err)]
    where = None
    if lns:
        src_lines = src.split("\n")
        check_start = next((k + 1 for k, l in enumerate(src_lines) if l.startswith("def check(")), None)
        final_call = len(src_lines) - 1  # "check(entry)" line
        ln = lns[-1]
        if check_start and ln >= check_start:
            where = "official test" if ln < final_call else "check call"
        else:
            where = "model code"
        # an assertion raised inside check() whose frame is check() itself
    return kind, last[:200], where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    doc = json.load(open(a.texts))
    he = {int(k): v for k, v in doc["humaneval"].items()}
    recs = [r for r in doc["records"] if r["lab"] in LABELS]
    assert len(recs) == 984, len(recs)

    # (1) fences
    fstat = Counter()
    pattern = []
    prog_diff = Counter()
    for r in recs:
        t, it = r["text"], he[r["i"]]
        bl = sf.fence_blocks(t)
        closed = [b for b in bl if b["closed"]]
        fstat["records"] += 1
        fstat[f"closed_blocks={min(len(closed), 3)}{'+' if len(closed) >= 3 else ''}"] += 1
        if any(b["tag"] not in ORIG_TAGS for b in closed):
            fstat["has_non_python_tag"] += 1
        if any(not b["closed"] for b in bl):
            fstat["has_unclosed_block"] += 1
        tags = [b["tag"] for b in closed]
        # defect pattern: a closed block whose tag the original regex does not accept, followed later by any closed block
        k_bad = [k for k, b in enumerate(closed) if b["tag"] not in ORIG_TAGS]
        if k_bad and k_bad[0] < len(closed) - 1:
            fstat["defect_pattern"] += 1
            pattern.append({"lab": r["lab"], "i": r["i"], "tags": tags})
        po = st.he_program(t, it)
        for rule in ("A", "B"):
            if sf.RULES[rule](t, it) != po:
                prog_diff[rule] += 1
        if not st.CODE.findall(t):
            fstat["orig_regex_no_block"] += 1
    print("FENCES", json.dumps(dict(sorted(fstat.items()))))
    print("PROGRAM_DIFF vs orig", json.dumps(dict(prog_diff)))
    print("DEFECT_PATTERN", json.dumps(pattern[:20]))

    # (2) failures under the original rule
    jobs = [(r, st.he_program(r["text"], he[r["i"]])) for r in recs]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: run_err(j[1]), jobs))
    fails = []
    tab = defaultdict(Counter)
    for (r, src), (rc, err) in zip(jobs, res):
        kind, last, where = classify(rc, err, src)
        if kind == "pass":
            continue
        trunc = int(not r["eog"])
        cat = "truncated" if trunc else kind
        tab[r["lab"]][cat] += 1
        tab[r["lab"]]["fail"] += 1
        fails.append({"lab": r["lab"], "i": r["i"], "task_id": he[r["i"]]["id"], "eog": r["eog"], "n": r["n"], "kind": kind,
                      "where": where, "last": last})
    for lab in LABELS:
        print("FAILS", lab, json.dumps(dict(tab[lab])))
    by_problem = defaultdict(list)
    for f in fails:
        by_problem[(f["i"], f["task_id"])].append(f["lab"])
    print("FAIL_PROBLEMS", len(by_problem))
    for (i, tid), labs in sorted(by_problem.items()):
        kinds = {f["lab"]: (("T:" if not f["eog"] else "") + f["kind"] + ("@" + f["where"] if f["where"] else "")) for f in fails if f["i"] == i}
        print("FP", i, tid, "".join(l if l in labs else "." for l in LABELS), json.dumps(kinds))
    if a.out:
        json.dump({"fences": fstat, "program_diff": prog_diff, "defect_pattern": pattern, "fails": fails}, open(a.out, "w"),
                  indent=1)


if __name__ == "__main__":
    main()
