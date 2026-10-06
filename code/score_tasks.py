"""Round 14 (Part 20): score the greedy generations of xcache-bench --gen (see PREREG_xc14.md).

Reads TASKS/tasks.json, the tokenizer, and RES/gen_<a>-<b>_<label>.jsonl. Labels: E (exact, reference), Q, S, T, C, H, and the
repeats Qr, Sr. Lines:
  G14  per label and task: accuracy, length, truncation, substituted mass, ms/token, agreement with E
  P14  per label and task: accuracy difference to E (paired bootstrap over problems, McNemar exact p)
  H14  the pre-registered checks H1-H6
  R14  repeats against their first run
  I14  per-item arrays;  X14  a few example texts
  J14  per-item records for pooling (after the analysis), one line per label, task and block of items, md5 at the end.
       Pooled runs leave out ms/token (it differs by pod) and the GSM8K answers (each pod prints them in its own I14).
--brief prints G14 and P14 only (progress during the run).
--from-logs F1 F2 ... runs the same analysis on the J14 records of several runs (the problems were split across pods; every
problem ran all its settings on one pod, so each record already holds its comparison with its own reference)."""
import argparse, glob, hashlib, json, math, os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np

LABELS = ["E", "Q", "S", "T", "C", "H"]
NONREF = ["Q", "S", "T", "C", "H"]
REPS = {"Qr": "Q", "Sr": "S"}
NB = 2000
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
CODE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)
JBLOCK = 40


def emit(tag, **kw):
    print(tag + " " + json.dumps(kw, separators=(",", ":"), ensure_ascii=False), flush=True)


def load(res):
    out = {}
    for f in sorted(glob.glob(os.path.join(res, "gen_*_*.jsonl"))):
        lab = os.path.basename(f)[:-len(".jsonl")].rsplit("_", 1)[1]
        for line in open(f):
            line = line.strip()
            if line:
                d = json.loads(line)
                out.setdefault(lab, {})[d["i"]] = d
    return out


def last_boxed(t):
    k = t.rfind("\\boxed{")
    if k < 0:
        return None
    i = j = k + len("\\boxed{")
    depth = 1
    while j < len(t) and depth:
        depth += {"{": 1, "}": -1}.get(t[j], 0)
        j += 1
    return t[i:j - 1] if depth == 0 else t[i:]


def gsm_answer(t):
    b = last_boxed(t)
    nums = NUM.findall(b if b is not None else t)
    if not nums:
        return None
    s = (nums[0] if b is not None else nums[-1]).replace(",", "")
    try:
        x = float(s)
    except ValueError:
        return None
    return int(round(x)) if abs(x - round(x)) < 1e-9 else round(x, 6)


def he_program(text, it):
    blocks = CODE.findall(text)
    if blocks:
        code = blocks[-1]
    else:
        k = text.rfind("```python")
        code = text[k + len("```python"):] if k >= 0 else text
    if f"def {it['entry_point']}" not in code:
        code = it["prompt"] + code
    imports = "\n".join(l for l in it["prompt"].splitlines() if l.startswith(("import ", "from ")))
    return imports + "\n" + code + "\n\n" + it["test"] + f"\n\ncheck({it['entry_point']})\n"


def run_prog(src):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prog.py")
        with open(p, "w") as f:
            f.write(src)
        try:
            r = subprocess.run([sys.executable, p], cwd=d, capture_output=True, timeout=10)
            return int(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0


def ranks(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def logit_fit(x, y, it=50):
    X = np.stack([np.ones_like(x), x], 1)
    w = np.zeros(2)
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (y - p)
        h = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return w


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def build(items, tok, gen):
    """Decode and score every generation; compare each with its reference (E, or Q/S for the repeats) on the same run."""
    rec = {}
    jobs = {}
    for lab, d in gen.items():
        for i, g in d.items():
            it = items[i]
            text = tok.decode(g["toks"], skip_special_tokens=True)
            r = {"text": text, "n": g["n"], "eog": g["eog"], "mass": g["mass"] / max(1, g["ltok"]), "ms": g["ms"],
                 "ndec": max(0, g["n"] - 1), "mass_sum": g["mass"], "ltok": g["ltok"], "toks": g["toks"]}
            if it["task"] == "gsm8k":
                r["ans"] = gsm_answer(text)
                try:
                    r["ok"] = int(r["ans"] is not None and abs(float(r["ans"]) - float(it["gold"])) < 1e-6)
                except ValueError:
                    r["ok"] = 0
            else:
                src = he_program(text, it)
                jobs.setdefault(src, []).append((lab, i))
            rec[(lab, i)] = r
    srcs = list(jobs)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(run_prog, srcs))
    for src, ok in zip(srcs, res):
        for key in jobs[src]:
            rec[key]["ok"] = ok
            rec[key]["ans"] = ok
    for (lab, i), r in rec.items():
        ref = REPS.get(lab, "E")
        if lab == "E" or (ref, i) not in rec:
            continue
        a, b = r["toks"], rec[(ref, i)]["toks"]
        r["same"] = int(r["ans"] == rec[(ref, i)]["ans"])
        r["ident"] = int(a == b)
        r["fdiv"] = None if a == b else next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return rec


def analyze(rec, items, brief):
    rng = np.random.default_rng(14)
    tasks = ["gsm8k", "humaneval"]
    summ = {}
    for t in tasks:
        ids_t = [i for i, m in enumerate(items) if m["task"] == t]
        for lab in LABELS + list(REPS):
            have = [i for i in ids_t if (lab, i) in rec]
            if not have:
                continue
            rr = [rec[(lab, i)] for i in have]
            ok = np.array([r["ok"] for r in rr], float)
            nd = sum(r["ndec"] for r in rr)
            g = {"label": lab, "task": t, "n_items": len(have), "acc": round(float(ok.mean()), 4),
                 "mean_tokens": round(float(np.mean([r["n"] for r in rr])), 1),
                 "trunc": round(float(np.mean([1 - r["eog"] for r in rr])), 4),
                 "mass": round(sum(r["mass_sum"] for r in rr) / max(1, sum(r["ltok"] for r in rr)), 5),
                 "mass_item_mean": round(float(np.mean([r["mass"] for r in rr])), 5),
                 "ms_tok": round(sum(r["ms"] for r in rr) / max(1, nd), 3) if all(r["ms"] is not None for r in rr) else None}
            ref = "E" if lab not in REPS else REPS[lab]
            both = [i for i in have if (ref, i) in rec]
            if lab != "E" and both:
                same = np.array([rec[(lab, i)]["same"] for i in both], float)
                ident = np.array([rec[(lab, i)]["ident"] for i in both], float)
                div = [rec[(lab, i)]["fdiv"] for i in both if not rec[(lab, i)]["ident"]]
                g.update({"ref": ref, "n_paired": len(both), "same_final": round(float(same.mean()), 4),
                          "identical": round(float(ident.mean()), 4),
                          "first_div_median": float(np.median(div)) if div else None})
            emit("G14", **g)
            summ[(lab, t)] = g
            if lab in REPS or lab == "E" or not both:
                if lab in REPS and both:
                    x = np.array([rec[(lab, i)]["ok"] for i in both], float)
                    y = np.array([rec[(ref, i)]["ok"] for i in both], float)
                    emit("R14", label=lab, ref=ref, task=t, n=len(both), dacc_pp=round(100 * float((x - y).mean()), 2),
                         same_final=g["same_final"], identical=g["identical"])
                continue
            x = np.array([rec[(lab, i)]["ok"] for i in both], float)
            y = np.array([rec[("E", i)]["ok"] for i in both], float)
            d = x - y
            pick = rng.integers(0, len(d), size=(NB, len(d)))
            bs = d[pick].mean(1)
            b, c = int(((y == 1) & (x == 0)).sum()), int(((y == 0) & (x == 1)).sum())
            emit("P14", label=lab, task=t, n=len(both), acc=round(float(x.mean()), 4), acc_E=round(float(y.mean()), 4),
                 dacc_pp=[round(100 * float(d.mean()), 2), round(100 * float(np.percentile(bs, 2.5)), 2),
                          round(100 * float(np.percentile(bs, 97.5)), 2)],
                 lost=b, gained=c, mcnemar_p=round(mcnemar(b, c), 4))
            summ[(lab, t, "dacc")] = [100 * float(d.mean()), 100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))]
    if brief:
        return

    # pre-registered checks
    for t in tasks:
        labs = [l for l in NONREF if (l, t) in summ and "same_final" in summ[(l, t)]]
        if len(labs) < 2:
            continue
        m = [summ[(l, t)]["mass"] for l in labs]
        order = [l for _, l in sorted(zip(m, labs))]
        emit("H14", check="H1", task=t, mass={l: summ[(l, t)]["mass"] for l in labs}, order=order,
             holds=order == [l for l in NONREF if l in labs])
        emit("H14", check="H2", task=t, spearman_mass_identical=round(spearman(m, [summ[(l, t)]["identical"] for l in labs]), 3),
             spearman_mass_same_final=round(spearman(m, [summ[(l, t)]["same_final"] for l in labs]), 3), threshold=-0.9)
        dd = [summ[(l, t, "dacc")][0] for l in labs]
        sl = float(np.polyfit(m, dd, 1)[0]) if len(labs) > 2 else float("nan")
        emit("H14", check="H6", task=t, slope_pp_per_0p01_mass=round(sl * 0.01, 3), points={l: [round(x, 5), round(y, 2)] for l, x, y in zip(labs, m, dd)})
        emit("H14", check="H4", task=t, dacc_pp={l: round(summ[(l, t, "dacc")][0], 2) for l in labs},
             holds_QSTC=all(-1.5 <= summ[(l, t, "dacc")][0] <= 0.5 for l in labs if l in "QSTC"),
             holds_H=(summ[("H", t, "dacc")][0] <= -2.0) if ("H", t, "dacc") in summ else None)
        emit("H14", check="H5", task=t, margin_pp=-2.0,
             lower={l: round(summ[(l, t, "dacc")][1], 2) for l in ("Q", "S") if (l, t, "dacc") in summ},
             noninferior={l: summ[(l, t, "dacc")][1] >= -2.0 for l in ("Q", "S") if (l, t, "dacc") in summ})
    # H3: per-problem logistic regression, rows = (label, problem) for the non-reference labels, y = final answer differs from E
    for scope in tasks + ["both"]:
        ids = [i for i, m in enumerate(items) if (scope == "both" or m["task"] == scope) and ("E", i) in rec]
        rows = [(i, rec[(l, i)]["mass"], float(1 - rec[(l, i)]["same"])) for i in ids for l in NONREF if (l, i) in rec]
        if len(rows) < 10:
            continue
        pid = np.array([r[0] for r in rows])
        x = np.array([r[1] for r in rows]) * 10.0  # per 0.1 mass
        y = np.array([r[2] for r in rows])
        w = logit_fit(x, y)
        uniq = np.unique(pid)
        pos = {p: np.where(pid == p)[0] for p in uniq}
        bs = []
        for _ in range(NB):
            sel = np.concatenate([pos[p] for p in rng.choice(uniq, size=len(uniq))])
            bs.append(logit_fit(x[sel], y[sel])[1])
        emit("H14", check="H3", scope=scope, rows=len(rows), problems=len(uniq), slope_per_0p1_mass=round(float(w[1]), 3),
             ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
             differs_rate=round(float(y.mean()), 4))
    # per-item arrays, then a few examples
    for t in tasks:
        ids_t = [i for i, m in enumerate(items) if m["task"] == t]
        for lab in LABELS + list(REPS):
            have = [i for i in ids_t if (lab, i) in rec]
            if not have:
                continue
            ref = "E" if lab not in REPS else REPS[lab]
            emit("I14", label=lab, task=t, i=have, ok=[rec[(lab, i)]["ok"] for i in have],
                 same=[rec[(lab, i)]["same"] if (ref, i) in rec and lab != "E" else -1 for i in have],
                 ident=[rec[(lab, i)]["ident"] if (ref, i) in rec and lab != "E" else -1 for i in have],
                 n=[rec[(lab, i)]["n"] for i in have], eog=[rec[(lab, i)]["eog"] for i in have],
                 mass_e4=[int(round(1e4 * rec[(lab, i)]["mass"])) for i in have],
                 ans=[rec[(lab, i)].get("ans") for i in have] if t == "gsm8k" else None)
    shown = 0
    for i, m in enumerate(items):
        if m["task"] != "gsm8k" or shown >= 3:
            continue
        if ("E", i) in rec and ("S", i) in rec and "text" in rec[("E", i)] and rec[("E", i)]["ok"] and not rec[("S", i)]["ok"]:
            emit("X14", i=i, gold=m["gold"], E=rec[("E", i)]["text"][-500:], S=rec[("S", i)]["text"][-500:],
                 S_ans=rec[("S", i)]["ans"], S_mass=round(rec[("S", i)]["mass"], 4))
            shown += 1
    for i, m in enumerate(items):
        if m["task"] == "humaneval" and ("E", i) in rec and "text" in rec[("E", i)]:
            emit("X14", i=i, task="humaneval", E=rec[("E", i)]["text"][-600:], E_ok=rec[("E", i)]["ok"])
            break


def dump(rec, items):
    """J14 lines: per label, task and block of consecutive problems, everything analyze() needs, then the md5 of the json."""
    for t in ["gsm8k", "humaneval"]:
        ids_t = [i for i, m in enumerate(items) if m["task"] == t]
        for lab in LABELS + list(REPS):
            have = [i for i in ids_t if (lab, i) in rec]
            runs, cur = [], []
            for i in have:  # blocks of consecutive problems, at most JBLOCK long
                if cur and (i != cur[-1] + 1 or len(cur) == JBLOCK):
                    runs.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                runs.append(cur)
            for blk in runs:
                rr = [rec[(lab, i)] for i in blk]
                cmp_ = lab != "E" and all("same" in r for r in rr)
                d = {"lab": lab, "task": t, "i0": blk[0], "k": len(blk),
                     "ok": "".join(str(r["ok"]) for r in rr), "eog": "".join(str(r["eog"]) for r in rr),
                     "same": "".join(str(r["same"]) for r in rr) if cmp_ else "",
                     "ident": "".join(str(r["ident"]) for r in rr) if cmp_ else "",
                     "n": [r["n"] for r in rr], "msum": [r["mass_sum"] for r in rr],
                     "ltx": {str(k): r["ltok"] for k, r in enumerate(rr) if r["ltok"] != 48 * r["ndec"]},
                     "fdiv": [r["fdiv"] if r.get("fdiv") is not None else -1 for r in rr] if cmp_ else []}
                s = json.dumps(d, separators=(",", ":"))
                print("J14 " + s + " " + hashlib.md5(s.encode()).hexdigest()[:8], flush=True)


def undump(files, items):
    rec, bad = {}, 0
    for f in files:
        for line in open(f):
            k = line.find("J14 {")
            if k < 0:
                continue
            s, h = line[k + 4:].rstrip("\n").rstrip().rsplit(" ", 1)
            if hashlib.md5(s.encode()).hexdigest()[:8] != h:
                print(f"J14 MD5 MISMATCH {f} {s[:60]}", file=sys.stderr)
                bad += 1
                continue
            d = json.loads(s)
            for k2 in range(d["k"]):
                i = d["i0"] + k2
                assert items[i]["task"] == d["task"], (i, d["task"])
                key = (d["lab"], i)
                n = d["n"][k2]
                r = {"n": n, "eog": int(d["eog"][k2]), "ok": int(d["ok"][k2]), "ms": None, "ndec": max(0, n - 1),
                     "mass_sum": d["msum"][k2], "ltok": d["ltx"].get(str(k2), 48 * max(0, n - 1))}
                r["mass"] = r["mass_sum"] / max(1, r["ltok"])
                r["ans"] = None if d["task"] == "gsm8k" else r["ok"]  # GSM8K answers stay in each pod's own I14 lines
                if d["same"]:
                    r["same"], r["ident"] = int(d["same"][k2]), int(d["ident"][k2])
                    r["fdiv"] = None if d["fdiv"][k2] < 0 else d["fdiv"][k2]
                if key in rec:
                    raise SystemExit(f"duplicate record {key} in {f}")
                rec[key] = r
    return rec, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tok")
    ap.add_argument("--res")
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--from-logs", nargs="+")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.tasks, "tasks.json")))
    items = meta["items"]
    if a.from_logs:
        rec, bad = undump(a.from_logs, items)
        cov = {}
        for (lab, i) in rec:
            cov[(lab, items[i]["task"])] = cov.get((lab, items[i]["task"]), 0) + 1
        emit("C14", records=len(rec), md5_mismatch=bad, coverage={f"{l}/{t}": n for (l, t), n in sorted(cov.items())})
        if bad:
            raise SystemExit("J14 lines with a wrong md5; fix the copies first")
        analyze(rec, items, False)
        print("SCORE14 POOLED DONE", flush=True)
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    rec = build(items, tok, load(a.res))
    analyze(rec, items, a.brief)
    if a.brief:
        return
    dump(rec, items)
    print("SCORE14 DONE", flush=True)


if __name__ == "__main__":
    main()
