"""Part 20 addendum (PREREG_xc14_addendum.md): score the HumanEval generations with the original code-block extraction and
with the corrected rules A and B, list every problem whose outcome depends on the rule, and rerun the pre-registered
analysis under rules A and B. Output tags carry the rule letter: G14A/P14A/H14A/I14A/X14A (rule A), ...B (rule B).
  FIXOK   per rule and label: the ok bits of all HumanEval problems in order (rule "orig" is score_tasks.py unchanged)
  FIXDIFF one line per (label, problem) whose ok bit differs between any two rules, with the fence count of the response
Only the code-block choice changes; everything after it (prompt prepended when the function is missing, prompt imports,
official test within 10 s) is the original."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_tasks as st

TAGS_A = {"python", "py", "Python", ""}


def accept_A(tag):
    return tag in TAGS_A


def accept_B(tag):
    return tag.lower() in {"python", "py", "python3"}


def fence_blocks(text):
    """Line-based fences. A fence line starts, after leading whitespace, with three or more backticks. Outside a block it
    opens one (the rest of the line, stripped, is the language tag); inside a block, a line of only backticks and
    whitespace closes it, and every other line is content."""
    out, cur = [], None
    for line in text.split("\n"):
        s = line.lstrip()
        if cur is None:
            if s.startswith("```"):
                n = len(s) - len(s.lstrip("`"))
                cur = {"tag": s[n:].strip(), "lines": [], "closed": False}
        elif s.startswith("```") and s.strip().strip("`") == "":
            cur["closed"] = True
            out.append(cur)
            cur = None
        else:
            cur["lines"].append(line)
    if cur is not None:
        out.append(cur)
    return out


def pick(text, accept):
    closed = [b for b in fence_blocks(text) if b["closed"] and accept(b["tag"])]
    if closed:
        return "".join(l + "\n" for l in closed[-1]["lines"])
    k = text.rfind("```python")  # unchanged fallback of the original rule
    return text[k + len("```python"):] if k >= 0 else text


def program(code, it):  # unchanged tail of score_tasks.he_program
    if f"def {it['entry_point']}" not in code:
        code = it["prompt"] + code
    imports = "\n".join(l for l in it["prompt"].splitlines() if l.startswith(("import ", "from ")))
    return imports + "\n" + code + "\n\n" + it["test"] + f"\n\ncheck({it['entry_point']})\n"


RULES = {"A": lambda text, it: program(pick(text, accept_A), it),
         "B": lambda text, it: program(pick(text, accept_B), it)}


def main(tok=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tok")
    ap.add_argument("--res", required=True)
    a = ap.parse_args()
    items = json.load(open(os.path.join(a.tasks, "tasks.json")))["items"]
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tok)
    gen = st.load(a.res)
    cache, run0 = {}, st.run_prog

    def run_cached(src):  # identical programs under different rules run once
        if src not in cache:
            cache[src] = run0(src)
        return cache[src]

    st.run_prog = run_cached
    he = [i for i, m in enumerate(items) if m["task"] == "humaneval"]
    labs = [l for l in st.LABELS if l in gen]
    recs = {"orig": st.build(items, tok, gen)}
    he_orig = st.he_program
    for r, fn in RULES.items():
        st.he_program = fn
        recs[r] = st.build(items, tok, gen)
    st.he_program = he_orig
    for r in ("orig", "A", "B"):
        for lab in labs:
            have = [i for i in he if (lab, i) in recs[r]]
            st.emit("FIXOK", rule=r, label=lab, i0=have[0] if have else None, n=len(have),
                    ok="".join(str(recs[r][(lab, i)]["ok"]) for i in have))
    ndiff = 0
    for lab in labs:
        for i in he:
            if (lab, i) not in recs["orig"]:
                continue
            oks = {r: recs[r][(lab, i)]["ok"] for r in ("orig", "A", "B")}
            if len(set(oks.values())) > 1:
                text = recs["orig"][(lab, i)]["text"]
                st.emit("FIXDIFF", label=lab, i=i, ok=oks, fences=sum(1 for l in text.split("\n") if l.lstrip().startswith("```")))
                ndiff += 1
    st.emit("FIXSUM", rule_dependent=ndiff, programs_run=len(cache))
    emit0 = st.emit
    for r in ("A", "B"):
        st.emit = lambda tag, _r=r, **kw: emit0(tag + _r, **kw)
        st.analyze(recs[r], items, False)
    st.emit = emit0
    print("SCOREFIX DONE", flush=True)


if __name__ == "__main__":
    main()
