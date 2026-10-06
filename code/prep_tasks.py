"""Round 14 (Part 20): prompts for greedy generation with xcache-bench --gen.

GSM8K test (first N of a seed-14 permutation) and HumanEval (all 164), Qwen chat template, one user message, no system prompt.
Writes OUT/tasks.bin (int32 N, then int32 len + tokens per prompt) and OUT/tasks.json (per prompt: task, id and what scoring needs).
PREP_FAKE_DIR=dir reads the parquet files from dir/<repo with / as __>/<file> instead of the Hub (local tests)."""
import argparse, json, os, struct
import numpy as np
import pyarrow.parquet as pq

GSM_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
HE_TEMPLATE = ("Complete the following Python function. Reply with the whole function, including any imports it needs, "
               "in one ```python code block.\n\n```python\n{prompt}```")


def fetch(repo, prefix):
    if os.environ.get("PREP_FAKE_DIR"):
        base = os.path.join(os.environ["PREP_FAKE_DIR"], repo.replace("/", "__"))
        files = sorted(os.path.relpath(os.path.join(r, f), base) for r, _, fs in os.walk(base) for f in fs)
        return [os.path.join(base, f) for f in files if f.startswith(prefix) and f.endswith(".parquet")]
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet"))
    return [hf_hub_download(repo, f, repo_type="dataset") for f in files]


def rows(paths):
    out = []
    for p in paths:
        out.extend(pq.read_table(p).to_pylist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gsm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=14)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)

    gsm = rows(fetch("openai/gsm8k", "main/test"))
    he = rows(fetch("openai/openai_humaneval", "openai_humaneval/test"))
    assert len(gsm) > 0 and len(he) > 0, (len(gsm), len(he))
    perm = np.random.default_rng(a.seed).permutation(len(gsm))[:a.gsm]

    items, prompts = [], []

    def add(text, meta):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        prompts.append([int(x) for x in ids])
        items.append(meta)

    for k in perm:
        r = gsm[int(k)]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        add(r["question"].strip() + GSM_SUFFIX, {"task": "gsm8k", "id": int(k), "gold": gold})
    for r in he:
        add(HE_TEMPLATE.format(prompt=r["prompt"]), {"task": "humaneval", "id": r["task_id"], "prompt": r["prompt"],
                                                     "test": r["test"], "entry_point": r["entry_point"]})

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "tasks.bin"), "wb") as f:
        f.write(struct.pack("<i", len(prompts)))
        for p in prompts:
            f.write(struct.pack("<i", len(p)))
            f.write(np.asarray(p, dtype=np.int32).tobytes())
    json.dump({"gsm_seed": a.seed, "n_gsm": int(len(perm)), "n_gsm_all": len(gsm), "items": items},
              open(os.path.join(a.out, "tasks.json"), "w"))
    for t in ("gsm8k", "humaneval"):
        n = [len(p) for p, m in zip(prompts, items) if m["task"] == t]
        print(f"TASKS {t} n={len(n)} prompt tokens min {min(n)} mean {np.mean(n):.0f} max {max(n)}", flush=True)
    print("TASKS first gsm ids", [m["id"] for m in items[:5]], "first prompt tail", repr(tok.decode(prompts[0][-12:])), flush=True)


if __name__ == "__main__":
    main()
