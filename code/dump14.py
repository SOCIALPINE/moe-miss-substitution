"""Part 20 addendum: print every generation of this run into the log, so the outputs survive the pod.
Two parts, each a JSON document compressed with xz and base64-encoded, split into lines of at most 4000 characters:
  D14T  texts: per (label, problem) the decoded text and the generation statistics, plus the HumanEval fields scoring
        needs (task id, prompt, test, entry point)
  D14K  tokens: per (label, problem) the generated token ids
Line format: "<tag> <k>/<n> <md5 of the line's payload, 8 hex> <payload>", then "<tag> END n=<lines> bytes=<compressed
bytes> md5=<md5 of the compressed bytes> raw=<md5 of the JSON>". Records are sorted by problem, then label."""
import argparse, base64, glob, hashlib, json, lzma, os

ORDER = ["E", "Q", "S", "T", "C", "H", "Qr", "Sr"]
WIDTH = 4000


def put(tag, doc):
    raw = json.dumps(doc, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode()
    comp = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    enc = base64.b64encode(comp).decode()
    parts = [enc[i:i + WIDTH] for i in range(0, len(enc), WIDTH)]
    for k, p in enumerate(parts):
        print(f"{tag} {k + 1}/{len(parts)} {hashlib.md5(p.encode()).hexdigest()[:8]} {p}", flush=True)
    print(f"{tag} END n={len(parts)} bytes={len(comp)} md5={hashlib.md5(comp).hexdigest()} raw={hashlib.md5(raw).hexdigest()}"
          f" rawbytes={len(raw)}", flush=True)


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
    recs = []
    for f in sorted(glob.glob(os.path.join(a.res, "gen_*_*.jsonl"))):
        lab = os.path.basename(f)[:-len(".jsonl")].rsplit("_", 1)[1]
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                d["lab"] = lab
                recs.append(d)
    recs.sort(key=lambda d: (d["i"], ORDER.index(d["lab"]) if d["lab"] in ORDER else 99))
    keep = ("i", "lab", "plan", "np", "n", "eog", "pf_ms", "cfg_ms", "ms", "mass", "ltok", "want", "hit", "sub", "bg", "bytes",
            "stall_ms")
    texts = [dict({k: d.get(k) for k in keep}, text=tok.decode(d["toks"], skip_special_tokens=True)) for d in recs]
    he = {str(i): {k: m.get(k) for k in ("id", "prompt", "test", "entry_point")}
          for i, m in enumerate(items) if m["task"] == "humaneval"}
    put("D14T", {"records": texts, "humaneval": he, "n_records": len(texts)})
    put("D14K", {"records": [{"i": d["i"], "lab": d["lab"], "toks": d["toks"]} for d in recs], "n_records": len(recs)})
    print("DUMP14 DONE", flush=True)


if __name__ == "__main__":
    main()
