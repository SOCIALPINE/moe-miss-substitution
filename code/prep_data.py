"""Build realistic single-topic 'sessions' (T tokens each) for 6 domains.
A session is one coherent context: one long article / one source file / one long chat /
one multi-turn math tutoring conversation. Every domain is optional: failures are logged
and skipped so one broken download can't kill the run."""
import os, sys, json, gzip, random, time, argparse, traceback
import numpy as np
import pyarrow.parquet as pq


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def fetch(repo, fn):
    if os.environ.get("PREP_FAKE_DIR"):
        return os.path.join(os.environ["PREP_FAKE_DIR"], repo.replace("/", "__"), fn)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fn, repo_type="dataset")


def list_files(repo, prefix):
    if os.environ.get("PREP_FAKE_DIR"):
        base = os.path.join(os.environ["PREP_FAKE_DIR"], repo.replace("/", "__"))
        out = []
        for r, _, fs in os.walk(base):
            for f in fs:
                out.append(os.path.relpath(os.path.join(r, f), base))
        return sorted(p for p in out if p.startswith(prefix))
    from huggingface_hub import list_repo_files
    return sorted(p for p in list_repo_files(repo, repo_type="dataset") if p.startswith(prefix))


def parquet_rows(path, columns, max_rows, pred=None):
    pf = pq.ParquetFile(path)
    rows = []
    for b in pf.iter_batches(batch_size=2000, columns=columns):
        for r in b.to_pylist():
            if pred is None or pred(r):
                rows.append(r)
        if len(rows) >= max_rows:
            break
    return rows


ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
        "system": "system", "chatgpt": "assistant", "bard": "assistant", "bing": "assistant"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--per_domain", type=int, default=32)
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    T, NS = args.seq_len, args.per_domain
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception as e:     # e.g. a config model_type this transformers version does not know
        log(f"AutoTokenizer failed ({type(e).__name__}: {str(e)[:120]}); using PreTrainedTokenizerFast")
        from transformers import PreTrainedTokenizerFast
        tc = {}
        p = os.path.join(args.model_dir, "tokenizer_config.json")
        if os.path.exists(p):
            tc = {k: (v["content"] if isinstance(v, dict) else v) for k, v in json.load(open(p)).items()
                  if k in ("bos_token", "eos_token", "pad_token", "unk_token", "chat_template")}
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.model_dir, "tokenizer.json"), **tc)

    def enc(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    def generic_msgs(m):              # DeepSeek-style fallback
        s = tok.bos_token or ""
        for r in m:
            if r["role"] == "system":
                s += r["content"]
            elif r["role"] == "user":
                s += "<｜User｜>" + r["content"]
            else:
                s += "<｜Assistant｜>" + r["content"] + (tok.eos_token or "")
        return s

    enc_msgs = None
    if not getattr(tok, "chat_template", None):
        try:
            sys.path.insert(0, os.path.join(args.model_dir, "encoding"))
            from encoding_dsv4 import encode_messages
            enc_msgs = lambda m: encode_messages(m, thinking_mode="chat")
            log("chat format: encoding/encoding_dsv4.encode_messages (thinking_mode=chat)")
            log("  example: " + repr(enc_msgs([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                                                {"role": "user", "content": "bye"}]))[:300])
        except Exception as e:
            log(f"no chat template / encoder ({type(e).__name__}: {str(e)[:120]}); generic DeepSeek chat format")
            enc_msgs = generic_msgs
    n_fallback = [0]

    def chat(msgs):
        if enc_msgs is None:
            return tok.apply_chat_template(msgs, tokenize=False)
        try:
            return enc_msgs(msgs)
        except Exception:
            n_fallback[0] += 1
            if n_fallback[0] <= 3:
                log(f"  encoder failed on a conversation ({n_fallback[0]}); generic format used")
            return generic_msgs(msgs)

    def take_long(texts, what):
        """texts: iterable of strings, each a full session candidate; keep those >= T tokens"""
        out, short = [], []
        for s in texts:
            ids = enc(s)
            if len(ids) >= T:
                out.append(ids[:T])
                if len(out) >= NS:
                    break
            elif len(short) < 400:
                short.append(ids)
        n_long = len(out)
        cur = []
        for ids in short:              # fallback: pack shorter items back-to-back
            if len(out) >= NS:
                break
            cur.extend(ids)
            if len(cur) >= T:
                out.append(cur[:T]); cur = []
        log(f"  {what}: {n_long} single-document sessions + {len(out)-n_long} packed")
        return out

    def pack(texts, what):
        """concatenate consecutive items (e.g. turns of one tutoring chat) until >= T tokens"""
        out, cur = [], []
        for s in texts:
            cur.extend(enc(s))
            if len(cur) >= T:
                out.append(cur[:T]); cur = []
                if len(out) >= NS:
                    break
        log(f"  {what}: {len(out)} packed sessions")
        return out

    rnd = random.Random(args.seed)

    def d_ko_wiki():
        p = fetch("wikimedia/wikipedia", "20231101.ko/train-00001-of-00003.parquet")
        rows = parquet_rows(p, ["title", "text"], 6000, lambda r: len(r["text"]) >= 3500)
        rnd.shuffle(rows)
        return take_long((r["title"] + "\n\n" + r["text"] for r in rows), "ko_wiki")

    def d_en_wiki():
        fs = [f for f in list_files("wikimedia/wikipedia", "20231101.en/") if f.endswith(".parquet")]
        p = fetch("wikimedia/wikipedia", fs[min(3, len(fs) - 1)])
        rows = parquet_rows(p, ["title", "text"], 6000, lambda r: len(r["text"]) >= 11000)
        rnd.shuffle(rows)
        return take_long((r["title"] + "\n\n" + r["text"] for r in rows), "en_wiki")

    def d_code():
        p = fetch("codeparrot/codeparrot-clean-valid", "file-000000000054.json.gz")
        rows = []
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                c = r.get("content", "")
                if len(c) >= 7000 and not r.get("autogenerated", False):
                    rows.append(c)
                if len(rows) >= 3000:
                    break
        rnd.shuffle(rows)
        return take_long(rows, "python_code")

    def d_chat_en():
        fs = [f for f in list_files("HuggingFaceH4/ultrachat_200k", "data/test_sft")]
        p = fetch("HuggingFaceH4/ultrachat_200k", fs[0])
        rows = parquet_rows(p, ["messages"], 20000,
                            lambda r: sum(len(m["content"]) for m in r["messages"]) >= 7000)
        rnd.shuffle(rows)
        texts = (chat([{"role": m["role"], "content": m["content"]} for m in r["messages"]]) for r in rows)
        return take_long(texts, "chat_en")

    def d_chat_ko():
        p = fetch("maywell/koVast", "data/train-00001-of-00003-76f1a8992c6a4ebe.parquet")

        def conv(r):
            ms = []
            for m in r["conversations"]:
                role = ROLE.get(str(m.get("from", "")).lower())
                if role and m.get("value"):
                    ms.append({"role": role, "content": m["value"]})
            return ms
        rows = parquet_rows(p, ["conversations"], 20000,
                            lambda r: sum(len(m.get("value") or "") for m in r["conversations"]) >= 3500)
        rnd.shuffle(rows)
        texts = (chat(conv(r)) for r in rows if len(conv(r)) >= 2)
        return take_long(texts, "chat_ko")

    def d_math():
        p = fetch("AI-MO/NuminaMath-CoT", "data/train-00000-of-00005.parquet")
        rows = parquet_rows(p, ["problem", "solution"], 20000)
        rnd.shuffle(rows)
        # one tutoring conversation = consecutive (problem, solution) turns
        texts = (chat([{"role": "user", "content": r["problem"]},
                       {"role": "assistant", "content": r["solution"]}]) for r in rows)
        return pack(texts, "math_chat")

    domains = [("ko_wiki", d_ko_wiki), ("en_wiki", d_en_wiki), ("python_code", d_code),
               ("chat_en", d_chat_en), ("chat_ko", d_chat_ko), ("math_chat", d_math)]
    toks, doms, names, info = [], [], [], {}
    for nm, fn in domains:
        t0 = time.time()
        try:
            ss = fn()
        except Exception as e:
            log(f"!! domain {nm} failed: {e!r}")
            traceback.print_exc()
            continue
        if len(ss) < 4:
            log(f"!! domain {nm} has only {len(ss)} sessions, skipped")
            continue
        if len(ss) % 2:
            ss = ss[:-1]
        d = len(names)
        names.append(nm)
        toks.extend(ss)
        doms.extend([d] * len(ss))
        info[nm] = {"sessions": len(ss), "sec": round(time.time() - t0, 1),
                    "sample": tok.decode(ss[0][:60]).replace("\n", " ")[:200]}
        log(f"  sample[{nm}]: {info[nm]['sample']}")
    assert len(names) >= 2, "too few domains succeeded"
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "sessions.npz"), tokens=np.array(toks, np.int32), domain=np.array(doms),
             domain_names=np.array(names), prompt_len=np.array(args.prompt_len))
    json.dump(info, open(os.path.join(args.out, "data_info.json"), "w"), ensure_ascii=False, indent=1)
    log(f"saved {len(toks)} sessions x {T} tokens, domains={names}")


if __name__ == "__main__":
    main()
