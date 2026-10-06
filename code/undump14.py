"""Part 20 addendum: rebuild the D14T / D14K documents that dump14.py printed, from saved pod logs (stream-pod-logs
tool-result files or plain logs). Checks every line's md5, the line count and the md5 of the compressed bytes and of the
JSON, then writes <out>/d14t.json and <out>/d14k.json.  Usage: python3 undump14.py --out DIR LOG [LOG ...]"""
import argparse, base64, hashlib, json, lzma, os, re, sys


def lines_of(path):
    txt = open(path, encoding="utf-8").read()
    try:
        d = json.loads(txt)
        return [x["line"] for x in d["items"]]
    except (ValueError, KeyError, TypeError):
        return txt.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    parts, ends, bad = {"D14T": {}, "D14K": {}}, {}, 0
    for p in a.logs:
        for line in lines_of(p):
            m = re.search(r"(D14[TK]) (\d+)/(\d+) ([0-9a-f]{8}) (\S+)$", line)
            if m:
                tag, k, n, h, payload = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
                if hashlib.md5(payload.encode()).hexdigest()[:8] != h:
                    bad += 1
                    continue
                parts[tag][k] = (n, payload)
                continue
            m = re.search(r"(D14[TK]) END n=(\d+) bytes=(\d+) md5=([0-9a-f]{32}) raw=([0-9a-f]{32})", line)
            if m:
                ends[m.group(1)] = (int(m.group(2)), int(m.group(3)), m.group(4), m.group(5))
    os.makedirs(a.out, exist_ok=True)
    ok_all = True
    for tag in ("D14T", "D14K"):
        if tag not in ends:
            print(tag, "END line missing"); ok_all = False; continue
        n, nbytes, md5c, md5r = ends[tag]
        have = parts[tag]
        miss = [k for k in range(1, n + 1) if k not in have]
        if miss:
            print(tag, "missing lines", len(miss), miss[:10]); ok_all = False; continue
        comp = base64.b64decode("".join(have[k][1] for k in range(1, n + 1)))
        if len(comp) != nbytes or hashlib.md5(comp).hexdigest() != md5c:
            print(tag, "compressed bytes mismatch"); ok_all = False; continue
        raw = lzma.decompress(comp)
        if hashlib.md5(raw).hexdigest() != md5r:
            print(tag, "raw md5 mismatch"); ok_all = False; continue
        out = os.path.join(a.out, tag.lower() + ".json")
        open(out, "wb").write(raw)
        doc = json.loads(raw)
        print(tag, "OK lines", n, "compressed", nbytes, "raw", len(raw), "records", doc.get("n_records"), "->", out)
    print("bad line md5:", bad)
    print("UNDUMP", "OK" if ok_all and not bad else "INCOMPLETE")
    sys.exit(0 if ok_all and not bad else 1)


if __name__ == "__main__":
    main()
