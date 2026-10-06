"""Token files for the round-9 router-logit dumps (same format as make_tokbin.py: int32 S, T, P, then S*T tokens).

all24:   every session (24 x 2048)
same8k:  3 sequences of 8192, four sessions of one domain back to back
cross8k: 3 sequences of 8192, four sessions of four different domains (the topic changes every 2048 tokens)
cross8kb: 3 more such sequences with other domain orders and other sessions (round 12)
cross24: 24 topic-change sequences, every session once in each measured position (round 13)
"""
import json, sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True)
out = sys.argv[2]
tok = d["tokens"].astype(np.int32)
dom = d["domain"].astype(np.int64)
names = [str(x) for x in d["domain_names"]]
P = int(d["prompt_len"])
by = {k: [int(i) for i in np.where(dom == k)[0]] for k in range(len(names))}


def write(name, seqs):
    t = np.stack([np.concatenate([tok[i] for i in s]) for s in seqs]).astype(np.int32)
    S, T = t.shape
    open(f"{out}/{name}.bin", "wb").write(np.array([S, T, P], np.int32).tobytes() + np.ascontiguousarray(t).tobytes())
    info = [[names[dom[i]] for i in s] for s in seqs]
    json.dump({"S": S, "T": T, "P": P, "sessions": seqs, "domains": info}, open(f"{out}/{name}.json", "w"))
    print("seqfile", name, S, T, P, info, flush=True)


write("all24", [[i] for k in range(len(names)) for i in by[k]])
n = len(names)
same = [k for k in (0, 2, 3) if len(by[k]) >= 4][:3]
write("same8k", [by[k][:4] for k in same])
order = [(0, 1, 2, 5), (3, 4, 0, 2), (5, 1, 4, 3)]
write("cross8k", [[by[a % n][j], by[b % n][j], by[c % n][j], by[e % n][j]] for j, (a, b, c, e) in enumerate(order)])
order2 = [(1, 3, 5, 0), (2, 0, 4, 1), (4, 5, 3, 2)]
write("cross8kb", [[by[x % n][(j + 3) % 4] for x in o] for j, o in enumerate(order2)])
# round 13: 24 topic-change sequences. With prompt 512 + 4096 generated, the first session gives the prompt and 1536 generated
# tokens, the second 2048 and the third 512 (the fourth is not measured). Every session is the first, the second and the
# third part of exactly one sequence; the three measured parts always come from three different domains.
s1, s2 = (1, 2, 4, 5), (3, 4, 2, 1)
write("cross24", [[by[d][k], by[(d + s1[k]) % n][(k + 1) % 4], by[(d + s2[k]) % n][(k + 2) % 4], by[(d + s1[k] + 3) % n][(k + 3) % 4]]
                  for d in range(n) for k in range(4)])
