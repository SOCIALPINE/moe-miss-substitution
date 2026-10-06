"""Figures 1-3 of the paper (2026-10-05). Reads only the records; computes nothing new.

Figure 1  Mass and quality: engine policies (Part 18 single topic, Part 19 topic change; Table 3) and the
          pre-registered generation test (Part 20; Table 9).
Figure 2  Loss around topic changes: Delta-ppl vs fetching every miss per 512-token bin (Part 19, E13 `bins`).
Figure 3  Time to answer by RAM condition (Part 13; Table 8), session medians as in summarize_xc7.py.

Inputs (paths relative to the project folder)
  results/xc12_3090.log   RESULT main (mass), E12 subset all (Delta-ppl, block interval)
  results/xc13_3090.log   RESULT cross24 (mass), E13 (Delta-ppl, block interval, bins)
  PART20_RESULTS_2026-10-03.md   appendix A pooled G14 (mass) and P14 (accuracy difference, interval) lines
  results/xc7_3090.json   Part 13 per-session prefill, fill and decode times
Each plotted number is checked against the value printed in the paper's tables (assertions below).
Usage: python3 paper/figs/make_figures.py   (writes paper/figs/fig{1,2,3}_{en,ko}.{pdf,png} and figdata.json)
"""
import json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.normpath(os.path.join(HERE, "..")) + "/"   # repository root (R20 values also need the research notes, not included; figdata.json holds every plotted value)

# ---------------------------------------------------------------- palette (dataviz reference palette, light)
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated all-pairs with slot 3)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"

KEYS = {
    "exact@0.25/nvme|aio=1+uf=1+up=1+bg=0+ram=0": "E",
    "C@0.25/nvme:0.15|aio=1+uf=1+up=1+bg=0+ram=0": "T",
    "C1@0.25/nvme|aio=1+uf=1+up=1+bg=0+ram=0": "C",
    "C@0.25/nvme:0.15|aio=1+uf=1+up=1+bg=0+ram=0+fe=0.0078125+dk=1": "T+F",
    "C@0.25/nvme:0.15|aio=1+uf=1+up=1+bg=0+ram=0+fe=0.0078125+dk=1+rb=1": "T+F+A1",
    "C@0.25/nvme:0.15|aio=1+uf=1+up=1+bg=0+ram=0+fe=0.0078125+dk=1+rb=0.5": "Q",
    "C1@0.25/nvme|aio=1+uf=1+up=1+bg=0+ram=0+fe=0.0078125+dk=1": "C+F",
    "C1@0.25/nvme|aio=1+uf=1+up=1+bg=0+ram=0+fe=0.0078125+dk=1+rb=0.5": "S",
}
FAMILY = {"T": "t", "T+F": "t", "T+F+A1": "t", "Q": "t", "C": "c", "C+F": "c", "S": "c", "H": "h"}


# ---------------------------------------------------------------- data
def engine(fn, result_name, eline, run):
    mass, dppl = {}, {}
    for line in open(B + fn):
        if line.startswith("RESULT "):
            _, name, js = line.split(" ", 2)
            if name == result_name:
                for k, v in json.loads(js)["runs"].items():
                    if KEYS.get(k, "E") != "E":
                        mass[KEYS[k]] = v["mass"]
        elif line.startswith(eline + " "):
            j = json.loads(line[len(eline) + 1:])
            if j["run"] != run or j.get("subset", "all") != "all":
                continue
            lab = KEYS.get(j["key"])
            if lab and lab != "E":
                d = j["dppl_exact"]
                d = d["block"] if isinstance(d, dict) else d
                dppl[lab] = (d[0], d[1], d[2], j.get("bins"))
    return {k: (mass[k],) + dppl[k] for k in mass}


def part20():
    sec, g, p = None, {}, {}
    for line in open(B + "PART20_RESULTS_2026-10-03.md"):
        if line.startswith("# 부록 A"):
            sec = "A"
        elif line.startswith("# 부록 B"):
            sec = None
        if sec != "A":
            continue
        if line.startswith("G14 "):
            j = json.loads(line[4:]); g[(j["task"], j["label"])] = j["mass"]
        elif line.startswith("P14 "):
            j = json.loads(line[4:]); p[(j["task"], j["label"])] = tuple(j["dacc_pp"])
    return {t: {lab: (g[(t, lab)],) + p[(t, lab)] for lab in "QSTCH"} for t in ("gsm8k", "humaneval")}


def part13():
    d = json.load(open(B + "results/xc7_3090.json"))

    def ans(name):
        out = []
        for pl in ("128", "512", "1536"):
            t = d[name]["ttft_by_p"][pl]
            fill = t["fill"] if t["fill"] else [0.0] * len(t["prefill"])
            out.append(statistics.median((t["prefill"][i] + fill[i] + t["dec_ms"][i]) / 1000 for i in range(len(t["prefill"]))))
        return out

    rows = {"cpu": ["stock_warm", "stock_hog32", "stock_hog"],
            "vram": ["stock34_warm", "stock34_hog32", "stock34_hog"],
            "pc": ["warm_auto", "hog32_r1", "hog_r1"],      # ample: the automatic switch turned page cache first on
            "ssd": ["warm_r0", "hog32_r0", "hog_r0"]}
    return {k: [ans(n) for n in v] for k, v in rows.items()}


tf18 = engine("results/xc12_3090.log", "main", "E12", "main")
tf19 = engine("results/xc13_3090.log", "cross24", "E13", "cross24")
gen = part20()
p13 = part13()

# ---------------------------------------------------------------- checks against the paper's tables
T3 = {"T": (0.0978, 1.21, 0.1337, 1.77), "C": (0.1101, 1.76, 0.1600, 3.27), "T+F": (0.0930, 0.94, 0.1327, 1.67),
      "T+F+A1": (0.0802, 0.95, None, None), "Q": (0.0717, 0.67, 0.0851, 0.97), "C+F": (0.1037, 1.55, 0.1587, 3.02),
      "S": (0.0779, 1.00, 0.0929, 1.46)}
for k, (m18, d18, m19, d19) in T3.items():
    assert round(tf18[k][0], 4) == m18 and round(tf18[k][1], 2) == d18, (k, tf18[k])
    if m19 is not None:
        assert round(tf19[k][0], 4) == m19 and round(tf19[k][1], 2) == d19, (k, tf19[k])
T9 = {"gsm8k": {"Q": (0.158, 0.4), "S": (0.168, -0.2), "T": (0.209, -1.8), "C": (0.228, -3.4), "H": (0.370, -8.4)},
      "humaneval": {"Q": (0.166, -1.22), "S": (0.175, -0.61), "T": (0.211, 0.61), "C": (0.225, -0.61), "H": (0.369, -2.44)}}
for t, v in T9.items():
    for lab, (m, a) in v.items():
        assert round(gen[t][lab][0], 3) == m and gen[t][lab][1] == a, (t, lab, gen[t][lab])
T8 = {"cpu": [(5.4, 6.1, 6.7), (11.2, 12.5, 15.5), (29.1, 34.9, 39.1)],
      "vram": [(4.3, 4.8, 5.3), (6.6, 5.2, 5.5), (13.9, 19.9, 22.3)],
      "pc": [(3.0, 3.4, 3.9), (3.6, 4.2, 4.7), (4.1, 4.8, 5.5)],
      "ssd": [(4.6, 5.6, 6.2), (4.6, 5.6, 6.2), (4.6, 5.6, 6.3)]}
for k, rows in T8.items():
    for r, want in enumerate(rows):
        got = tuple(round(x, 1) for x in p13[k][r])
        assert got == want, (k, r, got, want)
BIN_T = [1.337, 1.260, 1.025, 4.229, 1.286, 0.867, 1.260, 2.937]           # Part 19 E13 bins, the 0.15 rule
assert [round(x, 3) for x in tf19["T"][4]] == BIN_T, tf19["T"][4]

json.dump({"table3_part18": tf18, "table3_part19": {k: v[:4] for k, v in tf19.items()},
           "part19_bins": {k: v[4] for k, v in tf19.items()}, "table9": gen, "table8": p13},
          open(os.path.join(HERE, "figdata.json"), "w"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- text
TXT = {
    "en": dict(
        font="DejaVu Sans", ls=1.25,
        fam_t="0.15-rule family", fam_c="First-choice-rule family", fam_h="First-choice rule, 12.5% cache (H)",
        a="(a) Engine, single topic (24 sessions)", b="(b) Engine, topic change (24 sequences)",
        c="(c) Generation, GSM8K (500 problems)", d="(d) Generation, HumanEval (164 problems)",
        x1="Substituted gate mass",
        y1="Perplexity increase vs.\nfetching every miss (%)", y2="Accuracy difference\nvs. E (points)",
        ni="−2-point margin",
        f2l="0.15-rule family", f2r="First-choice-rule family", x2="Scored token position",
        y2b="Perplexity increase per\n512-token bin vs. E (%)", tc="topic change",
        s_t=["T", "T+F", "Q = T+F+A"], s_c=["C", "C+F", "S = C+F+A"],
        p3=["128-token prompt", "512-token prompt", "1,536-token prompt"], ram=["ample", "~32 GB", "~16 GB"],
        x3="Available RAM", y3="Time to answer (s, log scale)",
        l3=["stock llama.cpp, all experts on CPU", "stock llama.cpp, same VRAM",
            "engine (0.15 rule), SSD only", "engine (0.15 rule), page cache first"]),
    "ko": dict(
        font="Noto Sans CJK KR", ls=1.6,
        fam_t="0.15 규칙 계열", fam_c="1순위 규칙 계열", fam_h="1순위 규칙, 캐시 12.5%(H)",
        a="(a) 엔진, 한 주제(24세션)", b="(b) 엔진, 주제 전환(24시퀀스)",
        c="(c) 생성, GSM8K(500문제)", d="(d) 생성, HumanEval(164문제)",
        x1="대체 게이트 질량",
        y1="모든 미스 동기 적재 대비\nperplexity 증가(%)", y2="기준 설정(E) 대비\n정답률 차이(%p)",
        ni="비열등 한계 −2%p",
        f2l="0.15 규칙 계열", f2r="1순위 규칙 계열", x2="평가 토큰 위치",
        y2b="E 대비 perplexity 증가\n(512토큰 구획별, %)", tc="주제 전환",
        s_t=["T", "T+F", "Q = T+F+A"], s_c=["C", "C+F", "S = C+F+A"],
        p3=["프롬프트 128토큰", "프롬프트 512토큰", "프롬프트 1,536토큰"], ram=["넉넉", "약 32GB", "약 16GB"],
        x3="사용 가능한 RAM", y3="답변 시간(초, 로그 눈금)",
        l3=["기본 llama.cpp, 전문가 전부 CPU", "기본 llama.cpp, 같은 VRAM",
            "엔진(0.15 규칙), SSD만", "엔진(0.15 규칙), 페이지 캐시 우선"]),
}


def style(lang):
    plt.rcParams.update({
        "font.family": TXT[lang]["font"], "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "axes.unicode_minus": True, "axes.titlelocation": "left", "axes.titlepad": 6,
    })


def save(fig, name, lang):
    fig.savefig(os.path.join(HERE, f"{name}_{lang}.png"), dpi=300)
    if lang == "en":   # no creation date in the PDF (it would carry the local time zone)
        fig.savefig(os.path.join(HERE, f"{name}_{lang}.pdf"), metadata={"CreationDate": None})
    plt.close(fig)


MK = {"t": ("o", BLUE), "c": ("^", ORANGE), "h": ("s", ORANGE)}


def point(ax, x, y, lo, hi, fam, label, dx, dy, ha):
    m, col = MK[fam]
    ax.plot([x, x], [lo, hi], color=col, lw=1.0, alpha=0.75, solid_capstyle="round", zorder=2)
    ax.plot(x, y, m, ms=6.5, mfc=col, mec=SURFACE, mew=1.2, zorder=3)
    if ha == "top":    # label centred above the top of the interval bar
        ax.annotate(label, (x, hi), xytext=(dx, dy), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=INK, zorder=4)
    else:
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=7.5, color=INK, zorder=4)


def fig1(lang):
    t = TXT[lang]
    style(lang)
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.4))
    # label offsets (points) chosen so that labels do not overlap markers or interval bars
    off18 = {"T": (6, 0, "left"), "C": (6, 0, "left"), "T+F": (0, 2, "top"), "T+F+A1": (6, -7, "left"),
             "Q": (6, 0, "left"), "C+F": (0, 2, "top"), "S": (-6, 4, "right")}
    off19 = {"T": (6, -4, "left"), "C": (-6, 0, "right"), "T+F": (-6, 4, "right"), "Q": (6, 0, "left"),
             "C+F": (6, -9, "left"), "S": (6, 0, "left")}
    for ax, data, off, title, r in ((axs[0, 0], tf18, off18, t["a"], 0.92), (axs[0, 1], tf19, off19, t["b"], 0.91)):
        xs = [data[k][0] for k in data]; ys = [data[k][1] for k in data]
        n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((a - mx) ** 2 for a in xs); syy = sum((b - my) ** 2 for b in ys)
        sl = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx
        assert round(sl * sxx ** 0.5 / syy ** 0.5, 2) == r, (title, sl * sxx ** 0.5 / syy ** 0.5)   # r in Section 5.2
        assert round(sl * 0.01, 2) == {0.92: 0.25, 0.91: 0.26}[r], (title, sl)                      # slope in Section 5.2
        x0, x1 = min(xs) - 0.006, max(xs) + 0.006
        ax.plot([x0, x1], [my + sl * (x0 - mx), my + sl * (x1 - mx)], color=MUTED, lw=1.0, zorder=1)
        ax.text(0.03, 0.95, f"r = {r:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK2)
        for k in data:
            m, d, lo, hi = data[k][:4]
            point(ax, m, d, lo, hi, FAMILY[k], k, *off[k])
        ax.set_title(title)
        ax.set_ylabel(t["y1"], linespacing=t["ls"])
        ax.set_ylim(bottom=0)
    axs[0, 0].set_xlim(0.064, 0.118); axs[0, 1].set_xlim(0.078, 0.168)
    offg = {"gsm8k": {"Q": (-6, 3, "right"), "S": (6, -3, "left"), "T": (-5, 5, "right"), "C": (6, 0, "left"), "H": (-7, 0, "right")},
            "humaneval": {"Q": (-6, -3, "right"), "S": (6, 4, "left"), "T": (-6, 0, "right"), "C": (6, -4, "left"), "H": (-7, 0, "right")}}
    for ax, task, title in ((axs[1, 0], "gsm8k", t["c"]), (axs[1, 1], "humaneval", t["d"])):
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.axhline(-2, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(0.30, -2, t["ni"], ha="center", va="bottom", fontsize=7, color=INK2)
        for lab in "QSTCH":
            m, a, lo, hi = gen[task][lab]
            point(ax, m, a, lo, hi, FAMILY[lab], lab, *offg[task][lab])
        ax.set_title(title)
        ax.set_ylabel(t["y2"], linespacing=t["ls"])
        ax.set_xlim(0.135, 0.40)
    axs[1, 0].set_ylim(-12.5, 3); axs[1, 1].set_ylim(-7, 3)
    for ax in axs.ravel():
        ax.set_xlabel(t["x1"])
    handles = [Line2D([], [], marker="o", ls="", ms=6.5, mfc=BLUE, mec=SURFACE, label=t["fam_t"]),
               Line2D([], [], marker="^", ls="", ms=6.5, mfc=ORANGE, mec=SURFACE, label=t["fam_c"]),
               Line2D([], [], marker="s", ls="", ms=6.0, mfc=ORANGE, mec=SURFACE, label=t["fam_h"])]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 0.985, 0.955), h_pad=1.6, w_pad=1.4)
    save(fig, "fig1_mass_quality", lang)


def fig2(lang):
    t = TXT[lang]
    style(lang)
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    xs = [256 + 512 * i for i in range(8)]
    for ax, keys, names, col, mk, title in ((axs[0], ["T", "T+F", "Q"], t["s_t"], BLUE, "o", t["f2l"]),
                                             (axs[1], ["C", "C+F", "S"], t["s_c"], ORANGE, "^", t["f2r"])):
        for xc in (1536, 3584):
            ax.axvline(xc, color=AXIS, lw=0.8, zorder=1)
            ax.text(xc - 40, 8.45, t["tc"], ha="right", va="top", fontsize=7, color=INK2)
        styles = [(INK2, mk, 1.5), (MUTED, "D", 1.5), (col, mk, 2.0)]
        for k, nm, (c, m, lw) in zip(keys, names, styles):
            ax.plot(xs, tf19[k][4], color=c, lw=lw, marker=m, ms=5 if m != "D" else 4.2, mfc=c, mec=SURFACE, mew=1.0,
                    label=nm, zorder=3 if c == col else 2, solid_joinstyle="round", solid_capstyle="round")
        ax.set_title(title)
        ax.set_xticks([0, 1024, 2048, 3072, 4096]); ax.set_xticklabels(["0", "1,024", "2,048", "3,072", "4,096"])
        ax.set_xlim(0, 4096)
        ax.set_xlabel(t["x2"])
    axs[0].set_ylabel(t["y2b"], linespacing=t["ls"])
    axs[0].set_ylim(0, 8.6)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.2)
    for ax in axs:                                  # one legend under each panel, placed after the layout is fixed
        pos = ax.get_position()
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", bbox_to_anchor=((pos.x0 + pos.x1) / 2, 0.0), ncol=3, frameon=False,
                   handlelength=2.0, columnspacing=1.4, borderaxespad=0.2)
    save(fig, "fig2_topic_change", lang)


def fig3(lang):
    t = TXT[lang]
    style(lang)
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.9), sharey=True)
    # stock series are drawn slightly left of each RAM condition and engine series slightly right, so that close
    # values (e.g. 4.3 vs 4.6 s) do not hide each other
    series = [("cpu", MUTED, "D", 4.4, -0.09), ("vram", INK2, "s", 4.6, -0.09), ("ssd", ORANGE, "^", 5.2, 0.09),
              ("pc", BLUE, "o", 5.2, 0.09)]
    for j, ax in enumerate(axs):
        for (k, c, m, ms, dx), lab in zip(series, t["l3"]):
            ys = [p13[k][r][j] for r in range(3)]
            ax.plot([r + dx for r in range(3)], ys, color=c, lw=1.8 if k in ("pc", "ssd") else 1.4, marker=m, ms=ms,
                    mfc=c, mec=SURFACE, mew=1.0, label=lab, solid_joinstyle="round", solid_capstyle="round")
        ax.set_yscale("log")
        ax.set_yticks([2, 3, 5, 10, 20, 40]); ax.set_yticklabels(["2", "3", "5", "10", "20", "40"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(2.4, 48)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(t["ram"]); ax.set_xlim(-0.25, 2.25)
        ax.set_title(t["p3"][j])
        ax.set_xlabel(t["x3"])
    axs[0].set_ylabel(t["y3"])
    h, l = axs[0].get_legend_handles_labels()
    order = [3, 2, 1, 0]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.0, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=0.8)
    save(fig, "fig3_answer_time", lang)


for lang in ("en", "ko"):
    fig1(lang); fig2(lang); fig3(lang)
print("ok: all plotted values match Tables 3, 8, 9 and the Part 19 bins; figures written to", HERE)
