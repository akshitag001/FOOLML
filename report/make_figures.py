"""Rebuild every figure and table in the report from the saved result files. Run from the repo root:

    python report/make_figures.py

Reads:   results.csv, results/final/{summary,extra_metrics}.json, results/compare50/*.json, results/study50/log.json,
         site/data/frames.json (+ frame PNGs), p2_data/{clean,adv}_images
Writes:  report/figures/*.png, report/tables.tex, report/numbers.tex
Nothing here calls the target model, and no number is typed by hand: the LaTeX macros in numbers.tex are filled from the files above.
"""
import csv
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
INK, GREY, RULE = "#1A1A1A", "#7A7A74", "#DEDDD7"
RED, GREEN, BLUE = "#B3261E", "#2E7D32", "#2B5C8A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

# ------------------------------------------------------------------ data
rows = list(csv.DictReader(open("results.csv")))
per = {k: np.array([float(r[k]) for r in rows]) for k in ("L2", "Linf", "SSIM", "PSNR", "LPIPS", "queries")}
summary = json.load(open("results/final/summary.json"))
extra = json.load(open("results/final/extra_metrics.json"))
study = json.load(open("results/study50/log.json"))
traces = json.load(open("results/compare50/traces.json"))
frames = {x["name"]: x for x in json.load(open("site/data/frames.json"))["images"]}
cmp_ = {}
for a in ("hsja", "surfree", "cgba"):
    for b in (1000, 2000, 5000):
        cmp_[a, b] = json.load(open(f"results/compare50/{a}_{b}.json"))["summary"]
base50 = json.load(open("results/compare50/baseline.json"))["summary"]
img = lambda p: np.asarray(Image.open(p).convert("RGB"))


def px(ax, a):
    ax.imshow(a, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------ 1. architecture diagram
def architecture():
    fig = plt.figure(figsize=(14, 9.25))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 150); ax.set_ylim(0, 99.5); ax.axis("off")

    def box(x, y, w, h, fc="white", ec=INK, lw=1.0, hatch=None, r=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, hatch=hatch))

    def text(x, y, s, size=8, color=INK, weight="normal", ha="center", va="center", wrap=None, ls=1.35, style="normal"):
        if wrap:
            s = "\n".join(textwrap.fill(p, wrap) for p in s.split("\n"))
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=ls, fontstyle=style)

    def arrow(p, q, color=INK, lw=1.2, style="-|>", rad=0.0, ms=9):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    def thumb(path, cx, y, size=14):
        ax.imshow(img(path), extent=(cx - size / 2, cx + size / 2, y, y + size), interpolation="nearest", zorder=3)
        ax.add_patch(Rectangle((cx - size / 2, y), size, size, fill=False, ec=GREY, lw=0.6, zorder=4))

    fr = frames["cat_7"]["frames"]
    fdir = "site/data/frames/cat_7/"
    mid = next(i for i, f in enumerate(fr) if f["stage"] == "boundary walk" and f["l2"] < 1.0)
    stages = [
        ("Input", "clean image x\ntrue label y",
         "32x32x3 pixels in [0,1]. The only data the attack holds.", "site/data/frames/cat_7/clean_32.png", "cat, true label"),
        ("1  Start", "any wrong image",
         "random noise until the label is not y; else the nearest clean image of another class", fdir + "000_32.png",
         f"L2 {fr[0]['l2']:.1f}"),
        ("2  Binary search", "reach the boundary",
         "12 steps on the line from x to the start, on 8-bit points", fdir + "001_32.png", f"L2 {fr[1]['l2']:.1f}"),
        ("3  HSJA walk", "shrink the noise",
         "probe around the boundary, read the labels, estimate the normal, step, search back",
         fdir + f"{mid:03d}_32.png", f"L2 {fr[mid]['l2']:.2f}"),
        ("4  Survive 8 bits", "rounding-proof",
         "scale until the rounded image is still adversarial, then 5% further", fdir + f"{len(fr) - 1:03d}_32.png",
         f"L2 {fr[-1]['l2']:.2f}"),
        ("5  Save and verify", "PNG round trip",
         "write PNG, reload, re-query in FP32, TF32 and on CPU; push further if any fails",
         "p2_data/adv_images/cat_7_adv.png", "final PNG"),
    ]
    X0, W, GAP, Y0, H = 3, 20, 4, 52, 40
    for k, (title, sub, desc, path, cap) in enumerate(stages):
        x = X0 + k * (W + GAP)
        box(x, Y0, W, H, fc="#F4F3EE" if k else "white", ec=INK)
        text(x + W / 2, Y0 + H - 3.2, title, size=10, weight="bold")
        text(x + W / 2, Y0 + H - 7.0, sub, size=8, color=GREY, style="italic")
        text(x + W / 2, Y0 + H - 10.5, desc, size=7.6, wrap=26, va="top")
        thumb(path, x + W / 2, Y0 + 3.2, size=14)
        text(x + W / 2, Y0 + 1.4, cap, size=6.8, color=GREY)
        if k:
            arrow((x - GAP + 0.3, Y0 + H / 2), (x - 0.3, Y0 + H / 2))
    # HSJA loop marker
    x3 = X0 + 3 * (W + GAP)
    ax.add_patch(FancyArrowPatch((x3 + W - 3, Y0 + H + 0.2), (x3 + 3, Y0 + H + 0.2), arrowstyle="-|>", mutation_scale=9, color=RED,
                                 lw=1.2, connectionstyle="arc3,rad=0.35"))
    text(x3 + W / 2, Y0 + H + 5.6, "repeat until the query budget is used", size=7.2, color=RED)

    # oracle band
    OY, OH = 23, 17
    box(X0, OY, 140, OH, fc="#ECEBE6", ec=INK, lw=1.2)
    text(X0 + 1.5, OY + OH - 2.2, "Black-box oracle: QueryModel wrapper (the only way the attack touches the model)", size=9,
         weight="bold", ha="left")
    comp = [("image in [0,1]", 18), ("normalise\nmean / std", 18), ("target CNN, FP32\n(sealed: no weights,\nlogits or gradients)", 34),
            ("argmax", 12), ("one label\nout", 12)]
    cx = X0 + 4
    centres = []
    for k, (s, w) in enumerate(comp):
        sealed = k == 2
        box(cx, OY + 2.2, w, 9.6, fc="#C9C8C2" if sealed else "white", ec=INK, hatch="////" if sealed else None)
        if sealed:
            box(cx + 1.2, OY + 3.4, w - 2.4, 7.2, fc="white", ec=INK, lw=0.6)
        text(cx + w / 2, OY + 7, s, size=7.3 if sealed else 8)
        centres.append(cx + w / 2)
        if k:
            arrow((cx - 3.6, OY + 7), (cx - 0.2, OY + 7), lw=1.0)
        cx += w + 3.6
    # counter
    box(cx + 0.5, OY + 2.2, 140 + X0 - cx - 4, 9.6, fc="white", ec=RED, lw=1.0)
    text(cx + 0.5 + (140 + X0 - cx - 4) / 2, OY + 7, "query counter\n(per image)", size=8, color=RED)

    # query arrows from stages 1-5
    for k in range(1, 6):
        xc = X0 + k * (W + GAP) + W / 2
        arrow((xc - 1.2, Y0 - 0.2), (xc - 1.2, OY + OH + 0.2), lw=1.0, color=BLUE, ms=8)
        arrow((xc + 1.2, OY + OH + 0.2), (xc + 1.2, Y0 - 0.2), lw=1.0, color=GREEN, ms=8)
    text(X0 + 12, Y0 - 5.0, "blue: an image in [0,1]", size=7.5, color=BLUE, ha="center")
    text(X0 + 12, Y0 - 8.6, "green: a label, nothing else", size=7.5, color=GREEN, ha="center")

    # evaluation band
    EY, EH = 2, 15.5
    box(X0, EY, 140, EH, fc="white", ec=GREY, lw=1.0)
    text(X0 + 1.5, EY + EH - 2.2, "Offline evaluation (reads saved files only, never the model)", size=9, weight="bold", ha="left")
    ev = [("Submission", "p2_data/adv_images/\n<name>_adv.png (1000)\nresults.csv"),
          ("Metrics", "L2, Linf, SSIM, PSNR,\nLPIPS, queries\nper image"),
          ("Attack comparison", "HSJA vs SurFree vs CGBA\n50 images, same starts,\n1000 / 2000 / 5000 queries"),
          ("Ablations", "low-frequency probes, multi-start,\ntexture weighting, margin\n(kept or dropped by a rule)"),
          ("Website and report", "interactive walkthrough\n(site/), LaTeX report,\nfigures from result files")]
    w = 25.2
    for k, (t, s) in enumerate(ev):
        x = X0 + 2 + k * (w + 2.9)
        box(x, EY + 1.3, w, 9.6, fc="#F4F3EE", ec=GREY, lw=0.8)
        text(x + w / 2, EY + 9.2, t, size=8, weight="bold")
        text(x + w / 2, EY + 5.0, s, size=7.2)
    # stage 5 -> evaluation (right margin route)
    xr = X0 + 5 * (W + GAP) + W
    ax.plot([xr, 147.4, 147.4], [Y0 + H / 2 - 6, Y0 + H / 2 - 6, EY + 6], color=INK, lw=1.2)
    arrow((147.4, EY + 6), (X0 + 140 + 0.2, EY + 6), lw=1.2)
    fig.savefig(f"{OUT}/architecture.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ 2. HSJA inner loop schematic
def hsja_loop():
    """Schematic (not data): one HSJA iteration at a point on a curved decision boundary."""
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    bx = lambda y: 4.2 - 1.7 * ((y - 2.5) / 2.4) ** 2
    ys = np.linspace(0.1, 4.9, 200)
    ax.fill_betweenx(ys, 0, bx(ys), color="#F1F0EA", lw=0)
    ax.plot(bx(ys), ys, color=INK, lw=1.5)
    ax.text(0.12, 4.7, 'region "cat" (correct)', color=GREY, fontsize=9)
    ax.text(6.9, 0.2, 'region "frog" (wrong)', color=GREY, fontsize=9, ha="right")
    clean = np.array([0.8, 2.0])
    yb = 3.35; b = np.array([bx(yb), yb])
    n = np.array([1.0, 0.5]); n = n / np.linalg.norm(n)
    rng = np.random.default_rng(4)
    for _ in range(30):                                            # probes, coloured by the label each one returns
        u = rng.normal(size=2); u = u / np.linalg.norm(u) * rng.uniform(0.25, 0.5)
        p = b + u
        ax.scatter(*p, s=16, color=GREEN if p[0] < bx(p[1]) else RED, zorder=4, lw=0)
    ax.plot([clean[0], b[0]], [clean[1], b[1]], color=GREY, lw=0.9, ls=(0, (4, 3)))
    ax.scatter(*clean, s=70, color=GREEN, zorder=5); ax.text(clean[0], clean[1] - 0.42, "clean x", ha="center", fontsize=9.5)
    ax.scatter(*b, s=70, facecolor="white", edgecolor=RED, lw=1.8, zorder=6)
    ax.text(b[0] - 0.3, b[1] + 0.12, "current boundary point", ha="right", fontsize=9.5)
    ax.annotate("", xy=b + 1.05 * n, xytext=b, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8), zorder=7)
    ax.text(*(b + 1.05 * n + [0.1, 0.0]), "estimated normal\n(labels only)", color=BLUE, fontsize=9.5, va="center")
    yn = 2.35; new = np.array([bx(yn), yn])
    ax.annotate("", xy=new, xytext=b + [0.05, -0.08], arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5, connectionstyle="arc3,rad=-0.3"), zorder=7)
    ax.scatter(*new, s=70, color=RED, zorder=6)
    ax.text(new[0] + 0.25, new[1] - 0.15, "step along the boundary,\nthen search back:\nnext point is closer to x", color=RED, fontsize=9.5, va="top")
    ax.scatter([], [], s=16, color=GREEN, label="probe still labelled cat"); ax.scatter([], [], s=16, color=RED, label="probe labelled something else")
    ax.legend(loc="lower left", frameon=False, fontsize=8.8, bbox_to_anchor=(0.0, 0.0))
    ax.set_xlim(0, 7.0); ax.set_ylim(0, 5); ax.axis("off")
    fig.savefig(f"{OUT}/hsja_loop.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ 3. L2 vs queries
def l2_vs_queries():
    grid = np.arange(25, 5001, 25)

    def curve(tr):
        qs = np.array([q for q, _ in tr]); ds = np.minimum.accumulate([d for _, d in tr])
        k = np.searchsorted(qs, grid, side="right") - 1
        return np.where(k >= 0, ds[np.clip(k, 0, None)], np.nan)

    cols = {"hsja": ("#2a78d6", "HSJA"), "surfree": ("#eb6834", "SurFree"), "cgba": ("#1baf7a", "CGBA")}
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    meds = {}
    for a, (c, name) in cols.items():
        cv = np.array([curve(t) for t in traces[a].values()])
        med = np.nanmedian(cv, 0); lo, hi = np.nanpercentile(cv, [25, 75], 0); meds[a] = (cv, med)
        ax.fill_between(grid, lo, hi, color=c, alpha=0.12, lw=0)
        ax.plot(grid, med, color=c, lw=2, label=name)
    ax.set_yscale("log"); ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 3, 5]); ax.set_yticklabels(["0.2", "0.3", "0.5", "1", "2", "3", "5"])
    ax.minorticks_off(); ax.set_xlim(0, 5000)
    ax.set_xlabel("queries per image"); ax.set_ylabel("L2 distance (median of 50 images)")
    ax.grid(True, color=RULE, lw=0.6); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/l2_vs_queries.png", dpi=200); plt.close(fig)
    return grid, meds


# ------------------------------------------------------------------ 4. histograms
def histograms():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, k, lab, col in ((axs[0], "L2", "L2 per image", BLUE), (axs[1], "Linf", "L-inf per image, in 1/255 steps", BLUE)):
        v = per[k] * (255 if k == "Linf" else 1)
        ax.hist(v, bins=40, color=col, alpha=0.85, lw=0)
        ax.axvline(np.median(v), color=RED, lw=1.2); ax.text(np.median(v), ax.get_ylim()[1] * 0.93, f" median {np.median(v):.2f}" if k == "L2" else f" median {np.median(v):.0f}",
                                                              color=RED, fontsize=8.5, va="top")
        ax.set_xlabel(lab); ax.set_ylabel("images")
    fig.tight_layout(); fig.savefig(f"{OUT}/histograms.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ 5. easy / hard example grid
def examples():
    order = np.argsort(per["L2"])
    picks = list(order[:3]) + list(order[-3:])
    fig, axs = plt.subplots(3, 6, figsize=(8.4, 4.6))
    for c, i in enumerate(picks):
        name = rows[i]["image"].replace("_adv.png", "")
        cl, ad = img(f"p2_data/clean_images/{name}.png"), img(f"p2_data/adv_images/{name}_adv.png")
        nz = np.clip(127.5 + 10 * (ad.astype(float) - cl.astype(float)), 0, 255).astype(np.uint8)
        for r, (a, t, col) in enumerate(((cl, f"{name}\n{rows[i]['true_label']}", GREEN), (ad, f"model: {rows[i]['adv_label']}", RED),
                                         (nz, f"x10   L2 {per['L2'][i]:.2f}", GREY))):
            px(axs[r, c], a); axs[r, c].set_title(t, fontsize=7.6, color=col, pad=2)
    fig.text(0.015, 0.78, "original", rotation=90, va="center", fontsize=8.5); fig.text(0.015, 0.5, "adversarial", rotation=90, va="center", fontsize=8.5)
    fig.text(0.015, 0.22, "change", rotation=90, va="center", fontsize=8.5)
    fig.text(0.27, 0.985, "three easiest", ha="center", fontsize=9, va="top"); fig.text(0.74, 0.985, "three hardest", ha="center", fontsize=9, va="top")
    fig.tight_layout(rect=(0.03, 0, 1, 0.95)); fig.savefig(f"{OUT}/examples.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ 6. label flips
def flips():
    M = np.array([[extra["flip_table"][a][b] for b in CLASSES] for a in CLASSES])
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    ax.imshow(M, cmap="Greys", vmin=0, vmax=M.max() * 1.15)
    for i in range(10):
        for j in range(10):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8, color="white" if M[i, j] > M.max() * 0.55 else INK)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("class the model predicts after the attack"); ax.set_ylabel("true class")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/flips.png", dpi=200); plt.close(fig)
    return M


# ------------------------------------------------------------------ 7. tables + macros
def tex_tables(grid, meds, M):
    f = lambda v, d: f"{v:.{d}f}"
    t = []
    # main
    t.append("\\newcommand{\\tabMain}{\\begin{tabular}{lrrrrrr}\\toprule\n success & L2 & L$_\\infty$ & SSIM & PSNR (dB) & LPIPS & queries\\\\\\midrule\n"
             f"{int(round(summary['success_rate'] * 1000))}/1000 & {f(summary['avg_L2'], 3)} & {f(summary['avg_Linf'], 3)} & {f(summary['avg_SSIM'], 4)} & "
             f"{f(summary['avg_PSNR'], 1)} & {f(extra['avg_LPIPS'], 5)} & {int(round(summary['avg_queries']))}\\\\\\bottomrule\\end{{tabular}}}}")
    # comparison
    best = {b: min(cmp_[a, b]["avg_L2"] for a in ("hsja", "surfree", "cgba")) for b in (1000, 2000, 5000)}
    body = ""
    for a, name in (("hsja", "HSJA"), ("surfree", "SurFree"), ("cgba", "CGBA")):
        cells = []
        for b in (1000, 2000, 5000):
            v = cmp_[a, b]["avg_L2"]; s = f(v, 3)
            cells.append(f"\\textbf{{{s}}}" if v == best[b] else s)
        body += f"{name} & " + " & ".join(cells) + f" & {f(cmp_[a, 5000]['avg_Linf'], 3)} & {f(cmp_[a, 5000]['avg_SSIM'], 4)}\\\\\n"
    t.append("\\newcommand{\\tabCompare}{\\begin{tabular}{lrrrrr}\\toprule\n & \\multicolumn{3}{c}{mean L2 at budget} & \\multicolumn{2}{c}{at 5000}\\\\\n"
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n attack & 1000 & 2000 & 5000 & L$_\\infty$ & SSIM\\\\\\midrule\n" + body + "\\bottomrule\\end{tabular}}")
    # medians to reach thresholds
    thr = [2.0, 1.0, 0.5, 0.25]; body = ""
    for a, name in (("hsja", "HSJA"), ("surfree", "SurFree"), ("cgba", "CGBA")):
        cv = meds[a][0]; cells = []
        for th in thr:
            hit = cv <= th
            first = np.where(hit.any(1), grid[hit.argmax(1)], np.inf)
            m = np.median(first); cells.append("$>$5000" if np.isinf(m) else str(int(m)))
        body += f"{name} & " + " & ".join(cells) + "\\\\\n"
    t.append("\\newcommand{\\tabReach}{\\begin{tabular}{lrrrr}\\toprule\n attack & L2 $\\le$ 2 & $\\le$ 1 & $\\le$ 0.5 & $\\le$ 0.25\\\\\\midrule\n" + body + "\\bottomrule\\end{tabular}}")
    # study
    names = {"0_hsja": "HSJA, 1\\% margin", "1_lowfreq_16": "low-frequency probes, 16$\\times$16 DCT", "1_lowfreq_8": "low-frequency probes, 8$\\times$8 DCT",
             "2_multistart": "best of several starting points", "3_texture_0.5": "texture-weighted probes, $p=0.5$",
             "3_texture_1.0": "texture-weighted probes, $p=1$", "4_margin_0pct": "margin 0\\%", "4_margin_2pct": "margin 2\\%", "4_margin_5pct": "margin 5\\%"}
    body = ""
    for s in study["log"]:
        body += (f"{names[s['attack']]} & {f(s['avg_L2'], 3)} & {f(s['avg_Linf'], 3)} & {f(s['avg_SSIM'], 4)} & {s['contexts']}/50 & "
                 f"{int(round(s['jitter'] * 100))}\\% & {'yes' if s['attack'] in study['kept'] else 'no'}\\\\\n")
    t.append("\\newcommand{\\tabStudy}{\\begin{tabular}{lrrrrrl}\\toprule\n change & L2 & L$_\\infty$ & SSIM & holds & jitter & kept\\\\\\midrule\n" + body + "\\bottomrule\\end{tabular}}")
    open("report/tables.tex", "w").write("\n".join(t) + "\n")

    top = sorted(zip(CLASSES, M.sum(0)), key=lambda x: -x[1])
    # LaTeX macro names may not contain digits, so 0-9 become a-j (nL2 -> nLc)
    g = lambda name: f"\\newcommand{{\\{name.translate(str.maketrans('0123456789', 'abcdefghij'))}}}"
    n = []
    n.append(g("nL2") + f"{{{f(summary['avg_L2'], 3)}}}"); n.append(g("nLinf") + f"{{{f(summary['avg_Linf'], 3)}}}")
    n.append(g("nLinfSteps") + f"{{{f(summary['avg_Linf'] * 255, 1)}}}"); n.append(g("nSSIM") + f"{{{f(summary['avg_SSIM'], 4)}}}")
    n.append(g("nPSNR") + f"{{{f(summary['avg_PSNR'], 1)}}}"); n.append(g("nLPIPS") + f"{{{f(extra['avg_LPIPS'], 5)}}}")
    n.append(g("nQueries") + f"{{{int(round(summary['avg_queries']))}}}"); n.append(g("nTotalQueries") + f"{{{summary['total_queries']:,}}}".replace(",", "\\,"))
    n.append(g("nFixed") + f"{{{summary['fixed_after_saving']}}}")
    n.append(g("nGtEight") + f"{{{extra['n_linf_gt_8_255']}}}"); n.append(g("nGtSixteen") + f"{{{extra['n_linf_gt_16_255']}}}")
    n.append(g("nLinfMedian") + f"{{{int(extra['linf_steps_median'])}}}"); n.append(g("nLinfMax") + f"{{{extra['linf_steps_max']}}}")
    n.append(g("nMedL2") + f"{{{f(np.median(per['L2']), 3)}}}"); n.append(g("nMaxL2") + f"{{{f(per['L2'].max(), 2)}}}")
    n.append(g("nMinL2") + f"{{{f(per['L2'].min(), 3)}}}"); n.append(g("nPninety") + f"{{{f(np.percentile(per['L2'], 90), 2)}}}")
    n.append(g("nBaseL2") + f"{{{f(base50['avg_L2'], 2)}}}")
    n.append(g("nTopA") + f"{{{top[0][0]}}}"); n.append(g("nTopAn") + f"{{{int(top[0][1])}}}")
    n.append(g("nTopB") + f"{{{top[1][0]}}}"); n.append(g("nTopBn") + f"{{{int(top[1][1])}}}")
    n.append(g("nTopC") + f"{{{top[2][0]}}}"); n.append(g("nTopCn") + f"{{{int(top[2][1])}}}")
    n.append(g("nBottomA") + f"{{{top[-1][0]}}}"); n.append(g("nBottomAn") + f"{{{int(top[-1][1])}}}")
    n.append(g("nBottomB") + f"{{{top[-2][0]}}}"); n.append(g("nBottomBn") + f"{{{int(top[-2][1])}}}")
    sd = {s["attack"]: s for s in study["log"]}
    n.append(g("nStudyBaseL2") + f"{{{f(sd['0_hsja']['avg_L2'], 3)}}}"); n.append(g("nLowA") + f"{{{f(sd['1_lowfreq_16']['avg_L2'], 3)}}}")
    n.append(g("nLowB") + f"{{{f(sd['1_lowfreq_8']['avg_L2'], 3)}}}"); n.append(g("nMulti") + f"{{{f(sd['2_multistart']['avg_L2'], 3)}}}")
    n.append(g("nTexA") + f"{{{f(sd['3_texture_0.5']['avg_L2'], 3)}}}"); n.append(g("nTexB") + f"{{{f(sd['3_texture_1.0']['avg_L2'], 3)}}}")
    n.append(g("nMarginZeroJit") + f"{{{int(round(sd['4_margin_0pct']['jitter'] * 100))}}}"); n.append(g("nMarginFiveJit") + f"{{{int(round(sd['4_margin_5pct']['jitter'] * 100))}}}")
    n.append(g("nMarginZeroHolds") + f"{{{sd['4_margin_0pct']['contexts']}}}"); n.append(g("nMarginFiveL2") + f"{{{f(sd['4_margin_5pct']['avg_L2'], 3)}}}")
    for a, nm in (("hsja", "H"), ("surfree", "S"), ("cgba", "C")):
        for b in (1000, 2000, 5000):
            n.append(g(f"nL{nm}{b}") + f"{{{f(cmp_[a, b]['avg_L2'], 3)}}}")
    open("report/numbers.tex", "w").write("\n".join(n) + "\n")


architecture()
hsja_loop()
grid, meds = l2_vs_queries()
histograms()
examples()
M = flips()
tex_tables(grid, meds, M)
print("wrote", sorted(os.listdir(OUT)), "tables.tex numbers.tex")
