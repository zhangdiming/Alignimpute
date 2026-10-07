from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .style import BLUE, GREY, INK, RED

HW_VIEWS = ["profile\ncorr.", "Fourier", "Karhunen-\nLo\u00e8ve", "morpho-\nlogical", "pixels", "Zernike"]
CX_VIEWS = ["electro-\nphysiology", "morpho-\nlogy", "RNA", "GpC\naccess.", "CH\nmethyl.", "3C"]
CX_SRC = [("Patch-seq", 1), ("snmCAT-seq", 0), ("snm3C-seq", 2)]


def spark(ax, v, color=INK):
    v = np.asarray(v, float)[:48]
    ax.plot(np.arange(len(v)), v, lw=0.7, color=color)
    ax.set_xlim(-1, len(v)); lo, hi = np.percentile(v, [2, 98]); ax.set_ylim(lo - 0.2 * (hi - lo + 1e-6), hi + 0.2 * (hi - lo + 1e-6))


def cell(fig, x, y, w, h, filled):
    ax = fig.add_axes([x, y, w, h]); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(INK if filled else "#c9c8c3"); s.set_linewidth(0.6); s.set_linestyle("-" if filled else (0, (2, 2)))
    return ax


def graph(fig, x0, y0, h, n, far, labels, w=0.06, fs=7.0, ms=13, xr=1.4):
    ax = fig.add_axes([x0, y0, w, h]); ax.axis("off"); ax.set_xlim(-0.8, xr); ax.set_ylim(-0.6, n - 0.4)
    ys = np.arange(n)[::-1]
    for i in range(n - 1):
        ax.plot([0, 0], [ys[i], ys[i + 1]], color=INK, lw=1.0)
    a, b = far
    ax.annotate("", xy=(0.15, ys[b]), xytext=(0.15, ys[a]), arrowprops=dict(arrowstyle="-", color=RED, lw=1.0, ls=(0, (3, 2)), connectionstyle="arc3,rad=-0.5"))
    for i in range(n):
        ax.plot(0, ys[i], "o", ms=ms, mfc="white", mec=INK, mew=0.8); ax.text(0, ys[i], labels[i], ha="center", va="center", fontsize=fs)
    return ax


W, H = 5.4, 2.3
FT, FH, FR = 8.5, 7.0, 7.0
HW_SHORT = ["fac", "fou", "kar", "mor", "pix", "zer"]
CX_SHORT = ["ephys", "morph", "RNA", "GpC", "mCH", "3C"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); B = a.build
    hw = np.load(B / "hw_data.npz"); em = np.load(B / "hw_chain_s0_emb.npz"); mask = em["mask"].astype(bool); y = hw["y"]
    X = [hw[f"X{i}"] for i in range(6)]; Xs = [(v - v.mean(0)) / (v.std(0) + 1e-8) for v in X]
    cx = np.load(B / "cortex_stage2_data.npz"); cm = cx["mask"].astype(bool); src = cx["source"]; cy = cx["y"]
    C3 = np.load(B / "fig_3c.npz")
    fig = plt.figure(figsize=(W, H))
    gx, gy, cw, ch = 0.1, 0.03, 0.052, 0.152
    fig.text(0.0, 0.995, "(a) HandWritten, chain of cohorts", fontsize=FT, va="top", color=INK)
    for j, nm in enumerate(HW_SHORT):
        fig.text(gx + (j + 0.5) * cw, gy + 5 * ch + 0.015, nm, ha="center", va="bottom", fontsize=FH, color=INK)
    rng = np.random.default_rng(0)
    for r in range(5):
        rows = np.where(mask[:, r] & mask[:, r + 1] & (y == 3))[0]; i = rng.choice(rows)
        fig.text(gx - 0.008, gy + (4 - r + 0.5) * ch, f"cohort {r + 1}", ha="right", va="center", fontsize=FR, color=INK)
        for j in range(6):
            filled = j in (r, r + 1)
            ax = cell(fig, gx + j * cw + 0.003, gy + (4 - r) * ch + 0.01, cw - 0.006, ch - 0.02, filled)
            if not filled: continue
            if j == 4: ax.imshow(X[4][i].reshape(16, 15), cmap="Greys", interpolation="nearest", aspect="auto")
            else: spark(ax, Xs[j][i])
    graph(fig, gx + 6 * cw + 0.004, gy, 5 * ch, 6, (0, 5), [str(k) for k in range(1, 7)], w=0.06, fs=6.5, ms=10, xr=2.2)
    bx, by, bw, bh = 0.618, 0.03, 0.048, 0.253
    fig.text(0.49, 0.995, "(b) Human-cortex mosaic", fontsize=FT, va="top", color=INK)
    for j, nm in enumerate(CX_SHORT):
        fig.text(bx + (j + 0.5) * bw, by + 3 * bh + 0.015, nm, ha="center", va="bottom", fontsize=6.5, color=INK)
    for r, (nm, s_) in enumerate(CX_SRC):
        fig.text(bx - 0.008, by + (2 - r + 0.5) * bh, nm, ha="right", va="center", fontsize=FR, color=INK)
        cand = np.where((src == s_) & (cy == 0))[0]
        if s_ == 1: cand = cand[cm[cand, 1]] if cm[cand, 1].any() else cand
        i = cand[0]
        for j in range(6):
            filled = bool(cm[:, j][src == s_].any())
            ax = cell(fig, bx + j * bw + 0.003, by + (2 - r) * bh + 0.012, bw - 0.006, bh - 0.024, filled)
            if not filled: continue
            if j == 5:
                M = np.log1p(C3["Pvalb"]); ax.imshow(M, cmap="Reds", interpolation="nearest", aspect="auto", vmin=0, vmax=np.percentile(M[M > 0], 95))
            elif cm[i, j]: spark(ax, cx[f"X{j}"][i])
    graph(fig, bx + 6 * bw + 0.004, by, 3 * bh, 4, (0, 3), ["E", "R", "M", "3C"], w=0.06, fs=6.5, ms=10, xr=2.2)
    a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, bbox_inches="tight"); print("saved", a.out)


if __name__ == "__main__":
    main()
