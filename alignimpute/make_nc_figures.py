from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import family_analysis as FA
from .make_nc_tables import REAL, REAL_CANON, ROWS, rows_all

from .style import BLUE, ORANGE, GREEN, GREY, INK, RED
C = {"Alignment": BLUE, "Imputation": ORANGE, "Deep IMVC": GREEN, "No graph alignment": GREY}
MK = {"Alignment": "o", "Imputation": "s", "Deep IMVC": "^", "No graph alignment": "D"}
INK2, GRID = "#52514e", "#e4e3df"
W = 5.4
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix", "pdf.fonttype": 42})


def fig_graphs(B, out):
    D = json.load(open(B / "coobs_real.json"))
    fig, axs = plt.subplots(1, 6, figsize=(7.2, 1.75))
    for ax, (g, name) in zip(axs, REAL):
        d = D[g]; N = np.array(d["coobs"]); V = len(N); n = d["n"]; cov = np.diag(N) / n
        ang = np.pi / 2 - 2 * np.pi * np.arange(V) / V; xy = np.c_[np.cos(ang), np.sin(ang)]
        wmax = max(N[u, v] for u in range(V) for v in range(u + 1, V))
        for u in range(V):
            for v in range(u + 1, V):
                if N[u, v] > 0:
                    ax.plot(*xy[[u, v]].T, color=INK2, lw=0.4 + 2.6 * N[u, v] / wmax, zorder=1, solid_capstyle="round")
        hub = int(np.argmax(cov))
        for v in range(V):
            ax.scatter(*xy[v], s=14 + 70 * cov[v], color=RED if v == hub else "#ffffff", edgecolor=INK, lw=0.8, zorder=3)
            ax.text(*(xy[v] * 1.7), d["views"][v], ha="center", va="center", fontsize=6.5, color=INK2)
        A = (N > 0) & ~np.eye(V, dtype=bool); Dm = np.where(A, 1, np.inf); np.fill_diagonal(Dm, 0)
        for k in range(V): Dm = np.minimum(Dm, Dm[:, [k]] + Dm[[k], :])
        ax.set_title(f"{name}\nhub {cov.max():.2f}, diameter {int(Dm[np.isfinite(Dm)].max())}", fontsize=8, color=INK, pad=7)
        ax.set_xlim(-2.0, 2.0); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(w_pad=0.3); fig.savefig(out / "real_graphs.pdf", bbox_inches="tight"); plt.close(fig)


def fig_real(B, out):
    S = FA.real_settings(B); V = {g: S[("real", g)][0] for g, _ in REAL}
    RAW = FA.real_settings(B, raw=True); rng = np.random.default_rng(0)
    def ci(d):
        v = 100 * np.array(list(d.values()))
        if len(v) < 2: return None
        bs = v[rng.integers(0, len(v), (2000, len(v)))].mean(1); return np.percentile(bs, [2.5, 97.5])
    methods = [(fam, name, v) for fam, rs in ROWS for name, v in rs]
    CI = {}
    for g, _ in REAL_CANON:
        for fam, mname, v in methods:
            if V[g].get(v, float("nan")) == V[g].get(v, float("nan")):
                CI[(g, v)] = ci(RAW[("real", g)][0].get(v, {}))
    fig, axs = plt.subplots(1, 6, figsize=(W, 3.5), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (g, name) in zip(axs, REAL):
        best = max(x for x in V[g].values() if x == x)
        ax.axvline(best, color=GRID, lw=3, zorder=0)
        for yi, (fam, mname, v) in zip(y, methods):
            x = V[g].get(v, float("nan"))
            if x == x:
                c = CI[(g, v)]
                if c is not None: ax.plot(c, [yi, yi], color=C[fam], lw=1.0, zorder=2, solid_capstyle="butt")
                ax.scatter(x, yi, s=16, marker=MK[fam], color=C[fam], edgecolor="#fcfcfb", lw=0.6, zorder=3)
            else:
                ax.text(3, yi, "n/a", fontsize=6.5, color=INK2, ha="left", va="center")
        ax.set_title(name, fontsize=7.5, color=INK); ax.set_xlim(0, 100); ax.set_xticks([0, 50]); ax.axvline(100, color=GRID, lw=0.5); ax.tick_params(labelsize=6.5, length=2)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.tick_params(length=2)
    axs[0].set_yticks(y); axs[0].set_yticklabels([m for _, m, _ in methods], fontsize=6.8)
    fig.supxlabel("NMI (×100); bars: 95% bootstrap interval over seeds; grey band: best method", fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], ls="", marker=MK[f], color=C[f], label=f) for f in C]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.6, 1.03), handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.25); fig.savefig(out / "real_results.pdf", bbox_inches="tight"); plt.close(fig)


def fig_gap(B, out):
    R = [x for x in rows_all(B) if x["ugap"] == x["ugap"] and "hub_cov" in x]
    style = {"real": (RED, "o", "real assembly"), "bench": (BLUE, "s", "benchmark chain / thin edge"), "shift": (GREY, "^", "cohort shift")}
    names = dict(REAL)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    for ax, (key, xlab, title) in zip(axs, (("hub_cov", "hub coverage (largest fraction of\ninstances observing one view)", "(a) hub coverage"),
                                         ("nonedge_frac", "fraction of view pairs\nnever co-observed", "(b) never-co-observed pairs"))):
        ax.axhline(0, color=INK2, lw=0.8)
        for kind in ("shift", "bench", "real"):
            P = [x for x in R if x["kind"] == kind]; col, mk, lab = style[kind]
            ax.scatter([x[key] for x in P], [x["ugap"] for x in P], s=28 if kind == "real" else 16, marker=mk, color=col, edgecolor="#fcfcfb", lw=0.6, label=lab, zorder=3 if kind == "real" else 2)
        ax.set_xlabel(xlab); ax.grid(color=GRID, lw=0.5); ax.set_title(title, fontsize=8, color=INK, loc="left")
    off = {"cortex_stage2": (-4, -10), "cortex_stage2_no3c": (-4, 5), "tcga_legacy9": (-28, 16), "tcga_pancan": (-48, 6), "bmmc_site1": (-22, -16)}
    lab = dict(names, bmmc_site1="PBMC, BMMC")
    for x in R:
        if x["kind"] == "real" and x["name"] in off:
            axs[0].annotate(lab[x["name"]], (x["hub_cov"], x["ugap"]), xytext=off[x["name"]], textcoords="offset points", fontsize=6.5, color=INK, ha="right",
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=2))
    axs[0].set_ylabel("best of three alignment methods\n− kNN imputation (NMI points)")
    axs[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=1.0); fig.savefig(out / "gap_hub.pdf", bbox_inches="tight"); plt.close(fig)


def fig_shortfall(B, out):
    S = FA.real_settings(B); V = {g: S[("real", g)][0] for g, _ in REAL}
    best = {g: max(x for x in V[g].values() if x == x) for g, _ in REAL}
    rows = []
    for fam, rs in ROWS:
        for name, v in rs:
            sf = [best[g] - V[g][v] if V[g].get(v, float("nan")) == V[g].get(v, float("nan")) else np.nan for g, _ in REAL]
            rows.append((name, sf, np.nanmax(sf)))
    rows.sort(key=lambda r: r[2])
    M = np.array([r[1] + [r[2]] for r in rows])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("red", ["#fdf4f3", "#efb4ae", RED, "#6e1414"]); cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(W, 3.5))
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=80, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "n/a" if v != v else f"{v:.1f}", ha="center", va="center", fontsize=7, color="#ffffff" if v == v and v > 40 else INK)
    ax.axvline(M.shape[1] - 1.5, color="#fcfcfb", lw=3)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([n for _, n in REAL] + ["largest"], fontsize=7.5)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.tick_params(length=0); ax.xaxis.tick_top()
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel("shortfall to the best method of each assembly (NMI points; darker = further behind)", fontsize=7.5, color=INK2)
    fig.tight_layout(); fig.savefig(out / "shortfall.pdf", bbox_inches="tight"); plt.close(fig)


def fig_pitfalls(B, out):
    from .results import load, loads, seeds
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.4, 1]})
    taus = [0.1, 0.3, 0.5, 1.0, 2.0]; ax = axs[0]
    graphs = [("pbmc_mimitou", "PBMC", ":"), ("bmmc_site1", "BMMC", "-."), ("cortex_stage2_no3c", "Cortex 5", "-"), ("tcga_legacy9", "TCGA 9", "--")]
    for g, name, ls in graphs:
        hv = []
        for t in taus:
            if g in ("pbmc_mimitou", "bmmc_site1"):
                rows = loads(B / f"mosaic/main_{g}_s*.jsonl") if t == 0.1 else loads(B / f"mosaic/tau_{g}_t{t}_s*.jsonl")
            else:
                rows = load(B / f"opt3/results/real_{g}_t{t}.jsonl")
            hv.append(100 * np.mean(list(seeds(rows, "A").values())))
        ax.plot(taus, hv, ls=ls, color=INK2, lw=1.4, marker="o", ms=3)
        ax.text(taus[-1] * 1.1, hv[-1], name, fontsize=7, va="center", color=INK)
    ax.set_xscale("log"); ax.set_xticks(taus); ax.set_xticklabels([str(t) for t in taus]); ax.set_xlim(0.08, 4.6)
    ax.set_xlabel("temperature τ (the label-free rule selects 0.1)"); ax.set_ylabel("contrastive host, NMI"); ax.grid(color=GRID, lw=0.5)
    ax.set_title("(a) host NMI against temperature", fontsize=8, color=INK, loc="left")
    ax = axs[1]; lin = load(B / "opt3/results/lin_real.jsonl")
    gs = [("cortex_stage2_no3c", "Cortex 5"), ("cortex_stage2", "Cortex 6"), ("tcga_legacy9", "TCGA 9")]
    vals = {}
    for g, _ in gs:
        for std in (True, False):
            vals[(g, std)] = 100 * np.mean([r["nmi"] for r in lin if r["data"] == g and r["std"] == std and not r["whiten"] and r["method"] == "L1" and r["d"] == 32])
    x = np.arange(len(gs)); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, [vals[(g, False)] for g, _ in gs], w, color=C["Alignment"], label="scaled per view")
    b2 = ax.bar(x + w / 2 + 0.01, [vals[(g, True)] for g, _ in gs], w, color="#ffffff", edgecolor=C["Alignment"], hatch="////", lw=0.8, label="standardised per column")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in gs]); ax.set_ylabel("linear graph sync., NMI"); ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02)); ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("(b) standardising PC views", fontsize=8, color=INK, loc="left")
    fig.tight_layout(w_pad=1.5); fig.savefig(out / "pitfalls.pdf", bbox_inches="tight"); plt.close(fig)


def fig_bench(B, out):
    from .results import old_baselines
    S = FA.bench_settings(B); O = old_baselines(B, "nmi")
    rows = [("No graph alignment", [("Unaligned consensus", "PCAcat"), ("Contrastive host", "A")]),
            ("Imputation", [("kNN imputation + k-means", "Kimp"), ("Host on imputed views", "Aimp"), ("RecFormer", "RecFormer")]),
            ("Deep IMVC", [("MVP", "MVP"), ("CPSPAN", "CPSPAN"), ("CPM-Nets", "CPM-Nets"), ("DIMVC", "DIMVC"), ("TreeEIC", "TreeEIC"),
                           ("FreeCSL", "FreeCSL"), ("DVIMC", "DVIMC"), ("GHICMC", "GHICMC")]),
            ("Alignment", [("StabMap", "StabMap"), ("Joint WP, in loop", "Cwp"), ("Host, in-loop sync.", "S"), ("Linear graph sync.", "L1"),
                           ("Linear-sync first assignment", "ALw")])]
    def val(v, d, t):
        if v in O[d]:
            x = O[d][v].get("chain" if t == "chain" else "chain-sweep-0.05", {})
            return 100 * np.mean(list(x.values())) if x else float("nan")
        return S[("bench", f"{d}_{t}")][0].get(v, float("nan"))
    cols = [(d, t, f"{n}, {'chain' if t == 'chain' else 'thin edge'}") for d, n in (("handwritten", "HandWritten"), ("caltech101_7", "Caltech101-7"), ("caltech101_20", "Caltech101-20")) for t in ("chain", "thin05")]
    methods = [(fam, name, v) for fam, rs in rows for name, v in rs]
    fig, axs = plt.subplots(1, 6, figsize=(W, 3.6), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (d, t, title) in zip(axs, cols):
        vals = [val(v, d, t) for _, _, v in methods]; best = max(x for x in vals if x == x)
        ax.axvline(best, color=GRID, lw=3, zorder=0)
        for yi, (fam, _, _), x in zip(y, methods, vals):
            if x == x: ax.scatter(x, yi, s=16, marker=MK[fam], color=C[fam], edgecolor="#fcfcfb", lw=0.6, zorder=3)
            else: ax.text(3, yi, "n/a", fontsize=6.5, color=INK2, ha="left", va="center")
        ax.set_title(title.replace(", ", "\n"), fontsize=7.2, color=INK); ax.set_xlim(0, 90); ax.set_xticks([0, 40, 80])
        ax.grid(axis="x", color=GRID, lw=0.5); ax.tick_params(labelsize=6.5, length=2)
    axs[0].set_yticks(y); axs[0].set_yticklabels([m for _, m, _ in methods], fontsize=6.8)
    fig.supxlabel("NMI (×100), mean over seeds; grey band: best method", fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], ls="", marker=MK[f], color=C[f], label=f) for f in C]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.6, 1.03), handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.25); fig.savefig(out / "bench_results.pdf", bbox_inches="tight"); plt.close(fig)


def fig_star(B, out):
    S0 = json.load(open(B / "star/summary.json")); SH = json.load(open(B / "star_hub/summary.json")); H = json.load(open(B / "star_hub/hub_choice.json"))
    keys = {"HandWritten": "handwritten", "Caltech101-7": "caltech101_7", "Caltech101-20": "caltech101_20"}
    ds = [r["dataset"] for r in S0]
    series = [("chain (no hub)", [r["chain"]["gap"] for r in S0], GREY, None),
              ("star, hub = view 1", [r["star"]["gap"] for r in S0], ORANGE, None),
              ("star, hub predicts the other views", [r["star"]["gap"] for r in SH], BLUE, None)]
    fig, ax = plt.subplots(figsize=(W * 0.8, 2.4))
    x = np.arange(len(ds)); w = 0.26
    for k, (lab, v, col, _) in enumerate(series):
        bars = ax.bar(x + (k - 1) * (w + 0.02), v, w, color=col, label=lab, zorder=2)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + (0.6 if val >= 0 else -0.6), f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=6.5, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{d}\n(predictive hub = view {H[keys[d]]['hub'] + 1})" for d in ds], fontsize=6.8)
    ax.set_ylabel("best alignment − best imputation\n(NMI points)"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_ylim(-4, 21)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "star_gap.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--build", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    for f in (fig_graphs, fig_real, fig_gap, fig_shortfall, fig_pitfalls, fig_bench, fig_star):
        f(a.build, a.out); print("wrote", f.__name__)
