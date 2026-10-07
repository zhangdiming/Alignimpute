from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import family_analysis as FA
from .results import f1, old_baselines

REAL_CANON = [("cortex_stage2", "Cortex 6"), ("cortex_stage2_no3c", "Cortex 5"), ("tcga_legacy9", "TCGA 9"), ("tcga_pancan", "PanCan"),
              ("pbmc_mimitou", "PBMC"), ("bmmc_site1", "BMMC")]
REAL = [("pbmc_mimitou", "PBMC"), ("bmmc_site1", "BMMC"), ("tcga_pancan", "PanCan"), ("cortex_stage2_no3c", "Cortex 5"),
        ("cortex_stage2", "Cortex 6"), ("tcga_legacy9", "TCGA 9")]
ROWS = [("No graph alignment", [("Unaligned consensus", "PCAcat"), ("Contrastive host", "A")]),
        ("Imputation", [("kNN imputation + $k$-means", "Kimp"), ("Host on imputed views", "Aimp"), ("RecFormer", "RecFormer")]),
        ("Deep IMVC", [("FreeCSL", "FreeCSL"), ("DVIMC", "DVIMC"), ("GHICMC", "GHICMC")]),
        ("Alignment", [("StabMap", "StabMap"), ("Joint WP, in loop", "Cwp"), ("Host, in-loop sync.", "S"), ("Linear graph sync.", "L1"),
                       ("Linear-sync first assignment", "ALw")])]
OOM = {("GHICMC", "bmmc_site1")}


def wrap(L):
    return [r"\resizebox{\textwidth}{!}{" + x if x.startswith(r"\begin{tabular}") else (x + "}" if x == r"\end{tabular}" else x) for x in L]


def table_real(B, key):
    S = FA.real_settings(B, key); diag = {g: S[("real", g)][1] for g, _ in REAL}; V = {g: S[("real", g)][0] for g, _ in REAL}
    RAW = FA.real_settings(B, key, raw=True); SD = {g: {v: 100 * np.std(list(d.values()), ddof=1) if len(d) > 1 else float("nan") for v, d in RAW[("real", g)][0].items()} for g, _ in REAL}
    best = {g: max(v for v in V[g].values() if v == v) for g, _ in REAL}
    L = [r"\begin{table}[!htbp]",
         rf"\caption{{Real co-observation graphs ({key.upper()} $\times100$; ten seeds for the methods of our framework (five for the trained hosts on PanCan), five for the published deep IMVC methods; StabMap, linear graph synchronisation and kNN imputation are deterministic up to the $k$-means seed). $\pm$: standard deviation over seeds. Bold: best mean per column; --: not run (joint WP is too slow beyond about 5\,000 instances); OOM: out of memory on an 80\,GB GPU. Bottom: label-free descriptors of each graph.}}",
         rf"\label{{tab:real{key}}}", r"\centering\footnotesize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{l" + "c" * len(REAL) + "}", r"\toprule",
         "Method & " + " & ".join(n for _, n in REAL) + r"\\", r"\midrule"]
    for fam, rows in ROWS:
        L.append(r"\multicolumn{" + str(1 + len(REAL)) + r"}{l}{\textit{" + fam + "}}" + r"\\")
        for i, (name, v) in enumerate(rows):
            cells = []
            for g, _ in REAL:
                x = V[g].get(v, float("nan"))
                if (v, g) in OOM: cells.append("OOM"); continue
                s = "--" if x != x else f1(x)
                if x == x and abs(x - best[g]) < 1e-9: s = r"\textbf{" + s + "}"
                sd = SD[g].get(v, float("nan"))
                if x == x and sd == sd: s += r"{\scriptsize$\pm$" + f"{sd:.1f}" + "}"
                cells.append(s)
            L.append(r"\hspace{0.6em}" + name + " & " + " & ".join(cells) + r"\\")
        L.append(r"\midrule")
    for name, k, fmt in (("hub coverage", "hub_cov", "{:.2f}"), ("diameter", "diameter", "{:.0f}"), ("imputability", "imputability", "{:.2f}")):
        L.append(r"\textit{" + name + "} & " + " & ".join(fmt.format(diag[g][k]) for g, _ in REAL) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


UALIGN = ("L1", "ALw", "S")


def rows_all(B):
    S = {**FA.real_settings(B), **FA.bench_settings(B), **FA.shift_settings(B)}
    out = []
    for (kind, name), (r, dg) in S.items():
        ba, bi = FA.best(r, FA.ALIGN), FA.best(r, FA.IMPUTE)
        ugap = max(r[k] for k in UALIGN) - r["Kimp"]
        out.append(dict(kind=kind, name=name, gap=ba[0] - bi[0], ugap=ugap, ba=ba, bi=bi, r=r, **dg))
    return out


NAMES = {"L1": "linear sync.", "StabMap": "StabMap", "ALw": "lin.-sync. first assignment", "S": "host, in-loop sync.", "Cwp": "joint WP", "A2": "joint WP (post hoc)",
         "Kimp": "kNN imputation", "Aimp": "host on imputed views", "RecFormer": "RecFormer"}
DSN = {"handwritten": "HandWritten", "caltech101_7": "Caltech101-7", "caltech101_20": "Caltech101-20"}


def table_gap(B):
    order = {g: i for i, (g, _) in enumerate(REAL)}
    R = sorted([x for x in rows_all(B) if x["kind"] == "real"], key=lambda x: order[x["name"]]) + [x for x in rows_all(B) if x["kind"] == "bench"]
    L = [r"\begin{table}[!htbp]",
         r"\caption{Best alignment method against best imputation method (NMI $\times100$) on the real graphs and the benchmark chains and 5\% thin edges, with label-free descriptors of each setting. Gap: alignment minus imputation.}",
         r"\label{tab:gap}", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l l c l c r c c c}", r"\toprule",
         r"Setting & best alignment & & best imputation & & gap & hub & diam. & imput.\\", r"\midrule"]
    names = dict(REAL)
    for x in R:
        if x["kind"] == "real":
            nm = names[x["name"]]
        else:
            d, t = x["name"].rsplit("_", 1); nm = DSN[d] + (" chain" if t == "chain" else " thin")
        L.append(f"{nm} & {NAMES[x['ba'][1]]} & {x['ba'][0]:.1f} & {NAMES[x['bi'][1]]} & {x['bi'][0]:.1f} & {x['gap']:+.1f} & {x['hub_cov']:.2f} & {x['diameter']:.0f} & {x['imputability']:.2f}" + r"\\")
        if x["kind"] == "real" and x["name"] == REAL[-1][0]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return wrap(L)


def stats(B):
    R = rows_all(B); out = []
    def macro(name, val): out.append(rf"\newcommand{{\{name}}}{{{val}}}")
    groups = {"Real": (("real",), "gap"), "RealBench": (("real", "bench"), "gap"), "Uni": (("real", "bench", "shift"), "ugap")}
    for gname, (kinds, gk) in groups.items():
        X = [x for x in R if x["kind"] in kinds and x[gk] == x[gk] and "imputability" in x]
        macro(f"n{gname}", len(X))
        for k, kn in (("imputability", "Imp"), ("hub_cov", "Hub"), ("diameter", "Diam"), ("nonedge_frac", "Nonedge"), ("lambda2", "Lam")):
            rho, p = spearmanr([x[k] for x in X], [x[gk] for x in X])
            macro(f"rho{kn}{gname}", f"{rho:+.2f}"); macro(f"p{kn}{gname}", f"={p:.2g}" if p >= 1e-3 else r"<10^{-3}")
        imp = np.array([x["imputability"] for x in X]); win = np.array([x[gk] < 0 for x in X]); c = 0
        for i in range(len(X)):
            m = np.ones(len(X), bool); m[i] = False; cand = np.unique(imp[m])
            t = cand[int(np.argmax([np.mean((imp[m] > tt) == win[m]) for tt in cand]))]; c += (imp[i] > t) == win[i]
        macro(f"loo{gname}", f"{c}/{len(X)}"); macro(f"maj{gname}", f"{max(win.mean(), 1 - win.mean()):.2f}")
    RAW = FA.real_settings(B, "nmi", raw=True); gs = [g for g, _ in REAL_CANON]
    meths = [v for _, rs in ROWS for _, v in rs]
    nb = 2000
    def boot(paired):
        rng = np.random.default_rng(0); mx = {v: [] for v in meths}; l1best = 0; boot.fallback = 0; boot.fbres = 0
        for _ in range(nb):
            M = {}; fb = 0
            for g in gs:
                M[g] = {}; common = rng.integers(0, 10, 10)
                for v in meths:
                    d = RAW[("real", g)][0].get(v, {})
                    if not d: continue
                    if paired:
                        pick = [d[k] for k in common if k in d]
                        if not pick:
                            ks = list(d); pick = [d[ks[i]] for i in rng.integers(0, len(ks), len(ks))]; fb += 1
                    else:
                        vals = list(d.values()); pick = [vals[i] for i in rng.integers(0, len(vals), len(vals))]
                    M[g][v] = 100 * float(np.mean(pick))
            boot.fallback += fb; boot.fbres += fb > 0
            bests = {g: max(M[g].values()) for g in gs}
            cur = {v: max(bests[g] - M[g][v] for g in gs if v in M[g]) for v in meths if any(v in M[g] for g in gs)}
            for v, c in cur.items(): mx[v].append(c)
            l1best += min(cur, key=cur.get) == "L1"
        return mx, l1best
    for tag, paired in (("", True), ("Ind", False)):
        mx, l1best = boot(paired)
        lo, hi = np.percentile(mx["L1"], [2.5, 97.5]); macro(f"lOneMaxLo{tag}", f"{lo:.1f}"); macro(f"lOneMaxHi{tag}", f"{hi:.1f}")
        macro(f"lOneMostRobust{tag}", f"{100 * l1best / nb:.0f}")
        if paired: macro("bootFallbackRes", boot.fbres); macro("bootFallback", boot.fallback); macro("bootN", f"{nb:,}".replace(",", r"\,"))
        lo2 = min(np.percentile(mx[v], 2.5) for v in meths if v != "L1" and mx[v]); macro(f"otherMaxLo{tag}", f"{lo2:.1f}")
    rng = np.random.default_rng(0)
    def src(x):
        n = x["name"]
        if x["kind"] == "real":
            return "cortex" if n.startswith("cortex") else n
        return n.split("_chain")[0].split("_thin")[0]
    def merged(x):
        g = src(x); return "caltech101" if g.startswith("caltech101") else ("tcga" if g.startswith("tcga") else g)
    for gname, (kinds, gk, sf) in {"RealBench": (("real", "bench"), "gap", src), "Uni": (("real", "bench", "shift"), "ugap", src),
                                   "RealBenchM": (("real", "bench"), "gap", merged), "UniM": (("real", "bench", "shift"), "ugap", merged)}.items():
        X = [x for x in R if x["kind"] in kinds and x[gk] == x[gk] and "hub_cov" in x]
        groups = sorted({sf(x) for x in X}); macro(f"nGroups{gname}", len(groups))
        byg = {gg: [x for x in X if sf(x) == gg] for gg in groups}
        for k, kn in (("hub_cov", "Hub"), ("nonedge_frac", "Nonedge"), ("diameter", "Diam")):
            rs = []
            for _ in range(nb):
                pick = [x for gg in rng.choice(groups, len(groups)) for x in byg[gg]]
                vals = [x[k] for x in pick]
                if len(set(vals)) > 1: rs.append(spearmanr(vals, [x[gk] for x in pick])[0])
            lo, hi = np.nanpercentile(rs, [2.5, 97.5]); macro(f"ci{kn}{gname}", f"$[{lo:+.2f}, {hi:+.2f}]$")
    M = aw_means(B); rg = {d: max(m) - min(m) for d, m in M.items()}
    macro("awRangeTcga", f"{rg['tcga_legacy9']:.2f}"); macro("awRangeHW", f"{rg['handwritten']:.1f}")
    macro("awRangeOther", f"{max(v for d, v in rg.items() if d not in ('tcga_legacy9', 'handwritten')):.1f}")
    X = [x for x in R if x["kind"] in ("real", "bench") and x["gap"] == x["gap"]]
    lo = [x for x in X if x["hub_cov"] < 0.9]; hi = [x for x in X if x["hub_cov"] >= 0.9]
    macro("nLowHub", len(lo)); macro("nHighHub", len(hi))
    macro("hubLowMax", f"{max(x['hub_cov'] for x in lo):.2f}"); macro("hubHighMin", f"{min(x['hub_cov'] for x in hi):.2f}")
    macro("gapLowLo", f"{min(x['gap'] for x in lo):.1f}"); macro("gapLowHi", f"{max(x['gap'] for x in lo):.1f}")
    macro("gapHighLo", f"${min(x['gap'] for x in hi):+.1f}$"); macro("gapHighHi", f"${max(x['gap'] for x in hi):+.1f}$")
    _, SH, H = star_rows(B)
    drops = [r["gap_drop"] for r in SH]; ps = [r["p"] for r in SH]; gaps = {r["dataset"]: r["star"]["gap"] for r in SH}
    macro("starDropLo", f"{min(drops):.1f}"); macro("starDropHi", f"{max(drops):.1f}"); macro("starPMax", f"{max(ps):.3f}")
    macro("starGapHW", f"${gaps['HandWritten']:+.1f}$"); macro("starGapCalLo", f"{min(gaps['Caltech101-7'], gaps['Caltech101-20']):.1f}")
    macro("starGapCalHi", f"{max(gaps['Caltech101-7'], gaps['Caltech101-20']):.1f}")
    S0 = json.load(open(B / "star/summary.json")); g0 = {r["dataset"]: r["star"]["gap"] for r in S0}
    macro("starWeakCalLo", f"{min(g0['Caltech101-7'], g0['Caltech101-20']):.1f}"); macro("starWeakCalHi", f"{max(g0['Caltech101-7'], g0['Caltech101-20']):.1f}")
    O = obs_sens(B)
    macro("obsMaxChange", f"{O['max_change']:.1f}"); macro("obsGapLo", f"${O['gap_lo']:+.1f}$"); macro("obsGapHi", f"${O['gap_hi']:+.1f}$")
    macro("obsSignKept", f"{O['sign_kept']}"); macro("obsCells", f"{O['cells']}")
    macro("groupList", ", ".join(sorted({src(x) for x in R})).replace("_", r"\_"))
    sh = [x for x in R if x["kind"] == "shift"]
    macro("shiftAlignWins", sum(x["gap"] > 0 for x in sh)); macro("shiftN", len(sh)); macro("shiftGap", f"{np.mean([x['gap'] for x in sh]):+.1f}")
    return out


def table_regret(B, key="nmi"):
    S = FA.real_settings(B, key); V = {g: S[("real", g)][0] for g, _ in REAL}
    best = {g: max(v for v in V[g].values() if v == v) for g, _ in REAL}
    res = []
    for fam, rows in ROWS:
        for name, v in rows:
            r = [best[g] - V[g][v] for g, _ in REAL if V[g].get(v, float("nan")) == V[g].get(v, float("nan"))]
            res.append((fam, name, v, len(r), float(np.mean(r)) if r else float("nan"), float(np.max(r)) if r else float("nan")))
    L = [r"\begin{table}[!htbp]",
         rf"\caption{{Shortfall to the best method of each real graph ({key.upper()} $\times100$): mean and largest over the real graphs on which a method ran (of six).}}",
         rf"\label{{tab:regret}}", r"\centering\small", r"\begin{tabular}{l c c c}", r"\toprule", r"Method & graphs & mean shortfall & largest shortfall\\", r"\midrule"]
    for fam, name, v, n, mu, mx in sorted(res, key=lambda t: t[5]):
        L.append(f"{name} & {n} & {mu:.1f} & {mx:.1f}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L, res


def table_benchfam(B, key="nmi"):
    S = FA.bench_settings(B, key); O = old_baselines(B, key)
    cols = [(d, t) for d in ("handwritten", "caltech101_7", "caltech101_20") for t in ("chain", "thin05")]
    rows = [("No graph alignment", [("Unaligned consensus", "PCAcat"), ("Contrastive host", "A")]),
            ("Imputation", [("kNN imputation + $k$-means", "Kimp"), ("Host on imputed views", "Aimp"), ("RecFormer", "RecFormer")]),
            ("Deep IMVC", [("MVP", "MVP"), ("CPSPAN", "CPSPAN"), ("CPM-Nets", "CPM-Nets"), ("DIMVC", "DIMVC"), ("TreeEIC", "TreeEIC"),
                           ("FreeCSL", "FreeCSL"), ("DVIMC", "DVIMC"), ("GHICMC", "GHICMC")]),
            ("Alignment", [("StabMap", "StabMap"), ("Joint WP, in loop", "Cwp"), ("Host, in-loop sync.", "S"), ("Linear graph sync.", "L1"),
                           ("Linear-sync first assignment", "ALw")])]
    def val(v, d, t):
        if v in O[d]:
            x = O[d][v].get("chain" if t == "chain" else "chain-sweep-0.05", {})
            return 100 * np.mean(list(x.values())) if x else float("nan")
        return S[("bench", f"{d}_{t}")][0].get(v, float("nan"))
    best = {c: max(val(v, *c) for _, rs in rows for _, v in rs if val(v, *c) == val(v, *c)) for c in cols}
    L = [r"\begin{table}[!htbp]",
         rf"\caption{{Benchmarks under cohort missingness: chain and chain with a 5\% thin edge ({key.upper()} $\times100$; ten seeds for our framework, StabMap, kNN imputation and the host on imputed views; five for the published methods, RecFormer included). Every method withholds the features of unobserved views. Bold: best per column; --: no result.}}",
         rf"\label{{tab:bench{key}}}", r"\centering\small", r"\begin{tabular}{ll cc cc cc}", r"\toprule",
         r" & & \multicolumn{2}{c}{HandWritten} & \multicolumn{2}{c}{Caltech101-7} & \multicolumn{2}{c}{Caltech101-20}\\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}", r"Family & Method & chain & thin & chain & thin & chain & thin\\", r"\midrule"]
    for fam, rs in rows:
        for i, (name, v) in enumerate(rs):
            cells = []
            for c in cols:
                x = val(v, *c); s = "--" if x != x else f1(x)
                if x == x and abs(x - best[c]) < 1e-9: s = r"\textbf{" + s + "}"
                cells.append(s)
            L.append((fam if i == 0 else "") + " & " + name + " & " + " & ".join(cells) + r"\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    return wrap(L)


SHIFT_M = [("L1", "Linear graph sync."), ("ALw", "Linear-sync first assignment"), ("CLw", "Lin.-sync first assignment, in-loop sync."), ("S", "Host, in-loop sync."),
           ("B", "Host, post-hoc sync."), ("A", "Contrastive host"), ("Apl", "Host, deferred first assignment"), ("Anp", "Host, no prototypes"),
           ("Cwp", "Joint WP, in loop"), ("Cwpnp", "Joint WP, in loop, no proto."), ("A2", "Joint WP, post hoc"), ("A2np", "Joint WP, post hoc, no proto."),
           ("Kimp", "kNN imputation + $k$-means"), ("PCAcat", "Unaligned consensus")]


def shift_cells(B, key):
    from .results import load as L_, loads as Ls_, seeds as sd_
    cells = []
    for d in ("handwritten", "caltech101_7", "caltech101_20", "outdoorscene", "aloi"):
        for top, topo, tt in (("chain", "chain", "chain"), ("thin", "chain-sweep-0.05", "thin05")):
            for lv in ("a1", "a03", "s05", "s1", "b03"):
                R = L_(B / f"st2/shift_{d}_{lv}_{top}.jsonl"); W = Ls_(B / f"st2/shiftwp_{d}_{lv}_{top}_s*.jsonl")
                A2 = [r for r in L_(B / f"st2/a2_shift_{d}_{lv}.jsonl") if r.get("topology") == topo and r.get("variant") == "alaux_A2"]
                K = [r for r in L_(B / f"st2/shiftk_{d}_{lv}.jsonl") if r.get("topology") == topo]
                PH = [r for r in L_(B / "st2/posthoc_shift.jsonl") if r["dataset"] == d and r["level"] == lv and r["topology"] == topo]
                N = Ls_(B / f"ifE/shift_{d}_{lv}_{top}_s*.jsonl"); LS = L_(B / f"linshift/{d}_{tt}_{lv}.jsonl")
                c = {"ALw": sd_(N, "ALw", key), "CLw": sd_(N, "CLw", key), "S": sd_(R, "C" if top == "chain" else "Cthin", key),
                     "A": sd_(R, "A", key), "Anp": sd_(R, "Anp", key), "Apl": sd_(R, "Apl", key), "Cwp": sd_(W, "Cwp", key), "Cwpnp": sd_(W, "Cwpnp", key),
                     "A2": {r["seed"]: r[key] for r in A2 if r.get("host") == "A"}, "A2np": {r["seed"]: r[key] for r in A2 if r.get("host") == "Anp"},
                     "Kimp": {r["seed"]: r[key] for r in K}, "B": {r["seed"]: r[key] for r in PH},
                     "L1": {r["seed"]: r[key] for r in LS if r["method"] == "L1"}, "PCAcat": {r["seed"]: r[key] for r in LS if r["method"] == "PCAcat"}}
                cells.append(c)
    return [c for c in cells if all(len(c[k]) >= 10 for k, _ in SHIFT_M)]


def table_shift(B):
    from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon
    out = {}
    for key in ("nmi", "acc"):
        cells = shift_cells(B, key)
        M = np.array([[100 * np.mean(list(c[k].values())) for k, _ in SHIFT_M] for c in cells])
        rk = np.array([rankdata(-r) for r in M]).mean(0); _, p = friedmanchisquare(*M.T)
        k = len(SHIFT_M); cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * len(cells)))
        best = M.max(1); vs = []
        for i, (v, _) in enumerate(SHIFT_M):
            a = b = 0
            if v != "L1":
                for c in cells:
                    s_ = sorted(set(c["L1"]) & set(c[v])); g = np.array([c["L1"][j] - c[v][j] for j in s_])
                    pp = wilcoxon(g).pvalue if np.any(g) else 1.0; a += g.mean() > 0 and pp < 0.05; b += g.mean() < 0 and pp < 0.05
            vs.append((a, b))
        out[key] = dict(n=len(cells), rk=rk, p=p, cd=cd, mean=M.mean(0), within=[int(np.sum(best - M[:, i] <= 2)) for i in range(k)], worst=[(best - M[:, i]).max() for i in range(k)], vs=vs)
    sn, sa = out["nmi"], out["acc"]
    L = [r"\begin{table}[!htbp]",
         rf"\caption{{Cohort shift: {len(SHIFT_M)} methods over {sn['n']} settings (five datasets; chain and 5\% thin edge; class-composition, cohort-size and batch shifts), ten seeds each. Mean NMI rank (1 = best), mean NMI, settings within two points of the best method ($\le$2), largest gap to the best method (worst), settings in which linear graph synchronisation is significantly better / worse (L$\gtrless$; Wilcoxon, uncorrected), and mean ACC rank. Nemenyi critical difference {sn['cd']:.2f}; Friedman $p<10^{{-10}}$.}}",
         r"\label{tab:shift}", r"\centering\footnotesize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{l cccccc}", r"\toprule",
         r"Method & rank & mean & $\le$2 & worst & L$\gtrless$ & rank ACC\\", r"\midrule"]
    for i in np.argsort(sn["rk"]):
        v, name = SHIFT_M[i]
        L.append(f"{name} & {sn['rk'][i]:.2f} & {sn['mean'][i]:.1f} & {sn['within'][i]} & {sn['worst'][i]:.1f} & " + ("--" if v == "L1" else f"{sn['vs'][i][0]}/{sn['vs'][i][1]}") + f" & {sa['rk'][i]:.2f}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


def table_grid(B, tops, label, part):
    from .results import load as L_, loads as Ls_, seeds as sd_
    from .results import DS8, TOPS
    names = dict(TOPS)
    L = [r"\begin{table}[!htbp]",
         rf"\caption{{Topology grid, part {part} (NMI $\times100$, five seeds). H: contrastive host; H$_s$: host with in-loop synchronisation (thin-edge primitive on the thin edge); L: host with a linear-sync first assignment. PanCanAtlas without per-column standardisation (its views are principal-component scores); the other datasets standardised per column. Bold: best of the three.}}",
         rf"\label{{{label}}}", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l " + "ccc " * len(tops) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{names[t]}}}" for t in tops) + r"\\",
         "".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(tops))), "Dataset & " + " & ".join(["H & H$_s$ & L"] * len(tops)) + r"\\", r"\midrule"]
    for d, dn in DS8:
        cells = []
        for t in tops:
            if d == "tcga_pancan" and t == "thin05":
                cells += ["--"] * 3; continue
            new = Ls_(B / f"ifE/grid_{d}_{t}_s*.jsonl"); old = new if d == "tcga_pancan" else L_(B / f"st1/grid_{d}_{t}.jsonl")
            vals = [100 * np.mean(list(sd_(old, "A").values())), 100 * np.mean(list(sd_(old, "Cthin" if t == "thin05" else "C").values())), 100 * np.mean(list(sd_(new, "ALw").values()))]
            bm = max(vals); cells += [(r"\textbf{" + f1(x) + "}") if abs(x - bm) < 1e-9 else f1(x) for x in vals]
        L.append(dn + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


def obs_sens(B):
    R = json.load(open(B / "obs_sens/summary.json"))
    ch = [abs(r[m][2]) for r in R for m in ("A", "S", "ALw", "Aimp", "Kimp")]
    gd = [r["gap_obs"] - r["gap_all"] for r in R]
    return {"max_change": max(ch), "gap_lo": min(gd), "gap_hi": max(gd), "cells": len(R),
            "sign_kept": sum((r["gap_obs"] > 0) == (r["gap_all"] > 0) for r in R)}


STARM = {"S": "host, sync.", "ALw": "lin.-sync. host", "L1": "linear sync.", "Kimp": "kNN imp.", "Aimp": "host, imputed"}


def star_rows(B):
    S0 = json.load(open(B / "star/summary.json")); SH = json.load(open(B / "star_hub/summary.json")); H = json.load(open(B / "star_hub/hub_choice.json"))
    keys = {"HandWritten": "handwritten", "Caltech101-7": "caltech101_7", "Caltech101-20": "caltech101_20"}
    rows = []
    for r0, rh in zip(S0, SH):
        assert r0["dataset"] == rh["dataset"]
        d = keys[r0["dataset"]]
        rows.append((r0["dataset"], "chain", r0["chain"], None, None))
        rows.append((r0["dataset"], "star, hub 1", r0["star"], r0["gap_drop"], r0["p"]))
        rows.append((r0["dataset"], f"star, hub {H[d]['hub'] + 1}", rh["star"], rh["gap_drop"], rh["p"]))
    return rows, SH, H


def table_star(B):
    rows, _, _ = star_rows(B)
    L = [r"\begin{table}[!htbp]",
         r"\caption{Controlled topology on the same benchmarks (NMI $\times100$, ten seeds, one GPU): every instance observes two views, arranged as a chain or as a star whose hub view is observed on every instance. The hub is view 1 of the files or the view that best predicts the other views (label-free, \ref{app:baselines}). Alignment and imputation: the best method of each family. Host, sync.: host with in-loop synchronisation; lin.-sync. host: host with a linear-sync first assignment; kNN imp.: kNN imputation; host, imputed: host on imputed views. Gap: best alignment minus best imputation; change: gap of the chain minus gap of the star, paired over seeds (Wilcoxon $p$).}",
         r"\label{tab:star}", r"\centering\footnotesize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{l l l c l c r r}", r"\toprule",
         r"Dataset & Topology & alignment & & imputation & & gap & change ($p$)\\", r"\midrule"]
    last = None
    for ds, top, t, drop, p in rows:
        if last is not None and ds != last: L.append(r"\midrule")
        ch = "" if drop is None else f"${drop:+.1f}$ ({p:.3f})"
        L.append(f"{ds if ds != last else ''} & {top} & {STARM[t['best_align']]} & {t['mean'][t['best_align']]:.1f} & {STARM[t['best_impute']]} & {t['mean'][t['best_impute']]:.1f} & ${t['gap']:+.1f}$ & {ch}" + r"\\")
        last = ds
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


AW = (5, 10, 20, 40, 80)
AWDS = [("tcga_legacy9", "TCGA 9 (natural graph)"), ("handwritten", "HandWritten (thin chain)"), ("caltech101_7", "Caltech101-7"), ("caltech101_20", "Caltech101-20"),
        ("proteinfold", "ProteinFold")]


def aw_means(B):
    R = defaultdict(list)
    for f in glob.glob(str(B / "aw/*.jsonl")):
        for r in map(json.loads, open(f)):
            assert not r["whiten"] and r["method"] == "L1"
            R[(r["data"], int(r["tag"][2:]))].append(100 * r["nmi"])
    for k, v in R.items(): assert len(v) == 10, k
    return {d: [float(np.mean(R[(d, w)])) for w in AW] for d, _ in AWDS}


def table_aw(B):
    M = aw_means(B)
    L = [r"\begin{table}[!htbp]",
         r"\caption{Linear graph synchronisation with other anchor weights of the thin-edge primitive (NMI $\times100$, ten seeds): TCGA 9 on its natural graph (two edges with fewer than $d$ co-observed tumours) and the benchmarks whose chain with a 5\% thin edge leaves fewer than $d$ co-observed pairs on that edge (4 to 30). The weight acts only on such edges, so the other five real settings, and OutdoorScene, NUS-WIDE and ALOI, whose thin edge keeps at least $d$ pairs, do not depend on it. 20 is the value used throughout, chosen with labels on the HandWritten chain.}",
         r"\label{tab:aw}", r"\centering\small", r"\begin{tabular}{l ccccc c}", r"\toprule",
         "Anchor weight & " + " & ".join(map(str, AW)) + r" & range\\", r"\midrule"]
    for d, dn in AWDS:
        m = M[d]; L.append(dn + " & " + " & ".join(f"{x:.1f}" for x in m) + f" & {max(m) - min(m):.1f}" + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", type=Path, required=True); ap.add_argument("--sections", type=Path, required=True)
    a = ap.parse_args()
    for fn, L in (("table_real_nmi.tex", table_real(a.build, "nmi")), ("table_real_acc.tex", table_real(a.build, "acc")),
                  ("table_gap.tex", table_gap(a.build)), ("table_benchfam.tex", table_benchfam(a.build)), ("table_benchfam_acc.tex", table_benchfam(a.build, "acc")),
                  ("table_regret.tex", table_regret(a.build)[0]), ("table_grid1.tex", table_grid(a.build, ["complete", "random-2", "star", "ring"], "tab:grid", "1")), ("table_grid2.tex", table_grid(a.build, ["chain", "two-comp", "thin05"], "tab:grid2", "2")), ("table_shift.tex", table_shift(a.build)), ("stats.tex", stats(a.build)), ("table_aw.tex", table_aw(a.build)), ("table_star.tex", table_star(a.build))):
        (a.sections / fn).write_text("\n".join(L) + "\n"); print("wrote", fn)


if __name__ == "__main__":
    main()
