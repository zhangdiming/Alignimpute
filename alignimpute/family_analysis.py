from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .results import ext_seeds, load, loads, seeds

ALIGN = ["L1", "StabMap", "ALw", "S", "Cwp", "A2"]
IMPUTE = ["Kimp", "Aimp", "RecFormer"]


def mean(xs):
    xs = list(xs.values()) if isinstance(xs, dict) else list(xs)
    return 100 * float(np.mean(xs)) if xs else float("nan")


def real_settings(B, key="nmi", raw=False):
    agg = (lambda x: dict(x)) if raw else mean
    bys = lambda rows: {x["seed"]: x[key] for x in rows}
    out = {}
    names = ["cortex_stage2", "cortex_stage2_no3c", "tcga_legacy9", "pbmc_mimitou", "bmmc_site1", "tcga_pancan"]
    for g in names:
        r = {}
        if g in ("pbmc_mimitou", "bmmc_site1"):
            R = loads(B / f"mosaic/main_{g}_s*.jsonl")
            lin = load(B / f"mosaic/lin_{g}.jsonl"); st = load(B / f"mosaic/stabmap_{g}.jsonl")
        else:
            R = loads(B / f"ifE/real_{g}_s*.jsonl") + loads(B / f"ifE/realwp_{g}_s*.jsonl")
            lin = load(B / ("ifE/lin_pancan.jsonl" if g == "tcga_pancan" else "opt3/results/lin_real.jsonl"))
            lin = [x for x in lin if x.get("data", g) == g and not x.get("std", False) and not x.get("whiten", False) and x.get("d", 32) == 32]
            st = [x for x in load(B / "opt3/results/stabmap_noscale.jsonl") if x["data"].endswith(g + ".npz")]
        for v in ("A", "ALw", "Aimp", "Cwp"):
            r[v] = agg(seeds(R, v, key))
        r["S"] = agg(seeds(R, "C", key))
        r["L1"] = agg(bys([x for x in lin if x["method"] == "L1"])); r["PCAcat"] = agg(bys([x for x in lin if x["method"] == "PCAcat"]))
        if g == "tcga_pancan":
            st = load(B / "bench/stabmap_tcga_pancan.jsonl")
            R2 = loads(B / "ifE/real_tcga_pancan_aimp_s*.jsonl"); r["Aimp"] = agg(seeds(R2, "Aimp", key))
        r["StabMap"] = agg(bys([x for x in st if x["method"] == "StabMap[all_noscale]" and x["kmeans_on"] == "l2"]))
        r["Kimp"] = agg(bys(load(B / f"bench/kimp_real_{g}.jsonl")))
        for e in ("RecFormer", "FreeCSL", "DVIMC", "GHICMC"):
            x = ext_seeds(B, e, g, "natural", key)
            if not x:
                x = {rr["seed"]: rr[key] for f in glob.glob(str(B / f"ifB/ifB_{e}_mosaic*.jsonl")) for rr in load(f) if rr.get("dataset") == g and rr.get("status") == "ok"}
            r[e] = agg(x)
        d = load(B / f"diag/real_{g}.jsonl")
        out[("real", g)] = (r, d[0] if d else {})
    return out


def bench_settings(B, key="nmi"):
    out = {}
    for d in ("handwritten", "caltech101_7", "caltech101_20"):
        for t, ot, tk in (("chain", "chain", "chain"), ("thin05", "thin", "chain-sweep-0.05")):
            R = loads(B / f"optB/results/optB1/bench_{d}_{ot}_s*.jsonl")
            r = {"A": mean(seeds(R, "A", key)), "ALw": mean(seeds(R, "ALw", key)), "S": mean(seeds(R, "C" if t == "chain" else "Cthin", key))}
            lin = [x for x in load(B / "opt3/results/lin_bench.jsonl") if x["data"] == d and x["topology"] == t and x["std"] and not x["whiten"] and x["d"] == 32]
            r["L1"] = mean([x[key] for x in lin if x["method"] == "L1"]); r["PCAcat"] = mean([x[key] for x in lin if x["method"] == "PCAcat"])
            r["StabMap"] = mean([x[key] for x in load(B / f"bench/stabmap_{d}_{t}.jsonl") if x["kmeans_on"] == "l2"])
            r["Kimp"] = mean([x[key] for x in load(B / f"r18_kimp_{d}.jsonl") if x["topology"] == tk])
            r["Aimp"] = mean(seeds(load(B / f"st1/abl_{d}_{ot}_aimp.jsonl"), "Aimp", key))
            W = loads(B / f"st1/abl_{d}_{ot}_wp_s*.jsonl"); r["Cwp"] = mean(seeds(W, "Cwp", key))
            for e in ("RecFormer", "FreeCSL", "DVIMC", "GHICMC"):
                r[e] = mean(ext_seeds(B, e, d, t, key))
            dg = load(B / f"diag/bench_{d}_{t}.jsonl")
            out[("bench", f"{d}_{t}")] = (r, {k: float(np.mean([x[k] for x in dg])) for k in ("diameter", "hub_cov", "min_edge", "lambda2", "nonedge_frac", "imputability")} if dg else {})
    return out


def shift_settings(B, key="nmi"):
    out = {}
    for d in ("handwritten", "caltech101_7", "caltech101_20", "outdoorscene", "aloi"):
        for top, topo, tt in (("chain", "chain", "chain"), ("thin", "chain-sweep-0.05", "thin05")):
            for lv in ("a1", "a03", "s05", "s1", "b03"):
                R = load(B / f"st2/shift_{d}_{lv}_{top}.jsonl"); W = loads(B / f"st2/shiftwp_{d}_{lv}_{top}_s*.jsonl")
                A2 = [x for x in load(B / f"st2/a2_shift_{d}_{lv}.jsonl") if x.get("topology") == topo and x.get("variant") == "alaux_A2" and x.get("host") == "A"]
                K = [x for x in load(B / f"st2/shiftk_{d}_{lv}.jsonl") if x.get("topology") == topo]
                N = loads(B / f"ifE/shift_{d}_{lv}_{top}_s*.jsonl")
                r = {"A": mean(seeds(R, "A", key)), "ALw": mean(seeds(N, "ALw", key)), "S": mean(seeds(R, "C" if top == "chain" else "Cthin", key)),
                     "Cwp": mean(seeds(W, "Cwp", key)), "A2": mean([x[key] for x in A2]), "Kimp": mean([x[key] for x in K])}
                lin = [x for x in load(B / f"linshift/{d}_{tt}_{lv}.jsonl") if x["method"] == "L1" and x["std"] and not x["whiten"]]
                assert len(lin) == 10, (d, tt, lv, len(lin))
                r["L1"] = mean([x[key] for x in lin])
                dg = load(B / f"diag/shift_{d}_{tt}_{lv}.jsonl")
                out[("shift", f"{d}_{tt}_{lv}")] = (r, {k: float(np.mean([x[k] for x in dg])) for k in ("diameter", "hub_cov", "min_edge", "lambda2", "nonedge_frac", "imputability")} if dg else {})
    return out


def best(r, fam):
    xs = [(r[k], k) for k in fam if k in r and r[k] == r[k]]
    return max(xs) if xs else (float("nan"), "-")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", type=Path, default=Path("../build")); a = ap.parse_args()
    S = {**real_settings(a.build), **bench_settings(a.build), **shift_settings(a.build)}
    rows = []
    print(f"{'setting':42s} {'bestAlign':>16s} {'bestImpute':>16s} {'gap':>6s} | imput hub  diam minE  nonedge")
    for (kind, name), (r, dg) in S.items():
        ba, bi = best(r, ALIGN), best(r, IMPUTE); gap = ba[0] - bi[0]
        rows.append(dict(kind=kind, name=name, gap=gap, ba=ba, bi=bi, **dg, r=r))
        if kind != "shift":
            print(f"{kind+':'+name:42s} {ba[1]:>8s} {ba[0]:6.1f}  {bi[1]:>8s} {bi[0]:6.1f} {gap:+6.1f} | {dg.get('imputability', float('nan')):.2f}  {dg.get('hub_cov', float('nan')):.2f}  {dg.get('diameter', float('nan')):.0f}  {dg.get('min_edge', 0):5.0f}  {dg.get('nonedge_frac', float('nan')):.2f}")
    sh = [x for x in rows if x["kind"] == "shift"]
    print(f"shift: {len(sh)} cells; align wins {sum(x['gap'] > 0 for x in sh)}; mean gap {np.mean([x['gap'] for x in sh]):+.1f}")
    for kinds in (("real",), ("real", "bench"), ("real", "bench", "shift")):
        R = [x for x in rows if x["kind"] in kinds and "imputability" in x and x["gap"] == x["gap"]]
        print(f"== {'+'.join(kinds)} (n={len(R)}): Spearman of gap with descriptors")
        for k in ("imputability", "hub_cov", "diameter", "min_edge", "lambda2", "nonedge_frac"):
            v = [x[k] for x in R]; g = [x["gap"] for x in R]
            rho, p = spearmanr(v, g); print(f"   {k:13s} rho={rho:+.2f} p={p:.3g}")
        imp = np.array([x["imputability"] for x in R]); win = np.array([x["gap"] < 0 for x in R])
        correct = 0
        for i in range(len(R)):
            m = np.ones(len(R), bool); m[i] = False
            cand = np.unique(imp[m]); accs = [np.mean((imp[m] > t) == win[m]) for t in cand]
            t = cand[int(np.argmax(accs))]; correct += (imp[i] > t) == win[i]
        print(f"   LOO accuracy of 'impute if imputability > t': {correct}/{len(R)}  (majority baseline {max(win.mean(), 1 - win.mean()):.2f})")
    json.dump([{k: v for k, v in x.items()} for x in rows], open(a.build / "family_analysis.json", "w"), default=str, indent=0)


if __name__ == "__main__":
    main()
