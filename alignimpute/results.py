from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

EXT = ["FreeCSL", "DVIMC", "GHICMC", "RecFormer"]
DS3 = [("handwritten", "HandWritten"), ("caltech101_7", "Caltech101-7"), ("caltech101_20", "Caltech101-20")]
DS8 = DS3 + [("outdoorscene", "OutdoorScene"), ("nuswide", "NUS-WIDE"), ("aloi", "ALOI"), ("proteinfold", "ProteinFold"), ("tcga_pancan", "TCGA PanCan")]
TOPS = [("complete", "complete"), ("random-2", "random-2"), ("star", "star"), ("ring", "ring"), ("chain", "chain"), ("two-comp", "two comp."), ("thin05", r"thin 5\%")]
OLD = ["MVP", "CPSPAN", "CPM-Nets", "DIMVC", "TreeEIC"]


def load(p):
    p = Path(p)
    return [json.loads(l) for l in open(p)] if p.exists() else []


def loads(pattern):
    out = []
    for f in sorted(glob.glob(str(pattern))):
        out += load(f)
    return out


def seeds(rows, v, key="nmi", topo=None):
    return {r["seed"]: r[key] for r in rows if r.get("variant") == v and (topo is None or r.get("topology") == topo) and r.get(key) is not None}


def ext_rows(B, method):
    rows = []
    for f in sorted(glob.glob(str(B / f"ifB/ifB_{method}*.jsonl"))):
        if "sanity" in f:
            continue
        rows += [r for r in load(f) if r.get("status") == "ok"]
    return rows


def ext_seeds(B, method, dataset, topo, key="nmi"):
    return {r["seed"]: r[key] for r in ext_rows(B, method) if r["dataset"] == dataset and r["topology"] == topo}


def by_seed(rows, variant=None, key="acc", topo_key="topology", filt=None):
    d = defaultdict(dict)
    for r in rows:
        if r.get("status", "ok") != "ok" or key not in r: continue
        if variant and r.get("variant") != variant: continue
        if filt and not filt(r): continue
        d[r[topo_key]][r["seed"]] = r[key]
    return d


def old_baselines(B, key):
    out = {}
    for ds, _ in DS3:
        r = {}
        if ds == "handwritten":
            r["MVP"] = by_seed(load(B / "p1_mvp.jsonl"), key=key)
        else:
            r["MVP"] = by_seed(load(B / ("r7e9b_mvp_caltech101_20.jsonl" if ds == "caltech101_20" else "r7e8_mvp_caltech101_7.jsonl")), key=key)
        r["CPSPAN"] = by_seed(load(B / f"p4_cpspan_{ds}.jsonl") + load(B / f"r7e8_cpspan_{ds}.jsonl"), "CPSPAN", key=key)
        r["CPM-Nets"] = by_seed(load(B / f"r2_cpmnets_{ds}.jsonl"), key=key, filt=lambda q: q.get("tag") == "ep30")
        r["DIMVC"] = by_seed(load(B / f"r2_dimvc_{ds}.jsonl"), key=key)
        r["TreeEIC"] = by_seed(load(B / f"r7e8_treeeic_{ds}.jsonl"), key=key)
        out[ds] = r
    return out


def f1(x):
    return "--" if x != x else f"{x:.1f}"
