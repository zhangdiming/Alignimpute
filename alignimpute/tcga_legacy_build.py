from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .tcga_build import pca

COHORTS = ["GBM", "OV", "LUSC", "LUAD", "BRCA", "COAD", "READ", "KIRC", "UCEC"]
VIEWS = [("miRNA_GA", ["miRNA_GA_gene"]), ("U133A", ["HT_HG-U133A"]),
         ("Agilent", ["AgilentG4502A_07_1", "AgilentG4502A_07_2", "AgilentG4502A_07_3"]), ("RNAseq_GA", ["GA", "GAV2"]),
         ("HiSeqV2", ["HiSeqV2"]), ("Meth27", ["HumanMethylation27"]), ("Meth450", ["HumanMethylation450"]),
         ("CNV", ["Gistic2_CopyNumber_Gistic2_all_data_by_genes"]), ("miRNA_HiSeq", ["miRNA_HiSeq_gene"])]


def fpath(root, cohort, ds):
    return root / f"TCGA.{cohort}.sampleMap__{ds}.gz"


def read_matrix(p: Path, keep_rows: set | None = None) -> pd.DataFrame:
    if keep_rows is None:
        df = pd.read_csv(p, sep="\t", index_col=0, compression="gzip", low_memory=False)
    else:
        parts = [c[c.index.isin(keep_rows)] for c in pd.read_csv(p, sep="\t", index_col=0, compression="gzip", chunksize=20000, low_memory=False)]
        df = pd.concat(parts)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, [c for c in df.columns if len(c) >= 15 and c[13:15] == "01"]]
    df.columns = [c[:15] for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df.T


def build_view(root, datasets, keep_rows=None):
    frames = []
    for c in COHORTS:
        for ds in datasets:
            p = fpath(root, c, ds)
            if p.exists():
                X = read_matrix(p, keep_rows); X["_cohort"] = c; frames.append(X)
    if not frames:
        return None
    shared = set.intersection(*[set(f.columns) for f in frames])
    X = pd.concat([f[sorted(shared)] for f in frames])
    X = X[~X.index.duplicated(keep="first")]
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pca", type=int, default=256); ap.add_argument("--max-features", type=int, default=5000)
    a = ap.parse_args()
    p27 = next(fpath(a.root, c, "HumanMethylation27") for c in COHORTS if fpath(a.root, c, "HumanMethylation27").exists())
    probes27 = set(pd.read_csv(p27, sep="\t", index_col=0, usecols=[0], compression="gzip").index)
    views, info = {}, {}
    for name, dss in VIEWS:
        keep = probes27 if name.startswith("Meth") else None
        X = build_view(a.root, dss, keep)
        coh = X.pop("_cohort")
        X = X.loc[:, X.isna().mean(0) < 0.2]
        X = X.fillna(X.mean(0))
        if X.shape[1] > a.max_features:
            X = X[X.var(0).sort_values(ascending=False).index[:a.max_features]]
        views[name] = (X.astype(np.float32), coh)
        info[name] = {"samples": int(X.shape[0]), "features": int(X.shape[1])}
        print(name, X.shape, flush=True)
    samples = sorted(set().union(*[set(v[0].index) for v in views.values()]))
    idx = {s: i for i, s in enumerate(samples)}
    n, V = len(samples), len(VIEWS)
    cohort = np.full(n, "", dtype=object)
    mask = np.zeros((n, V), bool); out = {}
    for j, (name, _) in enumerate(VIEWS):
        X, coh = views[name]
        rows = np.array([idx[s] for s in X.index])
        Z = pca(X.values, a.pca)
        full = np.zeros((n, Z.shape[1]), np.float32); full[rows] = Z
        out[f"X{j}"] = full; mask[rows, j] = True; cohort[rows] = coh.values
    y = np.array([COHORTS.index(c) for c in cohort])
    N = mask.T.astype(int) @ mask.astype(int)
    info.update({"n": n, "views": [v for v, _ in VIEWS], "n_uv": N.tolist(), "cohorts": COHORTS})
    np.savez_compressed(a.out, **out, y=y, mask=mask, samples=np.array(samples))
    Path(str(a.out) + ".json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("n", "views")}), "\n", np.array(N))


if __name__ == "__main__":
    main()
