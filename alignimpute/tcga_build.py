from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import pandas as pd

FILES = {"rna": "rna.xena.gz", "mirna": "mirna.xena.gz", "rppa": "rppa.xena.gz", "cnv": "cnv.xena.gz", "meth27": "meth27.xena.gz"}


def read_xena(path: Path, max_features: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0, compression="gzip", low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    X = df.T
    X = X.loc[:, X.isna().mean(0) < 0.2]
    X = X.fillna(X.mean(0))
    if max_features and X.shape[1] > max_features:
        v = X.var(0).sort_values(ascending=False)
        X = X[v.index[:max_features]]
    return X.astype(np.float32)


def pca(X: np.ndarray, k: int) -> np.ndarray:
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    if X.shape[1] <= k:
        return X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:k].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pca", type=int, default=256); ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--views", nargs="+", default=list(FILES))
    a = ap.parse_args()
    ph = pd.read_csv(a.root / "phenotype.tsv.gz", sep="\t", compression="gzip")
    ph = ph.set_index(ph.columns[0])
    ctype_col = [c for c in ph.columns if "cancer type" in c.lower() or c.lower().startswith("_primary_disease") or "disease" in c.lower()][0]
    label_of = ph[ctype_col].dropna()
    views = {}
    for v in a.views:
        p = a.root / FILES[v]
        if not p.exists():
            print("missing", v); continue
        X = read_xena(p, a.max_features)
        X = X.loc[[s for s in X.index if isinstance(s, str) and len(s) >= 15 and s[13:15] == "01"]]
        views[v] = X
        print(v, X.shape, flush=True)
    samples = sorted(set().union(*[set(X.index) for X in views.values()]) & set(label_of.index))
    y_names = label_of.loc[samples].astype(str)
    classes = sorted(y_names.unique()); y = np.array([classes.index(c) for c in y_names])
    n = len(samples); V = len(views)
    mask = np.zeros((n, V), dtype=bool); feats = []
    for j, (v, X) in enumerate(views.items()):
        obs = [s for s in samples if s in X.index]
        idx = {s: i for i, s in enumerate(samples)}
        Z = pca(X.loc[obs].values.astype(np.float64), a.pca)
        F = np.zeros((n, Z.shape[1])); F[[idx[s] for s in obs]] = Z
        mask[[idx[s] for s in obs], j] = True
        feats.append(F)
    N = (mask.astype(float).T @ mask.astype(float)); np.fill_diagonal(N, 0)
    from .topology import fiedler, components
    print("views", list(views), "n", n, "classes", len(classes))
    print("per-view observed", mask.sum(0).tolist())
    print("co-observation matrix\n", N.astype(int))
    print("patterns", pd.Series(["".join(map(str, r.astype(int))) for r in mask]).value_counts().head(12).to_dict())
    print("lambda2 (L_n / n)", round(fiedler(N, normalize_by=n), 4), "components", components(N))
    np.savez(a.out, y=y, mask=mask, classes=np.array(classes), views=np.array(list(views)), samples=np.array(samples), **{f"X{i}": F for i, F in enumerate(feats)})
    print("saved", a.out)


if __name__ == "__main__":
    main()
