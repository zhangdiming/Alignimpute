from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

from . import linsync
from .topology import coobservation, fiedler, make_mask


def graph_stats(mask):
    n, V = mask.shape
    N = coobservation(mask)
    A = (N > 0) & ~np.eye(V, dtype=bool)
    D = np.full((V, V), np.inf); np.fill_diagonal(D, 0)
    D[A] = 1
    for k in range(V):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    finite = D[np.isfinite(D)]
    pairs = [(u, v) for u in range(V) for v in range(u + 1, V)]
    return {"diameter": float(finite.max()), "connected": bool(np.isfinite(D).all()),
            "hub_cov": float(mask.mean(0).max()), "min_edge": int(min(N[u, v] for u, v in pairs if N[u, v] > 0)),
            "lambda2": float(fiedler(N, normalize_by=n)), "nonedge_frac": float(np.mean([N[u, v] == 0 for u, v in pairs]))}


def imputability(Z, mask, seed=0, k=10, min_pairs=50, max_pairs=3000):
    V = mask.shape[1]; rng = np.random.default_rng(seed); out = []
    for u in range(V):
        for v in range(V):
            if u == v:
                continue
            cc = np.where(mask[:, u] & mask[:, v])[0]
            if len(cc) < min_pairs:
                continue
            if len(cc) > max_pairs:
                cc = rng.choice(cc, max_pairs, replace=False)
            X, Y = Z[u][cc], Z[v][cc]
            pred = np.zeros_like(Y)
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
                pred[te] = KNeighborsRegressor(n_neighbors=k).fit(X[tr], Y[tr]).predict(X[te])
            ss = ((Y - Y.mean(0)) ** 2).sum(); out.append(1 - ((Y - pred) ** 2).sum() / ss)
    return float(np.mean(out)) if out else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--topology", default="natural")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--no-std", action="store_true")
    ap.add_argument("--class-alpha", type=float, default=0.0); ap.add_argument("--size-sigma", type=float, default=0.0)
    ap.add_argument("--batch-scale", type=float, default=0.0)
    ap.add_argument("--tag", default=""); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    linsync.PRE.update(std=not a.no_std, whiten=False)
    for s in a.seeds:
        views, y, mask = linsync.load(a.data, a.topology, s)
        if a.class_alpha or a.size_sigma or a.batch_scale:
            V = len(views); w = None
            if a.topology.startswith("thin"):
                w = [1.0] * (V - 1); w[2] = float(a.topology[4:]) / 100
            mask, cohort = make_mask("chain", len(y), V, s, cohort_weights=w, y=y, class_alpha=a.class_alpha, size_sigma=a.size_sigma)
            if a.batch_scale > 0:
                views = [(v - v.mean(0)) / (v.std(0) + 1e-8) for v in views]
                br = np.random.default_rng(s + 15485863); vr = [v.copy() for v in views]
                for c in np.unique(cohort[cohort >= 0]):
                    rows = cohort == c
                    for j in range(V):
                        if mask[rows, j].any():
                            g = br.normal(size=views[j].shape[1]); b = br.normal(size=views[j].shape[1])
                            vr[j][rows] = views[j][rows] * (1 + a.batch_scale * g) + a.batch_scale * b
                views = vr
        Z = [linsync.whitened_pca(X, mask[:, v], 50)[:, :32] for v, X in enumerate(views)]
        rec = {"data": a.data.stem, "topology": a.topology, "seed": s, "tag": a.tag, **graph_stats(mask), "imputability": imputability(Z, mask, s)}
        with a.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
