from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import clustering_metrics, kmeans_np
from .sync import edge_estimates, spectral_sync
from .topology import make_mask


SHIFT = {"class_alpha": 0.0, "size_sigma": 0.0, "batch_scale": 0.0}


def load(path, topology, seed):
    d = np.load(path, allow_pickle=True); y = d["y"]
    views = [d[f"X{i}"].astype(np.float64) for i in range(len([k for k in d.files if k.startswith("X")]))]
    n, V = len(y), len(views)
    if topology == "natural":
        mask = d["mask"].astype(bool)
    else:
        w = None
        if topology.startswith("thin"):
            w = [1.0] * (V - 1); w[2] = float(topology[4:]) / 100
        mask, cohort = make_mask("chain" if topology.startswith("thin") else topology, n, V, seed, cohort_weights=w, y=y,
                                 class_alpha=SHIFT["class_alpha"], size_sigma=SHIFT["size_sigma"])
        if SHIFT["batch_scale"] > 0:
            views = [(v - v.mean(0)) / (v.std(0) + 1e-8) for v in views]
            br = np.random.default_rng(seed + 15485863); vr = [v.copy() for v in views]
            for c in np.unique(cohort[cohort >= 0]):
                rows = cohort == c
                for j in range(V):
                    if mask[rows, j].any():
                        g = br.normal(size=views[j].shape[1]); b = br.normal(size=views[j].shape[1])
                        vr[j][rows] = views[j][rows] * (1 + SHIFT["batch_scale"] * g) + SHIFT["batch_scale"] * b
            views = vr
    return views, y, mask


PRE = {"std": True, "whiten": True}


def whitened_pca(X, obs, p):
    Xo = X[obs]; mu = Xo.mean(0); sd = Xo.std(0); keep = sd > 1e-8
    Xo = (Xo[:, keep] - mu[keep]) / (sd[keep] if PRE["std"] else np.sqrt((sd[keep] ** 2).sum()))
    U, S, _ = np.linalg.svd(Xo, full_matrices=False)
    k = min(p, int((S > 1e-8 * S[0]).sum()))
    F = np.zeros((X.shape[0], k))
    F[obs] = U[:, :k] * np.sqrt(len(Xo)) if PRE["whiten"] else U[:, :k] * S[:k] / np.sqrt((S[:k] ** 2).sum() / len(Xo))
    return F


def l1(views, mask, d, p):
    Z = []
    for v, X in enumerate(views):
        F = whitened_pca(X, mask[:, v], p)[:, :d]
        if F.shape[1] < d:
            F = np.hstack([F, np.zeros((F.shape[0], d - F.shape[1]))])
        Z.append(F)
    edges = edge_estimates(Z, mask, min_pairs=d, thin=True)
    Q, comps = spectral_sync(edges, len(Z), d)
    cons = np.zeros_like(Z[0]); cnt = mask.sum(1, keepdims=True)
    for v in range(len(Z)):
        cons[mask[:, v]] += Z[v][mask[:, v]] @ Q[v]
    return cons / np.maximum(cnt, 1), {"n_edges": len(edges), "n_components": len(comps)}


def l2(views, mask, d, p, ridge):
    n = mask.shape[0]; deg = mask.sum(1).astype(float)
    blocks = []
    for v, X in enumerate(views):
        obs = mask[:, v]; F = whitened_pca(X, obs, p)
        if PRE["whiten"]:
            B = F / np.sqrt(len(np.where(obs)[0]) * (1.0 + ridge))
        else:
            B = F / np.sqrt(len(np.where(obs)[0]))
        blocks.append(B / np.sqrt(np.maximum(deg, 1))[:, None])
    Y = np.hstack(blocks)
    U, S, _ = np.linalg.svd(Y, full_matrices=False)
    return U[:, :d] * S[:d], {"top_sv": [round(float(s), 4) for s in S[:d + 2]]}


def pcacat(views, mask, d, p):
    cons = np.zeros((mask.shape[0], d)); cnt = mask.sum(1, keepdims=True)
    for v, X in enumerate(views):
        F = whitened_pca(X, mask[:, v], p)[:, :d]
        cons[:, :F.shape[1]] += F
    return cons / np.maximum(cnt, 1), {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--topologies", nargs="+", default=["chain"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--methods", nargs="+", default=["L1", "L2", "PCAcat"])
    ap.add_argument("--d", type=int, nargs="+", default=[32])
    ap.add_argument("--p", type=int, default=50)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--class-alpha", type=float, default=0.0); ap.add_argument("--size-sigma", type=float, default=0.0)
    ap.add_argument("--batch-scale", type=float, default=0.0); ap.add_argument("--tag", default="")
    ap.add_argument("--no-std", action="store_true", help="centre and scale each view globally instead of standardising columns")
    ap.add_argument("--no-whiten", action="store_true", help="keep the PCA spectrum in the scores")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    PRE["std"] = not a.no_std; PRE["whiten"] = not a.no_whiten
    SHIFT.update(class_alpha=a.class_alpha, size_sigma=a.size_sigma, batch_scale=a.batch_scale)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for topo in a.topologies:
        for seed in a.seeds:
            views, y, mask = load(a.data, topo, seed)
            K = int(len(np.unique(y)))
            for d in a.d:
                for m in a.methods:
                    if m == "L1":
                        E, info = l1(views, mask, d, a.p)
                    elif m == "L2":
                        E, info = l2(views, mask, d, a.p, a.ridge)
                    else:
                        E, info = pcacat(views, mask, d, a.p)
                    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
                    lab, _ = kmeans_np(En, K, seed=seed)
                    rec = {"data": a.data.stem, "topology": topo, "seed": seed, "method": m, "d": d, "p": a.p, "ridge": a.ridge, "std": PRE["std"], "whiten": PRE["whiten"], "tag": a.tag,
                           **clustering_metrics(y, lab), **info}
                    with a.out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(json.dumps({k: rec[k] for k in ("data", "topology", "seed", "method", "d", "nmi", "acc")}), flush=True)


if __name__ == "__main__":
    main()
