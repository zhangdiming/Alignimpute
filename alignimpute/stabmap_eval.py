from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import clustering_metrics, kmeans_np


def get_mask(a, d):
    if a.topology == "natural":
        return d["mask"].astype(bool)
    from .linsync import load as _load
    return _load(a.data, a.topology, a.seed)[2]


def prep(a):
    d = np.load(a.data, allow_pickle=True); M = get_mask(a, d); n, V = M.shape
    a.out.mkdir(parents=True, exist_ok=True); np.save(a.out / "mask.npy", M)
    a.out.mkdir(parents=True, exist_ok=True)
    pats, inv = np.unique(M, axis=0, return_inverse=True); inv = inv.ravel()
    cnt = np.bincount(inv, minlength=len(pats)); big = [p for p in range(len(pats)) if cnt[p] >= a.min_size]
    moved = 0; dropped = []; target = {}
    for p in range(len(pats)):
        if cnt[p] >= a.min_size:
            continue
        subs = [q for q in big if not (pats[q] & ~pats[p]).any()] or \
               [q for q in range(len(pats)) if q != p and cnt[q] >= 2 and not (pats[q] & ~pats[p]).any()]
        if subs:
            target[p] = max(subs, key=lambda q: (pats[q].sum(), cnt[q]))
        elif cnt[p] < 2:
            target[p] = -1
    for p in target:
        q = target[p]
        while q in target and q != -1:
            q = target[q]
        if q == -1:
            dropped.append(int(cnt[p]))
        else:
            moved += int(cnt[p])
        inv[inv == p] = q
    meta = []
    for p, pat in enumerate(pats):
        rows = np.where(inv == p)[0]
        if len(rows) == 0:
            continue
        views = [v for v in range(V) if pat[v]]
        X = np.hstack([d[f"X{v}"][rows] for v in views]).astype(float)
        names = [f"v{v}_{j}" for v in views for j in range(d[f"X{v}"].shape[1])]
        keep = X.std(0) > 1e-8
        X = X[:, keep]; names = [nm for nm, k in zip(names, keep) if k]
        tag = f"D{p}"
        np.savetxt(a.out / f"{tag}.csv", X.T, delimiter=",", fmt="%.6g")
        (a.out / f"{tag}.features.txt").write_text("\n".join(names) + "\n")
        (a.out / f"{tag}.cells.txt").write_text("\n".join(f"c{i}" for i in rows) + "\n")
        meta.append({"dataset": tag, "views": views, "n_cells": int(len(rows)), "n_features": int(len(names)),
                     "reference": bool(len(rows) >= a.min_ref)})
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    (a.out / "moved.json").write_text(json.dumps({"cells_reduced_to_subset_pattern": moved, "cells_not_embeddable": int(sum(dropped)), "min_size": a.min_size}))
    print(json.dumps(meta))


def score(a):
    d = np.load(a.data, allow_pickle=True); y = d["y"]; n = len(y)
    M = np.load(a.dir / "mask.npy") if (a.dir / "mask.npy").exists() else d["mask"].astype(bool)
    K = int(len(np.unique(y)))
    E = np.loadtxt(a.dir / f"embedding_{a.tag}.csv", delimiter=",", skiprows=1, dtype=str)
    cells = [int(c.strip('"')[1:]) for c in E[:, 0]]; Z = np.zeros((n, E.shape[1] - 1))
    Z[cells] = E[:, 1:].astype(float)
    emb = np.zeros(n, bool); emb[cells] = True
    Z, M, y = Z[emb], M[emb], y[emb]
    u, v = a.far
    A = Z[M[:, u]]; B = Z[M[:, v]]
    A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True)
    far = float((y[M[:, v]][(A @ B.T).argmax(1)] == y[M[:, u]]).mean())
    Zl2 = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    with a.out.open("a") as f:
        for post, ZZ in (("raw", Z), ("l2", Zl2)):
            for s in range(10):
                lab, _ = kmeans_np(ZZ, K, seed=s)
                rec = {"data": str(a.data), "topology": a.topology, "mask_seed": a.seed, "method": f"StabMap[{a.tag}]", "kmeans_on": post, "seed": s, "dim": int(Z.shape[1]), "n_scored": int(emb.sum()), "n": int(n), "far_cell": far, **clustering_metrics(y, lab)}
                f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--data", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--far", type=int, nargs=2, required=True); p.add_argument("--min-ref", type=int, default=100); p.add_argument("--min-size", type=int, default=10)
    p.add_argument("--topology", default="natural"); p.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("score"); s.add_argument("--data", type=Path, required=True); s.add_argument("--dir", type=Path, required=True)
    s.add_argument("--far", type=int, nargs=2, required=True); s.add_argument("--tag", required=True); s.add_argument("--out", type=Path, required=True)
    s.add_argument("--topology", default="natural"); s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else score(a)


if __name__ == "__main__":
    main()
