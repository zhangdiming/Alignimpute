from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import linsync
from .sync import procrustes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--topologies", nargs="+", default=["natural"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--d", type=int, default=32); ap.add_argument("--p", type=int, default=50)
    ap.add_argument("--no-std", action="store_true"); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    linsync.PRE.update(std=not a.no_std, whiten=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for topo in a.topologies:
        for seed in a.seeds:
            views, y, mask = linsync.load(a.data, topo, seed)
            Z = []
            for v, X in enumerate(views):
                F = linsync.whitened_pca(X, mask[:, v], a.p)[:, :a.d]
                Z.append(np.hstack([F, np.zeros((F.shape[0], a.d - F.shape[1]))]) if F.shape[1] < a.d else F)
            rng = np.random.default_rng(seed)
            V = len(Z)
            for u in range(V):
                for v in range(u + 1, V):
                    cc = mask[:, u] & mask[:, v]; n_uv = int(cc.sum())
                    if n_uv < a.d:
                        continue
                    Zu, Zv = Z[u][cc], Z[v][cc]
                    _, res = procrustes(Zu, Zv)
                    null = float(np.mean([procrustes(Zu, Zv[rng.permutation(n_uv)])[1] for _ in range(5)]))
                    B, *_ = np.linalg.lstsq(Zu, Zv, rcond=None)
                    ls = float(np.linalg.norm(Zu @ B - Zv) / max(np.linalg.norm(Zv), 1e-12))
                    rec = {"data": a.data.stem, "topology": topo, "seed": seed, "u": u, "v": v, "n_uv": n_uv, "res": res, "res_perm": null, "res_ls": ls}
                    with a.out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
