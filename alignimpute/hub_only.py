from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import linsync
from .metrics import clustering_metrics, kmeans_np

HUBS = [("pbmc_mimitou", 1), ("bmmc_site1", 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/bench/hub_only.json"))
    a = ap.parse_args()
    linsync.PRE.update(std=False, whiten=False)
    out = []
    for g, hub in HUBS:
        views, y, mask = linsync.load(str(a.data / f"{g}.npz"), "natural", 0)
        assert mask[:, hub].all()
        Z = linsync.whitened_pca(views[hub], mask[:, hub], 50)[:, :32]
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        K = len(np.unique(y))
        for s in range(10):
            out.append({"data": g, "method": "hub_only", "hub_view": hub, "seed": s, **clustering_metrics(y, kmeans_np(Z, K, seed=s)[0])})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
