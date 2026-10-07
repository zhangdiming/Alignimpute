from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .p0_handwritten import make_mask, obs_standardise
from .metrics import kmeans_np, clustering_metrics
from .bridge import bridge_impute


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10))); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--configs", nargs="+", default=["chain", "chain-sweep-0.05"])
    ap.add_argument("--class-alpha", type=float, default=0.0); ap.add_argument("--size-sigma", type=float, default=0.0)
    ap.add_argument("--batch-scale", type=float, default=0.0); ap.add_argument("--tag", default="")
    ap.add_argument("--obs-stats", action="store_true", help="preprocessing statistics on the observed rows of each view only")
    ap.add_argument("--no-std", action="store_true", help="real assemblies: centre and scale each view globally, as p0_handwritten --no-std")
    a = ap.parse_args()
    d = np.load(a.data); y = d["y"]; views = [d[f"X{i}"] for i in range(len([k for k in d.files if k.startswith("X")]))]
    raw_views = views
    if a.no_std:
        views = [(v - v.mean(0)) / (np.sqrt((v.std(0) ** 2).sum()) + 1e-8) * np.sqrt(v.shape[1]) for v in views]
    else:
        views = [(v - v.mean(0)) / (v.std(0) + 1e-8) for v in views]
    n, V = len(y), len(views); K = int(len(np.unique(y)))
    with open(a.out, "a") as fo:
        for cfg in a.configs:
            topo, w = ("chain", None) if cfg == "chain" else ("chain", None)
            if cfg in ("star", "ring", "two-comp", "complete"):
                topo = cfg
            if cfg.startswith("chain-sweep-"):
                p = float(cfg.split("-")[-1]); w = [1.0] * (V - 1); w[2] = p
            for seed in a.seeds:
                if cfg == "natural":
                    mask, cohort = d["mask"].astype(bool), np.full(n, -1)
                else:
                    mask, cohort = make_mask(topo, n, V, seed, cohort_weights=w, y=y, class_alpha=a.class_alpha, size_sigma=a.size_sigma)
                if a.obs_stats:
                    views = obs_standardise(raw_views, mask, a.no_std)
                vr = views
                if a.batch_scale > 0:
                    br = np.random.default_rng(seed + 15485863); vr = [v.copy() for v in views]
                    for c in np.unique(cohort[cohort >= 0]):
                        rows = cohort == c
                        for j in range(V):
                            if mask[rows, j].any():
                                g = br.normal(size=views[j].shape[1]); b = br.normal(size=views[j].shape[1])
                                vr[j][rows] = views[j][rows] * (1 + a.batch_scale * g) + a.batch_scale * b
                imp = bridge_impute(vr, mask, k=a.k)
                Xc = np.hstack([v / np.sqrt(v.shape[1]) for v in imp])
                lab, _ = kmeans_np(Xc, K, seed=seed)
                m = clustering_metrics(y, lab)
                rec = {"variant": "Kimp", "tag": a.tag, "topology": cfg, "seed": seed, "acc": m["acc"], "nmi": m.get("nmi"), "k": a.k, "obs_stats": a.obs_stats}
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                print(cfg, seed, round(m["acc"], 4), flush=True)


if __name__ == "__main__":
    main()
