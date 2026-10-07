from __future__ import annotations
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import silhouette_score
from .sync import edge_estimates, spectral_sync
from .metrics import kmeans_np, clustering_metrics


def cons(Z, mask):
    c = np.zeros(Z.shape[1:]); n = np.zeros(Z.shape[1])
    for v in range(Z.shape[0]):
        m = mask[:, v]; c[m] += Z[v][m]; n[m] += 1
    c = c / np.maximum(n, 1)[:, None]
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)


def pair_agreement(Z, mask):
    vals = []
    V = Z.shape[0]
    for u in range(V):
        for v in range(u + 1, V):
            cc = mask[:, u] & mask[:, v]
            if cc.sum() >= 2:
                a = Z[u][cc]; b = Z[v][cc]
                vals.append(float((np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)).mean()))
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--emb", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); res = defaultdict(dict)
    for f in sorted(glob.glob(str(a.emb / "gate_*" / "*.npz"))):
        cell = Path(f).parent.name[5:]; seed = int(re.search(r"_s(\d+)_A\.npz$", f).group(1))
        e = np.load(f); Z = e["Z"]; mask = e["mask"].astype(bool); y = e["y"]; K = int(len(np.unique(y)))
        d = Z.shape[2]
        edges = edge_estimates(list(Z), mask, min_pairs=d, thin=True)
        Q, _ = spectral_sync(edges, Z.shape[0], d)
        Zs = np.stack([Z[v] @ Q[v] for v in range(Z.shape[0])])
        ch, cs = cons(Z, mask), cons(Zs, mask)
        lh, _ = kmeans_np(ch, K, seed=seed); ls, _ = kmeans_np(cs, K, seed=seed)
        mh, ms = clustering_metrics(y, lh), clustering_metrics(y, ls)
        sub = np.random.default_rng(seed).choice(len(y), min(len(y), 3000), replace=False)
        sh = silhouette_score(ch[sub], lh[sub], metric="cosine"); ss = silhouette_score(cs[sub], ls[sub], metric="cosine")
        N = mask.T.astype(int) @ mask.astype(int)
        res[cell][seed] = {"host_nmi": mh["nmi"], "sync_nmi": ms["nmi"], "host_acc": mh["acc"], "sync_acc": ms["acc"],
                           "nonedge": bool((N == 0).any()), "sil_host": float(sh), "sil_sync": float(ss),
                           "pair_host": pair_agreement(Z, mask), "pair_sync": pair_agreement(Zs, mask)}
    a.out.write_text(json.dumps(res, indent=1)); print("cells", len(res))


if __name__ == "__main__":
    main()
