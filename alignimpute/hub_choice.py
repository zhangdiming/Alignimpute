import argparse, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from .metrics import clustering_metrics, kmeans_np


def pca(X, k=32):
    X = (X - X.mean(0)) / (X.std(0) + 1e-8); _, _, Vt = np.linalg.svd(X, full_matrices=False); return X @ Vt[:k].T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, default=Path("data")); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["handwritten", "caltech101_7", "caltech101_20"]); a = ap.parse_args()
    out = {}
    for d in a.datasets:
        z = np.load(a.data / f"{d}.npz"); V = len([k for k in z.files if k.startswith("X")]); y = z["y"]; K = len(np.unique(y))
        P = [pca(z[f"X{v}"].astype(float)) for v in range(V)]
        idx = np.random.default_rng(0).choice(len(y), min(1500, len(y)), replace=False)
        r2 = np.zeros((V, V))
        for u in range(V):
            for w in range(V):
                if u == w: continue
                pred = cross_val_predict(KNeighborsRegressor(10), P[u][idx], P[w][idx], cv=5)
                r2[u, w] = 1 - ((P[w][idx] - pred) ** 2).sum() / ((P[w][idx] - P[w][idx].mean(0)) ** 2).sum()
        pred_others = [float(r2[u][np.arange(V) != u].mean()) for u in range(V)]
        nmi = [clustering_metrics(y, kmeans_np(P[v] / np.maximum(np.linalg.norm(P[v], axis=1, keepdims=True), 1e-12), K, seed=0)[0])["nmi"] for v in range(V)]
        out[d] = {"predicts_others_r2": pred_others, "hub": int(np.argmax(pred_others)), "single_view_nmi": nmi}
        print(d, out[d]["hub"], [round(x, 3) for x in pred_others])
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
