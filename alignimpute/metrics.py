from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def kmeans_np(X, K, seed=0, iters=100, n_init=10):
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=n_init, max_iter=iters, random_state=seed).fit(np.asarray(X, dtype=np.float64))
        return km.labels_.astype(int), km.cluster_centers_
    except ImportError:
        pass
    best = None
    for t in range(n_init):
        rng = np.random.default_rng(seed * 1000 + t)
        n = X.shape[0]
        centers = np.empty((K, X.shape[1])); centers[0] = X[rng.integers(n)]
        d2 = ((X - centers[0]) ** 2).sum(1)
        for k in range(1, K):
            centers[k] = X[rng.choice(n, p=d2 / d2.sum())]; d2 = np.minimum(d2, ((X - centers[k]) ** 2).sum(1))
        lab = np.zeros(n, int)
        for _ in range(iters):
            D = ((X[:, None, :] - centers[None]) ** 2).sum(2); new = D.argmin(1)
            if np.array_equal(new, lab): break
            lab = new
            for k in range(K):
                m = lab == k
                centers[k] = X[m].mean(0) if m.any() else X[D.min(1).argmax()]
        inertia = ((X - centers[lab]) ** 2).sum()
        if best is None or inertia < best[0]: best = (inertia, lab, centers)
    return best[1], best[2]


def match_labels(y_true, y_pred):
    tu, ty = np.unique(y_true, return_inverse=True); pu, py = np.unique(y_pred, return_inverse=True)
    w = np.zeros((len(pu), len(tu)), int); np.add.at(w, (py, ty), 1)
    r, c = linear_sum_assignment(-w)
    lut = np.full(len(pu), tu.min() - 1); lut[r] = tu[c]
    return lut[py]


def clustering_metrics(y_true, y_pred):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    acc = float((match_labels(y_true, y_pred) == y_true).mean())
    return {"acc": acc, "nmi": float(normalized_mutual_info_score(y_true, y_pred)), "ari": float(adjusted_rand_score(y_true, y_pred))}


def retrieval_r1(Zu, Zv):
    A = Zu / np.linalg.norm(Zu, axis=1, keepdims=True); B = Zv / np.linalg.norm(Zv, axis=1, keepdims=True)
    S = A @ B.T
    return float((S.argmax(1) == np.arange(len(A))).mean())


def xview_knn_acc(Zu, Zv, y):
    A = Zu / np.linalg.norm(Zu, axis=1, keepdims=True); B = Zv / np.linalg.norm(Zv, axis=1, keepdims=True)
    S = A @ B.T; np.fill_diagonal(S, -np.inf)
    return float((y[S.argmax(1)] == y).mean())
