from __future__ import annotations
from collections import deque
import numpy as np


def _paths(N):
    V = N.shape[0]; P = {}
    for s in range(V):
        prev = {s: None}; q = deque([s])
        while q:
            u = q.popleft()
            for v in range(V):
                if v != u and N[u, v] > 0 and v not in prev:
                    prev[v] = u; q.append(v)
        for t in prev:
            path = [t]
            while prev[path[-1]] is not None: path.append(prev[path[-1]])
            P[(s, t)] = path[::-1]
    return P


def _knn_mean(Q, R, T, k):
    k = min(k, len(R))
    out = np.empty((len(Q), T.shape[1]), dtype=np.float64)
    rr = (R ** 2).sum(1)
    for s in range(0, len(Q), 1024):
        q = Q[s:s + 1024]
        D = (q ** 2).sum(1)[:, None] - 2 * q @ R.T + rr[None]
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        out[s:s + 1024] = T[idx].mean(1)
    return out


def bridge_impute(views, mask, k=10):
    mask = np.asarray(mask, bool); n, V = mask.shape
    N = mask.T.astype(int) @ mask.astype(int)
    P = _paths(N)
    out = [np.array(v, dtype=np.float64, copy=True) for v in views]
    for v in range(V):
        out[v][~mask[:, v]] = 0.0
    patterns = {}
    for i in range(n):
        patterns.setdefault(tuple(np.where(mask[i])[0]), []).append(i)
    for obs, rows in patterns.items():
        rows = np.array(rows)
        for v in range(V):
            if v in obs: continue
            cand = [P[(w, v)] for w in obs if (w, v) in P]
            if not cand: continue
            path = min(cand, key=len)
            est = np.asarray(views[path[0]], dtype=np.float64)[rows]
            for a, b in zip(path[:-1], path[1:]):
                ref = np.where(mask[:, a] & mask[:, b])[0]
                est = _knn_mean(est, np.asarray(views[a], dtype=np.float64)[ref], np.asarray(views[b], dtype=np.float64)[ref], k)
            out[v][rows] = est
    return [o.astype(np.float32) for o in out]


def _pca(X, k):
    X = X - X.mean(0)
    if X.shape[1] <= k: return X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:k].T


def bridge_pairs(views, mask, pca=50):
    mask = np.asarray(mask, bool); n, V = mask.shape
    N = mask.T.astype(int) @ mask.astype(int)
    out = []
    for u in range(V):
        for v in range(u + 1, V):
            if N[u, v] > 0: continue
            ws = [w for w in range(V) if w not in (u, v) and N[u, w] > 0 and N[w, v] > 0]
            if not ws: continue
            w = max(ws, key=lambda w: min(N[u, w], N[w, v]))
            I = np.where(mask[:, u] & mask[:, w] & ~mask[:, v])[0]; J = np.where(mask[:, w] & mask[:, v] & ~mask[:, u])[0]
            if len(I) == 0 or len(J) == 0: continue
            F = _pca(np.asarray(views[w], dtype=np.float64)[np.r_[I, J]], pca); A, B = F[:len(I)], F[len(I):]
            D = (A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None]
            ab, ba = D.argmin(1), D.argmin(0)
            mutual = np.where(ba[ab] == np.arange(len(I)))[0]
            out.append((u, v, I[mutual], J[ab[mutual]]))
    return out


def view_neighbours(views, mask, k=10, pca=50):
    mask = np.asarray(mask, bool); out = {}
    for v in range(mask.shape[1]):
        rows = np.where(mask[:, v])[0]
        if len(rows) <= k: continue
        F = _pca(np.asarray(views[v], dtype=np.float64)[rows], pca)
        sq = (F ** 2).sum(1); nbr = np.empty((len(rows), k), dtype=np.int64)
        for s in range(0, len(rows), 2048):
            D = sq[s:s + 2048, None] - 2 * F[s:s + 2048] @ F.T + sq[None]
            D[np.arange(D.shape[0]), np.arange(s, s + D.shape[0])] = np.inf
            nbr[s:s + 2048] = rows[np.argpartition(D, k, axis=1)[:, :k]]
        out[v] = (rows, nbr)
    return out
