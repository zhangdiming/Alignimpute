from __future__ import annotations

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(*a, **k): return nullcontext()


def procrustes(Zu: np.ndarray, Zv: np.ndarray) -> tuple[np.ndarray, float]:
    M = Zu.T @ Zv
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    res = np.linalg.norm(Zu @ R - Zv) / max(np.linalg.norm(Zv), 1e-12)
    return R, float(res)


def _sinkhorn(Cst: np.ndarray, reg: float, iters: int = 50) -> np.ndarray:
    K = np.exp(-(Cst - Cst.min()) / reg); a = np.ones(K.shape[0]) / K.shape[0]; b = np.ones(K.shape[1]) / K.shape[1]
    u = np.ones_like(a)
    for _ in range(iters):
        v = b / (K.T @ u + 1e-12); u = a / (K @ v + 1e-12)
    return (u[:, None] * K) * v[None, :]


THIN_CFG = {"n_sub": None, "reg": None, "anchor_weight": None, "threads": 1}


def thin_edge_estimate(Zu_all: np.ndarray, Zv_all: np.ndarray, Zu_cc: np.ndarray, Zv_cc: np.ndarray,
                       n_sub: int = 600, iters: int = 20, reg: float = 0.05, anchor_weight: float = 20.0,
                       seed: int = 0) -> tuple[np.ndarray, float]:
    n_sub = THIN_CFG["n_sub"] or n_sub; reg = THIN_CFG["reg"] or reg; anchor_weight = THIN_CFG["anchor_weight"] or anchor_weight
    with threadpool_limits(limits=THIN_CFG["threads"]):
        return _thin_edge_estimate(Zu_all, Zv_all, Zu_cc, Zv_cc, n_sub, iters, reg, anchor_weight, seed)


def _thin_edge_estimate(Zu_all, Zv_all, Zu_cc, Zv_cc, n_sub, iters, reg, anchor_weight, seed):
    rng = np.random.default_rng(seed)
    d = Zu_all.shape[1]
    su = Zu_all[rng.choice(len(Zu_all), min(n_sub, len(Zu_all)), replace=False)]
    sv = Zv_all[rng.choice(len(Zv_all), min(n_sub, len(Zv_all)), replace=False)]
    R = np.eye(d)
    if len(Zu_cc) >= d:
        R, _ = procrustes(Zu_cc, Zv_cc)
    else:
        R, _ = thin_edge_estimate_cov(Zu_all, Zv_all, Zu_cc, Zv_cc)
    for _ in range(iters):
        Cst = ((su @ R)[:, None, :] - sv[None, :, :]) ** 2
        Cst = Cst.sum(2)
        Cst = Cst / max(np.median(Cst), 1e-12)
        P = _sinkhorn(Cst, reg)
        M = su.T @ (P @ sv) * len(su)
        if len(Zu_cc):
            M = M + anchor_weight * (Zu_cc.T @ Zv_cc)
        U, _, Vt = np.linalg.svd(M); R = U @ Vt
    res = np.linalg.norm(Zu_cc @ R - Zv_cc) / max(np.linalg.norm(Zv_cc), 1e-12) if len(Zu_cc) else 1.0
    return R, float(res)


def thin_edge_estimate_cov(Zu_all: np.ndarray, Zv_all: np.ndarray, Zu_cc: np.ndarray, Zv_cc: np.ndarray) -> tuple[np.ndarray, float]:
    def eig(Z):
        C = np.cov(Z, rowvar=False); w, U = np.linalg.eigh(C); order = np.argsort(w)[::-1]; return U[:, order]
    Uu, Uv = eig(Zu_all), eig(Zv_all)
    Pu, Pv = Zu_cc @ Uu, Zv_cc @ Uv
    S = np.sign((Pu * Pv).sum(0)); S[S == 0] = 1.0
    R = Uu @ np.diag(S) @ Uv.T
    res = np.linalg.norm(Zu_cc @ R - Zv_cc) / max(np.linalg.norm(Zv_cc), 1e-12) if len(Zu_cc) else 1.0
    return R, float(res)


def edge_estimates(Z: list[np.ndarray], mask: np.ndarray, min_pairs: int | None = None, thin: bool = False) -> dict:
    V = len(Z); d = Z[0].shape[1]
    min_pairs = d if min_pairs is None else min_pairs
    out = {}
    for u in range(V):
        for v in range(u + 1, V):
            cc = mask[:, u] & mask[:, v]
            n_uv = int(cc.sum())
            if n_uv == 0:
                continue
            if n_uv < min_pairs:
                if not thin:
                    continue
                R, res = thin_edge_estimate(Z[u][mask[:, u]], Z[v][mask[:, v]], Z[u][cc], Z[v][cc])
                w = 0.5 * n_uv / (1.0 + 10.0 * res)
            else:
                R, res = procrustes(Z[u][cc], Z[v][cc])
                w = n_uv / (1.0 + 10.0 * res)
            out[(u, v)] = (R, w, n_uv)
    return out


def tree_sync(edges: dict, V: int, d: int) -> tuple[np.ndarray, list[list[int]]]:
    from .topology import components
    adj = np.zeros((V, V))
    for (u, v) in edges: adj[u, v] = adj[v, u] = 1
    comps = components(adj)
    Q = np.tile(np.eye(d), (V, 1, 1))
    for comp in comps:
        root = comp[0]; done = {root}
        while len(done) < len(comp):
            best = None
            for (u, v), (R, w, n) in edges.items():
                if (u in done) != (v in done) and (best is None or w > best[0]):
                    best = (w, u, v, R)
            if best is None: break
            w, u, v, R = best
            if u in done: Q[v] = R.T @ Q[u]
            else: Q[u] = R @ Q[v]
            done.add(u); done.add(v)
    return Q, comps


def spectral_sync(edges: dict, V: int, d: int) -> tuple[np.ndarray, list[list[int]]]:
    M = np.zeros((V * d, V * d))
    deg = np.zeros(V)
    adj = np.zeros((V, V))
    for (u, v), (R, w, n) in edges.items():
        M[u * d:(u + 1) * d, v * d:(v + 1) * d] = w * R
        M[v * d:(v + 1) * d, u * d:(u + 1) * d] = w * R.T
        deg[u] += w; deg[v] += w; adj[u, v] = adj[v, u] = 1
    Dinv = np.diag(1.0 / np.sqrt(np.maximum(np.repeat(deg, d), 1e-12)))
    Mn = Dinv @ M @ Dinv
    from .topology import components
    comps = components(adj)
    Q = np.tile(np.eye(d), (V, 1, 1))
    for comp in comps:
        if len(comp) == 1:
            continue
        idx = np.concatenate([np.arange(v * d, (v + 1) * d) for v in comp])
        sub = Mn[np.ix_(idx, idx)]
        ev, evec = np.linalg.eigh(sub)
        top = evec[:, -d:]
        for k, v in enumerate(comp):
            B = top[k * d:(k + 1) * d]
            U, _, Vt = np.linalg.svd(B)
            Q[v] = U @ Vt
    for comp in comps:
        g = Q[comp[0]].T.copy()
        for v in comp:
            Q[v] = Q[v] @ g
    return Q, comps


def sync_error(Q_hat: np.ndarray, Q_true: np.ndarray, comps: list[list[int]]) -> float:
    errs = []
    for comp in comps:
        v0 = comp[0]
        g = Q_hat[v0].T @ Q_true[v0]
        for v in comp:
            errs.append(np.linalg.norm(Q_hat[v] @ g - Q_true[v]) / np.sqrt(Q_true.shape[1]))
    return float(max(errs))
