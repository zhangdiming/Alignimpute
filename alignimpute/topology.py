from __future__ import annotations

import numpy as np


def patterns_for(topology: str, V: int) -> list[tuple[int, ...]]:
    if topology == "chain":
        return [(v, v + 1) for v in range(V - 1)]
    if topology == "ring":
        return [(v, (v + 1) % V) for v in range(V)]
    if topology == "star":
        return [(0, v) for v in range(1, V)]
    if topology == "two-comp":
        h = V // 2
        return [(v, v + 1) for v in range(h - 1)] + [(v, v + 1) for v in range(h, V - 1)]
    if topology == "complete":
        return [tuple(range(V))]
    raise ValueError(topology)


def make_mask(topology: str, n: int, V: int, seed: int,
              cohort_weights: list[float] | None = None, y: np.ndarray | None = None,
              class_alpha: float = 0.0, size_sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    if topology != "random-2" and (class_alpha > 0 or size_sigma > 0):
        return _make_mask_shift(topology, n, V, seed, cohort_weights, y, class_alpha, size_sigma)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    mask = np.zeros((n, V), dtype=bool)
    cohort = np.full(n, -1)
    if topology == "random-2":
        for i in range(n):
            vs = rng.choice(V, size=2, replace=False)
            mask[i, vs] = True
        return mask, cohort
    pats = patterns_for(topology, V)
    w = np.ones(len(pats)) if cohort_weights is None else np.asarray(cohort_weights, float)
    w = w / w.sum()
    bounds = np.round(np.cumsum(w) * n).astype(int)
    start = 0
    for c, (pat, end) in enumerate(zip(pats, bounds)):
        idx = order[start:end]
        for v in pat:
            mask[idx, v] = True
        cohort[idx] = c
        start = end
    return mask, cohort


def _make_mask_shift(topology, n, V, seed, cohort_weights, y, class_alpha, size_sigma):
    rng = np.random.default_rng(seed + 104729)
    pats = patterns_for(topology, V)
    w = np.ones(len(pats)) if cohort_weights is None else np.asarray(cohort_weights, float)
    if size_sigma > 0:
        w = w * rng.lognormal(0.0, size_sigma, len(pats))
    w = w / w.sum()
    cap = np.maximum(np.round(w * n).astype(int), 1)
    cap[-1] += n - cap.sum()
    order = rng.permutation(n)
    cohort = np.full(n, -1)
    if class_alpha > 0:
        K = int(y.max()) + 1
        pi = rng.dirichlet(np.full(K, class_alpha), size=len(pats))
        rem = cap.astype(float).copy()
        for i in order:
            p = pi[:, y[i]] * rem
            if p.sum() <= 0: p = rem.copy()
            c = rng.choice(len(pats), p=p / p.sum())
            cohort[i] = c; rem[c] -= 1
    else:
        bounds = np.cumsum(cap); start = 0
        for c, end in enumerate(bounds):
            cohort[order[start:end]] = c; start = end
    mask = np.zeros((n, V), dtype=bool)
    for c, pat in enumerate(pats):
        for v in pat:
            mask[cohort == c, v] = True
    return mask, cohort


def coobservation(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(float)
    N = m.T @ m
    np.fill_diagonal(N, 0.0)
    return N


def fiedler(N: np.ndarray, normalize_by: float | None = None) -> float:
    W = N.copy()
    if normalize_by:
        W = W / normalize_by
    L = np.diag(W.sum(1)) - W
    ev = np.linalg.eigvalsh(L)
    return float(ev[1])


def components(N: np.ndarray) -> list[list[int]]:
    V = N.shape[0]
    seen = [False] * V
    comps = []
    for s in range(V):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop(); comp.append(u)
            for v in range(V):
                if N[u, v] > 0 and not seen[v]:
                    seen[v] = True; stack.append(v)
        comps.append(sorted(comp))
    return comps
