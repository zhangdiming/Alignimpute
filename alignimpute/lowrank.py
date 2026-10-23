from __future__ import annotations

import numpy as np
import torch

PATH = tuple(float(x) for x in np.geomspace(0.5, 0.005, 12))


def _svt(Z, lam, q):
    U, S, V = torch.svd_lowrank(Z, q=q, niter=2)
    S = torch.clamp(S - lam, min=0.0); k = int((S > 0).sum())
    return (U[:, :k] * S[:k]) @ V[:, :k].T, k


def soft_impute_path(X, M, fracs=PATH, iters=100, tol=1e-4, q_max=100, seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float64, device=device); Mt = torch.as_tensor(M, dtype=torch.bool, device=device)
    P = torch.where(Mt, Xt, torch.zeros_like(Xt))
    q = min(q_max, min(P.shape))
    lam0 = float(torch.svd_lowrank(P, q=min(5, q), niter=4)[1][0])
    L = torch.zeros_like(P); out = []
    for fr in fracs:
        lam = fr * lam0
        for it in range(1, iters + 1):
            Lnew, k = _svt(torch.where(Mt, Xt, L), lam, q)
            delta = torch.linalg.norm(Lnew - L) / (torch.linalg.norm(L) + 1e-12)
            L = Lnew
            if delta < tol:
                break
        out.append((fr, L.cpu().numpy(), k, it))
    return out


def holdout_blocks(mask, seed, frac=0.1):
    rng = np.random.default_rng(seed + 104729)
    elig = np.where(mask.sum(1) >= 2)[0]
    rows = rng.choice(elig, int(round(frac * len(elig))), replace=False) if len(elig) else np.array([], int)
    held = [(int(i), int(rng.choice(np.where(mask[i])[0]))) for i in rows]
    m2 = mask.copy()
    for i, v in held:
        m2[i, v] = False
    return m2, held


def lowrank_impute(Xc, mask, dims, seed=0, all_fits=False, device=None):
    off = np.concatenate([[0], np.cumsum(dims)]); E = lambda m: np.repeat(m, dims, axis=1)
    m2, held = holdout_blocks(mask, seed)
    mu = (np.where(E(m2), Xc, 0.0).sum(0) / np.maximum(E(m2).sum(0), 1))
    Xh = np.where(E(m2), Xc - mu, 0.0)
    err = {}
    for fr, L, k, it in soft_impute_path(Xh, E(m2), seed=seed, device=device):
        num = sum(((L[i, off[v]:off[v + 1]] - (Xc[i, off[v]:off[v + 1]] - mu[off[v]:off[v + 1]])) ** 2).sum() for i, v in held)
        den = sum(((Xc[i, off[v]:off[v + 1]] - mu[off[v]:off[v + 1]]) ** 2).sum() for i, v in held)
        err[fr] = float(num / max(den, 1e-12))
    best = min(err, key=err.get)
    Mf = E(mask); mu = np.where(Mf, Xc, 0.0).sum(0) / np.maximum(Mf.sum(0), 1)
    X0 = np.where(Mf, Xc - mu, 0.0)
    path = soft_impute_path(X0, Mf, fracs=PATH if all_fits else tuple(f for f in PATH if f >= best), seed=seed, device=device)
    fits = {fr: (np.where(Mf, Xc, L + mu), k, it) for fr, L, k, it in path}
    info = {"lam_frac": round(best, 5), "rank": int(fits[best][1]), "iters": int(fits[best][2]), "n_heldout": len(held),
            "heldout_err": {f"{k:.4f}": round(v, 5) for k, v in err.items()}}
    return fits[best][0], info, ({f"{fr:.4f}": f[0] for fr, f in fits.items()} if all_fits else None)
