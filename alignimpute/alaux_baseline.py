from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from .sync import _sinkhorn, edge_estimates, spectral_sync, threadpool_limits
from .metrics import clustering_metrics, kmeans_np


def polar(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def align(Z, mask, supervised, iters=30, n_sub=600, reg=0.05, pair_weight=1.0, seed=0, init="spectral", W0=None):
    V, n, d = Z.shape
    rng = np.random.default_rng(seed)
    sub = []
    for v in range(V):
        rows = np.where(mask[:, v])[0]
        sub.append(Z[v][rng.choice(rows, min(n_sub, len(rows)), replace=False)])
    cc = {(u, v): np.where(mask[:, u] & mask[:, v])[0] for u in range(V) for v in range(u + 1, V)}
    cc = {e: r for e, r in cc.items() if len(r) > 0}
    if W0 is not None and init == "given":
        W = [np.array(W0[v]) for v in range(V)]
    elif supervised and init == "spectral":
        edges = edge_estimates(list(Z), mask, min_pairs=d, thin=True)
        Q, _ = spectral_sync(edges, V, d)
        W = [Q[v] for v in range(V)]
    else:
        W = [np.eye(d) for _ in range(V)]
    for _ in range(iters):
        for v in range(V):
            M = np.zeros((d, d))
            for u in range(V):
                if u == v:
                    continue
                a, b = sub[v] @ W[v], sub[u] @ W[u]
                C = ((a[:, None, :] - b[None, :, :]) ** 2).sum(2); C = C / max(np.median(C), 1e-12)
                P = _sinkhorn(C, reg)
                M += sub[v].T @ (P @ b) * len(sub[v])
                e = (min(u, v), max(u, v))
                if supervised and e in cc:
                    r = cc[e]
                    M += pair_weight * Z[v][r].T @ (Z[u][r] @ W[u])
            W[v] = polar(M)
    return np.stack(W)


def evaluate(Z, mask, y, W, K, seed):
    V = Z.shape[0]
    acc = np.zeros((Z.shape[1], Z.shape[2])); cnt = np.zeros(Z.shape[1])
    Zf = [Z[v] @ W[v] for v in range(V)]
    for v in range(V):
        acc[mask[:, v]] += Zf[v][mask[:, v]]; cnt[mask[:, v]] += 1
    c = acc / np.maximum(cnt, 1)[:, None]; c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    lab, _ = kmeans_np(c, K, seed=seed)
    met = clustering_metrics(y, lab)
    u, v = 0, V - 1
    ru, rv = np.where(mask[:, u])[0], np.where(mask[:, v])[0]
    A_ = Zf[u][ru] / np.linalg.norm(Zf[u][ru], axis=1, keepdims=True); B_ = Zf[v][rv] / np.linalg.norm(Zf[v][rv], axis=1, keepdims=True)
    S = A_ @ B_.T; S[ru[:, None] == rv[None, :]] = -np.inf
    met["xkobs_0L"] = float((y[rv[S.argmax(1)]] == y[ru]).mean())
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--init", default="spectral", choices=["spectral", "identity"], help="initialisation of the semi-supervised A2 (identity = not from linear graph synchronisation)")
    ap.add_argument("--hosts", nargs="*", default=None, help="only embeddings of these host variants (e.g. A Anp)")
    a = ap.parse_args()
    for f in sorted(a.emb_dir.glob("*.npz")):
        dat = np.load(f); Z, mask, y = dat["Z"], dat["mask"].astype(bool), dat["y"]
        topo, seed, host = f.stem.rsplit("_", 2); seed = int(seed[1:])
        if a.hosts and host not in a.hosts: continue
        K = int(len(np.unique(y)))
        runs = (("alaux_A1", False), ("alaux_A2", True)) if a.init == "spectral" else (("alaux_A2id", True),)
        for name, sup in runs:
            t0 = time.time()
            with threadpool_limits(limits=1):
                W = align(Z, mask, sup, iters=a.iters, seed=seed, init=a.init)
            met = evaluate(Z, mask, y, W, K, seed)
            rec = {"topology": topo, "seed": seed, "host": host, "variant": name, "host_acc": float(dat["acc"]),
                   **met, "iters": a.iters, "seconds": round(time.time() - t0, 1)}
            with a.out.open("a") as fo:
                fo.write(json.dumps(rec) + "\n")
            print(f"{topo:18s} s{seed} {host:4s} {name}: acc={met['acc']:.4f} (host {float(dat['acc']):.4f}) xkobs_0L={met['xkobs_0L']:.3f} [{rec['seconds']}s]", flush=True)


if __name__ == "__main__":
    main()
