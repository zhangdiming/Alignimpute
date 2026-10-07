from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


def rna_pcs(X, n_hvg=2000, k=100, seed=0):
    X = sp.csr_matrix(X, dtype=np.float64)
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    X = sp.diags(1e4 / lib) @ X; X.data = np.log1p(X.data)
    mu = np.asarray(X.mean(0)).ravel(); var = np.asarray(X.multiply(X).mean(0)).ravel() - mu ** 2
    hv = np.argsort(-var)[:n_hvg]
    D = X[:, hv].toarray(); D -= D.mean(0)
    U, S, _ = randomized_svd(D, n_components=k, random_state=seed)
    return U * S


def adt_pcs(X, k=50, seed=0):
    X = np.asarray(sp.csr_matrix(X).toarray(), dtype=np.float64)
    L = np.log1p(X); L = L - L.mean(1, keepdims=True)
    L -= L.mean(0)
    U, S, _ = randomized_svd(L, n_components=min(k, L.shape[1] - 1), random_state=seed)
    return U * S


def atac_lsi(X, k=100, seed=0):
    X = sp.csr_matrix(X, dtype=np.float64); X.data[:] = 1.0
    tf = sp.diags(1.0 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X
    idf = np.log1p(X.shape[0] / np.maximum(np.asarray(X.sum(0)).ravel(), 1))
    T = tf @ sp.diags(idf); T.data = np.log1p(T.data * 1e4)
    U, S, _ = randomized_svd(T, n_components=k + 1, random_state=seed)
    return (U * S)[:, 1:]


def assemble(blocks, n, V):
    out = {}; mask = np.zeros((n, V), bool)
    for v, (rows, Z) in enumerate(blocks):
        X = np.zeros((n, Z.shape[1]), np.float32); X[rows] = Z; out[f"X{v}"] = X; mask[rows, v] = True
    out["mask"] = mask
    return out


def mimitou(a):
    S = Path(a.src)
    import pandas as pd
    load = lambda f: sp.load_npz(S / f).T.tocsr()
    G = [load("GxC1.npz"), load("GxC2.npz")]; P = [load(f"PxC{i}.npz") for i in (1, 2, 3, 4)]; R = [load("RxC3.npz"), load("RxC4.npz")]
    meta = [pd.read_csv(S / f"meta_c{i}.csv", index_col=0) for i in (1, 2, 3, 4)]
    sizes = [m.shape[0] for m in meta]; off = np.cumsum([0] + sizes); n = off[-1]
    lab = np.concatenate([m["coarse_cluster"].values for m in meta]); classes = sorted(set(lab))
    y = np.array([classes.index(c) for c in lab]); src = np.repeat(np.arange(4), sizes)
    rna_rows = np.arange(off[0], off[2]); atac_rows = np.arange(off[2], off[4]); adt_rows = np.arange(n)
    Z_rna = rna_pcs(sp.vstack(G)); Z_adt = adt_pcs(sp.vstack(P)); Z_atac = atac_lsi(sp.vstack(R))
    out = assemble([(rna_rows, Z_rna), (adt_rows, Z_adt), (atac_rows, Z_atac)], n, 3)
    out.update(y=y, source=src)
    np.savez_compressed(a.out, **out)
    info = {"views": ["RNA", "ADT", "ATAC"], "classes": classes, "n": int(n), "n_per_class": np.bincount(y).tolist(),
            "sources": ["CITE control", "CITE stim", "ASAP control", "ASAP stim"], "n_per_source": sizes,
            "coobs": (out["mask"].T.astype(int) @ out["mask"].astype(int)).tolist()}
    Path(str(a.out) + ".json").write_text(json.dumps(info, indent=1)); print(json.dumps(info))


COARSE = {
    "CD14+ Mono": "CD14 Mono", "CD16+ Mono": "CD16 Mono", "cDC2": "cDC", "pDC": "pDC",
    "Naive CD20+ B IGKC+": "B", "Naive CD20+ B IGKC-": "B", "Naive CD20+ B": "B", "B1 B IGKC+": "B", "B1 B IGKC-": "B", "B1 B": "B",
    "Transitional B": "B", "Plasma cell IGKC+": "Plasma", "Plasma cell IGKC-": "Plasma", "Plasma cell": "Plasma",
    "CD4+ T naive": "CD4 T", "CD4+ T activated": "CD4 T", "CD4+ T CD314+ CD45RA+": "CD4 T", "CD4+ T activated integrinB7+": "CD4 T", "CD4+ T CD314+ CD45RA+ ": "CD4 T",
    "CD8+ T naive": "CD8 T", "CD8+ T": "CD8 T", "CD8+ T CD49f+": "CD8 T", "CD8+ T CD57+ CD45RA+": "CD8 T", "CD8+ T CD57+ CD45RO+": "CD8 T",
    "CD8+ T CD69+ CD45RA+": "CD8 T", "CD8+ T CD69+ CD45RO+": "CD8 T", "CD8+ T TIGIT+ CD45RA+": "CD8 T", "CD8+ T TIGIT+ CD45RO+": "CD8 T",
    "CD8+ T naive CD127+ CD26- CD101-": "CD8 T", "Plasmablast IGKC+": "Plasma", "Plasmablast IGKC-": "Plasma",
    "MAIT": "CD8 T", "gdT CD158b+": "CD8 T", "gdT TCRVD2+": "CD8 T", "T reg": "CD4 T", "dnT": "CD8 T", "T prog cycling": "T prog",
    "NK": "NK", "NK CD158e1+": "NK", "ILC": "ILC", "ILC1": "ILC",
    "HSC": "HSC/MPP", "MPP": "HSC/MPP", "Lymph prog": "Lymph prog", "G/M prog": "G/M prog", "MK/E prog": "MK/E prog",
    "Proerythroblast": "Erythroid", "Erythroblast": "Erythroid", "Normoblast": "Erythroid", "Reticulocyte": "Erythroid",
    "ID2-hi myeloid prog": "Myeloid prog", "Myeloid prog": "Myeloid prog",
}


def bmmc(a):
    import anndata as ad
    C = ad.read_h5ad(a.cite); M = ad.read_h5ad(a.multiome)
    C = C[C.obs["Site"].astype(str) == a.site].copy(); M = M[M.obs["Site"].astype(str) == a.site].copy()
    unmapped = sorted(set(C.obs["cell_type"].astype(str)) - set(COARSE)) + sorted(set(M.obs["cell_type"].astype(str)) - set(COARSE))
    cc = C.obs["cell_type"].astype(str).map(COARSE); mc = M.obs["cell_type"].astype(str).map(COARSE)
    keep = sorted(k for k in set(cc.dropna()) & set(mc.dropna()) if (cc == k).sum() >= 50 and (mc == k).sum() >= 50)
    C = C[cc.isin(keep).values].copy(); M = M[mc.isin(keep).values].copy()
    rng = np.random.default_rng(0)
    if a.n_per_source and C.n_obs > a.n_per_source: C = C[np.sort(rng.choice(C.n_obs, a.n_per_source, replace=False))].copy()
    if a.n_per_source and M.n_obs > a.n_per_source: M = M[np.sort(rng.choice(M.n_obs, a.n_per_source, replace=False))].copy()
    cc = C.obs["cell_type"].astype(str).map(COARSE).values; mc = M.obs["cell_type"].astype(str).map(COARSE).values
    ft = lambda A: A.var["feature_types"].astype(str).values
    counts = lambda A: A.layers["counts"] if "counts" in A.layers else A.X
    gC = C.var_names[ft(C) == "GEX"]; gM = M.var_names[ft(M) == "GEX"]; genes = sorted(set(gC) & set(gM))
    XC = counts(C); XM = counts(M)
    iC = {g: i for i, g in enumerate(C.var_names) if ft(C)[i] == "GEX"}; iM = {g: i for i, g in enumerate(M.var_names) if ft(M)[i] == "GEX"}
    rna = sp.vstack([sp.csr_matrix(XC)[:, [iC[g] for g in genes]], sp.csr_matrix(XM)[:, [iM[g] for g in genes]]])
    adt = sp.csr_matrix(XC)[:, np.where(ft(C) == "ADT")[0]]; atac = sp.csr_matrix(XM)[:, np.where(ft(M) == "ATAC")[0]]
    nC, nM = C.n_obs, M.n_obs; n = nC + nM
    out = assemble([(np.arange(n), rna_pcs(rna)), (np.arange(nC), adt_pcs(adt)), (np.arange(nC, n), atac_lsi(atac))], n, 3)
    lab = np.concatenate([cc, mc]); y = np.array([keep.index(c) for c in lab]); src = np.repeat([0, 1], [nC, nM])
    out.update(y=y, source=src)
    np.savez_compressed(a.out, **out)
    info = {"views": ["RNA", "ADT", "ATAC"], "classes": keep, "site": a.site, "n_per_source": a.n_per_source, "n": int(n), "n_per_class": np.bincount(y).tolist(),
            "n_cite": int(nC), "n_multiome": int(nM), "unmapped_types": unmapped, "n_genes": len(genes), "n_adt": int(adt.shape[1]), "n_peaks": int(atac.shape[1]),
            "coobs": (out["mask"].T.astype(int) @ out["mask"].astype(int)).tolist()}
    Path(str(a.out) + ".json").write_text(json.dumps(info, indent=1)); print(json.dumps(info))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mimitou"); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    q = sub.add_parser("bmmc"); q.add_argument("--cite", required=True); q.add_argument("--multiome", required=True); q.add_argument("--site", default="site1"); q.add_argument("--n-per-source", type=int, default=10000); q.add_argument("--out", required=True)
    a = ap.parse_args()
    mimitou(a) if a.cmd == "mimitou" else bmmc(a)
