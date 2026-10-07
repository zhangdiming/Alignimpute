from __future__ import annotations
import argparse, gzip, json, re
from pathlib import Path
import numpy as np
import pandas as pd

MAP = {"Inh_MGE_PVALB": 0, "Inh_MGE_CALB1": 1, "Inh_MGE_B3GAT2": 1, "Inh_CGE_VIP": 2, "Inh_CGE_LAMP5": 3, "Inh_CGE_NDNF": 3}
PATCH = {"PVALB": 0, "SST": 1, "VIP": 2, "LAMP5/PAX6/Other": 3}
CLASSES = ["Pvalb", "Sst", "Vip", "Lamp5/Pax6/other"]


def pca(X, k):
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, X.shape[1], X.shape[0])
    return (U[:, :k] * S[:k]).astype(np.float32)


def std(X):
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)


def short_id(s):
    return re.sub(r"^.*_BA10_", "", s).replace("_indexed", "")


def read_rows(path, keep, dtype=np.float32, key=None):
    key = key or short_id
    out = []
    for ch in pd.read_csv(path, index_col=0, chunksize=500, dtype={c: dtype for c in []}):
        ch.index = [key(i) for i in ch.index]
        ch = ch[ch.index.isin(keep)]
        if len(ch): out.append(ch.astype(dtype))
    return pd.concat(out)


def meth_view(raw, kind, cells, min_cov, n_genes, k):
    mc = read_rows(raw / f"GSE140493_snmC2T-seq.{kind}_mc_gene_da.csv.gz", cells)
    cov = read_rows(raw / f"GSE140493_snmC2T-seq.{kind}_cov_gene_da.csv.gz", cells).loc[mc.index, mc.columns]
    ratio = (mc / cov.where(cov >= min_cov)).astype(np.float32)
    ratio = ratio.loc[:, ratio.isna().mean() < 0.2]
    glob = (mc.sum(1) / cov.sum(1)).values[:, None]
    ratio = ratio / glob
    ratio = ratio.fillna(ratio.mean())
    top = ratio.var().sort_values(ascending=False).index[:n_genes]
    return ratio.index.tolist(), pca(std(ratio[top].values), k), len(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-cov", type=int, default=20); ap.add_argument("--genes", type=int, default=2000); ap.add_argument("--pca", type=int, default=100)
    a = ap.parse_args(); raw = a.raw
    lab = pd.read_excel(raw / "luo2022_tableS6.xlsx", header=2)
    lab = lab[lab["Cell ID"].astype(str).str.contains("mCTseq")].copy()
    lab["id"] = lab["Cell ID"].map(short_id); lab = lab[lab["MajorType"].isin(MAP)]
    cat_y = lab.set_index("id")["MajorType"].map(MAP)
    cat_cells = set(cat_y.index)
    sym = pd.read_csv(raw / "snmcat_gene_symbols.csv", index_col=0).iloc[:, 0].fillna("")
    tot = pd.read_csv(raw / "snmcat_rna_totals.csv", index_col=0)["total"]
    R = read_rows(raw / "GSE140493_snmC2T-seq.gene_rna_counts.4358cell.60606gene.csv.gz", cat_cells)
    R = np.log2(R.div(tot.loc[R.index], axis=0) * 1e6 + 1)
    R.columns = [sym.get(c, "") for c in R.columns]
    R = R.loc[:, (R.columns != "") & ~R.columns.duplicated()]
    import pyreadr
    d1 = pyreadr.read_r(raw / "complete_patchseq_data_sets1.RData")["datPatch1"]
    d2 = pyreadr.read_r(raw / "complete_patchseq_data_sets2.RData")["datPatch2"]
    P = pd.concat([d1, d2], axis=1).T
    meta = pd.read_csv(raw / "LeeDalley_manuscript_metadata_v2.csv")
    meta = meta[meta["Revised_subclass_label"].isin(PATCH)]
    meta = meta[meta["patched_cell_container"].isin(P.index)].drop_duplicates("patched_cell_container")
    P = P.loc[meta["patched_cell_container"]]; P.index = meta["specimen_id_x"].astype(str).values
    pat_y = pd.Series(meta["Revised_subclass_label"].map(PATCH).values, index=P.index)
    ephys = pd.read_csv(raw / "LeeDalley_ephys_fx.csv").set_index("specimen_id"); ephys.index = ephys.index.astype(str)
    morph = pd.read_csv(raw / "LeeDalley_morpho_features.csv", index_col=0).set_index("specimen_id"); morph.index = morph.index.astype(str)
    cat_ids = list(R.index); pat_ids = list(P.index)
    samples = [f"snmCAT:{i}" for i in cat_ids] + [f"patchseq:{i}" for i in pat_ids]
    n, V = len(samples), 5
    y = np.r_[cat_y.loc[cat_ids].values, pat_y.loc[pat_ids].values].astype(int)
    assert len(y) == n, (len(y), n)
    source = np.r_[np.zeros(len(cat_ids), int), np.ones(len(pat_ids), int)]
    mask = np.zeros((n, V), bool); out = {}; info = {"views": ["ephys", "morph", "RNA", "mCH", "GpC"], "classes": CLASSES}
    genes = sorted(set(R.columns) & set(P.columns))
    Rs, Ps = R[genes], P[genes]
    score = Rs.var().rank(ascending=False) + Ps.var().rank(ascending=False)
    top = score.sort_values().index[:a.genes]
    Xr = np.vstack([std(Rs[top].values), std(Ps[top].values)])
    out["X2"] = pca(Xr, a.pca); mask[:, 2] = True; info["rna_genes_shared"] = len(genes)
    for j, kind in ((3, "HCHN"), (4, "GCYN")):
        ids, Z, ng = meth_view(raw, kind, set(cat_ids), a.min_cov, a.genes, a.pca)
        pos = {c: i for i, c in enumerate(cat_ids)}; full = np.zeros((n, Z.shape[1]), np.float32)
        rows = [pos[c] for c in ids]; full[rows] = Z; mask[rows, j] = True; out[f"X{j}"] = full; info[f"{kind}_genes"] = ng
    for j, T in ((0, ephys), (1, morph)):
        T = T.select_dtypes("number"); T = T.loc[:, T.isna().mean() < 0.2]; T = T.fillna(T.mean())
        have = [i for i in pat_ids if i in T.index]
        Z = std(T.loc[have].values).astype(np.float32)
        full = np.zeros((n, Z.shape[1]), np.float32); rows = [len(cat_ids) + pat_ids.index(i) for i in have]
        full[rows] = Z; mask[rows, j] = True; out[f"X{j}"] = full
    N = mask.T.astype(int) @ mask.astype(int)
    info.update(n=int(n), n_per_class=np.bincount(y).tolist(), n_per_source=np.bincount(source).tolist(), coobs=N.tolist())
    np.savez_compressed(a.out, **out, y=y, mask=mask, source=source, samples=np.array(samples))
    (a.out.with_suffix(".json")).write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
