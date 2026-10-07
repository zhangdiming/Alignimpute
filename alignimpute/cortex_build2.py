from __future__ import annotations
import argparse, gzip, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from .cortex_build import MAP, PATCH, CLASSES, pca, std, short_id, read_rows

CHROMS = [f"chr{i}" for i in range(1, 23)]
HG19 = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516,
        133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566]
RES, BAND, MIN_DIST = 1_000_000, 10, 20_000


def full_id(s):
    return str(s).replace("_indexed", "")


def cell_contacts(path):
    nb = {c: L // RES + 1 for c, L in zip(CHROMS, HG19)}
    M = {c: np.zeros((nb[c], nb[c]), np.float32) for c in CHROMS}
    with gzip.open(path, "rt") as f:
        for line in f:
            t = line.split()
            if len(t) < 5 or t[1] != t[3] or t[1] not in M: continue
            p1, p2 = int(t[2]), int(t[4])
            if abs(p1 - p2) < MIN_DIST: continue
            i, j = sorted((p1 // RES, p2 // RES)); M[t[1]][i, j] += 1
    return M


def impute(A, rp=0.5, iters=30):
    A = A + A.T - np.diag(np.diag(A))
    P = np.pad(A, 1); A = sum(P[1 + di:1 + di + A.shape[0], 1 + dj:1 + dj + A.shape[1]] for di in (-1, 0, 1) for dj in (-1, 0, 1)) / 9.0
    A = A + np.eye(len(A)) * 1e-6
    W = A / A.sum(1, keepdims=True)
    Q = np.eye(len(A)); I = np.eye(len(A))
    for _ in range(iters): Q = rp * I + (1 - rp) * Q @ W
    return Q


def cell_features(path):
    M = cell_contacts(path); out = []
    for c in CHROMS:
        Q = impute(M[c]); n = len(Q)
        iu = [(i, j) for i in range(n) for j in range(i, min(n, i + BAND + 1))]
        out.append(np.log1p(1e3 * np.array([Q[i, j] for i, j in iu], np.float32)))
    return out


def threec_view(raw3c, cells, k, workers):
    files = {full_id(p.name.split("_", 1)[1].replace("_contacts.txt.gz", "")): p for p in raw3c.glob("*_contacts.txt.gz")}
    have = [c for c in cells if c in files]
    with ProcessPoolExecutor(workers) as ex: feats = list(ex.map(cell_features, [files[c] for c in have], chunksize=4))
    per_chrom = [pca(std(np.vstack([f[ci] for f in feats])), 20) for ci in range(len(CHROMS))]
    return have, pca(std(np.hstack(per_chrom)), k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True); ap.add_argument("--raw3c", type=Path, required=True)
    ap.add_argument("--meth3c", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-cov", type=int, default=20); ap.add_argument("--genes", type=int, default=2000); ap.add_argument("--pca", type=int, default=100)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args(); raw = a.raw
    lab = pd.read_excel(raw / "luo2022_tableS6.xlsx", header=2); lab = lab[lab["MajorType"].isin(MAP)]
    catl = lab[lab["Cell ID"].astype(str).str.contains("mCTseq")]; m3l = lab[lab["Cell ID"].astype(str).str.contains("snm3Cseq")]
    cat_y = pd.Series(catl["MajorType"].map(MAP).values, index=catl["Cell ID"].map(short_id))
    m3_y = pd.Series(m3l["MajorType"].map(MAP).values, index=m3l["Cell ID"].map(full_id))
    sym = pd.read_csv(raw / "snmcat_gene_symbols.csv", index_col=0).iloc[:, 0].fillna("")
    tot = pd.read_csv(raw / "snmcat_rna_totals.csv", index_col=0)["total"]
    R = read_rows(raw / "GSE140493_snmC2T-seq.gene_rna_counts.4358cell.60606gene.csv.gz", set(cat_y.index))
    R = np.log2(R.div(tot.loc[R.index], axis=0) * 1e6 + 1); R.columns = [sym.get(c, "") for c in R.columns]
    R = R.loc[:, (R.columns != "") & ~R.columns.duplicated()]
    import pyreadr
    P = pd.concat([pyreadr.read_r(raw / "complete_patchseq_data_sets1.RData")["datPatch1"],
                   pyreadr.read_r(raw / "complete_patchseq_data_sets2.RData")["datPatch2"]], axis=1).T
    meta = pd.read_csv(raw / "LeeDalley_manuscript_metadata_v2.csv")
    meta = meta[meta["Revised_subclass_label"].isin(PATCH) & meta["patched_cell_container"].isin(P.index)].drop_duplicates("patched_cell_container")
    P = P.loc[meta["patched_cell_container"]]; P.index = meta["specimen_id_x"].astype(str).values
    pat_y = pd.Series(meta["Revised_subclass_label"].map(PATCH).values, index=P.index)
    cat_ids, pat_ids = list(R.index), list(P.index)
    m3_ids = [c for c in m3_y.index]
    samples = [f"snmCAT:{i}" for i in cat_ids] + [f"patchseq:{i}" for i in pat_ids] + [f"snm3C:{i}" for i in m3_ids]
    n, V = len(samples), 6; off_p, off_3 = len(cat_ids), len(cat_ids) + len(pat_ids)
    y = np.r_[cat_y.loc[cat_ids].values, pat_y.loc[pat_ids].values, m3_y.loc[m3_ids].values].astype(int); assert len(y) == n
    source = np.r_[np.zeros(len(cat_ids), int), np.ones(len(pat_ids), int), 2 * np.ones(len(m3_ids), int)]
    mask = np.zeros((n, V), bool); out = {}
    info = {"views": ["ephys", "morph", "RNA", "GpC", "mCH", "3C"], "classes": CLASSES}
    genes = sorted(set(R.columns) & set(P.columns)); Rs, Ps = R[genes], P[genes]
    top = (Rs.var().rank(ascending=False) + Ps.var().rank(ascending=False)).sort_values().index[:a.genes]
    X = np.zeros((n, a.pca), np.float32); X[:off_3] = pca(np.vstack([std(Rs[top].values), std(Ps[top].values)]), a.pca)
    out["X2"] = X; mask[:off_3, 2] = True
    def ratio(mc_path, cov_path, keep, key=None):
        mc = read_rows(mc_path, keep, key=key); cov = read_rows(cov_path, keep, key=key).loc[mc.index, mc.columns]
        r = (mc / cov.where(cov >= a.min_cov)).astype(np.float32); g = (mc.sum(1) / cov.sum(1)).values[:, None]
        return r / g
    G = ratio(raw / "GSE140493_snmC2T-seq.GCYN_mc_gene_da.csv.gz", raw / "GSE140493_snmC2T-seq.GCYN_cov_gene_da.csv.gz", set(cat_ids))
    G = G.loc[:, G.isna().mean() < 0.2]; G = G.fillna(G.mean()); topg = G.var().sort_values(ascending=False).index[:a.genes]
    X = np.zeros((n, a.pca), np.float32); pos = {c: i for i, c in enumerate(cat_ids)}; rows = [pos[c] for c in G.index]
    X[rows] = pca(std(G[topg].values), a.pca); mask[rows, 3] = True; out["X3"] = X
    Mc = ratio(raw / "GSE140493_snmC2T-seq.HCHN_mc_gene_da.csv.gz", raw / "GSE140493_snmC2T-seq.HCHN_cov_gene_da.csv.gz", set(cat_ids))
    M3 = ratio(a.meth3c / "GSE140493_snm3C-seq.CHN_mc_gene_da.csv.gz", a.meth3c / "GSE140493_snm3C-seq.CHN_cov_gene_da.csv.gz", set(m3_ids), key=full_id)
    shared = [g for g in Mc.columns if g in M3.columns and Mc[g].isna().mean() < 0.2 and M3[g].isna().mean() < 0.2]
    Mc, M3 = Mc[shared].fillna(Mc[shared].mean()), M3[shared].fillna(M3[shared].mean())
    topm = (Mc.var().rank(ascending=False) + M3.var().rank(ascending=False)).sort_values().index[:a.genes]
    Z = pca(np.vstack([std(Mc[topm].values), std(M3[topm].values)]), a.pca)
    X = np.zeros((n, a.pca), np.float32); pos = {c: i for i, c in enumerate(samples)}
    rows = [pos[f"snmCAT:{c}"] for c in Mc.index] + [pos[f"snm3C:{c}"] for c in M3.index]
    X[rows] = Z; mask[rows, 4] = True; out["X4"] = X; info["mCH_genes_shared"] = len(shared)
    have, Z = threec_view(a.raw3c, m3_ids, a.pca, a.workers)
    X = np.zeros((n, Z.shape[1]), np.float32); rows = [pos[f"snm3C:{c}"] for c in have]; X[rows] = Z; mask[rows, 5] = True; out["X5"] = X
    info["3C_cells"] = len(have)
    ephys = pd.read_csv(raw / "LeeDalley_ephys_fx.csv").set_index("specimen_id"); ephys.index = ephys.index.astype(str)
    morph = pd.read_csv(raw / "LeeDalley_morpho_features.csv", index_col=0).set_index("specimen_id"); morph.index = morph.index.astype(str)
    for j, T in ((0, ephys), (1, morph)):
        T = T.select_dtypes("number"); T = T.loc[:, T.isna().mean() < 0.2]; T = T.fillna(T.mean())
        hv = [i for i in pat_ids if i in T.index]; Zt = std(T.loc[hv].values).astype(np.float32)
        X = np.zeros((n, Zt.shape[1]), np.float32); rows = [pos[f"patchseq:{i}"] for i in hv]; X[rows] = Zt; mask[rows, j] = True; out[f"X{j}"] = X
    keep = mask.sum(1) >= 2
    out = {k: v[keep] for k, v in out.items()}
    y, mask, source, samples = y[keep], mask[keep], source[keep], np.array(samples)[keep]
    N = mask.T.astype(int) @ mask.astype(int)
    info.update(n=int(len(y)), n_per_class=np.bincount(y).tolist(), n_per_source=np.bincount(source).tolist(), coobs=N.tolist())
    np.savez_compressed(a.out, **out, y=y, mask=mask, source=source, samples=samples)
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=1)); print(json.dumps({k: v for k, v in info.items() if k != "coobs"}, indent=1))
    print(np.array(N))


if __name__ == "__main__":
    main()
