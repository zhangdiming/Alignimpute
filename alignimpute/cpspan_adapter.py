from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np

from .topology import coobservation, fiedler, make_mask


def build_inputs(views, y, mask, zero_unobserved=True):
    from sklearn.preprocessing import MinMaxScaler
    import torch
    V = len(views); n = len(y)
    X = [MinMaxScaler().fit_transform(v).astype(np.float32) for v in views]
    if zero_unobserved:
        for v in range(V):
            X[v][~mask[:, v]] = 0.0
    Y = [y.astype(np.int64).copy() for _ in range(V)]
    missindex = mask.astype(np.float64)
    index_complete = [list(np.where(mask[:, v])[0]) for v in range(V)]
    max_len = max(len(ic) for ic in index_complete)
    filled = []
    for v in range(V):
        ic = index_complete[v]
        if len(ic) < max_len:
            ic = ic + random.sample(ic, max_len - len(ic)) if max_len - len(ic) <= len(ic) else ic + list(np.random.choice(ic, max_len - len(ic)))
        filled.append(ic)
    X_com = [torch.from_numpy(X[v][filled[v]]) for v in range(V)]
    Y_com = [Y[v][filled[v]] for v in range(V)]
    Xt = [torch.from_numpy(X[v]) for v in range(V)]
    return Xt, Y, missindex, X_com, Y_com


def run_one(repo: Path, views, y, mask, K, seed, device, pre_epochs=200, align_epochs=50, zero_unobserved=True, impute="released"):
    import torch, torch.nn.functional as F
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import main as M
    from network import Network
    from datasets import Data_Sampler, TrainDataset_Com
    from utils import get_Similarity
    from Nmetrics import evaluate
    from sklearn.cluster import KMeans
    M.seed_everything(seed)
    V = len(views)
    X, Y, missindex, X_com, Y_com = build_inputs(views, y, mask, zero_unobserved)

    class A: pass
    args = A(); args.V = V; args.K = K; args.batch_size = 256; args.Batch_Align = 256
    args.pretrain_epochs = pre_epochs; args.align_epochs = align_epochs; args.feature_dim = K
    args.view_dims = [v.shape[1] for v in views]
    M.para_loss = [1e-3, 1e-3]
    M.X = X; M.Y = Y
    model = Network(V, args.view_dims, args.feature_dim).to(device)
    opt_pre = torch.optim.Adam(model.parameters(), lr=0.0005)
    M.pretrain(model, opt_pre, args, device, X_com, Y_com)
    opt_al = torch.optim.Adam(model.parameters(), lr=0.0001)
    Miss_vecs = [missindex[:, v] for v in range(V)]
    fea_end = M.train_align(model, opt_al, args, device, X, Y, Miss_vecs)
    fea_end = [f.cpu() for f in fea_end]
    fea_final = [[] for _ in range(V)]
    final_batch = 2000
    ds2 = TrainDataset_Com(fea_end, Y)
    loader2 = torch.utils.data.DataLoader(dataset=ds2, batch_sampler=Data_Sampler(ds2, shuffle=False, batch_size=final_batch, drop_last=False))
    for batch_idx, (xs, ys) in enumerate(loader2):
        for v in range(V): xs[v] = torch.squeeze(xs[v]).to(device)
        cos = []
        for v in range(V):
            sm = get_Similarity(xs[v], xs[v]); sm = sm - torch.diag_embed(torch.diag(sm))
            for i in range(xs[0].shape[0]):
                if missindex[final_batch * batch_idx + i, v] == 0: sm[:, i] = 0
            cos.append(sm)
        for i in range(xs[0].shape[0]):
            for v in range(V):
                if missindex[final_batch * batch_idx + i, v] == 0:
                    _, idx = torch.sort(cos[v][i], descending=True); xs[v][i] = xs[v][idx[0]]
        for v in range(V): fea_final[v] += xs[v].tolist()
    if impute == "observed":
        fe = [torch.tensor(np.array(f), dtype=torch.float32, device=device) for f in fea_final]
        E = [f.cpu() for f in fea_end]
        for v in range(V):
            obs_v = np.where(missindex[:, v] == 1)[0]
            Bv = F.normalize(E[v][obs_v].to(device), dim=1)
            for i in np.where(missindex[:, v] == 0)[0]:
                us = [u for u in range(V) if missindex[i, u] == 1]
                if not us: continue
                q = F.normalize(E[us[0]][i:i + 1].to(device), dim=1)
                sim = (q @ Bv.T).squeeze(0); sim[obs_v == i] = -1e9
                fe[v][i] = E[v][obs_v[int(sim.argmax())]].to(device)
        fea_final = [f.cpu().tolist() for f in fe]
    fea = np.concatenate([np.array(f) for f in fea_final], axis=1)
    pred = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(fea).labels_
    acc, nmi, purity, fscore, precision, recall, ari = evaluate(Y[0], pred)
    return {"acc": float(acc), "nmi": float(nmi), "ari": float(ari)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--topologies", nargs="*", default=["chain"]); ap.add_argument("--sweep", type=float, nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pre-epochs", type=int, default=200); ap.add_argument("--align-epochs", type=int, default=50)
    ap.add_argument("--leaky", action="store_true", help="reproduce the released protocol (unobserved views keep their features)")
    ap.add_argument("--impute", choices=["released", "observed"], default="released", help="final imputation query: released (same-view embedding of the zeroed input) or observed (embedding of an observed view of the instance)")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.data); y = d["y"]; views = [d[f"X{i}"] for i in range(len([k for k in d.files if k.startswith("X")]))]
    n, V = len(y), len(views); K = int(len(np.unique(y)))
    out = a.out.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    configs = [(t, None) for t in a.topologies] + [("chain", p) for p in a.sweep]
    for topo, p in configs:
        for seed in a.seeds:
            w = None
            if p is not None: w = [1.0] * (V - 1); w[2] = p
            mask, _ = make_mask(topo, n, V, seed, cohort_weights=w)
            N = coobservation(mask); lam = fiedler(N, normalize_by=n)
            t0 = time.time()
            try:
                met = run_one(a.repo, views, y, mask, K, seed, device, a.pre_epochs, a.align_epochs, zero_unobserved=not a.leaky, impute=a.impute); status = "ok"
            except Exception as e:
                met = {}; status = f"failed: {type(e).__name__}: {e}"[:300]
            rec = {"topology": topo if p is None else f"chain-sweep-{p}", "seed": seed, "variant": "CPSPAN" + ("-leaky" if a.leaky else "") + ("-obsimp" if a.impute == "observed" else ""), "lambda2": lam,
                   "status": status, **met, "seconds": round(time.time() - t0, 1)}
            with out.open("a") as f: f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    main()
