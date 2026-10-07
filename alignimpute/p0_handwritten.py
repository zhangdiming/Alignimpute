from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import clustering_metrics, kmeans_np, match_labels, retrieval_r1, xview_knn_acc
from .sync import edge_estimates, spectral_sync, tree_sync
from .topology import coobservation, components, fiedler, make_mask


class MV(nn.Module):
    def __init__(self, dims, d=32, hid=256):
        super().__init__()
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(di, hid), nn.BatchNorm1d(hid), nn.ReLU(),
                                                nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, d)) for di in dims])
        self.dec = nn.ModuleList([nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, di)) for di in dims])
        self.register_buffer("Q", torch.eye(d).repeat(len(dims), 1, 1))

    def embed(self, v, x, framed=True):
        z = F.normalize(self.enc[v](x), dim=1)
        return z @ self.Q[v] if framed else z


def info_nce(zu, zv, tau):
    logits = zu @ zv.T / tau
    lab = torch.arange(len(zu), device=zu.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))


def train(views, mask, variant, seed, device, epochs=300, tau=0.1, lr=1e-3, sync_every=25, lam_rec=1.0, lam_proto=1.0, K=10,
          hid=256, d=32, wd=0.0):
    use_proto = variant not in ("Cnp", "Anp", "Cwpnp")
    wp_from_current = variant == "Cwpid"
    use_edges = variant != "PM"
    proto_start = 150 if variant == "Ad" else (50 if variant in ("Cpl", "Cthinpl", "Apl") else 0)
    if getattr(train, "proto_start", None) is not None and variant in ("A", "C", "Cthin"):
        proto_start = train.proto_start
    late = variant in ("Clate", "Cthinlate")
    thin = variant in ("Cthin", "Ctthin", "Cg", "Cthinlate", "Cthinpl", "Bthin", "Cb", "Cbs", "CL", "CLw", "CLg")
    linit = variant in ("AL", "ALw", "CL", "CLw", "CLg")
    warm = variant in ("ALw", "CLw", "CLg", "Aw")
    pseudo = getattr(train, "pseudo", None) or []
    pseudo_loss = variant in ("Bp", "Cb"); pseudo_sync = variant in ("Cb", "Cbs"); beta = getattr(train, "bridge_beta", 0.5)
    syncer = tree_sync if variant in ("Ct", "Ctthin") else spectral_sync
    gated = variant in ("Cg", "CLg")
    torch.manual_seed(seed); np.random.seed(seed)
    n, V = mask.shape
    X = [torch.tensor(v, dtype=torch.float32, device=device) for v in views]
    M = torch.tensor(mask, device=device)
    model = MV([v.shape[1] for v in views], d=d, hid=hid).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    N = coobservation(mask)
    edges_idx = [(u, v) for u in range(V) for v in range(u + 1, V) if N[u, v] > 0]
    framed = variant in ("C", "Cnp", "Ct", "Cthin", "Ctthin", "Cg", "Cwp", "Cwpnp", "Cwpid", "Clate", "Cthinlate", "Cpl", "Cthinpl", "Cb", "Cbs", "CL", "CLw", "CLg")
    joint_wp = variant in ("Cwp", "Cwpnp", "Cwpid")
    gate_open = not gated
    GATE = train.gate if hasattr(train, "gate") else (0.7, 0.9)
    protos, assign = None, None
    ptype = getattr(train, "proto_type", "ours")
    mu, Ptgt, head = None, None, None
    train.last_trace = []
    train.last_churn = []
    for ep in range(epochs):
        model.train()
        loss = 0.0
        for v in range(V):
            idx = M[:, v].nonzero().squeeze(1)
            z = model.enc[v](X[v][idx])
            loss = loss + lam_rec * F.mse_loss(model.dec[v](z), X[v][idx])
        Z = [None] * V
        for (u, v) in (edges_idx if use_edges else []):
            cc = (M[:, u] & M[:, v]).nonzero().squeeze(1)
            if len(cc) < 2:
                continue
            zu = model.embed(u, X[u][cc], framed=framed); zv = model.embed(v, X[v][cc], framed=framed)
            loss = loss + info_nce(zu, zv, tau)
        nbr = getattr(train, "nbr", None); gamma = getattr(train, "nbr_gamma", 0.0)
        if nbr and gamma > 0:
            for v, (rows, nb) in nbr.items():
                pick = nb[np.arange(len(rows)), torch.randint(nb.shape[1], (len(rows),)).numpy()]
                ri = torch.as_tensor(rows, device=device); pj = torch.as_tensor(pick, device=device)
                loss = loss + gamma * info_nce(model.embed(v, X[v][ri], framed=False), model.embed(v, X[v][pj], framed=False), tau)
        if pseudo_loss:
            for (u, v, I, J) in pseudo:
                if len(I) < 2: continue
                Ii = torch.as_tensor(I, device=device); Jj = torch.as_tensor(J, device=device)
                loss = loss + beta * info_nce(model.embed(u, X[u][Ii], framed=framed), model.embed(v, X[v][Jj], framed=framed), tau)
        if protos is not None and use_proto:
            if variant == "PM":
                for v in range(V):
                    idx = M[:, v].nonzero().squeeze(1)
                    P = torch.tensor(protos[v], dtype=torch.float32, device=device)
                    zf = model.embed(v, X[v][idx], framed=False)
                    loss = loss + lam_proto * F.cross_entropy(zf @ P.T / tau, torch.tensor(assign[v][idx.cpu().numpy()], device=device))
            elif ptype == "dec":
                Pt = torch.tensor(Ptgt, dtype=torch.float32, device=device)
                for v in range(V):
                    idx = M[:, v].nonzero().squeeze(1)
                    zf = model.embed(v, X[v][idx], framed=framed)
                    q = 1.0 / (1.0 + torch.cdist(zf, mu) ** 2); q = q / q.sum(1, keepdim=True)
                    loss = loss + lam_proto * F.kl_div(q.log(), Pt[idx], reduction="batchmean")
            elif ptype == "deepcluster":
                lab_t = torch.tensor(assign, device=device); cnt = torch.bincount(lab_t, minlength=K).float()
                wcls = torch.where(cnt > 0, 1.0 / cnt.clamp(min=1), torch.zeros_like(cnt)); wcls = wcls / wcls[cnt > 0].mean()
                for v in range(V):
                    idx = M[:, v].nonzero().squeeze(1)
                    zf = model.embed(v, X[v][idx], framed=framed)
                    loss = loss + lam_proto * F.cross_entropy(head(zf), lab_t[idx], weight=wcls)
            else:
                P = torch.tensor(protos, dtype=torch.float32, device=device)
                for v in range(V):
                    idx = M[:, v].nonzero().squeeze(1)
                    zf = model.embed(v, X[v][idx], framed=framed)
                    loss = loss + lam_proto * F.cross_entropy(zf @ P.T / tau, torch.tensor(assign[idx.cpu().numpy()], device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % sync_every == 0 and ep + 1 < epochs:
            if getattr(train, "trace", False):
                dg0 = frame_drift_diagnostics(model, X, mask)
                rec_t = {"epoch": ep + 1, "path_dev": dg0.get("diag_path_dev"), "path_dev_energy": dg0.get("diag_path_dev_energy"), "edge_resid": dg0.get("diag_edge_resid")}
                if getattr(train, "acc_trace_y", None) is not None:
                    cons_t = consensus(model, X, mask, device, framed=framed and gate_open)
                    lab_t, _ = kmeans_np(cons_t, K, seed=seed); rec_t["acc"] = clustering_metrics(train.acc_trace_y, lab_t)["acc"]
                train.last_trace.append(rec_t)
            if gated and not gate_open:
                dg = frame_drift_diagnostics(model, X, mask)
                if dg.get("diag_path_len", 1) >= 2 and dg.get("diag_path_dev", 0) >= GATE[0] and dg.get("diag_edge_resid", 1.0) <= GATE[1]:
                    gate_open = True
                    print(f"    [gate] opened at epoch {ep+1}: path_dev={dg.get('diag_path_dev'):.2f} edge_resid={dg.get('diag_edge_resid'):.2f}", flush=True)
            if framed and gate_open and not (late and ep + 1 == sync_every):
                if getattr(train, "trace", False) and train.last_trace:
                    cons_b = consensus(model, X, mask, device, framed=True)
                Q, comps = synchronise_wp(model, X, mask, seed, from_current=wp_from_current) if joint_wp else synchronise(model, X, mask, device, thin=thin, syncer=syncer, pseudo=pseudo if pseudo_sync else None)
                model.Q.copy_(torch.tensor(Q, dtype=torch.float32, device=device))
                if getattr(train, "trace", False) and train.last_trace:
                    cons_s = consensus(model, X, mask, device, framed=True)
                    jump = np.linalg.norm(cons_s - cons_b, axis=1)
                    lab_s, _ = kmeans_np(cons_s, K, seed=seed)
                    rec_s = train.last_trace[-1]; rec_s["jump_mean"] = float(jump.mean()); rec_s["jump_max"] = float(jump.max())
                    if protos is not None and not isinstance(protos, dict):
                        Pn = np.asarray(protos); D = np.linalg.norm(Pn[:, None] - Pn[None], axis=2); rec_s["proto_sep_min"] = float(D[~np.eye(len(Pn), dtype=bool)].min())
                        from scipy.optimize import linear_sum_assignment
                        Cm = np.zeros((K, K)); np.add.at(Cm, (assign, lab_s), 1); rr, cc = linear_sum_assignment(-Cm)
                        rec_s["reassign_post"] = float(1 - Cm[rr, cc].sum() / len(lab_s))
                    if getattr(train, "acc_trace_y", None) is not None:
                        rec_s["acc_post"] = clustering_metrics(train.acc_trace_y, lab_s)["acc"]
            if ep + 1 >= proto_start and variant == "PM":
                assign, protos = {}, {}
                Zr = raw_embeddings(model, X, device)
                for v in range(V):
                    idx = np.where(mask[:, v])[0]
                    lab, cen = kmeans_np(Zr[v][idx], K, seed=seed)
                    a_v = np.full(n, -1); a_v[idx] = lab
                    assign[v] = a_v; protos[v] = cen / np.linalg.norm(cen, axis=1, keepdims=True)
            elif ep + 1 >= proto_start and ptype == "dec":
                cons = consensus(model, X, mask, device, framed=framed); ct = torch.tensor(cons, dtype=torch.float32, device=device)
                prev = assign
                if mu is None:
                    _, cen = kmeans_np(cons, K, seed=seed)
                    mu = torch.nn.Parameter(torch.tensor(cen, dtype=torch.float32, device=device)); opt.add_param_group({"params": [mu]})
                elif framed and gate_open:
                    with torch.no_grad():
                        mu.data = (torch.tensor(Ptgt, dtype=torch.float32, device=device).T @ ct) / torch.tensor(Ptgt, dtype=torch.float32, device=device).sum(0)[:, None].clamp(min=1e-8)
                with torch.no_grad():
                    q = 1.0 / (1.0 + torch.cdist(ct, mu) ** 2); q = q / q.sum(1, keepdim=True)
                    pt = q ** 2 / q.sum(0); pt = pt / pt.sum(1, keepdim=True)
                Ptgt = pt.cpu().numpy(); assign = q.argmax(1).cpu().numpy(); protos = mu.detach().cpu().numpy()
                if prev is not None:
                    from scipy.optimize import linear_sum_assignment
                    C = np.zeros((K, K)); np.add.at(C, (prev, assign), 1)
                    rr, cc = linear_sum_assignment(-C); train.last_churn.append(float(1 - C[rr, cc].sum() / len(assign)))
            elif ep + 1 >= proto_start:
                cons = consensus(model, X, mask, device, framed=framed)
                prev = assign
                if linit and prev is None:
                    assign = np.asarray(train.init_assign).copy()
                    protos = np.stack([cons[assign == k].mean(0) if (assign == k).any() else cons[np.random.randint(len(cons))] for k in range(K)])
                elif warm and prev is not None:
                    assign, protos = kmeans_warm(cons, protos, K)
                else:
                    assign, protos = kmeans_np(cons, K, seed=seed)
                if ptype == "deepcluster":
                    if head is None:
                        head = torch.nn.Linear(model.Q.shape[1], K).to(device); opt.add_param_group({"params": list(head.parameters())})
                    else:
                        head.reset_parameters()
                        for prm in head.parameters(): opt.state.pop(prm, None)
                if prev is not None and not isinstance(prev, dict):
                    from scipy.optimize import linear_sum_assignment
                    C = np.zeros((K, K)); np.add.at(C, (prev, assign), 1)
                    rr, cc = linear_sum_assignment(-C); train.last_churn.append(float(1 - C[rr, cc].sum() / len(assign)))
                protos = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    if variant in ("B", "Bthin") or (framed and gate_open):
        Q, comps = synchronise_wp(model, X, mask, seed, from_current=wp_from_current) if joint_wp else synchronise(model, X, mask, device, thin=thin, syncer=syncer)
        model.Q.copy_(torch.tensor(Q, dtype=torch.float32, device=device))
    else:
        comps = components(N)
    train.last_gate_open = gate_open
    train.last_protos = protos
    return model, comps


def kmeans_warm(X, init, K, iters=100):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, init=np.asarray(init, dtype=np.float64), n_init=1, max_iter=iters).fit(np.asarray(X, dtype=np.float64))
    return km.labels_.astype(int), km.cluster_centers_


@torch.no_grad()
def raw_embeddings(model, X, device=None):
    model.eval()
    return [model.embed(v, X[v], framed=False).cpu().numpy() for v in range(len(X))]


def synchronise(model, X, mask, device, thin=False, syncer=spectral_sync, pseudo=None):
    Z = raw_embeddings(model, X, device)
    edges = edge_estimates(Z, mask, min_pairs=Z[0].shape[1], thin=thin)
    for (u, v, I, J) in (pseudo or []):
        if len(I) < 2: continue
        from .sync import procrustes
        R, res = procrustes(Z[u][I], Z[v][J]); edges[(u, v)] = (R, 0.5 * len(I) / (1.0 + 10.0 * res), len(I))
    return syncer(edges, len(X), Z[0].shape[1])


def synchronise_wp(model, X, mask, seed, from_current=False):
    from .alaux_baseline import align
    from .sync import threadpool_limits
    Z = np.stack(raw_embeddings(model, X))
    with threadpool_limits(limits=1):
        W = align(Z, mask, True, seed=seed, init="given" if from_current else "spectral",
                  W0=model.Q.detach().cpu().numpy() if from_current else None)
    return W, components(coobservation(mask))


@torch.no_grad()
def consensus(model, X, mask, device, framed):
    model.eval()
    n, V = mask.shape
    acc = np.zeros((n, model.Q.shape[1])); cnt = np.zeros(n)
    for v in range(V):
        z = model.embed(v, X[v], framed=framed).cpu().numpy()
        acc[mask[:, v]] += z[mask[:, v]]; cnt[mask[:, v]] += 1
    c = acc / np.maximum(cnt, 1)[:, None]
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)


def prototype_graph_matching(model, X, mask, K, seed):
    from scipy.optimize import linear_sum_assignment
    Zr = raw_embeddings(model, X, None)
    n, V = mask.shape
    labs, cens = [], []
    for v in range(V):
        idx = np.where(mask[:, v])[0]
        lab, cen = kmeans_np(Zr[v][idx], K, seed=seed)
        a_v = np.full(n, -1); a_v[idx] = lab
        labs.append(a_v); cens.append(cen / np.linalg.norm(cen, axis=1, keepdims=True))
    D = [np.linalg.norm(c[:, None] - c[None], axis=2) for c in cens]
    ref = 0
    maps = [np.arange(K)]
    for v in range(1, V):
        prof_r = np.sort(D[ref], axis=1); prof_v = np.sort(D[v], axis=1)
        cost = np.linalg.norm(prof_r[:, None] - prof_v[None], axis=2)
        r, c = linear_sum_assignment(cost)
        perm = np.empty(K, int); perm[c] = r
        def qcost(p):
            Dp = D[v][np.ix_(np.argsort(p), np.argsort(p))]
            return np.linalg.norm(D[ref] - Dp)
        improved = True
        while improved:
            improved = False
            for i in range(K):
                for j in range(i + 1, K):
                    q = perm.copy(); q[i], q[j] = q[j], q[i]
                    if qcost(q) < qcost(perm) - 1e-9:
                        perm = q; improved = True
        maps.append(perm)
    votes = np.zeros((n, K))
    for v in range(V):
        obs = mask[:, v]
        votes[np.where(obs)[0], maps[v][labs[v][obs]]] += 1
    return votes.argmax(1)


def frame_drift_diagnostics(model, X, mask):
    Z = raw_embeddings(model, X)
    d = Z[0].shape[1]; V = len(Z)
    edges = edge_estimates(Z, mask, min_pairs=d, thin=True)
    if not edges:
        return {"diag_edge_dev": float("nan"), "diag_path_dev": float("nan")}
    dev = {e: float(np.linalg.norm(R - np.eye(d)) / np.sqrt(d)) for e, (R, w, n) in edges.items()}
    resid = {e: float(max(0.0, ((1.0 if n >= d else 0.5) * n / max(w, 1e-9) - 1.0) / 10.0)) for e, (R, w, n) in edges.items()}
    adj = {v: [] for v in range(V)}
    for (u, v) in edges: adj[u].append(v); adj[v].append(u)
    best = None
    for s0 in range(V):
        prev_s = {s0: None}; order_s = [s0]
        for u in order_s:
            for v in adj[u]:
                if v not in prev_s: prev_s[v] = u; order_s.append(v)
        far_s = order_s[-1]; L = 0; v = far_s
        while prev_s[v] is not None: L += 1; v = prev_s[v]
        if best is None or L > best[0]: best = (L, prev_s, far_s)
    _, prev, far = best
    path = []; v = far
    while prev[v] is not None: path.append((prev[v], v)); v = prev[v]
    P = np.eye(d)
    for (u, v) in reversed(path):
        R = edges[(u, v)][0] if (u, v) in edges else edges[(v, u)][0].T
        P = P @ R
    src = path[-1][0] if path else 0
    Zs = Z[src][mask[:, src]]
    e_dev = float(np.linalg.norm(Zs @ (P - np.eye(d))) / max(np.linalg.norm(Zs), 1e-12))
    return {"diag_edge_dev": float(max(dev.values())), "diag_path_dev": float(np.linalg.norm(P - np.eye(d)) / np.sqrt(d)), "diag_path_len": len(path),
            "diag_edge_resid": float(np.mean(list(resid.values()))), "diag_path_dev_energy": e_dev}


def proto_match_baseline(cons, mask, comps, y, K, seed):
    n = len(y); pred = np.full(n, -1); ref = None
    for ci, comp in enumerate(comps):
        rows = mask[:, comp].any(1)
        lab, cen = kmeans_np(cons[rows], K, seed=seed)
        D = np.linalg.norm(cen[:, None] - cen[None], axis=2)
        prof = np.sort(D, axis=1)
        if ref is None:
            ref = prof; pred[rows] = lab
        else:
            from scipy.optimize import linear_sum_assignment
            cost = np.linalg.norm(ref[:, None] - prof[None], axis=2)
            r, c = linear_sum_assignment(cost); lut = np.empty(K, int); lut[c] = r
            pred[rows] = lut[lab]
    return clustering_metrics(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topologies", nargs="*", default=["complete", "random-2", "chain", "two-comp"])
    ap.add_argument("--sweep", type=float, nargs="*", default=[], help="chain edge (2,3) cohort fractions")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--trace", action="store_true", help="record composed-drift diagnostics at every synchronisation epoch")
    ap.add_argument("--acc-trace", action="store_true", help="with --trace: also record consensus k-means accuracy at every synchronisation epoch (diagnostic)")
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--K", type=int, default=0, help="0 = number of classes")
    ap.add_argument("--no-std", action="store_true", help="option 3: centre each view and scale it globally (unit mean column variance) instead of standardising every column")
    ap.add_argument("--pca", type=int, default=0, help="reduce every view to at most this many PCA dims (0 = off)")
    ap.add_argument("--lam-rec", type=float, default=1.0)
    ap.add_argument("--lam-proto", type=float, default=1.0); ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--hid", type=int, default=256); ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--sync-every", type=int, default=25); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--tag", default="")
    ap.add_argument("--complete-only", action="store_true", help="keep only instances whose stored mask has every view (natural datasets)")
    ap.add_argument("--cohort-weights", default="", help="comma-separated relative cohort sizes for the chosen topology, e.g. 1,1,0.05,1,0.05,1 for a ring with two thin edges")
    ap.add_argument("--gate-path", type=float, default=0.7, help="Cg: synchronise only if the no-sync path drift exceeds this")
    ap.add_argument("--gate-resid", type=float, default=0.9, help="Cg: ... and the mean edge Procrustes residual is below this")
    ap.add_argument("--holdout-frac", type=float, default=0.0, help="withhold this fraction of the co-observed instances of every edge with n_uv >= 2d from training; they are still clustered at evaluation and give a label-free held-out pair retrieval score")
    ap.add_argument("--label-free-diag", action="store_true", help="also record the silhouette of the final consensus clustering (label-free)")
    ap.add_argument("--class-alpha", type=float, default=0.0, help="Dirichlet concentration of per-cohort class proportions (0 = off)")
    ap.add_argument("--size-sigma", type=float, default=0.0, help="lognormal sigma of cohort size factors (0 = off)")
    ap.add_argument("--batch-scale", type=float, default=0.0, help="per-cohort, per-view affine batch effect x*(1+s*g)+s*b on standardised features (0 = off)")
    ap.add_argument("--thin-anchor-weight", type=float, default=None, help="override the thin-edge anchor weight (default 20)")
    ap.add_argument("--thin-reg", type=float, default=None, help="override the thin-edge Sinkhorn regularisation (default 0.05)")
    ap.add_argument("--thin-nsub", type=int, default=None, help="override the thin-edge subsample size (default 600)")
    ap.add_argument("--thin-threads", type=int, default=1, help="BLAS threads inside the thin-edge primitive (1 = pinned, as reported)")
    ap.add_argument("--nbr-gamma", type=float, default=0.0, help="weight of the within-view raw-kNN neighbour contrast (0 = off)")
    ap.add_argument("--nbr-k", type=int, default=10, help="neighbours per instance for the neighbour contrast")
    ap.add_argument("--bridge-beta", type=float, default=0.5, help="weight of the InfoNCE term on bridged pseudo-pairs (Bp, Cb)")
    ap.add_argument("--bridge-pca", type=int, default=50, help="dimensions of the shared view used for mutual nearest neighbours")
    ap.add_argument("--bridge-k", type=int, default=10, help="neighbours per hop of the chained kNN imputation (variant Aimp)")
    ap.add_argument("--proto-type", default="ours", choices=["ours", "dec", "deepcluster"], help="prototype term of the host")
    ap.add_argument("--proto-start", type=int, default=None, help="epoch of the first prototype assignment for A, C and Cthin (multiple of --sync-every; default: first synchronisation epoch)")
    ap.add_argument("--save-emb", type=Path, default=None, help="save final raw (unframed) embeddings, mask and labels per run to this directory")
    a = ap.parse_args()
    from .sync import THIN_CFG
    THIN_CFG.update({"anchor_weight": a.thin_anchor_weight, "reg": a.thin_reg, "n_sub": a.thin_nsub, "threads": a.thin_threads})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.data); y = d["y"]; views = [d[f"X{i}"] for i in range(len([k for k in d.files if k.startswith("X")]))]
    if a.complete_only and "mask" in d.files:
        keep = d["mask"].astype(bool).all(1)
        views = [v[keep] for v in views]; y = y[keep]
        print(f"complete-only: kept {keep.sum()} of {len(keep)} instances", flush=True)
    if a.no_std:
        views = [(v - v.mean(0)) / (np.sqrt((v.std(0) ** 2).sum()) + 1e-8) * np.sqrt(v.shape[1]) for v in views]
    else:
        views = [(v - v.mean(0)) / (v.std(0) + 1e-8) for v in views]
    if a.pca > 0:
        red = []
        for v in views:
            if v.shape[1] > a.pca:
                U, S, Vt = np.linalg.svd(v - v.mean(0), full_matrices=False)
                v = (v - v.mean(0)) @ Vt[:a.pca].T
                v = v / (v.std(0) + 1e-8)
            red.append(v)
        views = red
    n, V = len(y), len(views)
    if a.K == 0:
        a.K = int(len(np.unique(y)))
    configs = [(t, None) for t in a.topologies] + [("chain", p) for p in a.sweep]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for topo, p in configs:
        for seed in a.seeds:
            w = None
            if p is not None:
                w = [1.0] * (V - 1); w[2] = p
            if a.cohort_weights:
                w = [float(x) for x in a.cohort_weights.split(",")]
            if topo == "natural":
                mask, cohort = d["mask"].astype(bool), np.full(n, -1)
                if a.complete_only:
                    raise SystemExit("natural topology is incompatible with --complete-only")
            else:
                mask, cohort = make_mask(topo, n, V, seed, cohort_weights=w, y=y, class_alpha=a.class_alpha, size_sigma=a.size_sigma)
            views_run = views
            if a.batch_scale > 0 and topo != "natural":
                br = np.random.default_rng(seed + 15485863); views_run = [v.copy() for v in views]
                for c in np.unique(cohort[cohort >= 0]):
                    rows = cohort == c
                    for j in range(V):
                        if mask[rows, j].any():
                            g = br.normal(size=views[j].shape[1]); b = br.normal(size=views[j].shape[1])
                            views_run[j][rows] = views[j][rows] * (1 + a.batch_scale * g) + a.batch_scale * b
            N = coobservation(mask); lam2 = fiedler(N, normalize_by=n)
            train_mask, held = mask, {}
            if a.holdout_frac > 0:
                hr = np.random.default_rng(seed + 7919); train_mask = mask.copy()
                thick = [(u, v) for u in range(V) for v in range(u + 1, V) if N[u, v] >= 2 * a.d]
                on_thin = np.zeros(n, bool)
                for u in range(V):
                    for v in range(u + 1, V):
                        if 0 < N[u, v] < 2 * a.d:
                            on_thin |= mask[:, u] & mask[:, v]
                on_thick = np.zeros(n, bool)
                for (u, v) in thick:
                    on_thick |= mask[:, u] & mask[:, v]
                elig = np.where(on_thick & ~on_thin)[0]
                H = np.zeros(n, bool)
                H[hr.choice(elig, int(round(a.holdout_frac * len(elig))), replace=False)] = True
                train_mask[H] = False
                for (u, v) in thick:
                    h = np.where(H & mask[:, u] & mask[:, v])[0]
                    if len(h) >= 2:
                        held[(u, v)] = h
            imp_cache = None; pairs_cache = None; nbr_cache = None; lin_cache = None
            for variant in a.variants:
                t0 = time.time()
                vrun, tmask, emask, vtrain = views_run, train_mask, mask, variant
                if variant == "Aimp":
                    from .bridge import bridge_impute
                    if imp_cache is None: imp_cache = bridge_impute(views_run, train_mask, k=a.bridge_k)
                    vrun, tmask, emask, vtrain = imp_cache, np.ones_like(train_mask), np.ones_like(mask), "A"
                train.pseudo = None; train.nbr = None; train.nbr_gamma = a.nbr_gamma
                if a.nbr_gamma > 0:
                    from .bridge import view_neighbours
                    if nbr_cache is None: nbr_cache = view_neighbours(vrun, tmask, k=a.nbr_k)
                    train.nbr = nbr_cache
                if variant in ("Bp", "Cb", "Cbs"):
                    from .bridge import bridge_pairs
                    if pairs_cache is None: pairs_cache = bridge_pairs(views_run, train_mask, pca=a.bridge_pca)
                    train.pseudo = pairs_cache; train.bridge_beta = a.bridge_beta
                if variant in ("AL", "ALw", "CL", "CLw", "CLg"):
                    if lin_cache is None:
                        from . import linsync
                        linsync.PRE.update(std=False, whiten=False)
                        E, _ = linsync.l1(vrun, tmask, a.d, 50)
                        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
                        lin_cache = kmeans_np(E, a.K, seed=seed)[0]
                    train.init_assign = lin_cache
                train.gate = (a.gate_path, a.gate_resid); train.proto_start = a.proto_start; train.proto_type = a.proto_type; train.trace = a.trace; train.acc_trace_y = y if a.acc_trace else None
                model, comps = train(vrun, tmask, vtrain, seed, device, epochs=a.epochs, K=a.K, lam_rec=a.lam_rec,
                                     lam_proto=a.lam_proto, tau=a.tau, hid=a.hid, d=a.d, sync_every=a.sync_every, lr=a.lr, wd=a.wd)
                framed = vtrain not in ("A", "Ad", "Apl", "Bp", "AL", "ALw", "Aw") and (vtrain not in ("Cg", "CLg") or getattr(train, "last_gate_open", False))
                X = [torch.tensor(v, dtype=torch.float32, device=device) for v in vrun]
                cons = consensus(model, X, emask, device, framed=framed)
                if variant == "PM":
                    lab = prototype_graph_matching(model, X, mask, a.K, seed)
                elif variant in ("ALw", "CLw", "CLg", "Aw") and getattr(train, "last_protos", None) is not None:
                    lab, _ = kmeans_warm(cons, train.last_protos, a.K)
                else:
                    lab, _ = kmeans_np(cons, a.K, seed=seed)
                met = clustering_metrics(y, lab)
                with torch.no_grad():
                    model.eval(); Zf = [model.embed(v, X[v], framed=framed).cpu().numpy() for v in range(V)]
                pairs = {k: p for k, p in {"01": (0, 1), "02": (0, 2), "0L": (0, V - 1), "23": (2, 3)}.items() if max(p) < V}
                if topo == "natural":
                    def xk_nat(u, v):
                        ru, rv = np.where(mask[:, u])[0], np.where(mask[:, v])[0]
                        A_ = Zf[u][ru] / np.linalg.norm(Zf[u][ru], axis=1, keepdims=True); B_ = Zf[v][rv] / np.linalg.norm(Zf[v][rv], axis=1, keepdims=True)
                        S = A_ @ B_.T
                        same = ru[:, None] == rv[None, :]; S[same] = -np.inf
                        return float((y[rv[S.argmax(1)]] == y[ru]).mean())
                    r1 = {f"r1_{k}": float("nan") for k in pairs}
                    r1.update({f"xk_{k}": xk_nat(u, v) for k, (u, v) in pairs.items()})
                else:
                    r1 = {f"r1_{k}": retrieval_r1(Zf[u], Zf[v]) for k, (u, v) in pairs.items()}
                    r1.update({f"xk_{k}": xview_knn_acc(Zf[u], Zf[v], y) for k, (u, v) in pairs.items()})
                    def xk_obs(u, v):
                        ru, rv = np.where(mask[:, u])[0], np.where(mask[:, v])[0]
                        if len(ru) == 0 or len(rv) == 0: return float("nan")
                        A_ = Zf[u][ru] / np.linalg.norm(Zf[u][ru], axis=1, keepdims=True); B_ = Zf[v][rv] / np.linalg.norm(Zf[v][rv], axis=1, keepdims=True)
                        S = A_ @ B_.T; S[ru[:, None] == rv[None, :]] = -np.inf
                        return float((y[rv[S.argmax(1)]] == y[ru]).mean())
                    r1.update({f"xkobs_{k}": xk_obs(u, v) for k, (u, v) in pairs.items()})
                rec = {"topology": (topo if p is None else f"chain-sweep-{p}") + (f"-w{a.cohort_weights}" if a.cohort_weights else ""), "seed": seed, "variant": variant, "tag": a.tag,
                       "cfg": {"no_std": a.no_std, "proto_type": a.proto_type, "nbr_gamma": a.nbr_gamma, "proto_start": a.proto_start, "hid": a.hid, "d": a.d, "tau": a.tau, "lam_proto": a.lam_proto, "lam_rec": a.lam_rec, "sync_every": a.sync_every, "lr": a.lr, "wd": a.wd, "epochs": a.epochs, "pca": a.pca,
                               **({"class_alpha": a.class_alpha, "size_sigma": a.size_sigma, "batch_scale": a.batch_scale} if (a.class_alpha or a.size_sigma or a.batch_scale) else {}),
                               **({"thin": {k: v for k, v in THIN_CFG.items()}} if (a.thin_anchor_weight or a.thin_reg or a.thin_nsub or a.thin_threads != 1) else {})},
                       "lambda2": lam2, "n_edge_23": int(N[2, 3]) if V > 3 else -1, "n_components": len(comps), **met, **r1,
                       "seconds": round(time.time() - t0, 1)}
                rec.update(frame_drift_diagnostics(model, X, train_mask))
                if held:
                    sc = []
                    for (u, v), h in held.items():
                        A_ = Zf[u][h] / np.linalg.norm(Zf[u][h], axis=1, keepdims=True); B_ = Zf[v][h] / np.linalg.norm(Zf[v][h], axis=1, keepdims=True)
                        S = A_ @ B_.T; t = np.arange(len(h))
                        sc.append(0.5 * ((S.argmax(1) == t).mean() + (S.argmax(0) == t).mean()))
                    rec["holdout_r1"] = float(np.mean(sc)); rec["holdout_frac"] = a.holdout_frac; rec["n_heldout"] = int(sum(len(h) for h in held.values()))
                if a.label_free_diag:
                    from sklearn.metrics import silhouette_score
                    rec["silhouette"] = float(silhouette_score(cons, lab, metric="cosine", sample_size=min(len(lab), 5000), random_state=seed))
                if a.save_emb is not None:
                    a.save_emb.mkdir(parents=True, exist_ok=True)
                    Zr = raw_embeddings(model, X)
                    np.savez_compressed(a.save_emb / f"{rec['topology']}_s{seed}_{variant}.npz", Z=np.stack(Zr), Zf=np.stack(Zf), framed=framed, mask=mask, y=y, acc=met["acc"])
                if getattr(train, "trace", False): rec["drift_trace"] = train.last_trace
                if getattr(train, "last_churn", None): rec["churn"] = train.last_churn
                if variant == "Cg": rec["gate_open"] = bool(getattr(train, "last_gate_open", False))
                if len(comps) > 1 and variant in ("C", "Cthin"):
                    pm = proto_match_baseline(cons, mask, comps, y, a.K, seed)
                    rec["proto_match_acc"] = pm["acc"]
                with a.out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{rec['topology']:16s} seed={seed} {variant}: acc={met['acc']:.4f} nmi={met['nmi']:.4f} "
                      f"xk_0L={r1['xk_0L']:.3f} xk_02={r1['xk_02']:.3f} xk_01={r1['xk_01']:.3f} r1_01={r1['r1_01']:.3f} lam2={lam2:.4f} "
                      f"comps={len(comps)} {rec.get('proto_match_acc', '')} [{rec['seconds']}s]", flush=True)


if __name__ == "__main__":
    main()
