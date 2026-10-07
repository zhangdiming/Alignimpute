from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

from .topology import coobservation, fiedler, make_mask
from .metrics import kmeans_np, clustering_metrics


def xavier(shape, gen):
    fan_in, fan_out = shape
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return torch.tensor(np.random.default_rng(gen).uniform(-a, a, size=shape), dtype=torch.float32)


def run_one(views, mask, K, seed, device, lsd=128, hidden=150, epochs=30, steps=(5, 5), lr=(0.01, 0.01), dropout=0.1):
    torch.manual_seed(seed)
    n, V = mask.shape
    X = [torch.tensor(v, dtype=torch.float32, device=device) for v in views]
    S = torch.tensor(mask, dtype=torch.float32, device=device)
    H = nn.Parameter(xavier((n, lsd), seed).to(device))
    nets = nn.ModuleList([nn.Sequential(nn.Linear(lsd, hidden), nn.Dropout(dropout), nn.Linear(hidden, X[v].shape[1])) for v in range(V)]).to(device)
    opt_net = torch.optim.Adam(nets.parameters(), lr=lr[0]); opt_h = torch.optim.Adam([H], lr=lr[1])
    def loss():
        return sum(((nets[v](H) - X[v]) ** 2).sum(1).mul(S[:, v]).sum() for v in range(V))
    for ep in range(epochs):
        nets.train()
        for _ in range(steps[0]):
            opt_net.zero_grad(); l = loss(); l.backward(); opt_net.step()
        for _ in range(steps[1]):
            opt_h.zero_grad(); l = loss(); l.backward(); opt_h.step()
    Hn = H.detach().cpu().numpy()
    lab, _ = kmeans_np(Hn, K, seed=seed)
    return lab, float(l.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topologies", nargs="*", default=["chain"]); ap.add_argument("--sweep", type=float, nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lsd", type=int, default=128); ap.add_argument("--tag", default="cpmnets")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.data); y = d["y"]; views = [d[f"X{i}"] for i in range(len([k for k in d.files if k.startswith("X")]))]
    views = [(v - v.mean(0)) / (v.std(0) + 1e-8) for v in views]
    n, V = len(y), len(views); K = int(len(np.unique(y)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    configs = [(t, None) for t in a.topologies] + [("chain", p) for p in a.sweep]
    for topo, p in configs:
        for seed in a.seeds:
            w = None
            if p is not None: w = [1.0] * (V - 1); w[2] = p
            mask, _ = make_mask(topo, n, V, seed, cohort_weights=w)
            N = coobservation(mask); lam = fiedler(N, normalize_by=n)
            vz = [v.copy() for v in views]
            for v in range(V): vz[v][~mask[:, v]] = 0.0
            t0 = time.time()
            lab, l = run_one(vz, mask, K, seed, device, lsd=a.lsd, epochs=a.epochs)
            met = clustering_metrics(y, lab)
            rec = {"topology": topo if p is None else f"chain-sweep-{p}", "seed": seed, "variant": "CPMNets-unsup", "tag": a.tag,
                   "cfg": {"lsd": a.lsd, "epochs": a.epochs}, "lambda2": lam, "n_edge_23": int(N[2, 3]), **met, "final_loss": l, "seconds": round(time.time() - t0, 1)}
            with a.out.open("a") as f: f.write(json.dumps(rec) + "\n")
            print(f"{rec['topology']:16s} seed={seed} acc={met['acc']:.4f} nmi={met['nmi']:.4f} loss={l:.1f} [{rec['seconds']}s]", flush=True)


if __name__ == "__main__": main()
