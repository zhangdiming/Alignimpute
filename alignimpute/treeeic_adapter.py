from __future__ import annotations
import argparse, json, os, sys, time, types
from pathlib import Path
import numpy as np
from .topology import make_mask, coobservation, fiedler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--name", required=True, choices=["HandWritten", "Caltech101_7", "Caltech101_20"])
    ap.add_argument("--topologies", nargs="*", default=["chain"]); ap.add_argument("--sweep", type=float, nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke-epochs", type=int, default=0, help="smoke test only: override init_epoch/epoch")
    a = ap.parse_args()
    a.out = a.out.resolve(); a.data = a.data.resolve()
    root = a.root.resolve(); sys.path.insert(0, str(root)); os.chdir(root)
    try:
        import docutils.parsers.rst.directives.tables
    except ImportError:
        for m in ("docutils", "docutils.parsers", "docutils.parsers.rst", "docutils.parsers.rst.directives", "docutils.parsers.rst.directives.tables"):
            sys.modules[m] = types.ModuleType(m)
        sys.modules["docutils.parsers.rst.directives.tables"].align = None
    import torch, itertools, random
    from sklearn.preprocessing import MinMaxScaler
    import model as TM
    from configure import get_default_config

    _orig_group = TM.TreeEIC.missing_pattern_tree_group
    def capped_group(self, masks, k):
        kmax = int(masks.sum(0).max().item())
        groups, combos = _orig_group(self, masks, min(k, max(kmax, 2)))
        keep = [i for i, g in enumerate(groups) if len(g) > 0]
        return [groups[i] for i in keep], [combos[i] for i in keep]
    TM.TreeEIC.missing_pattern_tree_group = capped_group
    _orig_match = TM.TreeEIC.Match
    def safe_match(self, y_true, y_pred):
        if np.asarray(y_true).size == 0:
            K = self.n_clusters
            return np.zeros(0), np.arange(K), np.arange(K), np.eye(K, dtype=np.int64)
        return _orig_match(self, y_true, y_pred)
    TM.TreeEIC.Match = safe_match

    d = np.load(a.data)
    y = d["y"].astype(int); y = y - y.min(); n = len(y)
    views = [d[f"X{i}"].astype("float32") for i in range(len([k for k in d.files if k.startswith("X")]))]
    V = len(views)
    views = [MinMaxScaler().fit_transform(v).astype("float32") for v in views]
    base = get_default_config("Caltech101_7" if a.name == "Caltech101_20" else a.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [(t, None) for t in a.topologies] + [("chain", p) for p in a.sweep]
    for topo, p in configs:
        for seed in a.seeds:
            w = None
            if p is not None:
                w = [1.0] * (V - 1); w[2] = p
            mask, _ = make_mask(topo, n, V, seed, cohort_weights=w)
            N = coobservation(mask); lam2 = fiedler(N, normalize_by=n)
            X = [torch.from_numpy(views[v] * mask[:, v:v + 1]).float().to(device) for v in range(V)]
            index = mask.T.astype(int)
            cfg = json.loads(json.dumps(base)); cfg["dataset"] = a.name; cfg["print_num"] = 100; cfg["cuda"] = 0
            cfg["miss_rate"] = float((mask.sum(1) < V).mean())
            cfg["training"]["best_view"] = int(np.argmax([v.shape[1] for v in views]))
            if a.smoke_epochs:
                cfg["training"]["init_epoch"] = cfg["training"]["epoch"] = a.smoke_epochs
            s0 = cfg["training"]["seed"]
            random.seed(s0); np.random.seed(s0); torch.manual_seed(s0); torch.cuda.manual_seed_all(s0)
            torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
            t0 = time.time()
            M = TM.TreeEIC(cfg, V, [v.shape[1] for v in views], n_clusters=int(len(np.unique(y))), seed=s0, data_size=n)
            M.to_device(device)
            opts = [torch.optim.Adam(itertools.chain(M.autoencoders[v].parameters()), lr=cfg["training"]["lr"]) for v in range(V)]
            rec = {"topology": topo if p is None else f"chain-sweep-{p}", "seed": seed, "variant": "TreeEIC", "dataset": a.name, "lambda2": lam2,
                   "miss_rate": cfg["miss_rate"], "best_view": cfg["training"]["best_view"]}
            try:
                acc, nmi, ari = M.train(cfg, X, [y] * V, index, opts, device, n)
                rec.update({"status": "ok", "acc": float(acc), "nmi": float(nmi), "ari": float(ari)})
            except Exception as e:
                rec.update({"status": "failed", "error": repr(e)[:500]})
            rec["seconds"] = round(time.time() - t0, 1)
            with a.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    main()
