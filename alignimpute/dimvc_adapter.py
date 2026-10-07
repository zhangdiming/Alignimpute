from __future__ import annotations
import argparse, json, os, sys, time, random
from pathlib import Path
import numpy as np

from .topology import coobservation, fiedler, make_mask
from .metrics import clustering_metrics


def run_one(repo: Path, views, y, mask, K, seed, pretrain_epochs=500, maxiter=10000, update=1000, batch=256, lr=1e-3, dname="ours"):
    sys.path.insert(0, str(repo)); os.chdir(repo)
    import tensorflow as tf
    from tensorflow.keras.optimizers.legacy import Adam
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    from DIMVC import MvDEC
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    n, V = mask.shape
    X = [MinMaxScaler().fit_transform(v).astype(np.float32) for v in views]
    rows = [np.where(mask[:, v])[0] for v in range(V)]
    core = np.where(mask.all(1))[0]
    m = max(len(r) for r in rows)
    padded = [np.concatenate([r, np.random.choice(r, m - len(r))]) if len(r) < m else r for r in rows]
    x = [X[v][padded[v]] for v in range(V)]; yy = [y[padded[v]] for v in range(V)]
    model = MvDEC(n_clusters=K, view_shape=[x[v].shape[1:] for v in range(V)], data="ours")
    loss = []; lw = []
    for v in range(V): loss += ["categorical_crossentropy", "mse"]; lw += [1.0, 1.0]
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, loss_weights=lw)
    save_dir = str(repo / "results" / f"ours_{os.getpid()}"); os.makedirs(save_dir, exist_ok=True)
    model.pretrain(x, yy, optimizer=Adam(learning_rate=lr), epochs=pretrain_epochs, batch_size=batch, save_dir=save_dir, verbose=0)

    class A: pass
    arg = A(); arg.UpdateCoo = update; arg.dataset = "Caltech" if "caltech" in dname.lower() else "ours"
    if len(core) >= K:
        xc = [X[v][core] for v in range(V)]; yc = [y[core] for v in range(V)]
        model.new_fit(arg=arg, x=xc, y=yc, maxiter=maxiter, batch_size=batch, UpdateCoo=update, save_dir=save_dir)
        stage = "full"
    else:
        feats = model.encoder.predict({f"input{v+1}": x[v] for v in range(V)}, verbose=0)
        for v in range(V):
            km = KMeans(n_clusters=K, n_init=100, random_state=seed).fit(feats[v])
            model.model.get_layer(name=f"clustering{v+1}").set_weights([km.cluster_centers_])
        stage = "no-core"
    xa = [X[v] for v in range(V)]
    QX = model.model.predict({f"input{v+1}": xa[v] for v in range(V)}, verbose=0)
    soft = np.zeros((n, K))
    for v in range(V): soft[mask[:, v]] += QX[2 * v][mask[:, v]]
    pred = soft.argmax(1)
    import shutil; shutil.rmtree(save_dir, ignore_errors=True)
    return pred, stage, len(core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--topologies", nargs="*", default=["chain"]); ap.add_argument("--sweep", type=float, nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pretrain-epochs", type=int, default=500); ap.add_argument("--maxiter", type=int, default=10000)
    a = ap.parse_args()
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
                pred, stage, ncore = run_one(a.repo, views, y, mask, K, seed, a.pretrain_epochs, a.maxiter, dname=a.data.stem)
                met = clustering_metrics(y, pred); status = "ok"
            except Exception as e:
                met = {}; stage = "error"; ncore = -1; status = f"failed: {type(e).__name__}: {e}"[:300]
            rec = {"topology": topo if p is None else f"chain-sweep-{p}", "seed": seed, "variant": "DIMVC", "stage": stage, "n_core": ncore,
                   "lambda2": lam, "status": status, **met, "seconds": round(time.time() - t0, 1)}
            with out.open("a") as f: f.write(json.dumps(rec) + "\n")
            print(f"{rec['topology']:16s} seed={seed} {stage} core={ncore} {status} acc={met.get('acc', float('nan')):.4f} [{rec['seconds']}s]", flush=True)


if __name__ == "__main__": main()
