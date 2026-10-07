from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ALIGNIMPUTE_DATA", ROOT / "data"))
EXTERNAL = Path(os.environ.get("ALIGNIMPUTE_EXTERNAL", ROOT / "external"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from alignimpute.topology import make_mask
from alignimpute.metrics import clustering_metrics, kmeans_np

BENCH = {"handwritten", "caltech101_7", "caltech101_20"}
REAL = {"cortex_stage2", "cortex_stage2_no3c", "tcga_legacy9", "pbmc_mimitou", "bmmc_site1", "tcga_pancan"}


def load(dataset: str):
    d = np.load(DATA / f"{dataset}.npz")
    V = len([k for k in d.files if k.startswith("X")])
    views = [np.asarray(d[f"X{i}"], dtype=np.float64) for i in range(V)]
    y = np.asarray(d["y"]).astype(int)
    if dataset in REAL:
        views = [(v - v.mean(0)) / (np.sqrt((v.std(0) ** 2).sum()) + 1e-8) * np.sqrt(v.shape[1]) for v in views]
    else:
        views = [(v - v.mean(0)) / (v.std(0) + 1e-8) for v in views]
    nat = d["mask"].astype(bool) if "mask" in d.files else None
    return views, y, nat


def get_mask(dataset: str, topology: str, n: int, V: int, seed: int, nat=None) -> np.ndarray:
    if topology == "chain":
        m, _ = make_mask("chain", n, V, seed)
    elif topology == "thin05":
        w = [1.0] * (V - 1); w[2] = 0.05
        m, _ = make_mask("chain", n, V, seed, cohort_weights=w)
    elif topology == "natural":
        m = nat.copy()
    elif topology == "complete":
        m = np.ones((n, V), dtype=bool)
    else:
        raise ValueError(topology)
    assert m.sum(1).min() >= 1, "every instance needs at least one observed view"
    return m


def zero_fill(views, mask):
    out = []
    for v, x in enumerate(views):
        x = np.nan_to_num(x.astype(np.float32), copy=True)
        x[~mask[:, v]] = 0.0
        out.append(x)
    return out


def write(out: Path, rec: dict):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def done_keys(out: Path):
    s = set()
    if Path(out).exists():
        for line in open(out):
            try:
                r = json.loads(line); s.add((r["dataset"], r["topology"], r["seed"]))
            except Exception:
                pass
    return s


def cells(datasets, topologies):
    for ds in datasets:
        if ds in REAL:
            yield ds, "natural"
        else:
            for t in topologies:
                yield ds, t


def run_cells(method, run_one, args, config_note):
    done = done_keys(args.out)
    for ds, topo in cells(args.datasets, args.topologies):
        views, y, nat = load(ds)
        n, V = len(y), len(views); K = int(len(np.unique(y)))
        for seed in args.seeds:
            if (ds, topo, seed) in done:
                print("skip", ds, topo, seed, flush=True); continue
            mask = get_mask(ds, topo, n, V, seed, nat)
            Xz = zero_fill(views, mask)
            t0 = time.time(); rec = {"method": method, "dataset": ds, "topology": topo, "seed": seed}
            try:
                pred, emb, extra = run_one(Xz, mask, K, seed)
                if pred is None:
                    pred, _ = kmeans_np(np.asarray(emb, dtype=np.float64), K, seed=seed)
                pred = np.asarray(pred).astype(int)
                met = clustering_metrics(y, pred)
                rec.update(status="ok", **met, n_pred_clusters=int(len(np.unique(pred))))
                rec.update(extra or {})
            except Exception as e:
                import traceback; traceback.print_exc()
                oom = "out of memory" in str(e).lower() or "CUDA error: out of memory" in str(e)
                rec.update(status="oom" if oom else "failed", acc=None, nmi=None, ari=None,
                           error=f"{type(e).__name__}: {e}"[:400])
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception:
                    pass
            rec.update(seconds=round(time.time() - t0, 1), config_note=config_note)
            write(args.out, rec)


def base_parser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["handwritten"])
    ap.add_argument("--topologies", nargs="+", default=["chain", "thin05"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    return ap
