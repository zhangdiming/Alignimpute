from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np
from .sync import edge_estimates, spectral_sync
from .metrics import kmeans_np, clustering_metrics
from .st2_gate import cons


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--emb", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); n = 0
    with open(a.out, "w") as fo:
        for f in sorted(glob.glob(str(a.emb / "shift_*" / "*_A.npz"))):
            m = re.match(r"shift_(.+)_(a1|a03|s05|s1|b03)$", Path(f).parent.name); topo, seed = re.match(r"(.+)_s(\d+)_A\.npz$", Path(f).name).groups(); seed = int(seed)
            e = np.load(f); Z = e["Z"]; mask = e["mask"].astype(bool); y = e["y"]; K = int(len(np.unique(y))); d = Z.shape[2]
            Q, _ = spectral_sync(edge_estimates(list(Z), mask, min_pairs=d, thin=True), Z.shape[0], d)
            Zs = np.stack([Z[v] @ Q[v] for v in range(Z.shape[0])])
            lh, _ = kmeans_np(cons(Z, mask), K, seed=seed); ls, _ = kmeans_np(cons(Zs, mask), K, seed=seed)
            mh, ms = clustering_metrics(y, lh), clustering_metrics(y, ls)
            fo.write(json.dumps({"dataset": m.group(1), "level": m.group(2), "topology": topo, "seed": seed, "variant": "B",
                                 "host_nmi": mh["nmi"], "host_acc": mh["acc"], **ms}) + "\n"); n += 1
    print("records", n)


if __name__ == "__main__":
    main()
