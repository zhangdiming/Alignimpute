from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import clustering_metrics

METHODS = {"PCAcat": "none", "A": "none", "Kimp": "impute", "Aimp": "impute", "RecFormer": "impute", "LRimp": "impute",
           "FreeCSL": "deep", "DVIMC": "deep", "GHICMC": "deep",
           "StabMap": "align", "Cwp": "align", "C": "align", "L1": "align", "ALw": "align"}
SUBSETS = {"all": None, "patchseq": [1], "methylome": [0, 2]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, required=True); ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    out = {}
    for g in ("cortex_stage2_no3c", "cortex_stage2"):
        d = np.load(a.data / f"{g}.npz", allow_pickle=True); y = d["y"]; src = d["source"]
        res = {}
        for m in METHODS:
            files = sorted(a.preds.glob(f"{m}_{g}_natural_s*.npy"))
            per = {k: [] for k in SUBSETS}
            for f in files:
                lab = np.load(f).astype(int)
                for k, codes in SUBSETS.items():
                    sel = (lab >= 0) & (np.ones(len(y), bool) if codes is None else np.isin(src, codes))
                    per[k].append(clustering_metrics(y[sel], lab[sel]))
            if files:
                res[m] = {k: {"nmi": 100 * float(np.mean([x["nmi"] for x in v])), "acc": 100 * float(np.mean([x["acc"] for x in v])),
                              "nmi_sd": 100 * float(np.std([x["nmi"] for x in v], ddof=1)) if len(v) > 1 else float("nan"), "n_seeds": len(v)}
                          for k, v in per.items()}
        gap = {}
        for k in SUBSETS:
            al = max((res[m][k]["nmi"], m) for m in res if METHODS[m] == "align")
            im = max((res[m][k]["nmi"], m) for m in res if METHODS[m] == "impute")
            gap[k] = {"best_align": al[1], "align": al[0], "best_impute": im[1], "impute": im[0], "gap": al[0] - im[0]}
        out[g] = {"methods": res, "gap": gap}
        print(f"== {g}")
        print(f"{'method':10s} " + " ".join(f"{k:>10s}" for k in SUBSETS))
        for m in res:
            print(f"{m:10s} " + " ".join(f"{res[m][k]['nmi']:10.1f}" for k in SUBSETS) + f"   (n={res[m]['all']['n_seeds']})")
        for k, v in gap.items():
            print(f"gap[{k}]: {v['best_align']} {v['align']:.1f} - {v['best_impute']} {v['impute']:.1f} = {v['gap']:+.1f}")
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
