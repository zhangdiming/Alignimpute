from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VIEWS = {"cortex_stage2": ["ephys", "morph", "RNA", "GpC", "mCH", "3C"], "cortex_stage2_no3c": ["ephys", "morph", "RNA", "GpC", "mCH"],
         "tcga_legacy9": ["miRNA-GA", "U133A", "Agilent", "RNA-GA", "HiSeq", "Meth27", "Meth450", "CNV", "miRNA-HiSeq"],
         "tcga_pancan": ["RNA", "miRNA", "RPPA", "CNV"], "pbmc_mimitou": ["RNA", "protein", "ATAC"], "bmmc_site1": ["RNA", "protein", "ATAC"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/coobs_real.json"))
    a = ap.parse_args()
    out = {}
    for g, vn in VIEWS.items():
        M = np.load(a.data / f"{g}.npz")["mask"].astype(bool)
        N = M.T.astype(int) @ M.astype(int)
        out[g] = {"views": vn, "n": int(M.shape[0]), "coobs": N.tolist()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
