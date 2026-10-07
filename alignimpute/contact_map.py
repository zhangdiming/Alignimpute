from __future__ import annotations

import argparse
import glob
import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

TYPES = (("Inh_MGE_PVALB", "Pvalb"), ("Inh_MGE_CALB1", "Sst"), ("Inh_CGE_VIP", "Vip"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("raw/snm3c_inh_labels.csv"))
    ap.add_argument("--contacts", type=Path, default=Path("raw3c"))
    ap.add_argument("--out", type=Path, default=Path("results/st2/fig_3c.npz"))
    a = ap.parse_args()
    lab = pd.read_csv(a.labels)
    files = {os.path.basename(f): f for f in glob.glob(str(a.contacts / "*contacts.txt.gz"))}
    out = {}
    for mt, name in TYPES:
        cid = lab[lab.MajorType == mt]["Cell ID"].iloc[0]; short = cid.replace("_indexed", "").split("_BA10_")[1]
        f = [v for k, v in files.items() if cid.split("_indexed")[0].split("hs_21yr_")[-1] in k or short in k][0]
        M = np.zeros((160, 160))
        with gzip.open(f, "rt") as h:
            for line in h:
                p = line.split()
                if p[1] == "chr7" and p[3] == "chr7":
                    i0, j0 = int(p[2]), int(p[4])
                    if abs(i0 - j0) >= 20000:
                        i, j = i0 // 1000000, j0 // 1000000; M[i, j] += 1; M[j, i] += 1
        out[name] = uniform_filter(M, 3)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
