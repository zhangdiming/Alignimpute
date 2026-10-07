from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io as sio

MAT = {
    "caltech101_7": ("Caltech101-7", "https://raw.githubusercontent.com/ChuanbinZhang/Multi-view-datasets/master/Caltech101-7.mat"),
    "outdoorscene": ("OutdoorScene", "https://raw.githubusercontent.com/ChuanbinZhang/Multi-view-datasets/master/OutdoorScene.mat"),
    "nuswide": ("NUS-WIDE", "https://raw.githubusercontent.com/ChuanbinZhang/Multi-view-datasets/master/NUS-WIDE.mat"),
    "aloi": ("ALOI", "https://raw.githubusercontent.com/ChuanbinZhang/Multi-view-datasets/master/ALOI.mat"),
    "proteinfold": ("ProteinFold", "https://raw.githubusercontent.com/ChuanbinZhang/Multi-view-datasets/master/ProteinFold.mat"),
    "caltech101_20": ("Caltech101-20", "https://raw.githubusercontent.com/sudalvxin/2019-PR-Sparse-Multi-view-clustering/master/Data/Caltech101-20.mat"),
}
MFEAT = "https://raw.githubusercontent.com/gaoxin492/MVP/main/data/Handwritten/mfeat-{}"
MFEAT_VIEWS = ["fac", "fou", "kar", "mor", "pix", "zer"]


def fetch(url, path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def from_mat(path):
    m = sio.loadmat(path); X = m["X"]
    for k in ("Y", "y", "gt", "label", "labels", "truth"):
        if k in m:
            y = np.asarray(m[k]).ravel(); break
    cells = [X[i, 0] for i in range(X.shape[0])] if X.shape[1] == 1 else [X[0, i] for i in range(X.shape[1])]
    views = []
    for c in cells:
        c = np.asarray(c.toarray() if hasattr(c, "toarray") else c, dtype=np.float64)
        views.append(c if c.shape[0] == len(y) else c.T)
    return views, y


def handwritten(raw):
    views = [np.loadtxt(fetch(MFEAT.format(f), Path(raw) / f"mfeat-{f}")).astype(np.float64) for f in MFEAT_VIEWS]
    return views, np.repeat(np.arange(10), 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--datasets", nargs="+", default=["handwritten"] + list(MAT))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in a.datasets:
        if d == "handwritten":
            views, y = handwritten(a.raw / "mfeat")
        else:
            name, url = MAT[d]
            views, y = from_mat(fetch(url, a.raw / f"{name}.mat"))
        np.savez(a.out / f"{d}.npz", y=y, **{f"X{i}": v for i, v in enumerate(views)})
        print(d, [v.shape for v in views], "classes", len(np.unique(y)))


if __name__ == "__main__":
    main()
