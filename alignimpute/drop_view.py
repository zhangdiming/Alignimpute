from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/cortex_stage2.npz"))
    ap.add_argument("--view", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("data/cortex_stage2_no3c.npz"))
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=True)
    V = d["mask"].shape[1]
    assert a.view == V - 1
    out = {k: d[k] for k in d.files if k not in (f"X{a.view}", "mask")}
    out["mask"] = d["mask"][:, :a.view].copy()
    np.savez_compressed(a.out, **out)


if __name__ == "__main__":
    main()
