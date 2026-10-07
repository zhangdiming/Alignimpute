from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .topology import coobservation, fiedler, make_mask

MVP_ORDER = [1, 0, 2, 5, 4, 3]

def total_permutation(mask: list, num: int, cyc):
    idx = [i for i, m in enumerate(mask) if m == 1]
    import random
    random.seed(42)
    perms = cyc.double_cyclic_permutation(idx, num)
    out = []
    for perm in perms:
        res = list(range(len(mask)))
        for i, p in enumerate(perm):
            res[idx[i]] = p
        out.append(res)
    return out


CALTECH_VIEWS = ['0_Gabor48.csv', '1_WM40.csv', '2_cenhist254.csv', '3_HOG1984.csv', '4_GIST512.csv', '5_LBP928.csv']


MVP_STYLES = {
    "hw": {"hid": None, "top": 1024, "latent_dim": 16, "likelihood": "Laplace", "normalizing_factor": [2.5, 2], "epochs": 150},
    "scene15": {"hid": 512, "top": 2048, "latent_dim": 16, "likelihood": "Normal", "normalizing_factor": [10, 1], "epochs": 250},
    "reuters": {"hid": 512, "top": 2048, "latent_dim": 16, "likelihood": "Normal", "normalizing_factor": [5, 1], "epochs": 60},
}


def write_caltech(mvp_root: Path, d, data_dir: str, style: str = "hw") -> dict:
    out = mvp_root / "data" / data_dir; out.mkdir(parents=True, exist_ok=True)
    widths, dropped = [], []
    for j, name in enumerate(CALTECH_VIEWS):
        X = d[f"X{j}"].astype(float); keep = X.std(0) > 0
        np.savetxt(out / name, X[:, keep], delimiter=",", fmt="%.6g")
        widths.append(int(keep.sum())); dropped.append(int((~keep).sum()))
    st = MVP_STYLES[style]
    hid = lambda w: st["hid"] or (512 if w >= 200 else 256)
    arch = {f"m{j}": [w, hid(w), hid(w), st["top"]] for j, w in enumerate(widths)}
    cfg = {"data_dir": data_dir, "num_views": 6, "in_channels": widths, "arch": arch, "num_classes": int(len(np.unique(d["y"]))),
           "latent_dim": st["latent_dim"], "likelihood": st["likelihood"], "normalizing_factor": st["normalizing_factor"],
           "style": style, "dropped_constant_columns": dropped}
    (out / "mvp_cfg.json").write_text(json.dumps(cfg))
    return cfg


def write_fingerprint(mvp_root: Path, mask: np.ndarray, y: np.ndarray, tag: str, data_dir: str = "Handwritten"):
    import torch
    sys.path.insert(0, str(mvp_root / "dataprovider"))
    import cyclic_permutation as cyc
    V = mask.shape[1]
    num_pt = max(V, math.factorial(V - 1))
    fp = {}
    for i in range(len(y)):
        m = [int(x) for x in mask[i]]
        fp[str(i)] = {"mask": m, "pt": total_permutation(m, num_pt, cyc), "labels": int(y[i])}
    p = mvp_root / "data" / data_dir / "fingerprint_0.0.pth"
    torch.save(fp, p)
    return p


def parse_final(log_txt: Path) -> dict | None:
    txt = log_txt.read_text().splitlines()
    for i, line in enumerate(txt):
        if re.search(r"epoch: \d+$", line.strip()) and i + 2 < len(txt):
            m = re.search(r"\[([^\]]+)\]", txt[i + 2])
            if m:
                vals = [float(x) for x in m.group(1).split(",")]
                last = {"acc": vals[0], "nmi": vals[1], "ari": vals[2]}
    return last if "last" in dir() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvp-root", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--topologies", nargs="+", default=["chain"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--wrapper", default="run_mvp_patched.py", help="run_mvp_patched.py (baseline) or run_mvp_sync.py (plug-in)")
    ap.add_argument("--dataset", default="Handwritten", choices=["Handwritten", "Caltech101", "Caltech101_20"], help="Caltech101 = our Caltech101-7 (MVP's loader name for the six Caltech views), via run_mvp_generic.py")
    ap.add_argument("--mvp-style", default="hw", choices=list(MVP_STYLES), help="configuration style for Caltech (epochs follow the style unless --epochs is given explicitly)")
    ap.add_argument("--variant", default="mvp"); ap.add_argument("--extra", nargs="*", default=[], help="extra wrapper options as key=value, e.g. lam_proto=20 sync_every=10")
    a = ap.parse_args()
    d = np.load(a.data); y = d["y"]; n = len(y); V = 6
    env = dict(os.environ); order = MVP_ORDER; data_dir = "Handwritten"
    if a.dataset != "Handwritten":
        data_dir = a.dataset
        env["MVP_CFG"] = json.dumps(write_caltech(a.mvp_root, d, data_dir, a.mvp_style)); order = list(range(V))
        if a.mvp_style != "hw" and a.epochs == 150: a.epochs = MVP_STYLES[a.mvp_style]["epochs"]
        a.wrapper = "run_mvp_generic.py"
    for topo in a.topologies:
        for seed in a.seeds:
            mask, _ = make_mask(topo, n, V, seed)
            N = coobservation(mask); lam2 = fiedler(N, normalize_by=n)
            tag = f"{topo}_s{seed}" if a.dataset == "Handwritten" else f"{a.dataset}_{topo}_s{seed}"
            write_fingerprint(a.mvp_root, mask[:, order], y, tag, data_dir)
            logdir = a.mvp_root / "results" / f"run_{a.variant}_{'_'.join(a.extra).replace('=', '')}_{tag}"
            t0 = time.time()
            cmd = [a.python, a.wrapper, "--dataset", a.dataset, "--missing_rate", "0.0",
                   "--log_path", str(logdir), "--epochs", str(a.epochs)]
            if a.wrapper == "run_mvp_sync.py":
                cmd += ["--variant", a.variant]
                for kv in a.extra:
                    k, v = kv.split("=", 1); cmd += [f"--{k}", v]
            proc = subprocess.run(cmd, cwd=a.mvp_root, capture_output=True, text=True, env=env)
            logs = sorted(logdir.glob("log_*/train_log.txt")) if logdir.exists() else []
            met = parse_final(logs[-1]) if logs else None
            rec = {"topology": topo, "seed": seed, "dataset": a.dataset, "mvp_style": a.mvp_style, "epochs": a.epochs, "variant": ("MVP" if a.wrapper in ("run_mvp_patched.py", "run_mvp_generic.py") else f"MVP+{a.variant}"), "extra": a.extra, "lambda2": lam2, "seconds": round(time.time() - t0, 1),
                   "status": "ok" if met else "failed", **(met or {}), "stderr_tail": proc.stderr[-500:] if not met else ""}
            with a.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    main()
