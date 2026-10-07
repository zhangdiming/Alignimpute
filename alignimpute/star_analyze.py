import argparse, glob, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

DS = [("handwritten", "HandWritten"), ("caltech101_7", "Caltech101-7"), ("caltech101_20", "Caltech101-20")]
ALIGN, IMPUTE = ("S", "ALw", "L1"), ("Kimp", "Aimp")


def per_seed(B, d, top):
    R = {}
    host = [json.loads(l) for f in sorted(glob.glob(str(B / f"host_{d}_{top}_s*.jsonl"))) for l in open(f)]
    aimp = [json.loads(l) for f in sorted(glob.glob(str(B / f"aimp_{d}_{top}_s*.jsonl"))) for l in open(f)]
    for v, name in (("A", "A"), ("C", "S"), ("ALw", "ALw")):
        R[name] = {r["seed"]: 100 * r["nmi"] for r in host if r["variant"] == v}
    R["Aimp"] = {r["seed"]: 100 * r["nmi"] for r in aimp if r["variant"] == "Aimp"}
    R["Kimp"] = {r["seed"]: 100 * r["nmi"] for r in map(json.loads, open(B / f"kimp_{d}.jsonl")) if r["topology"] == top}
    lin = [json.loads(l) for l in open(B / f"lin_{d}.jsonl")]
    for m in ("L1", "PCAcat"):
        R[m] = {r["seed"]: 100 * r["nmi"] for r in lin if r["topology"] == top and r["method"] == m}
    for k, v in R.items(): assert sorted(v) == list(range(10)), (d, top, k, sorted(v))
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--star-build", type=Path, default=None, help="star results with another hub view (default: --build)")
    ap.add_argument("--out", default="summary.json")
    a = ap.parse_args(); B = a.build; BS = a.star_build or B
    out = []
    for d, dn in DS:
        row = {"dataset": dn}
        for top in ("chain", "star"):
            R = per_seed(B if top == "chain" else BS, d, top); mean = {k: float(np.mean(list(v.values()))) for k, v in R.items()}
            ba = max(ALIGN, key=lambda m: mean[m]); bi = max(IMPUTE, key=lambda m: mean[m])
            row[top] = {"mean": mean, "best_align": ba, "best_impute": bi, "gap": mean[ba] - mean[bi],
                        "gap_seeds": [R[ba][s] - R[bi][s] for s in range(10)]}
        dg = np.array(row["chain"]["gap_seeds"]) - np.array(row["star"]["gap_seeds"])
        row["gap_drop"] = float(dg.mean()); row["p"] = float(wilcoxon(dg).pvalue); out.append(row)
    meths = ("A", "S", "ALw", "L1", "Aimp", "Kimp", "PCAcat")
    print(f"{'dataset':15s} {'top':6s}" + "".join(f"{m:>8s}" for m in meths) + "   best align - best impute = gap")
    for r in out:
        for top in ("chain", "star"):
            t = r[top]; print(f"{r['dataset']:15s} {top:6s}" + "".join(f"{t['mean'][m]:8.1f}" for m in meths) + f"   {t['best_align']:>4s} - {t['best_impute']:<4s} = {t['gap']:+6.1f}")
        print(f"{'':15s} gap chain - star = {r['gap_drop']:+.1f} (Wilcoxon over seeds p = {r['p']:.3g})")
    json.dump(out, open(BS / a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
