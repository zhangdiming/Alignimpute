import argparse, glob, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

DS = [("handwritten", "HandWritten"), ("caltech101_7", "Caltech101-7"), ("caltech101_20", "Caltech101-20")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--build", type=Path, required=True); a = ap.parse_args(); B = a.build
    rows = []
    for d, dn in DS:
        for top in ("chain", "thin"):
            S = {}
            for cond in ("all", "obs"):
                R = [json.loads(l) for f in sorted(glob.glob(str(B / f"host_{cond}_{d}_{top}_s*.jsonl"))) for l in open(f)]
                for v in ("A", "C" if top == "chain" else "Cthin", "ALw"):
                    S[(cond, "S" if v.startswith("C") else v)] = {r["seed"]: 100 * r["nmi"] for r in R if r["variant"] == v}
                RA = [json.loads(l) for f in sorted(glob.glob(str(B / f"aimp_{cond}_{d}_{top}_s*.jsonl"))) for l in open(f)]
                S[(cond, "Aimp")] = {r["seed"]: 100 * r["nmi"] for r in RA if r["variant"] == "Aimp"}
                K = [json.loads(l) for l in open(B / f"kimp_{cond}_{d}.jsonl")]
                S[(cond, "Kimp")] = {r["seed"]: 100 * r["nmi"] for r in K if r["topology"] == ("chain" if top == "chain" else "chain-sweep-0.05")}
            out = {"setting": f"{dn} {top}"}
            for m in ("A", "S", "ALw", "Aimp", "Kimp"):
                x, y = S[("all", m)], S[("obs", m)]; ks = sorted(set(x) & set(y)); assert len(ks) == 10, (d, top, m, len(ks))
                dlt = np.array([y[k] - x[k] for k in ks])
                out[m] = (np.mean([x[k] for k in ks]), np.mean([y[k] for k in ks]), dlt.mean(), wilcoxon(dlt).pvalue if np.any(dlt) else 1.0)
            for cond in ("all", "obs"):
                ba = max(np.mean(list(S[(cond, m)].values())) for m in ("S", "ALw"))
                bi = max(np.mean(list(S[(cond, m)].values())) for m in ("Kimp", "Aimp"))
                out[f"gap_{cond}"] = ba - bi
            rows.append(out)
    print(f"{'setting':22s}" + "".join(f"{m:>22s}" for m in ("A", "S", "ALw", "Aimp", "Kimp")) + f"{'gap all':>9s}{'gap obs':>9s}")
    for r in rows:
        print(f"{r['setting']:22s}" + "".join(f"  {r[m][0]:5.1f}->{r[m][1]:5.1f} ({r[m][2]:+4.1f}{'*' if r[m][3] < 0.05 else ' '})" for m in ("A", "S", "ALw", "Aimp", "Kimp")) + f"{r['gap_all']:+9.1f}{r['gap_obs']:+9.1f}")
    allm = {m: np.array([r[m][2] for r in rows]) for m in ("A", "S", "ALw", "Aimp", "Kimp")}
    print("mean change over the six cells:", {m: round(float(v.mean()), 2) for m, v in allm.items()}, "largest |change|:", {m: round(float(np.abs(v).max()), 2) for m, v in allm.items()})
    print("family gap: min/max change", round(min(r["gap_obs"] - r["gap_all"] for r in rows), 2), round(max(r["gap_obs"] - r["gap_all"] for r in rows), 2),
          "| sign kept in", sum((r["gap_obs"] > 0) == (r["gap_all"] > 0) for r in rows), "of", len(rows))
    json.dump(rows, open(B / "summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
