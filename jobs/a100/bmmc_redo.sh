#!/bin/bash
conda activate alignimpute; 
python - <<'PY'
import os, re
jobs = [l.rstrip("\n") for l in open("mosaic_bmmc_site1_jobs.txt") if l.strip()]
redo = []
for j in jobs:
    out = re.search(r"--out (\S+)", j).group(1)
    nv = len(re.search(r"--variants ((?:\S+ )+?)--", j).group(1).split())
    have = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    if have < nv:
        if os.path.exists(out): os.remove(out)
        redo.append(j)
open("mosaic_bmmc_redo.txt", "w").write("\n".join(redo) + "\n")
print(len(redo), "of", len(jobs), "jobs to redo")
PY
cat mosaic_bmmc_redo.txt | xargs -P 5 -I{} sh -c "{} >> logs/mosaic_bmmc_redo.log 2>&1"
echo DONE > logs/mosaic_bmmc_redo.done
