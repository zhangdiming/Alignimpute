#!/bin/bash
conda activate alignimpute; 
while [ ! -f logs/ifE.done ]; do sleep 60; done
python - <<'PY'
import os, re
jobs = []
seen = set()
rest = [l.rstrip("\n") for l in open("ifE_jobs_rest.txt") if l.strip()]
rest_out = {re.search(r"--out (\S+)", j).group(1) for j in rest}
for j in rest + [l.rstrip("\n") for l in open("ifE_jobs.txt") if l.strip()]:
    out = re.search(r"--out (\S+)", j).group(1)
    if out in seen or (out in rest_out and j not in rest):
        continue
    seen.add(out)
    nv = len(re.search(r"--variants ((?:\S+ )+?)--", j).group(1).split())
    have = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    if have < nv:
        if os.path.exists(out): os.remove(out)
        jobs.append(j)
open("ifE_jobs_redo.txt", "w").write("\n".join(jobs) + ("\n" if jobs else ""))
print(len(jobs), "jobs to redo")
PY
[ -s ifE_jobs_redo.txt ] && cat ifE_jobs_redo.txt | xargs -P 4 -I{} sh -c "{} >> logs/ifE_redo.log 2>&1"
echo DONE > logs/ifE.done2
