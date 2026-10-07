#!/bin/bash
conda activate alignimpute; export OMP_NUM_THREADS=4
H="--lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"
mkdir -p results/st0; rm -f results/st0/*.jsonl logs/st0.done; J=st0a_jobs.txt; rm -f $J
cfg() { case $1 in proteinfold) echo "--topologies complete";; tcga_pancan) echo "--complete-only --topologies random-2";;
  tcga_legacy9|cortex_stage2|cortex_stage2_no3c) echo "--topologies natural";; *) echo "--topologies random-2";; esac; }
DS="handwritten caltech101_7 caltech101_20 outdoorscene nuswide aloi proteinfold tcga_pancan tcga_legacy9 cortex_stage2 cortex_stage2_no3c"
for d in $DS; do for t in 0.1 0.3 0.5 1.0 2.0; do
  echo "$P --data data/$d.npz $(cfg $d) --seeds 0 1 --variants A --tau $t $H --holdout-frac 0.1 --tag st0tune --out results/st0/tune_${d}_t$t.jsonl" >> $J
done; done
cat $J | xargs -P 18 -I{} sh -c "{} >> logs/st0.log 2>&1"
python - <<PY > results/st0/tau.txt
import json,glob,collections
for d in "$DS".split():
    m=collections.defaultdict(list)
    for f in glob.glob(f"results/st0/tune_{d}_t*.jsonl"):
        for r in map(json.loads,open(f)): m[r["cfg"]["tau"]].append(r["holdout_r1"])
    print(d, max(m,key=lambda k:sum(m[k])/len(m[k])), " ".join(f"{k}:{sum(v)/len(v):.3f}" for k,v in sorted(m.items())))
PY
echo TAU_DONE >> logs/st0.done
J=st0b_jobs.txt; rm -f $J
for d in handwritten caltech101_7 caltech101_20; do t=$(awk -v d=$d "\$1==d{print \$2}" results/st0/tau.txt)
  echo "$P --data data/$d.npz --topologies chain --seeds 0 1 2 3 4 5 6 7 8 9 --variants A Anp C Clate Apl Cpl --tau $t $H --trace --acc-trace --tag st0lock --out results/st0/lock_${d}_chain.jsonl" >> $J
  echo "$P --data data/$d.npz --topologies --sweep 0.05 --seeds 0 1 2 3 4 5 6 7 8 9 --variants A Anp Cthin Cthinlate Apl Cthinpl --tau $t $H --trace --acc-trace --tag st0lock --out results/st0/lock_${d}_thin.jsonl" >> $J
done
cat $J | xargs -P 6 -I{} sh -c "{} >> logs/st0.log 2>&1"
echo ALL_DONE >> logs/st0.done
