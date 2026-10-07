#!/bin/bash
conda activate alignimpute; 
S10="0 1 2 3 4 5 6 7 8 9"; J=r18_p1_joblist.txt; rm -f $J
for spec in "handwritten 0.5" "caltech101_7 0.1" "caltech101_20 0.5"; do set -- $spec; d=$1; t=$2
  echo "python -m alignimpute.r18_kimp --data data/$d.npz --out results/r18_kimp_$d.jsonl --seeds $S10" >> $J
  echo "python -m alignimpute.p0_handwritten --data data/$d.npz --tau $t --lam-proto 2 --topologies chain --seeds $S10 --variants Aimp --tag r18aimp --out results/r18_aimp_${d}_chain.jsonl" >> $J
  echo "python -m alignimpute.p0_handwritten --data data/$d.npz --tau $t --lam-proto 2 --topologies --sweep 0.05 --seeds $S10 --variants Aimp --tag r18aimp --out results/r18_aimp_${d}_thin.jsonl" >> $J
done
export OMP_NUM_THREADS=4; rm -f logs/r18p1.done results/r18_kimp_*.jsonl results/r18_aimp_*.jsonl
cat $J | xargs -P 9 -I{} sh -c "{} >> logs/r18p1.log 2>&1; echo DONE >> logs/r18p1.done"
echo ALL_DONE >> logs/r18p1.done
