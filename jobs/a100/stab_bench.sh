#!/bin/bash

J=stab_bench_jobs.txt; rm -f $J
for d in handwritten caltech101_7 caltech101_20; do V=6; for t in chain thin05; do for s in 0 1 2 3 4 5 6 7 8 9; do
  dir=stabmap/bench_${d}_${t}_s$s
  echo "conda activate alignimpute && python -m alignimpute.stabmap_eval prep --data data/$d.npz --topology $t --seed $s --far 0 5 --out $dir >/dev/null && conda activate stabmap && Rscript alignimpute/stabmap_run.R $dir all >/dev/null 2>&1 && conda activate alignimpute && python -m alignimpute.stabmap_eval score --data data/$d.npz --topology $t --seed $s --far 0 5 --dir $dir --tag all --out results/bench/stabmap_${d}_${t}.jsonl" >> $J
done; done; done
cat $J | xargs -P 12 -I{} bash -c "{} >> logs/stab_bench.log 2>&1"
echo DONE > logs/stab_bench.done
