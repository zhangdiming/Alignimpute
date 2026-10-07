#!/bin/bash
conda activate alignimpute
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/aw logs
run() { python -m alignimpute.aw_sens --aw $1 -- "${@:2}" > logs/aw_$1_$(basename $3 .npz).log 2>&1; }
for w in 5 10 20 40 80; do
  for d in handwritten caltech101_7 caltech101_20 outdoorscene nuswide aloi proteinfold; do
    run $w --data data/$d.npz --topologies thin05 --methods L1 --no-whiten --out results/aw/aw$w.jsonl --tag aw$w &
  done
  run $w --data data/tcga_legacy9.npz --topologies natural --methods L1 --no-std --no-whiten --out results/aw/aw${w}_tcga.jsonl --tag aw$w &
done
wait; touch logs/aw.done
