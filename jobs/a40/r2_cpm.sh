#!/bin/bash
conda activate mvl-mid; 
for ds in handwritten caltech101_7 caltech101_20; do for ep in 30 300; do
  CUDA_VISIBLE_DEVICES=0 python -m alignimpute.cpmnets_baseline --data data/$ds.npz --out results/r2_cpmnets_$ds.jsonl --topologies complete random-2 chain two-comp --sweep 0.05 --seeds 0 1 2 3 4 --epochs $ep --tag ep$ep >> logs/r2_cpmnets_$ds.log 2>&1
done; done
echo R2_CPM_DONE >> logs/r2_cpm.done
