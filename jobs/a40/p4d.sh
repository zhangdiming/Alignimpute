#!/bin/bash
conda activate mvl-mid

for ds in handwritten caltech101_7 caltech101_20; do
  CUDA_VISIBLE_DEVICES=1 python -m alignimpute.cpspan_adapter --repo baselines/CPSPAN --data data/$ds.npz --topologies complete random-2 star ring chain two-comp --sweep 0.1 0.05 0.02 --seeds 0 1 2 3 4 --out results/p4_cpspan_$ds.jsonl > logs/p4_cpspan_$ds.log 2>&1
done
