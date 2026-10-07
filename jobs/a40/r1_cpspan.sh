#!/bin/bash
conda activate mvl-mid

CUDA_VISIBLE_DEVICES=1 python -m alignimpute.cpspan_adapter --repo baselines/CPSPAN --data data/handwritten.npz --topologies complete chain two-comp --sweep 0.05 --seeds 0 1 2 3 4 --impute observed --out results/r1_cpspan_obsimp_handwritten.jsonl > logs/r1_cpspan_hw.log 2>&1
for ds in caltech101_7 caltech101_20; do
  CUDA_VISIBLE_DEVICES=1 python -m alignimpute.cpspan_adapter --repo baselines/CPSPAN --data data/$ds.npz --topologies complete chain --seeds 0 1 2 3 4 --impute observed --out results/r1_cpspan_obsimp_$ds.jsonl > logs/r1_cpspan_$ds.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python -m alignimpute.cpspan_adapter --repo baselines/CPSPAN --data data/$ds.npz --topologies complete chain --seeds 3 4 --out results/p4_cpspan_$ds.jsonl >> logs/p4_cpspan_$ds.log 2>&1
done
echo R1_CPSPAN_DONE >> logs/r1_cpspan.done
