#!/bin/bash
conda activate mvl-mid; 
P="python -m alignimpute.p0_handwritten"; S10="0 1 2 3 4 5 6 7 8 9"
( for d in handwritten caltech101_7 caltech101_20; do
    for t in 0.1 0.3 0.5 1.0 2.0; do CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $P --data data/$d.npz --out results/r7e9_tune_$d.jsonl --topologies random-2 --seeds 0 1 --variants A --tau $t --lam-proto 2 --tag r7e9tune --holdout-frac 0.1 >> logs/r7_e9.log 2>&1; done
    T=$(python -c "
import json,collections;m=collections.defaultdict(list)
[m[r[\"cfg\"][\"tau\"]].append(r[\"holdout_r1\"]) for r in map(json.loads,open(\"results/r7e9_tune_$d.jsonl\"))]
print(max(m,key=lambda k:sum(m[k])/len(m[k])))")
    echo "$d TAU=$T" >> logs/r7_e9.done
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $P --data data/$d.npz --out results/r7e9_$d.jsonl --topologies chain --seeds $S10 --variants A Anp C --tau $T --lam-proto 2 --tag r7e9 >> logs/r7_e9.log 2>&1
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $P --data data/$d.npz --out results/r7e9_$d.jsonl --topologies --sweep 0.05 --seeds $S10 --variants A Anp Cthin --tau $T --lam-proto 2 --tag r7e9 >> logs/r7_e9.log 2>&1
  done; echo E9_DONE >> logs/r7_e9.done ) &
( C="python -m alignimpute.cpspan_adapter --repo external/CPSPAN"
  for d in handwritten caltech101_7 caltech101_20; do
    for imp in released observed; do CUDA_VISIBLE_DEVICES=1 $C --data data/$d.npz --topologies random-2 --seeds 0 1 2 3 4 --impute $imp --out results/r7e8_cpspan_$d.jsonl >> logs/r7_e8a.log 2>&1; done
  done
  CUDA_VISIBLE_DEVICES=1 $C --data data/handwritten.npz --topologies chain two-comp --sweep 0.05 --seeds 3 4 --impute released --out results/r7e8_cpspan_handwritten.jsonl >> logs/r7_e8a.log 2>&1
  for d in caltech101_7 caltech101_20; do CUDA_VISIBLE_DEVICES=1 $C --data data/$d.npz --topologies two-comp --sweep 0.05 --seeds 3 4 --impute released --out results/r7e8_cpspan_$d.jsonl >> logs/r7_e8a.log 2>&1; done
  echo E8A_DONE >> logs/r7_e8a.done ) &
wait
