#!/bin/bash
conda activate mvl-mid; 
while ! grep -q E8B_C7_DONE logs/r7_e8b.done 2>/dev/null; do sleep 60; done
M="python -m alignimpute.mvp_adapter --mvp-root external/MVP --python python"
tune() { d=$1; nm=$2; g=$3
  for st in hw scene15 reuters; do CUDA_VISIBLE_DEVICES=$g $M --data data/$d.npz --dataset $nm --mvp-style $st --topologies complete --seeds 0 1 --out results/r7e9b_mvptune_$d.jsonl >> logs/r7_e9b.log 2>&1; done
  ST=$(python -c "
import json,collections;m=collections.defaultdict(list)
[m[r[\"mvp_style\"]].append(r[\"acc\"]) for r in map(json.loads,open(\"results/r7e9b_mvptune_$d.jsonl\")) if r[\"status\"]==\"ok\"]
print(max(m,key=lambda k:sum(m[k])/len(m[k])))")
  echo "$d STYLE=$ST" >> logs/r7_e9b.done
  if [ "$ST" != hw ]; then CUDA_VISIBLE_DEVICES=$g $M --data data/$d.npz --dataset $nm --mvp-style $ST --topologies complete random-2 chain two-comp --seeds 0 1 2 3 4 --out results/r7e9b_mvp_$d.jsonl >> logs/r7_e9b.log 2>&1; fi; }
tune caltech101_7 Caltech101 0 & tune caltech101_20 Caltech101_20 1 & wait
echo E9B_DONE >> logs/r7_e9b.done
