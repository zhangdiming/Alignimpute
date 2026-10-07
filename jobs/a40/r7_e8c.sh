#!/bin/bash
conda activate mvl-mid; 
jobs=""; k=0
for ds in handwritten:HandWritten caltech101_7:Caltech101_7 caltech101_20:Caltech101_20; do d=${ds%%:*}; nm=${ds##*:}
  for s in 0 1 2 3 4; do for top in complete random-2 chain two-comp sweep; do jobs="$jobs$d $nm $s $top $((k%2))\n"; k=$((k+1)); done; done; done
printf "$jobs" | OMP_NUM_THREADS=2 xargs -P 6 -L 1 sh -c 'd=$0; nm=$1; s=$2; top=$3; g=$4
  if [ $top = sweep ]; then TOP="--topologies --sweep 0.05"; else TOP="--topologies $top"; fi
  CUDA_VISIBLE_DEVICES=$g python -m alignimpute.treeeic_adapter --root baselines/TreeEIC --data data/$d.npz --name $nm $TOP --seeds $s --out results/r7e8_treeeic_$d.jsonl >> logs/r7_e8c.log 2>&1'
echo E8C_DONE >> logs/r7_e8c.done
