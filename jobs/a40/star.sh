#!/bin/bash
conda activate alignimpute; 
mkdir -p results/star logs; rm -f logs/star.done
H="--tau 0.1 --lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"
J=star_jobs.txt; rm -f $J; i=0
for d in handwritten caltech101_7 caltech101_20; do for top in chain star; do for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "CUDA_VISIBLE_DEVICES=$((i % 2)) OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies $top --seeds $s --variants A C ALw $H --tag star_ctl --out results/star/host_${d}_${top}_s$s.jsonl" >> $J; i=$((i + 1))
  echo "CUDA_VISIBLE_DEVICES=$((i % 2)) OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies $top --seeds $s --variants Aimp $H --tag star_ctl --out results/star/aimp_${d}_${top}_s$s.jsonl" >> $J; i=$((i + 1))
done; done; done
for d in handwritten caltech101_7 caltech101_20; do
  OMP_NUM_THREADS=4 python -m alignimpute.r18_kimp --data data/$d.npz --configs chain star --out results/star/kimp_$d.jsonl > logs/kimp_$d.log 2>&1
  OMP_NUM_THREADS=4 python -m alignimpute.linsync --data data/$d.npz --topologies chain star --methods L1 PCAcat --no-whiten --out results/star/lin_$d.jsonl > logs/lin_$d.log 2>&1
done
cat $J | xargs -P 8 -I{} sh -c "{} >> logs/star.log 2>&1"
echo DONE > logs/star.done
