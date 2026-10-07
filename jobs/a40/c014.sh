#!/bin/bash
conda activate alignimpute; 
mkdir -p results/c014 results/obs_sens logs; rm -f logs/c014*.done
H="--tau 0.1 --lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"
J=c014_small.txt; rm -f $J; i=0
for g in cortex_stage2 cortex_stage2_no3c tcga_legacy9; do for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "CUDA_VISIBLE_DEVICES=$((i % 2)) OMP_NUM_THREADS=2 $P --data data/$g.npz --topologies natural --seeds $s --variants Aimp $H --no-std --tag c014 --out results/c014/real_${g}_aimp_s$s.jsonl" >> $J; i=$((i + 1))
done; done
for d in handwritten caltech101_7 caltech101_20; do for top in chain thin; do
  [ $top = chain ] && T="--topologies chain" || T="--topologies --sweep 0.05"
  for s in 0 1 2 3 4 5 6 7 8 9; do for cond in all obs; do
    [ $cond = obs ] && F="--obs-stats" || F=""
    echo "CUDA_VISIBLE_DEVICES=$((i % 2)) OMP_NUM_THREADS=2 $P --data data/$d.npz $T --seeds $s --variants Aimp $H $F --tag obs_$cond --out results/obs_sens/aimp_${cond}_${d}_${top}_s$s.jsonl" >> $J; i=$((i + 1))
  done; done
done; done
J2=c014_big.txt; rm -f $J2; i=0
for g in pbmc_mimitou bmmc_site1; do for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "CUDA_VISIBLE_DEVICES=$((i % 2)) OMP_NUM_THREADS=4 $P --data data/$g.npz --topologies natural --seeds $s --variants Aimp $H --no-std --tag c014 --out results/c014/real_${g}_aimp_s$s.jsonl" >> $J2; i=$((i + 1))
done; done
cat $J | xargs -P 8 -I{} sh -c "{} >> logs/c014_small.log 2>&1"; echo DONE > logs/c014_small.done
cat $J2 | xargs -P 2 -I{} sh -c "{} >> logs/c014_big.log 2>&1"; echo DONE > logs/c014_big.done
