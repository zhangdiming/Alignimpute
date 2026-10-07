#!/bin/bash
conda activate alignimpute; 
H="--lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"; S10="0 1 2 3 4 5 6 7 8 9"; S5="0 1 2 3 4"
mkdir -p results/optB1; J=optB1_jobs.txt; rm -f $J logs/optB1.done
VR="A C AL ALw CL CLw"
for g in cortex_stage2 cortex_stage2_no3c tcga_legacy9; do for s in $S10; do
  echo "OMP_NUM_THREADS=2 $P --data data/$g.npz --topologies natural --seeds $s --variants $VR --tau 0.1 $H --no-std --tag optB1 --out results/optB1/real_${g}_s$s.jsonl" >> $J
done; done
for d in handwritten caltech101_7 caltech101_20; do for s in $S10; do
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies chain --seeds $s --variants A C AL ALw CL CLw --tau 0.1 $H --tag optB1 --out results/optB1/bench_${d}_chain_s$s.jsonl" >> $J
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies --sweep 0.05 --seeds $s --variants A Cthin AL ALw CL CLw --tau 0.1 $H --tag optB1 --out results/optB1/bench_${d}_thin_s$s.jsonl" >> $J
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies chain --seeds $s --variants A C AL CL --tau 0.1 $H --no-std --tag optB1ns --out results/optB1/benchns_${d}_chain_s$s.jsonl" >> $J
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies --sweep 0.05 --seeds $s --variants A Cthin AL CL --tau 0.1 $H --no-std --tag optB1ns --out results/optB1/benchns_${d}_thin_s$s.jsonl" >> $J
done; done
for d in outdoorscene nuswide aloi proteinfold; do for s in $S5; do
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies chain --seeds $s --variants A C AL ALw CL CLw --tau 0.1 $H --tag optB1 --out results/optB1/bench_${d}_chain_s$s.jsonl" >> $J
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies --sweep 0.05 --seeds $s --variants A Cthin AL ALw CL CLw --tau 0.1 $H --tag optB1 --out results/optB1/bench_${d}_thin_s$s.jsonl" >> $J
done; done
wc -l $J
cat $J | xargs -P 16 -I{} sh -c "{} >> logs/optB1.log 2>&1"
echo DONE > logs/optB1.done
