#!/bin/bash
conda activate alignimpute; 
mkdir -p results/obs_sens logs; rm -f logs/obs.done
H="--tau 0.1 --lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"
J=obs_jobs.txt; rm -f $J; i=0
for d in handwritten caltech101_7 caltech101_20; do
  for top in chain thin; do
    if [ $top = chain ]; then T="--topologies chain"; VAR="A C ALw Aimp"; else T="--topologies --sweep 0.05"; VAR="A Cthin ALw Aimp"; fi
    for s in 0 1 2 3 4 5 6 7 8 9; do
      for cond in all obs; do
        [ $cond = obs ] && F="--obs-stats" || F=""
        echo "CUDA_VISIBLE_DEVICES=$((i % 2)) OMP_NUM_THREADS=2 $P --data data/$d.npz $T --seeds $s --variants $VAR $H $F --tag obs_$cond --out results/obs_sens/host_${cond}_${d}_${top}_s$s.jsonl" >> $J
        i=$((i + 1))
      done
    done
  done
done
for d in handwritten caltech101_7 caltech101_20; do
  OMP_NUM_THREADS=4 python -m alignimpute.r18_kimp --data data/$d.npz --out results/obs_sens/kimp_all_$d.jsonl > logs/kimp_all_$d.log 2>&1
  OMP_NUM_THREADS=4 python -m alignimpute.r18_kimp --data data/$d.npz --obs-stats --out results/obs_sens/kimp_obs_$d.jsonl > logs/kimp_obs_$d.log 2>&1
done
cat $J | xargs -P 8 -I{} sh -c "{} >> logs/obs.log 2>&1"
echo DONE > logs/obs.done
