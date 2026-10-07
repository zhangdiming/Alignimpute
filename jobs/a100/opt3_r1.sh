#!/bin/bash
conda activate alignimpute
H="--lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"; S10="0 1 2 3 4 5 6 7 8 9"
mkdir -p results/opt3 logs; J=opt3_r1_jobs.txt; rm -f $J logs/opt3_r1.done
for g in cortex_stage2 cortex_stage2_no3c tcga_legacy9; do for t in 0.1 0.3 0.5 1.0 2.0; do
  echo "OMP_NUM_THREADS=2 $P --data data/$g.npz --topologies natural --seeds 0 1 --variants A --tau $t $H --no-std --holdout-frac 0.1 --tag opt3tune --out results/opt3/tune_${g}_t$t.jsonl" >> $J
  echo "OMP_NUM_THREADS=2 $P --data data/$g.npz --topologies natural --seeds $S10 --variants A B C --tau $t $H --no-std --tag opt3real --out results/opt3/real_${g}_t$t.jsonl" >> $J
done; done
for d in handwritten caltech101_7 caltech101_20 outdoorscene nuswide aloi proteinfold; do for fl in "" "--no-whiten" "--no-std --no-whiten"; do
  echo "OMP_NUM_THREADS=2 python -m alignimpute.linsync --data data/$d.npz --topologies chain thin05 --seeds $S10 $fl --out results/opt3/lin_bench.jsonl" >> $J
done; done
for g in cortex_stage2 cortex_stage2_no3c tcga_legacy9; do for fl in "" "--no-whiten" "--no-std --no-whiten"; do
  echo "OMP_NUM_THREADS=2 python -m alignimpute.linsync --data data/$g.npz --topologies natural --seeds $S10 $fl --out results/opt3/lin_real.jsonl" >> $J
done
  echo "OMP_NUM_THREADS=2 python -m alignimpute.linsync --data data/$g.npz --topologies natural --seeds $S10 --no-std --no-whiten --d 16 64 --out results/opt3/lin_real_dsens.jsonl" >> $J
done
for d in handwritten caltech101_7 caltech101_20; do
  echo "OMP_NUM_THREADS=2 python -m alignimpute.linsync --data data/$d.npz --topologies chain thin05 --seeds $S10 --no-std --no-whiten --d 16 64 --out results/opt3/lin_bench_dsens.jsonl" >> $J
done
wc -l $J
cat $J | xargs -P 16 -I{} sh -c "{} >> logs/opt3_r1.log 2>&1"
conda activate stabmap
for t in cortex6:D1 cortex5:D1 tcga9:D9; do dir=${t%%:*}; big=${t##*:}
  Rscript alignimpute/stabmap_run.R stabmap/$dir all noscale > stabmap/$dir.all_noscale.log 2>&1
  Rscript alignimpute/stabmap_run.R stabmap/$dir $big noscale > stabmap/$dir.big_noscale.log 2>&1
done
conda activate alignimpute
for t in cortex6:cortex_stage2:0:5:D1 cortex5:cortex_stage2_no3c:0:4:D1 tcga9:tcga_legacy9:0:8:D9; do IFS=: read dir g u v big <<< "$t"
  for tag in all_noscale ref_${big}_noscale; do python -m alignimpute.stabmap_eval score --data data/$g.npz --far $u $v --dir stabmap/$dir --tag $tag --out results/opt3/stabmap_noscale.jsonl; done
done
echo DONE > logs/opt3_r1.done
