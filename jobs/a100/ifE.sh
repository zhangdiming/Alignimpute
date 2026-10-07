#!/bin/bash
conda activate alignimpute; 
H="--lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"; S10="0 1 2 3 4 5 6 7 8 9"; S5="0 1 2 3 4"
mkdir -p results/ifE logs; rm -f logs/ifE.done
J0=ifE0_jobs.txt; rm -f $J0
for t in 0.1 0.3 0.5 1.0 2.0; do echo "OMP_NUM_THREADS=2 $P --data data/tcga_pancan.npz --complete-only --topologies random-2 --seeds 0 1 --variants A --tau $t $H --no-std --holdout-frac 0.1 --tag ifEtune --out results/ifE/tune_pancan_t$t.jsonl" >> $J0; done
cat $J0 | xargs -P 5 -I{} sh -c "{} >> logs/ifE.log 2>&1"
TP=$(python -c "
import json,glob
m={}
for f in glob.glob('results/ifE/tune_pancan_t*.jsonl'):
    rs=[json.loads(l) for l in open(f)]; m[rs[0]['cfg']['tau']]=sum(r['holdout_r1'] for r in rs)/len(rs)
print(max(m,key=m.get))")
echo "pancan_nostd_tau $TP" > results/ifE/tau_pancan.txt
J=ifE_jobs.txt; rm -f $J
for g in cortex_stage2 cortex_stage2_no3c tcga_legacy9; do for s in $S10; do
  echo "OMP_NUM_THREADS=2 $P --data data/$g.npz --topologies natural --seeds $s --variants A B C ALw CLw Aw AL Aimp --tau 0.1 $H --no-std --tag ifE1 --out results/ifE/real_${g}_s$s.jsonl" >> $J
  echo "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $P --data data/$g.npz --topologies natural --seeds $s --variants Cwp --tau 0.1 $H --no-std --tag ifE1 --out results/ifE/realwp_${g}_s$s.jsonl" >> $J
  for t in 0.3 0.5 1.0 2.0; do echo "OMP_NUM_THREADS=2 $P --data data/$g.npz --topologies natural --seeds $s --variants ALw --tau $t $H --no-std --tag ifE1tau --out results/ifE/realtau_${g}_t${t}_s$s.jsonl" >> $J; done
done; done
for s in $S5; do echo "OMP_NUM_THREADS=2 $P --data data/tcga_pancan.npz --topologies natural --seeds $s --variants A B C ALw CLw --tau $TP $H --no-std --tag ifE1 --out results/ifE/real_tcga_pancan_s$s.jsonl" >> $J; done
for d in handwritten caltech101_7 caltech101_20 outdoorscene nuswide aloi proteinfold tcga_pancan; do
  if [ $d = tcga_pancan ]; then EX="--complete-only --no-std"; t=$TP; else EX=""; t=0.1; fi
  for top in complete random-2 star ring chain two-comp; do for s in $S5; do
    echo "OMP_NUM_THREADS=2 $P --data data/$d.npz $EX --topologies $top --seeds $s --variants A C ALw CLw --tau $t $H --tag ifE2 --out results/ifE/grid_${d}_${top}_s$s.jsonl" >> $J
  done; done
  [ $d != tcga_pancan ] && for s in $S5; do echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies --sweep 0.05 --seeds $s --variants A Cthin ALw CLw --tau 0.1 $H --tag ifE2 --out results/ifE/grid_${d}_thin05_s$s.jsonl" >> $J; done
done
LEVELS="a1:--class-alpha=1.0 a03:--class-alpha=0.3 s05:--size-sigma=0.5 s1:--size-sigma=1.0 b03:--batch-scale=0.3"
for d in handwritten caltech101_7 caltech101_20 outdoorscene aloi; do for lv in $LEVELS; do n=${lv%%:*}; opt=${lv#*:}; for s in $S10; do
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies chain --seeds $s --variants A ALw CLw --tau 0.1 $H $opt --tag ifE3_$n --out results/ifE/shift_${d}_${n}_chain_s$s.jsonl" >> $J
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies --sweep 0.05 --seeds $s --variants A ALw CLw --tau 0.1 $H $opt --tag ifE3_$n --out results/ifE/shift_${d}_${n}_thin_s$s.jsonl" >> $J
done; done; done
for d in handwritten caltech101_7 caltech101_20; do for s in $S10; do
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies chain --seeds $s --variants Aw --tau 0.1 $H --tag ifE4 --out results/ifE/abl_${d}_chain_s$s.jsonl" >> $J
  echo "OMP_NUM_THREADS=2 $P --data data/$d.npz --topologies --sweep 0.05 --seeds $s --variants Aw --tau 0.1 $H --tag ifE4 --out results/ifE/abl_${d}_thin_s$s.jsonl" >> $J
done; done
wc -l $J
cat $J | xargs -P 16 -I{} sh -c "{} >> logs/ifE.log 2>&1"
python -m alignimpute.linsync --data data/tcga_pancan.npz --topologies natural --seeds $S10 --no-std --no-whiten --out results/ifE/lin_pancan.jsonl >> logs/ifE.log 2>&1
echo DONE > logs/ifE.done
