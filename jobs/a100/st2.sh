#!/bin/bash
conda activate alignimpute; 
H="--lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"; S10="0 1 2 3 4 5 6 7 8 9"; S5="0 1 2 3 4"
tau() { awk -v d=$1 "\$1==d{print \$2}" results/st0/tau.txt; }
LEVELS="a1:--class-alpha=1.0 a03:--class-alpha=0.3 s05:--size-sigma=0.5 s1:--size-sigma=1.0 b03:--batch-scale=0.3"
mkdir -p results/st2 emb/st2; rm -f logs/st2.done
J=st2_fast.txt; rm -f $J
for d in handwritten caltech101_7 caltech101_20 outdoorscene nuswide aloi proteinfold tcga_pancan; do t=$(tau $d); EX=""; [ $d = tcga_pancan ] && EX="--complete-only"
  for top in complete random-2 star ring chain two-comp; do
    echo "$P --data data/$d.npz $EX --topologies $top --seeds $S5 --variants A --tau $t $H --save-emb emb/st2/gate_${d}_$top --tag st2gate --out results/st2/gate_${d}_$top.jsonl" >> $J; done
  [ $d != tcga_pancan ] && echo "$P --data data/$d.npz --topologies --sweep 0.05 --seeds $S5 --variants A --tau $t $H --save-emb emb/st2/gate_${d}_thin --tag st2gate --out results/st2/gate_${d}_thin.jsonl" >> $J
done
for d in handwritten caltech101_7 caltech101_20 outdoorscene aloi; do t=$(tau $d)
  for lv in $LEVELS; do n=${lv%%:*}; opt=${lv#*:}
    echo "$P --data data/$d.npz --topologies chain --seeds $S10 --variants A Anp C Apl --tau $t $H $opt --save-emb emb/st2/shift_${d}_$n --tag st2shift_$n --out results/st2/shift_${d}_${n}_chain.jsonl" >> $J
    echo "$P --data data/$d.npz --topologies --sweep 0.05 --seeds $S10 --variants A Anp Cthin Apl --tau $t $H $opt --save-emb emb/st2/shift_${d}_$n --tag st2shift_$n --out results/st2/shift_${d}_${n}_thin.jsonl" >> $J
    echo "python -m alignimpute.r18_kimp --data data/$d.npz --seeds $S10 $opt --tag st2shift_$n --out results/st2/shiftk_${d}_$n.jsonl" >> $J
  done; done
t=$(tau tcga_legacy9); echo "$P --data data/tcga_legacy9.npz --topologies natural --seeds $S10 --variants A Anp B C Cg Clate Apl --tau $t $H --trace --save-emb emb/st2/real_tcga9 --tag st2real --out results/st2/real_tcga9.jsonl" >> $J
t=$(tau tcga_pancan); echo "$P --data data/tcga_pancan.npz --topologies natural --seeds $S5 --variants A C --tau $t $H --tag st2real --out results/st2/real_pancan.jsonl" >> $J
for g in cortex_stage2 cortex_stage2_no3c; do for t in 0.3 0.5; do
  echo "$P --data data/$g.npz --topologies natural --seeds $S10 --variants A Anp B C Clate Apl Cpl --tau $t $H --trace --acc-trace --save-emb emb/st2/real_${g}_t$t --tag st2real --out results/st2/real_${g}_t$t.jsonl" >> $J
done; done
wc -l $J; export OMP_NUM_THREADS=2
cat $J | xargs -P 30 -I{} sh -c "{} >> logs/st2.log 2>&1; echo F >> logs/st2.done"
echo FAST_DONE >> logs/st2.done
J2=st2_slow.txt; rm -f $J2
for d in handwritten caltech101_7 caltech101_20 outdoorscene aloi; do t=$(tau $d)
  for lv in $LEVELS; do n=${lv%%:*}; opt=${lv#*:}; for s in $S10; do
    echo "$P --data data/$d.npz --topologies chain --seeds $s --variants Cwp Cwpnp --tau $t $H $opt --tag st2shift_$n --out results/st2/shiftwp_${d}_${n}_chain_s$s.jsonl" >> $J2
    echo "$P --data data/$d.npz --topologies --sweep 0.05 --seeds $s --variants Cwp Cwpnp --tau $t $H $opt --tag st2shift_$n --out results/st2/shiftwp_${d}_${n}_thin_s$s.jsonl" >> $J2
  done; done; done
for s in $S10; do t=$(tau tcga_legacy9); echo "$P --data data/tcga_legacy9.npz --topologies natural --seeds $s --variants Cwp --tau $t $H --tag st2real --out results/st2/realwp_tcga9_s$s.jsonl" >> $J2
  for g in cortex_stage2 cortex_stage2_no3c; do for t in 0.3 0.5; do
    echo "$P --data data/$g.npz --topologies natural --seeds $s --variants Cwp Cwpnp --tau $t $H --tag st2real --out results/st2/realwp_${g}_t${t}_s$s.jsonl" >> $J2; done; done; done
wc -l $J2
cat $J2 | OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 xargs -P 44 -I{} sh -c "{} >> logs/st2wp.log 2>&1; echo S >> logs/st2.done"
echo SLOW_DONE >> logs/st2.done
for dir in emb/st2/shift_* emb/st2/real_*; do b=$(basename $dir)
  echo "python -m alignimpute.alaux_baseline --emb-dir $dir --out results/st2/a2_$b.jsonl --hosts A Anp"; done | OMP_NUM_THREADS=2 xargs -P 24 -I{} sh -c "{} >> logs/st2a2.log 2>&1"
echo ALL_DONE >> logs/st2.done
