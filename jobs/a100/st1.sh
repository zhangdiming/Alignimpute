#!/bin/bash
conda activate alignimpute; export OMP_NUM_THREADS=4
H="--lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"; S10="0 1 2 3 4 5 6 7 8 9"; S5="0 1 2 3 4"
tau() { awk -v d=$1 "\$1==d{print \$2}" results/st0/tau.txt; }
mkdir -p results/st1 emb/st1; rm -f logs/st1.done; J=st1_jobs.txt; rm -f $J
for d in handwritten caltech101_7 caltech101_20; do t=$(tau $d)
  for top in chain thin; do
    if [ $top = chain ]; then TOP="--topologies chain"; X="Cnp"; else TOP="--topologies --sweep 0.05"; X="C Cnp"; fi
    echo "$P --data data/$d.npz $TOP --seeds $S10 --variants A Anp --tau $t $H --save-emb emb/st1/${d}_$top --tag st1abl --out results/st1/abl_${d}_${top}_host.jsonl" >> $J
    echo "$P --data data/$d.npz $TOP --seeds $S10 --variants $X --tau $t $H --tag st1abl --out results/st1/abl_${d}_${top}_np.jsonl" >> $J
    echo "$P --data data/$d.npz $TOP --seeds $S10 --variants Aimp --tau $t $H --tag st1abl --out results/st1/abl_${d}_${top}_aimp.jsonl" >> $J
    for s in $S10; do
      echo "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $P --data data/$d.npz $TOP --seeds $s --variants Cwp Cwpnp --tau $t $H --tag st1abl --out results/st1/abl_${d}_${top}_wp_s$s.jsonl" >> $J
    done
  done
done
for d in handwritten caltech101_7 caltech101_20 outdoorscene nuswide aloi proteinfold tcga_pancan; do t=$(tau $d)
  EX=""; [ $d = tcga_pancan ] && EX="--complete-only"
  for top in complete random-2 star ring chain two-comp; do
    echo "$P --data data/$d.npz $EX --topologies $top --seeds $S5 --variants A Anp C --tau $t $H --tag st1grid --out results/st1/grid_${d}_$top.jsonl" >> $J
  done
  [ $d != tcga_pancan ] && echo "$P --data data/$d.npz --topologies --sweep 0.05 --seeds $S5 --variants A Anp C Cthin --tau $t $H --tag st1grid --out results/st1/grid_${d}_thin05.jsonl" >> $J
done
t=$(tau handwritten); echo "$P --data data/handwritten.npz --topologies --sweep 0.1 0.02 --seeds $S5 --variants A C Cthin --tau $t $H --tag st1grid --out results/st1/grid_handwritten_sweep.jsonl" >> $J
wc -l $J
cat $J | xargs -P 20 -I{} sh -c "{} >> logs/st1.log 2>&1; echo DONE >> logs/st1.done"
for d in handwritten caltech101_7 caltech101_20; do for top in chain thin; do
  python -m alignimpute.alaux_baseline --emb-dir emb/st1/${d}_$top --out results/st1/abl_${d}_${top}_a2.jsonl --hosts A Anp >> logs/st1.log 2>&1 &
done; done; wait
echo ALL_DONE >> logs/st1.done
