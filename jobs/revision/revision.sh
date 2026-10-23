#!/bin/bash
conda activate alignimpute
mkdir -p results/eswa logs preds/eswa
S10="0 1 2 3 4 5 6 7 8 9"; REAL="cortex_stage2 cortex_stage2_no3c tcga_legacy9 tcga_pancan pbmc_mimitou bmmc_site1"; BENCH="handwritten caltech101_7 caltech101_20"
LR="python -m alignimpute.r18_kimp --method lowrank"; H="--tau 0.1 --lam-proto 2 --lam-rec 0 --nbr-gamma 50 --no-std"

A=revision_jobsA.txt; rm -f $A
for g in $REAL; do for s in $S10; do
  echo "OMP_NUM_THREADS=4 $LR --data data/$g.npz --configs natural --no-std --seeds $s --all-fits --save-pred preds/eswa --out results/eswa/lrimp_real_${g}_s$s.jsonl" >> $A; done; done
for d in $BENCH; do for s in $S10; do
  echo "OMP_NUM_THREADS=4 $LR --data data/$d.npz --configs chain chain-sweep-0.05 --seeds $s --all-fits --out results/eswa/lrimp_bench_${d}_s$s.jsonl" >> $A
  echo "OMP_NUM_THREADS=4 $LR --data data/$d.npz --configs chain star --seeds $s --out results/eswa/lrimp_star_${d}_s$s.jsonl" >> $A
  echo "OMP_NUM_THREADS=4 $LR --data data_hub/${d}_hub.npz --configs star --seeds $s --out results/eswa/lrimp_starhub_${d}_s$s.jsonl" >> $A; done; done
for d in handwritten caltech101_7 caltech101_20 outdoorscene aloi; do for s in $S10; do
  for lv in "a1 --class-alpha=1.0" "a03 --class-alpha=0.3" "s05 --size-sigma=0.5" "s1 --size-sigma=1.0" "b03 --batch-scale=0.3"; do set -- $lv
    echo "OMP_NUM_THREADS=4 $LR --data data/$d.npz --seeds $s $2 --tag eswa_shift_$1 --out results/eswa/lrimp_shift_${d}_$1_s$s.jsonl" >> $A; done; done; done

B=revision_jobsB.txt; rm -f $B
for g in cortex_stage2 cortex_stage2_no3c; do for s in $S10; do
  echo "OMP_NUM_THREADS=2 python -m alignimpute.p0_handwritten --data data/$g.npz --topologies natural --seeds $s --variants A C ALw Aimp $H --tag eswa_c007 --save-pred preds/eswa --out results/eswa/c007_host_${g}_s$s.jsonl" >> $B; done
  for m in recformer:RecFormer freecsl:FreeCSL dvimc:DVIMC ghicmc:GHICMC; do for s in 0 1 2 3 4; do
    echo "ALIGNIMPUTE_SAVE_PRED=preds/eswa OMP_NUM_THREADS=4 python baselines/${m%%:*}_adapter.py --datasets $g --seeds $s --out results/eswa/c007_${m##*:}_${g}_s$s.jsonl" >> $B; done; done; done

C=revision_jobsC.txt; rm -f $C
for k in 5 20 50; do
  for g in $REAL; do echo "OMP_NUM_THREADS=4 python -m alignimpute.r18_kimp --data data/$g.npz --configs natural --no-std --k $k --out results/eswa/kimp_k${k}_real_$g.jsonl" >> $C; done
  for d in $BENCH; do echo "OMP_NUM_THREADS=4 python -m alignimpute.r18_kimp --data data/$d.npz --k $k --seeds $S10 --out results/eswa/kimp_k${k}_bench_$d.jsonl" >> $C; done; done
for g in cortex_stage2 cortex_stage2_no3c; do for s in $S10; do
  echo "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m alignimpute.p0_handwritten --data data/$g.npz --topologies natural --seeds $s --variants Cwp $H --tag eswa_c007 --save-pred preds/eswa --out results/eswa/c007_wp_${g}_s$s.jsonl" >> $C; done
  echo "OMP_NUM_THREADS=4 python -m alignimpute.linsync --data data/$g.npz --topologies natural --seeds $S10 --methods L1 PCAcat --no-std --no-whiten --save-pred preds/eswa --out results/eswa/c007_lin_$g.jsonl" >> $C
  echo "OMP_NUM_THREADS=4 python -m alignimpute.r18_kimp --data data/$g.npz --configs natural --no-std --save-pred preds/eswa --out results/eswa/c007_kimp_$g.jsonl" >> $C; done
for gf in "cortex_stage2 5" "cortex_stage2_no3c 4"; do set -- $gf
  echo "OMP_NUM_THREADS=4 python -m alignimpute.stabmap_eval prep --data data/$1.npz --far 0 $2 --out stabmap/$1 && conda activate stabmap && Rscript alignimpute/stabmap_run.R stabmap/$1 all noscale && conda activate alignimpute && OMP_NUM_THREADS=4 python -m alignimpute.stabmap_eval score --data data/$1.npz --far 0 $2 --dir stabmap/$1 --tag all_noscale --save-pred preds/eswa --out results/eswa/c007_stabmap_$1.jsonl" >> $C; done
for g in $REAL; do echo "OMP_NUM_THREADS=2 python -m alignimpute.edge_resid --data data/$g.npz --topologies natural --no-std --out results/eswa/edge_resid.jsonl" >> $C; done
for d in $BENCH; do echo "OMP_NUM_THREADS=2 python -m alignimpute.edge_resid --data data/$d.npz --topologies chain thin05 --seeds $S10 --out results/eswa/edge_resid.jsonl" >> $C; done

(cat $A | xargs -P 6 -I{} sh -c "{} >> logs/revisionA.log 2>&1"; echo DONE > logs/revisionA.done) &
(cat $B | xargs -P 4 -I{} sh -c "{} >> logs/revisionB.log 2>&1"; echo DONE > logs/revisionB.done) &
(cat $C | xargs -P 12 -I{} sh -c "{} >> logs/revisionC.log 2>&1"; echo DONE > logs/revisionC.done) &
wait
python -m alignimpute.c007_analyze --data data --preds preds/eswa --out results/eswa/c007_summary.json
for f in results/eswa/lrimp_star_*_s*.jsonl; do b=$(basename $f); cp $f results/star/${b/lrimp_star_/lrimp_}; done
for f in results/eswa/lrimp_starhub_*_s*.jsonl; do b=$(basename $f); cp $f results/star_hub/${b/lrimp_starhub_/lrimp_}; done
python -m alignimpute.star_analyze --build results/star
python -m alignimpute.star_analyze --build results/star --star-build results/star_hub
