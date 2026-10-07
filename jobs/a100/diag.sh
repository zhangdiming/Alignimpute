#!/bin/bash
conda activate alignimpute
P="python -m alignimpute.graph_diag"; S10="0 1 2 3 4 5 6 7 8 9"; J=diag_jobs.txt; rm -f $J; mkdir -p results/diag
for g in cortex_stage2 cortex_stage2_no3c tcga_legacy9 pbmc_mimitou bmmc_site1 tcga_pancan; do echo "$P --data data/$g.npz --topology natural --no-std --tag real --out results/diag/real_$g.jsonl" >> $J; done
for d in handwritten caltech101_7 caltech101_20; do for t in chain thin05; do echo "$P --data data/$d.npz --topology $t --seeds $S10 --tag bench --out results/diag/bench_${d}_$t.jsonl" >> $J; done; done
for d in handwritten caltech101_7 caltech101_20 outdoorscene aloi; do for t in chain thin05; do
  for lv in "a1 --class-alpha 1.0" "a03 --class-alpha 0.3" "s05 --size-sigma 0.5" "s1 --size-sigma 1.0" "b03 --batch-scale 0.3"; do set -- $lv
    echo "$P --data data/$d.npz --topology $t --seeds $S10 $2 $3 --tag shift_$1 --out results/diag/shift_${d}_${t}_$1.jsonl" >> $J; done; done; done
wc -l $J; cat $J | xargs -P 16 -I{} sh -c "OMP_NUM_THREADS=2 {} >> logs/diag.log 2>&1"; echo DONE > logs/diag.done
