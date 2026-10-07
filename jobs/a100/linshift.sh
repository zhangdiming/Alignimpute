#!/bin/bash
conda activate alignimpute; mkdir -p results/linshift; J=linshift_jobs.txt; rm -f $J
for d in handwritten caltech101_7 caltech101_20 outdoorscene aloi; do for t in chain thin05; do
  for lv in "a1 --class-alpha 1.0" "a03 --class-alpha 0.3" "s05 --size-sigma 0.5" "s1 --size-sigma 1.0" "b03 --batch-scale 0.3"; do set -- $lv
    echo "python -m alignimpute.linsync --data data/$d.npz --topologies $t --seeds 0 1 2 3 4 5 6 7 8 9 --methods L1 PCAcat --no-whiten $2 $3 --tag $1 --out results/linshift/${d}_${t}_$1.jsonl" >> $J; done; done; done
cat $J | xargs -P 16 -I{} sh -c "OMP_NUM_THREADS=2 {} > /dev/null 2>> logs/linshift.log"; echo DONE > logs/linshift.done
