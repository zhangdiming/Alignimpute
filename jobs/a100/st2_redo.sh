#!/bin/bash
conda activate alignimpute; rm -f logs/st2redo.done
cat st2_redo.txt | OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 xargs -P 12 -I{} sh -c "{} >> logs/st2redo.log 2>&1; echo R >> logs/st2redo.done"
for dir in emb/st2/real_*; do b=$(basename $dir); [ -s results/st2/a2_$b.jsonl ] || echo "python -m alignimpute.alaux_baseline --emb-dir $dir --out results/st2/a2_$b.jsonl --hosts A Anp"; done | OMP_NUM_THREADS=2 xargs -r -P 6 -I{} sh -c "{} >> logs/st2a2.log 2>&1"
echo ALL_DONE >> logs/st2redo.done
