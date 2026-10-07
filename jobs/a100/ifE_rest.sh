#!/bin/bash
conda activate alignimpute; 
while [ $(pgrep -c -f "python -m alignimpute.p0_handwritten") -gt 6 ]; do sleep 30; done
cat ifE_jobs_rest.txt | xargs -P 14 -I{} sh -c "{} >> logs/ifE.log 2>&1"
while [ $(pgrep -c -f "python -m alignimpute.p0_handwritten") -gt 0 ]; do sleep 30; done
python -m alignimpute.linsync --data data/tcga_pancan.npz --topologies natural --seeds 0 1 2 3 4 5 6 7 8 9 --no-std --no-whiten --out results/ifE/lin_pancan.jsonl >> logs/ifE.log 2>&1
echo DONE > logs/ifE.done
