#!/bin/bash
conda activate mvl-mid; 
M="python -m alignimpute.mvp_adapter --mvp-root external/MVP --python python --epochs 150 --topologies complete random-2 chain two-comp --seeds 0 1 2 3 4"
( CUDA_VISIBLE_DEVICES=0 $M --data data/caltech101_7.npz --dataset Caltech101 --out results/r7e8_mvp_caltech101_7.jsonl >> logs/r7_e8b.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 $M --data data/caltech101_20.npz --dataset Caltech101_20 --out results/r7e8_mvp_caltech101_20.jsonl >> logs/r7_e8b.log 2>&1 ) &
wait; echo E8B_DONE >> logs/r7_e8b.done
