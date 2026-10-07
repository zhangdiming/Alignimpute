#!/bin/bash
conda activate tf; 
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=10 TF_NUM_INTRAOP_THREADS=10 TF_NUM_INTEROP_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
for ds in handwritten caltech101_7 caltech101_20; do
  python -m alignimpute.dimvc_adapter --repo baselines/DIMVC --data data/$ds.npz --topologies complete random-2 chain two-comp --sweep 0.05 --seeds 0 1 2 3 4 --out results/r2_dimvc_$ds.jsonl > logs/r2_dimvc_$ds.log 2>&1 &
done
wait
echo R2_DIMVC_DONE >> logs/r2_dimvc.done
