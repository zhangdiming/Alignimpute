#!/bin/bash
conda activate alignimpute; 
mkdir -p results/star_hub data_hub logs; rm -f logs/star_hub.done
python - <<PY
import numpy as np
for d,h in (("handwritten",4),("caltech101_7",5),("caltech101_20",5)):
    z=np.load(f"data/{d}.npz"); V=len([k for k in z.files if k.startswith("X")])
    order=[h]+[v for v in range(V) if v!=h]
    np.savez(f"data_hub/{d}_hub.npz", y=z["y"], **{f"X{i}": z[f"X{v}"] for i,v in enumerate(order)})
    print(d, "view order", order)
PY
H="--tau 0.1 --lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"
J=star_hub_jobs.txt; rm -f $J; i=0
for d in handwritten caltech101_7 caltech101_20; do for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "CUDA_VISIBLE_DEVICES=$((i % 2)) OMP_NUM_THREADS=2 $P --data data_hub/${d}_hub.npz --topologies star --seeds $s --variants A C ALw $H --tag star_hub --out results/star_hub/host_${d}_star_s$s.jsonl" >> $J; i=$((i + 1))
  echo "CUDA_VISIBLE_DEVICES=$((i % 2)) OMP_NUM_THREADS=2 $P --data data_hub/${d}_hub.npz --topologies star --seeds $s --variants Aimp $H --tag star_hub --out results/star_hub/aimp_${d}_star_s$s.jsonl" >> $J; i=$((i + 1))
done; done
for d in handwritten caltech101_7 caltech101_20; do
  OMP_NUM_THREADS=4 python -m alignimpute.r18_kimp --data data_hub/${d}_hub.npz --configs star --out results/star_hub/kimp_$d.jsonl > logs/kimp_hub_$d.log 2>&1
  OMP_NUM_THREADS=4 python -m alignimpute.linsync --data data_hub/${d}_hub.npz --topologies star --methods L1 PCAcat --no-whiten --out results/star_hub/lin_$d.jsonl > logs/lin_hub_$d.log 2>&1
done
cat $J | xargs -P 8 -I{} sh -c "{} >> logs/star_hub.log 2>&1"
echo DONE > logs/star_hub.done
