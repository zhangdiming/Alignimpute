#!/bin/bash
conda activate alignimpute; 
g=$1; H="--lam-proto 2 --lam-rec 0 --nbr-gamma 50"; P="python -m alignimpute.p0_handwritten"; S10="0 1 2 3 4 5 6 7 8 9"
mkdir -p results/mosaic logs; rm -f logs/mosaic_$g.done
J=mosaic_${g}_tune.txt; rm -f $J
for t in 0.1 0.3 0.5 1.0 2.0; do echo "OMP_NUM_THREADS=2 $P --data data/$g.npz --topologies natural --seeds 0 1 --variants A --tau $t $H --no-std --holdout-frac 0.1 --tag mtune --out results/mosaic/tune_${g}_t$t.jsonl" >> $J; done
cat $J | xargs -P 5 -I{} sh -c "{} >> logs/mosaic_$g.log 2>&1"
T=$(python -c "
import json,glob
m={}
for f in glob.glob('results/mosaic/tune_${g}_t*.jsonl'):
    rs=[json.loads(l) for l in open(f)]; m[rs[0]['cfg']['tau']]=sum(r['holdout_r1'] for r in rs)/len(rs)
print(max(m,key=m.get))")
echo "$g $T" > results/mosaic/tau_$g.txt
J=mosaic_${g}_jobs.txt; rm -f $J
for s in $S10; do
  echo "OMP_NUM_THREADS=2 $P --data data/$g.npz --topologies natural --seeds $s --variants A B C ALw CLw Aimp --tau $T $H --no-std --tag mmain --out results/mosaic/main_${g}_s$s.jsonl" >> $J
  for t in 0.1 0.3 0.5 1.0 2.0; do [ $t = $T ] || echo "OMP_NUM_THREADS=2 $P --data data/$g.npz --topologies natural --seeds $s --variants A ALw --tau $t $H --no-std --tag mtau --out results/mosaic/tau_${g}_t${t}_s$s.jsonl" >> $J; done
done
cat $J | xargs -P 8 -I{} sh -c "{} >> logs/mosaic_$g.log 2>&1"
python -m alignimpute.linsync --data data/$g.npz --topologies natural --seeds $S10 --no-std --no-whiten --out results/mosaic/lin_$g.jsonl >> logs/mosaic_$g.log 2>&1
python -m alignimpute.stabmap_eval prep --data data/$g.npz --far 0 2 --out stabmap/$g >> logs/mosaic_$g.log 2>&1
BIG=$(python -c "import json; m=json.load(open('stabmap/$g/meta.json')); print(max(m,key=lambda r:r['n_cells'])['dataset'])")
conda activate stabmap
Rscript alignimpute/stabmap_run.R stabmap/$g all noscale >> logs/mosaic_$g.log 2>&1
Rscript alignimpute/stabmap_run.R stabmap/$g $BIG noscale >> logs/mosaic_$g.log 2>&1
conda activate alignimpute
for tag in all_noscale ref_${BIG}_noscale; do python -m alignimpute.stabmap_eval score --data data/$g.npz --far 0 2 --dir stabmap/$g --tag $tag --out results/mosaic/stabmap_$g.jsonl >> logs/mosaic_$g.log 2>&1; done
echo DONE > logs/mosaic_$g.done
