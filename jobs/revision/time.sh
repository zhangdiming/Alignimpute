#!/bin/bash
conda activate alignimpute
mkdir -p results/eswa/time_out logs/time stabmap_time
T="OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8"; O=results/eswa/time_out
H="--tau 0.1 --lam-proto 2 --lam-rec 0 --nbr-gamma 50 --no-std"; ACT="conda activate"
J=time_jobs.txt; rm -f $J
for g in pbmc_mimitou bmmc_site1 tcga_pancan cortex_stage2_no3c cortex_stage2 tcga_legacy9; do
  printf "Diag\t$g\t$T python -m alignimpute.graph_diag --data data/$g.npz --topology natural --no-std --out $O/diag_$g.jsonl\n" >> $J
  printf "PCAcat\t$g\t$T python -m alignimpute.linsync --data data/$g.npz --topologies natural --seeds 0 --methods PCAcat --no-std --no-whiten --out $O/pcacat_$g.jsonl\n" >> $J
  printf "L1\t$g\t$T python -m alignimpute.linsync --data data/$g.npz --topologies natural --seeds 0 --methods L1 --no-std --no-whiten --out $O/l1_$g.jsonl\n" >> $J
  printf "Kimp\t$g\t$T python -m alignimpute.r18_kimp --data data/$g.npz --configs natural --no-std --seeds 0 --out $O/kimp_$g.jsonl\n" >> $J
  printf "LRimp\t$g\t$T python -m alignimpute.r18_kimp --method lowrank --data data/$g.npz --configs natural --no-std --seeds 0 --out $O/lrimp_$g.jsonl\n" >> $J
  for v in A C ALw Aimp; do
    printf "$v\t$g\t$T python -m alignimpute.p0_handwritten --data data/$g.npz --topologies natural --seeds 0 --variants $v $H --tag eswa_time --out $O/host_${v}_$g.jsonl\n" >> $J; done
  case $g in cortex_stage2|cortex_stage2_no3c|tcga_legacy9)
    printf "Cwp\t$g\tOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m alignimpute.p0_handwritten --data data/$g.npz --topologies natural --seeds 0 --variants Cwp $H --tag eswa_time --out $O/host_Cwp_$g.jsonl\n" >> $J;; esac
  printf "StabMap\t$g\t$T python -m alignimpute.stabmap_eval prep --data data/$g.npz --far 0 1 --out stabmap_time/$g && $ACT stabmap && $T Rscript alignimpute/stabmap_run.R stabmap_time/$g all noscale && $ACT alignimpute && $T python -m alignimpute.stabmap_eval score --data data/$g.npz --far 0 1 --dir stabmap_time/$g --tag all_noscale --kmeans-seeds 1 --posts l2 --out $O/stabmap_$g.jsonl\n" >> $J
  for m in recformer:RecFormer freecsl:FreeCSL dvimc:DVIMC ghicmc:GHICMC; do
    [ "$g" = bmmc_site1 ] && [ "${m##*:}" = GHICMC ] && continue
    printf "${m##*:}\t$g\t$T python baselines/${m%%:*}_adapter.py --datasets $g --seeds 0 --out $O/ext_${m##*:}_$g.jsonl\n" >> $J; done
done
wc -l $J
python -m alignimpute.time_run --jobs $J --out results/eswa/time.jsonl > logs/time_run.log 2>&1
echo DONE > logs/time.done
