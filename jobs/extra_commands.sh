#!/bin/bash
conda activate alignimpute

python -m alignimpute.prepare_benchmarks --raw data/raw --out data
python -m alignimpute.drop_view --data data/cortex_stage2.npz --view 5 --out data/cortex_stage2_no3c.npz

for g in cortex_stage2 cortex_stage2_no3c tcga_legacy9 pbmc_mimitou bmmc_site1 tcga_pancan; do
  OMP_NUM_THREADS=8 python -m alignimpute.r18_kimp --data data/$g.npz --configs natural --no-std --out results/bench/kimp_real_$g.jsonl
done

python -m alignimpute.stabmap_eval prep --data data/tcga_pancan.npz --far 0 3 --out stabmap/tcga_pancan
conda activate stabmap
Rscript alignimpute/stabmap_run.R stabmap/tcga_pancan all noscale
conda activate alignimpute
python -m alignimpute.stabmap_eval score --data data/tcga_pancan.npz --far 0 3 --dir stabmap/tcga_pancan --tag all_noscale --out results/bench/stabmap_tcga_pancan.jsonl

for s in 0 1 2 3 4; do
  python -m alignimpute.p0_handwritten --data data/tcga_pancan.npz --topologies natural --seeds $s --variants Aimp --tau 0.1 --lam-proto 2 --lam-rec 0 --nbr-gamma 50 --no-std --tag ifE1_a40 --out results/ifE/real_tcga_pancan_aimp_s$s.jsonl
done

python -m alignimpute.mvp_adapter --mvp-root external/MVP --data data/handwritten.npz --topologies chain two-comp complete random-2 ring star --seeds 0 1 2 3 4 --epochs 150 --out results/p1_mvp.jsonl

for m in freecsl dvimc ghicmc recformer; do
  M=$(python -c "print({'freecsl':'FreeCSL','dvimc':'DVIMC','ghicmc':'GHICMC','recformer':'RecFormer'}['$m'])")
  python baselines/${m}_adapter.py --datasets handwritten caltech101_7 caltech101_20 cortex_stage2 cortex_stage2_no3c tcga_legacy9 --out results/ifB/ifB_$M.jsonl
  python baselines/${m}_adapter.py --datasets pbmc_mimitou bmmc_site1 tcga_pancan --out results/ifB/ifB_${M}_mosaic.jsonl
done

python -m alignimpute.coobs --data data --out results/coobs_real.json
python -m alignimpute.hub_only --data data --out results/bench/hub_only.json
python -m alignimpute.contact_map --labels raw/snm3c_inh_labels.csv --contacts raw3c --out results/st2/fig_3c.npz
