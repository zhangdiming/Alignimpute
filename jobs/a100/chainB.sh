#!/bin/bash
while [ ! -f logs/mosaic_pbmc_mimitou.done ]; do sleep 60; done
bash mosaicE.sh bmmc_site1 > logs/mosaicE_bmmc.out 2>&1
conda activate alignimpute; /baselines_if
ln -sf data/bmmc_site1.npz data/bmmc_site1.npz
CUDA_VISIBLE_DEVICES=0 python ghicmc_adapter.py --datasets pbmc_mimitou --out ../results/ifB_GHICMC_mosaic_a100.jsonl > ../logs/ghicmc_a100.log 2>&1
CUDA_VISIBLE_DEVICES=0 python ghicmc_adapter.py --datasets bmmc_site1 --out ../results/ifB_GHICMC_mosaic_a100.jsonl >> ../logs/ghicmc_a100.log 2>&1
echo DONE > logs/chainB.done
