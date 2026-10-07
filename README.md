# align-or-impute

Code and per-seed results for the paper

> **When to Align and When to Impute: Multi-View Clustering of Assembled Data with Views Never Observed Together**

Multi-view data assembled from several studies record different views on different cohorts, so some pairs of views are never observed on the same instance. This repository contains the methods, baselines and analysis scripts used in the paper:

- training-free linear graph synchronisation;
- a contrastive host and its variants;
- kNN imputation through shared views;
- adapters that run published incomplete multi-view clustering methods without the features of unobserved views;
- the result files from which every table and figure of the paper is generated.

## Repository layout

| Path | Content |
|---|---|
| `alignimpute/` | Python package: methods, data builders, adapters, analysis |
| `baselines/` | adapters for FreeCSL, DVIMC, GHICMC and RecFormer (run against their released code) |
| `results/` | per-seed result records (JSON lines) used by the tables and figures |
| `jobs/` | the command lists and launch scripts that produced `results/` |
| `requirements.txt` | Python dependencies |

## Installation

```bash
conda create -n alignimpute python=3.11
conda activate alignimpute
pip install -r requirements.txt
```

StabMap runs in a separate R environment, `stabmap`, with:

- R 4.5;
- Bioconductor `StabMap` 1.4.0;
- `jsonlite`.

All Python commands below are run from the repository root.

## Reproducing the tables and figures from the released results

```bash
python -m alignimpute.make_nc_tables --build results --sections out/sections
python -m alignimpute.make_nc_figures --build results --out out/figures
python -m alignimpute.fig_teaser --build results/st2 --out out/figures/teaser.pdf
```

`make_nc_tables` writes the LaTeX tables and `stats.tex`. `stats.tex` defines the statistics quoted in the text:

- the seed bootstrap of the largest shortfall;
- source-grouped bootstrap intervals of the Spearman correlations;
- anchor-weight sensitivities.

These outputs are identical to the ones in the manuscript.

## Data

### Benchmarks

```bash
python -m alignimpute.prepare_benchmarks --raw data/raw --out data
```

This command downloads and converts seven benchmarks:

- HandWritten;
- Caltech101-7 and Caltech101-20;
- OutdoorScene;
- NUS-WIDE;
- ALOI;
- ProteinFold.

The sources are the public `.mat` collections `ChuanbinZhang/Multi-view-datasets` and `sudalvxin/2019-PR-Sparse-Multi-view-clustering`, and the UCI `mfeat` files as distributed with MVP.

### Real assemblies

All raw data are public. The builders reduce every view to principal components fitted on the instances that observe it, and store the natural observation mask.

| Setting | Builder | Sources |
|---|---|---|
| Cortex 6 | `cortex_build2` (uses `cortex_build`) | snmCAT-seq (GSE140493), human Patch-seq GABAergic neurons (Allen Institute), snm3C-seq (GSE130711) |
| Cortex 5 | `drop_view` (Cortex 6 without the 3C view) | as above |
| TCGA 9 | `tcga_legacy_build` | UCSC Xena TCGA legacy hub, nine platforms |
| PanCan | `tcga_build` | UCSC Xena PanCanAtlas hub |
| PBMC | `mosaic_build mimitou` | Mimitou et al. 2021 (GSE156478), as preprocessed by scMoMaT |
| BMMC | `mosaic_build bmmc` | NeurIPS 2021 BMMC (GSE194122), site 1 |

```bash
python -m alignimpute.cortex_build2 --raw raw --raw3c raw3c --meth3c data --out data/cortex_stage2.npz
python -m alignimpute.drop_view --data data/cortex_stage2.npz --view 5 --out data/cortex_stage2_no3c.npz
python -m alignimpute.tcga_legacy_build --root data/tcga_legacy --out data/tcga_legacy9.npz
python -m alignimpute.tcga_build --root data/tcga --out data/tcga_pancan.npz --pca 256
python -m alignimpute.mosaic_build mimitou --src data/mimitou --out data/pbmc_mimitou.npz
python -m alignimpute.mosaic_build bmmc --cite raw/bmmc/cite.h5ad --multiome raw/bmmc/multiome.h5ad --out data/bmmc_site1.npz
```

The files under `results/st2/` used by Figure 1 are released with the results:

- `hw_data.npz`;
- `hw_chain_s0_emb.npz`;
- `cortex_stage2_data.npz`;
- `fig_3c.npz`.

`fig_3c.npz` is recomputed by `alignimpute.contact_map` from the snm3C-seq contact files.

## Methods and where they are implemented

| Name in the paper | Code |
|---|---|
| Linear graph synchronisation | `linsync`, method `L1` (all reported runs use `--no-whiten`; principal-component views use `--no-std`) |
| Unaligned consensus | `linsync`, method `PCAcat` |
| Contrastive host | `p0_handwritten`, variant `A` |
| Host with a linear-sync first assignment | `p0_handwritten`, variant `ALw` |
| Host with in-loop synchronisation | `p0_handwritten`, variant `C` (`Cthin` on a thin edge) |
| Joint Wasserstein–Procrustes, in loop | `p0_handwritten`, variant `Cwp` |
| Joint Wasserstein–Procrustes, post hoc | `alaux_baseline` (record variant `alaux_A2`) |
| Host on imputed views | `p0_handwritten`, variant `Aimp` |
| kNN imputation + k-means | `r18_kimp` (imputation in `bridge`) |
| StabMap | `stabmap_eval` (`prep` / `score`) with `stabmap_run.R` |
| MVP, CPSPAN, CPM-Nets, DIMVC, TreeEIC | `mvp_adapter`, `cpspan_adapter`, `cpmnets_baseline`, `dimvc_adapter`, `treeeic_adapter` |
| FreeCSL, DVIMC, GHICMC, RecFormer | `baselines/*_adapter.py` |
| Graph descriptors and imputability | `graph_diag` |
| Hub-only clustering | `hub_only` |
| Anchor-weight sensitivity | `aw_sens` |
| Post-hoc synchronisation under cohort shift | `st4_posthoc_shift` |
| Preprocessing statistics on observed rows only (sensitivity) | `--obs-stats` of `p0_handwritten` and `r18_kimp`; summary by `obs_sens_analyze` |

Shared components:

- edge estimates and the thin-edge primitive: `sync`;
- cohort masks and cohort shift: `topology`;
- metrics and seeded k-means: `metrics`.

The host configuration used throughout is `--tau 0.1 --lam-proto 2 --lam-rec 0 --nbr-gamma 50`, with 300 full-batch epochs and `d = 32`. The temperature is selected by the label-free rule `--holdout-frac 0.1` (see `jobs/a100/ifE.sh` and `jobs/a100/mosaicE.sh`).

Every method receives zero or no features for unobserved views. The hosts, kNN imputation and the published methods compute their preprocessing statistics over all instances before the mask is applied; `--obs-stats` computes them on the observed rows of each view only (sensitivity analysis in `results/obs_sens/`).

## Re-running the experiments

`jobs/` contains the command lists and launch scripts that produced `results/`:

- `jobs/a100/` and `jobs/a40/`: lists and scripts, one file per experiment family. In each list, every line is one run with its full arguments; the launch scripts generate the lists and run them in parallel.
- `jobs/extra_commands.sh`: the runs that were launched individually.

The result records of the trained hosts also carry their full configuration in the field `cfg`.

The host on imputed views builds its neighbour graph on the imputed views of all instances. Runs that trained it after other variants in the same process had reused the neighbour graph of those variants (observed rows of the original views); its results on Cortex 6, Cortex 5, TCGA 9, PBMC and BMMC, and in the sensitivity analysis, were therefore rerun alone with the fixed code (`results/c014/`, `results/obs_sens/aimp_*`, `jobs/a40/c014.sh`). The tables and figures use these reruns.

Seeds:

- ten seeds for linear graph synchronisation, kNN imputation, StabMap and the hosts;
- five seeds for the published deep methods, for the trained hosts on PanCan and for the topology grid.

On benchmarks, a seed fixes the mask, the network initialisation and the k-means initialisation.

### External methods

External methods are run against their released code, which is not included here.

1. Clone each repository into `external/`:

   | Method | Repository | Revision |
   |---|---|---|
   | FreeCSL | `zoyadai/2025_CVPR_FreeCSL` | `191fbc6` |
   | DVIMC | `tkkxgh/DVIMC-pytorch` | `bc1b05f` |
   | GHICMC | `KelvinXuu/GHICMC` | `2f5c562` |
   | RecFormer | `justsmart/RecFormer` | `4a7f6ee` |
   | TreeEIC | `SubmissionsIn/TreeEIC` | `10e838b` |
   | MVP | `gaoxin492/MVP` | — |
   | CPSPAN | `jinjiaqi1998/CPSPAN` | — |
   | DIMVC | `SubmissionsIn/DIMVC` | — |

2. Set the paths the adapters read:
   - `ALIGNIMPUTE_EXTERNAL` points to `external/`;
   - `ALIGNIMPUTE_DATA` points to the directory of the `.npz` files.

   The defaults are `external/` and `data/`.

3. Install each method's own dependencies in a separate environment (for example, TensorFlow for DIMVC).

CPM-Nets is reimplemented in PyTorch in `cpmnets_baseline`.

## Result files

| Directory | Content |
|---|---|
| `results/ifE/` | real settings, topology grid and cohort shift for the hosts |
| `results/mosaic/` | PBMC and BMMC |
| `results/st1/`, `results/st2/` | ablations, cohort shift, joint WP and post-hoc runs |
| `results/opt3/`, `results/linshift/` | linear graph synchronisation (real, benchmarks, shift) |
| `results/optB/` | host variants on the benchmark chains |
| `results/bench/` | kNN imputation and StabMap on the real settings, StabMap on the benchmarks, hub-only clustering |
| `results/diag/` | label-free descriptors |
| `results/ifB/` | FreeCSL, DVIMC, GHICMC, RecFormer |
| `results/aw/` | anchor-weight sensitivity |
| `results/c014/` | host on imputed views on the real settings, rerun alone with its own neighbour graph |
| `results/obs_sens/` | preprocessing statistics over all instances against observed rows only (paired runs, benchmarks) |
| `results/*.jsonl` (top level) | MVP, CPSPAN, CPM-Nets, DIMVC, TreeEIC, kNN imputation on the benchmarks |
| `results/p1_mvp_leaky.jsonl`, `results/p4_cpspan_leaky_*.jsonl` | MVP and CPSPAN with their released loaders, which feed unobserved views to the encoders (Appendix D) |
| `results/ifB/*_sanity.jsonl` | FreeCSL, DVIMC, GHICMC and RecFormer on complete HandWritten, seed 0 (Appendix C) |

Each line of a `.jsonl` file is one run. Common fields:

- `dataset`/`data`;
- `topology`;
- `seed`;
- `variant`/`method`;
- `nmi`, `acc`, `ari`.

## License

MIT, see `LICENSE`. The released result files derived from public datasets follow the terms of those datasets.
