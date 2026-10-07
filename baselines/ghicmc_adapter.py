from __future__ import annotations
import os, sys, collections
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

REPO = common.EXTERNAL / "GHICMC"
sys.path.insert(0, str(REPO))
import models.GHICMC as G
from config.config import get_config
from utils.graph_adjacency import get_adjacency
from utils.util import setup_seed

NOTE = ("GHICMC@2f5c562; config=flag 3 handwritten (shipped): epoch=200,lr=1e-3,topk=10,latent=128,hidden=1024x3,relu,"
        "batchnorm; one encoder per view for all V views; adapt: alignimpute mask, zero-filled input + per-view kNN on observed "
        "rows (as released), our preprocessing (not MinMax), per-epoch label evaluation/plot stubbed out, "
        "final = argmax of summed soft assignments at last epoch")

_LAST = {}


def _eval_stub(y_pred, y_true, accumulated_metrics):
    _LAST["pred"] = np.asarray(y_pred).copy()
    for k in ("acc", "nmi", "ARI"):
        accumulated_metrics[k].append(0.0)
    return {"accuracy": 0.0, "NMI": 0.0, "ARI": 0.0}


G.evaluation = _eval_stub
G.loss_plot = lambda *a, **k: None


class _Log:
    def info(self, *a, **k): pass


def make_config(dims, K):
    base = get_config(3)
    ae = base["Autoencoder"]; lat = ae["gcnEncoder1"][-1]
    cfg = dict(dataset="ifB", seed=base["seed"], v_num=len(dims), topk=base["topk"], n_clusters=K,
               training=dict(base["training"]), device=base["device"])
    A = {}
    for i, d in enumerate(dims):
        A[f"gcnEncoder{i + 1}"] = [d] + ae["gcnEncoder1"][1:]
        A[f"graphEncoder{i + 1}"] = list(ae["graphEncoder1"])
        A[f"activations{i + 1}"] = ae["activations1"]
    A["graphEncoderf"] = list(ae["graphEncoderf"]); A["activationsf"] = ae["activationsf"]; A["batchnorm"] = ae["batchnorm"]
    assert all(ae[f"gcnEncoder{i}"][-1] == lat for i in range(1, 5))
    cfg["Autoencoder"] = A
    return cfg


def run_one(Xz, mask_np, K, seed):
    n, V = mask_np.shape
    config = make_config([x.shape[1] for x in Xz], K)
    setup_seed(seed)
    train_miss = [torch.from_numpy(x).to(torch.float32) for x in Xz]
    mask = torch.from_numpy(mask_np.astype(np.int64))
    adj, adj_add = [], []
    for i in range(V):
        features = train_miss[i][mask[:, i].bool()]
        adj_i, _ = get_adjacency(features, features.shape[0], config["topk"])
        adj.append(adj_i)
        mask_idx = mask[:, i].view(-1, 1) * mask[:, i]
        result = torch.zeros(n, n)
        result[mask_idx.bool()] = adj_i.to_dense().view(-1)
        idx = torch.nonzero(result); vals = result[idx[:, 0], idx[:, 1]]
        adj_add.append(torch.sparse_coo_tensor(idx.t(), vals, result.size()))
        del result
    setup_seed(seed)
    model = G.GHICMC(config).to(config["device"])
    _LAST.clear()
    model.run_train(train_miss, torch.zeros(n, dtype=torch.long), adj, adj_add, mask,
                    collections.defaultdict(list), _Log())
    pred = _LAST["pred"]
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.cuda.reset_peak_memory_stats()
    return pred, None, {"peak_gpu_gb": round(peak, 2)}


if __name__ == "__main__":
    a = common.base_parser().parse_args()
    common.run_cells("GHICMC", run_one, a, NOTE)
