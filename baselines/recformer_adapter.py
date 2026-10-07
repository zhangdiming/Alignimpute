from __future__ import annotations
import argparse, os, random, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

torch.zeros(1).cuda()
REPO = common.EXTERNAL / "RecFormer"
sys.path.insert(0, str(REPO))
import scipy.io as scio
import mydataset, myloss, train as rf
from model import get_model
from constructGraph import getMvKNNGraph

CFG = dict(epochs=101, rec_epochs=50, dim=512, n_layers=1, heads=4, dropout=0.0, batch_size=128, lr=1e-3, beta=1.0, gamma=15)
NOTE = ("RecFormer@4a7f6ee; config=train.py defaults for shipped handwritten-5view: " + ",".join(f"{k}={v}" for k, v in CFG.items()) +
        "; adapt: alignimpute mask written as folds .mat, data .mat holds zero-filled views (released also loads complete "
        "views), our preprocessing (loader StandardScaler -> identity), labels replaced by row ids, evaluate stubbed, "
        "seeded, final = kmeans_np on H at epoch 100 (last)")
MATDIR = Path(__file__).resolve().parent / "recformer_mats"


class _Identity:
    def fit_transform(self, x): return x


_LAST = {}


def _evaluate_stub(H, all_label, estimator, classes_num, epoch, logger):
    _LAST["H"] = np.asarray(H).copy(); _LAST["epoch"] = epoch
    return {'ACC': 0., 'AMI': 0., 'NMI': 0., 'ARI': 0., 'PUR': 0.}


rf.evaluate = _evaluate_stub


class _Log:
    def info(self, *a, **k): pass


def run_one(Xz, mask, K, seed):
    n, V = mask.shape
    MATDIR.mkdir(exist_ok=True)
    tag = f"ifB_{os.getpid()}_{seed}"
    dpath, fpath = MATDIR / f"{tag}.mat", MATDIR / f"{tag}_percentDel.mat"
    Xc = np.empty((1, V), dtype=object)
    for v in range(V): Xc[0, v] = Xz[v].astype(np.float32)
    scio.savemat(dpath, {"X": Xc, "Y": np.arange(n, dtype=np.float64).reshape(-1, 1)})
    folds = np.empty((1, 1), dtype=object); folds[0, 0] = mask.astype(np.uint8)
    scio.savemat(fpath, {"folds": folds})

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    orig = mydataset.StandardScaler; mydataset.StandardScaler = _Identity
    try:
        loader, ds = mydataset.getIncDataloader(str(dpath), str(fpath), fold_idx=0, is_train=True,
                                                batch_size=CFG["batch_size"], shuffle=False, num_workers=4)
    finally:
        mydataset.StandardScaler = orig
    order = ds.cur_labels.reshape(-1).astype(int)
    assert sorted(order.tolist()) == list(range(n))
    assert np.array_equal(ds.cur_inc_V_ind.astype(bool), mask[order])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rf.args = argparse.Namespace(batch_size=CFG["batch_size"], beta=CFG["beta"])
    model = get_model(ds.d_list, d_model=CFG["dim"], n_layers=CFG["n_layers"], heads=CFG["heads"],
                      classes_num=K, dropout=CFG["dropout"])
    loss_model = myloss.MyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    all_newX = [torch.tensor(v, dtype=torch.float) for v in ds.inc_mv_data]
    all_encX = torch.ones((len(ds), ds.view_num, CFG["dim"])).to('cuda:0')
    all_graph = None
    _LAST.clear(); log = _Log()
    for epoch in range(CFG["epochs"]):
        if epoch < CFG["rec_epochs"]:
            _, model, all_encX, all_newX = rf.train_1(loader, ds, model, all_graph, all_encX, loss_model, opt, None, None, epoch, log)
            all_graph = torch.tensor(getMvKNNGraph(all_newX, k=CFG["gamma"]), device=torch.device('cuda:0'), dtype=torch.float32)
        else:
            _, model, _, all_encX = rf.train_2(loader, ds, model, all_graph, all_encX, all_newX, loss_model, opt, None, None, epoch, log, 0)
    assert _LAST["epoch"] == CFG["epochs"] - 1
    H = np.empty_like(_LAST["H"]); H[order] = _LAST["H"]
    for p in (dpath, fpath): p.unlink(missing_ok=True)
    if not np.isfinite(H).all():
        raise FloatingPointError("non-finite H")
    return None, H, {}


if __name__ == "__main__":
    a = common.base_parser().parse_args()
    common.run_cells("RecFormer", run_one, a, NOTE)
