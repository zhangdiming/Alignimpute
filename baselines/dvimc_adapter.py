from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

REPO = common.EXTERNAL / "DVIMC-pytorch"
sys.path.insert(0, str(REPO))
from base_model import DVIMC
from datasets import SingleviewDataset, Incomplete_MultiviewDataset
from base_fn import vade_trick
import main as dvimc_main
from sklearn.cluster import KMeans

CFG = dict(epochs=300, initial_epochs=200, batch_size=256, learning_rate=0.0005, prior_learning_rate=0.05, z_dim=10,
           lr_decay_step=10, lr_decay_factor=0.9, interval=100, alpha=5, likelihood="Gaussian")
NOTE = ("DVIMC@bc1b05f; config=Caltech7-5V (only shipped config/default): " + ",".join(f"{k}={v}" for k, v in CFG.items()) +
        "; adapt: alignimpute mask, zero-filled input (as released), our preprocessing (not pixel_normalize), GMM prior init = "
        "seeded KMeans(n_init=10) on mask-averaged latent mean of observed views of all instances (released: complete "
        "instances only, none exist), run seed replaces released seed 5, final = released argmax q(c|PoE mean) at epoch 300")


def initialization_patched(model, sv_loaders, Xz, mask, args, seed):
    criterion = nn.MSELoss()
    for v in range(args.num_views):
        optimizer = optim.Adam([{"params": model.encoders[f'view_{v}'].parameters(), 'lr': 0.001},
                                {"params": model.decoders[f'view_{v}'].parameters(), 'lr': 0.001}])
        for e in range(1, args.initial_epochs + 1):
            for xv in sv_loaders[v]:
                optimizer.zero_grad()
                xv = xv.reshape(xv.shape[0], -1).to(args.device)
                _, xvr = model.sv_encode(xv, v)
                criterion(xvr, xv).backward()
                optimizer.step()
    with torch.no_grad():
        data = [torch.tensor(x, dtype=torch.float32, device=args.device) for x in Xz]
        lat = model.mv_encode(data)
        m = torch.tensor(mask, dtype=torch.float32, device=args.device)
        fused = sum(lat[v] * m[:, v:v + 1] for v in range(args.num_views)) / m.sum(1, keepdim=True)
        km = KMeans(n_clusters=args.class_num, n_init=10, random_state=seed).fit(fused.cpu().numpy())
        model.prior_mu.data = torch.tensor(km.cluster_centers_, dtype=torch.float32).to(args.device)


def run_one(Xz, mask, K, seed):
    n, V = mask.shape
    args = argparse.Namespace(**CFG, device=torch.device("cuda"), multiview_dims=[x.shape[1] for x in Xz],
                              num_views=V, class_num=K, data_size=n, seed=seed)
    np.random.seed(seed)
    mk = mask.astype(np.float64)
    imv = Incomplete_MultiviewDataset(Xz, mk, np.zeros(n, dtype=int), V)
    svs = [SingleviewDataset(Xz[v][mask[:, v]]) for v in range(V)]
    dvimc_main.setup_seed(seed)
    imv_loader = DataLoader(imv, batch_size=args.batch_size, shuffle=True)
    sv_loaders = [DataLoader(s, batch_size=args.batch_size, shuffle=True) for s in svs]
    model = DVIMC(args).to(args.device)
    optimizer = optim.Adam([{"params": model.encoders.parameters(), 'lr': args.learning_rate},
                            {"params": model.decoders.parameters(), 'lr': args.learning_rate},
                            {"params": model.prior_weight, 'lr': args.prior_learning_rate},
                            {"params": model.prior_mu, 'lr': args.prior_learning_rate},
                            {"params": model.prior_var, 'lr': args.prior_learning_rate}])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor)
    initialization_patched(model, sv_loaders, Xz, mask, args, seed)
    dvimc_main.train(model, optimizer, scheduler, imv_loader, args)
    with torch.no_grad():
        data = [torch.tensor(x, dtype=torch.float32, device=args.device) for x in imv.data_list]
        msk = [torch.tensor(m, dtype=torch.float32, device=args.device) for m in imv.mask_list]
        _, _, _, agg_mu, _, _, _ = model(data, msk)
        c = vade_trick(agg_mu.cpu(), model.prior_weight.data.cpu(), model.prior_mu.data.cpu(), model.prior_var.data.cpu())
        pred = torch.argmax(c, dim=1).numpy()
    return pred, None, {}


if __name__ == "__main__":
    a = common.base_parser().parse_args()
    common.run_cells("DVIMC", run_one, a, NOTE)
