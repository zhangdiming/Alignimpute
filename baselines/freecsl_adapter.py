from __future__ import annotations
import argparse, os, random, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import normalize
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

REPO = common.EXTERNAL / "2025_CVPR_FreeCSL"
sys.path.insert(0, str(REPO))
from dataloader import TrainDataset_All
from network import FreeCSL
from sklearn.cluster import KMeans

DEFAULTS = dict(Pre_epochs=50, epochs=100, batch_pre=512, batch=512, lr_pre=0.0003, lr_train=0.0005, recon_fea_dim=64,
                gamma=1.0, alpha=1.0, tau=0.2, epsilon=0.05, sinkhorn_iterations=3, z_dim=64, K_neighber=3,
                collapse_regularization=0.2, lamda=0.1, graph_out_dim=64, graph_h_dim=128)
NOTE = ("FreeCSL@191fbc6; config=main.py argparse defaults (no HandWritten/our-dataset entry in config.yaml): " +
        ",".join(f"{k}={v}" for k, v in DEFAULTS.items()) +
        "; adapt: alignimpute mask, unobserved views zero-filled at input (released keeps full X), our preprocessing "
        "(not MinMax), skip view pairs with 0 co-observed rows in batch (Constr), skip views with <K_neighber+1 "
        "observed rows in batch (Graph), in-loop KMeans seeded, final = kmeans_np on mask-averaged Z at last epoch")


class FreeCSLGuarded(FreeCSL):
    def train_Constr(self, xs, miss_vecs):
        _, z_spec_all = self.AE.forward_singleh(xs)
        V = self.args.view_num
        loss = 0.0
        for m in range(V):
            for n in range(m + 1, V):
                mask_mn = (miss_vecs[m] > 0) & (miss_vecs[n] > 0)
                if int(mask_mn.sum()) == 0:
                    continue
                Z_m = z_spec_all[m][mask_mn]; Z_n = z_spec_all[n][mask_mn]
                loss = loss + self.Constr_loss(Z_m, Z_n, self.clu_H_layer, self.args.tau, Z_m.shape[0])
        if not torch.is_tensor(loss):
            loss = torch.zeros((), device=xs[0].device)
        return z_spec_all, loss

    def train_Graph(self, xs, miss_vecs):
        h_spec = self.get_Single_reconHs(xs)
        z_spec = self.get_Single_constrZs(xs)
        assign_result, MGC_loss = [], 0.0
        for v in range(self.args.view_num):
            valid_indices = torch.nonzero(miss_vecs[v].squeeze() != 0, as_tuple=False).squeeze(dim=1)
            if valid_indices.numel() < self.args.K_neighber + 1:
                assign_result.append(torch.zeros(len(xs[v]), self.args.cluster_num, device=xs[v].device)); continue
            z_valid = z_spec[v][valid_indices]; x_valid = xs[v][valid_indices]; h_valid = h_spec[v][valid_indices]
            A, A_sparse = self.KNN_Graph.construct_knn_graph_spec(x_valid)
            x_embedding = self.GCN_list[v].forward_spec(h_valid, A_sparse)
            q_assignment = self.compute_q(z_valid)
            assign_x, loss_spec = self.MGC.compute_loss(x_embedding, A)
            eps = 1e-10
            assign_x = assign_x + eps; q_assignment = q_assignment + eps
            kl_loss = F.kl_div(assign_x.log(), q_assignment, reduction='batchmean')
            assignments = torch.zeros(len(xs[v]), self.args.cluster_num, dtype=torch.float32).cuda()
            assignments[valid_indices] = assign_x
            assign_result.append(assignments)
            MGC_loss = MGC_loss + loss_spec + self.args.collapse_regularization * kl_loss
        if not torch.is_tensor(MGC_loss):
            MGC_loss = torch.zeros((), device=xs[0].device)
        return assign_result, MGC_loss


def seed_everything(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; torch.backends.cudnn.enabled = False


def run_one(Xz, mask, K, seed):
    seed_everything(seed)
    device = torch.device("cuda:0")
    n, V = mask.shape
    args = argparse.Namespace(**DEFAULTS, device=device, cluster_num=K, data_num=n, view_num=V,
                              view_dims=[x.shape[1] for x in Xz])
    X = [torch.from_numpy(x) for x in Xz]
    Y = [np.zeros(n, dtype=int) for _ in range(V)]
    Miss_vecs = [torch.tensor(mask[:, v].astype(np.int32)) for v in range(V)]
    idxs = np.arange(n)
    estimator = KMeans(n_clusters=K, random_state=seed)
    model = FreeCSLGuarded(args).to(device)
    opt_pre = torch.optim.Adam(model.parameters(), lr=args.lr_pre)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_train)
    All_loader = torch.utils.data.DataLoader(TrainDataset_All(X, Y, Miss_vecs, idxs), batch_size=args.batch, shuffle=False, drop_last=False)

    def get_single(loader, which):
        out = [[] for _ in range(V)]
        with torch.no_grad():
            for xs, _, _, _ in loader:
                xs = [t.to(device) for t in xs]
                hs = model.get_Single_reconHs(xs) if which == "H" else model.get_Single_constrZs(xs)
                for v in range(V): out[v] += hs[v].cpu().tolist()
        return out

    def fea_com(fea):
        with torch.no_grad():
            fea = [torch.tensor(f).to(device) for f in fea]
            mv = [m.to(device) for m in Miss_vecs]
            spec = [torch.mul(fea[v].t(), mv[v].t()).float() for v in range(V)]
            return torch.mul(sum(spec), (1 / sum(mv))).t().float().cpu().numpy()

    pre_loader = torch.utils.data.DataLoader(TrainDataset_All(X, Y, Miss_vecs, idxs), batch_size=args.batch_pre, shuffle=True, drop_last=False)
    for _ in range(args.Pre_epochs):
        for xs, _, mvs, _ in pre_loader:
            xs = [t.to(device) for t in xs]; mvs = [t.to(device) for t in mvs]
            loss = model.train_Recon(xs, mvs)
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()
    tr_loader = torch.utils.data.DataLoader(TrainDataset_All(X, Y, Miss_vecs, idxs), batch_size=args.batch, shuffle=True, drop_last=False)
    for epoch in range(args.epochs):
        cH = fea_com(get_single(tr_loader, "H")); cZ = fea_com(get_single(tr_loader, "Z"))
        estimator.fit(cH); estimator.fit(cZ)
        cen_H = estimator.cluster_centers_; cen_Z = estimator.cluster_centers_
        with torch.no_grad():
            model.clu_H_layer.data = normalize(torch.tensor(cen_H).float().cuda(), dim=1, p=2)
            model.clu_Z_layer.data = normalize(torch.tensor(cen_Z).float().cuda(), dim=1, p=2)
        for xs, _, mvs, _ in tr_loader:
            xs = [t.to(device) for t in xs]; mvs = [t.to(device) for t in mvs]
            l_rec = model.train_Recon(xs, mvs)
            _, l_con = model.train_Constr(xs, mvs)
            _, l_mgc = model.train_Graph(xs, mvs)
            tot = l_rec + l_con + l_mgc
            opt.zero_grad(); tot.backward(); opt.step()
        if not np.isfinite(float(tot.item())):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
    Zc = fea_com(get_single(All_loader, "Z"))
    if not np.isfinite(Zc).all():
        raise FloatingPointError("non-finite embedding")
    return None, Zc, {"final_loss": float(tot.item())}


if __name__ == "__main__":
    a = common.base_parser().parse_args()
    common.run_cells("FreeCSL", run_one, a, NOTE)
