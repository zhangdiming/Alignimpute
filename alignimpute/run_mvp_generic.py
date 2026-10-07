import sys, os, json
sys.path.insert(0, os.getcwd())
import main as mvp_main

CFG = json.loads(os.environ["MVP_CFG"])
_orig = mvp_main.get_trainer_from_args


def get_trainer_from_args(args):
    log_path, epochs = args.log_path, args.epochs
    for k in ("num_views", "in_channels", "arch", "num_classes", "latent_dim", "likelihood", "normalizing_factor"):
        setattr(args, k, CFG[k])
    import training.trainer as T
    real_ctor = T.ICMVtrainer
    def fake_ctor(a):
        a.data_path = os.path.join(os.getcwd(), "data", CFG["data_dir"])
        a.log_path = log_path
        a.seed = [42]
        a.epochs = epochs
        a.interval = epochs
        return real_ctor(a)
    mvp_main.ICMVtrainer = fake_ctor
    return _orig(args)


mvp_main.get_trainer_from_args = get_trainer_from_args

import dataprovider.MvDataset as MVD
_orig_getitem = MVD.MvDataset.__getitem__
def _masked_getitem(self, index):
    datas_dict, labels, masks, permutations = _orig_getitem(self, index)
    for m in range(self.num_views):
        datas_dict["m%d" % m] = datas_dict["m%d" % m] * float(masks[m])
    return datas_dict, labels, masks, permutations
MVD.MvDataset.__getitem__ = _masked_getitem

if __name__ == "__main__":
    mvp_main.run_training_entry()
