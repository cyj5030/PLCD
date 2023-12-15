import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np
import shap
from tqdm import tqdm
import pickle

import sys
import os
sys.path.append(os.getcwd())

from core.model import LoopModule
from core.datasets.synthetic import Synthetic
from evaluation_scripts.eval_utils import *

def vo_feature_loader(network, dataset, seq, voname):
    from tools.tool_utils import norm_trajectory, resize
    path = "./data"
    vo_filename = os.path.join(path, "odometry", voname, dataset, f"{seq}.txt")
    
    raw_vo = torch.from_numpy(np.loadtxt(vo_filename)).float()
    vo_trajectory = raw_vo[:, [1, 3]] if raw_vo.shape[1] == 8 else raw_vo[:, [3, 11]]
    vo_trajectory = norm_trajectory(vo_trajectory[None])
    vo_trajectory = resize(vo_trajectory, 1024)
    outputs = network.infer(vo_trajectory.to("cuda"))
    feature = outputs["feature"]#.cpu()
    return feature

class ExplaineNet(nn.Module):
    def __init__(self, net):
        super(ExplaineNet, self).__init__()
        self.exNet = net

    def forward(self, x):
        sim = self.exNet(x)
        loop = sim.max(-1)[0]
        return loop

def main(args):
    # db = Synthetic(args.datapath, mode="train")

    # # build dataloader
    # train_loader = DataLoader(db, batch_size=1, num_workers=0, shuffle=False)

    model = LoopModule(args)
    model.cuda()
    model.eval()

    if args.ckpt is not None:
        ckpt_path = os.path.join("./train_log", args.name, "checkpoints", f"{args.ckpt:0>6d}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    # collect background
    with open(os.path.join("./train_log", args.name, "feature_score", "features.pkl"), "rb") as f:
        background = torch.from_numpy(pickle.load(f)["feature"][:100])
    
    # explainer
    net = ExplaineNet(model.match_net)
    net.cpu()
    e = shap.DeepExplainer(net, background)
    
    save_file = os.path.join("./train_log", args.name, "shap_values")
    if not os.path.exists(save_file):
        os.makedirs(save_file)

    with open(os.path.join(save_file, f"expected_value.pkl"), "wb") as f:
        pickle.dump(e.expected_value, f)

    sequences = {}
    sequences["kitti"] = ['00', '02', '05', '06', '07', '08', '09']
    # sequences["kaist"] = [
    #     'urban26-dongtan', 
    #     'urban27-dongtan', 
    #     'urban28-pankyo', 
    #     'urban30-gangnam', 
    #     'urban32-yeouido', 
    #     'urban33-yeouido', 
    #     'urban34-yeouido', 
    #     'urban38-pankyo', 
    #     'urban39-pankyo'
    # ]
    datasets_info = [
        ["kitti", "ORB-SLAM_mono"],
        # ["kitti", "ORB-SLAM_stereo"],
        # ["kitti", "RVO"],
        # ["kaist", "RVO"],
    ]

    model.cuda()
    test_features = {}
    for dataset, vo_name in datasets_info:
        for seq in sequences[dataset]:
            test_feature = vo_feature_loader(model, dataset, seq, vo_name)
            test_features[dataset, seq, vo_name] = test_feature

    model.cpu()
    for dataset, vo_name in datasets_info:
        save_folder = os.path.join(save_file, dataset, vo_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for seq in sequences[dataset]:
            shap_values = e.shap_values(test_feature)
            with open(os.path.join(save_folder, f"{seq}.pkl"), "wb") as f:
                pickle.dump(shap_values, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    """ environment """
    parser.add_argument('--name', default='attention_12_16_hidden_cell_mix_fbatt', help='name your experiment')
    parser.add_argument('--ckpt', default=100000, type=int, help='checkpoint to restore')
    parser.add_argument('--gpu_id', type=str, default="3", help="set gpu id if gpus is 1")
    parser.add_argument('--noise_model', type=str, default="None") # gaussian, mix, None
    # parser.add_argument('--size', type=int, default=1024, help="trajectory size")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    args_old = load_args(os.path.join("./train_log", args.name, "config.yaml"))
    args_new = vars(args)
    for key, value in args_old.items():
        if key not in args_new.keys():
            args_new[key] = value
    args = argparse.Namespace(**args_new)

    main(args)