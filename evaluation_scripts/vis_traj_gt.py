import collections
import os
import numpy as np
import torch

import sys
sys.path.append(os.getcwd())

from core.model import LoopModule
from evaluation_scripts.eval_utils import *
from evaluation_scripts.eval_visual import sequences, pgt_loader, vgt_loader

def load_gt_traj(dataset, seq):
    filename = os.path.join("data", dataset, seq, "trajectory.txt")
    traj = np.loadtxt(filename)
    return traj

def main(args):       
    needed_datasets = ["kitti"]

    scores, poses = {}, {}
    for dataset in needed_datasets:
        for seq in sequences[dataset]:
            scores[dataset, seq, 'pgt'] = pgt_loader(dataset, seq).max(-1)[0] > 0.5
            scores[dataset, seq, 'vgt'] = vgt_loader(dataset, seq).max(-1)[0] > 0
            poses[dataset, seq] = load_gt_traj(dataset, seq)

            pgt = scores[dataset, seq, 'pgt'].float()
            vgt = F.interpolate(scores[dataset, seq, 'vgt'][None][None].float(), 1024, mode="linear")[0][0] > 0.5
            gt = (pgt + vgt.float()).numpy()
            save_filename = os.path.join("train_log", args.name, "plots", "traj_gt", dataset)
            if not os.path.exists(save_filename):
                os.makedirs(save_filename)
            plot_traj_gt(poses[dataset, seq], gt, os.path.join(save_filename, f"{seq}.png"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    """ environment """
    parser.add_argument('--name', default='attention_12_16_hidden_cell_mix_fbatt', help='name your experiment')
    parser.add_argument('--ckpt', default=100000, type=int, help='checkpoint to restore')
    parser.add_argument('--gpu_id', type=str, default="1", help="set gpu id if gpus is 1")
    parser.add_argument('--noise_model', type=str, default="None") # gaussian, mix, None


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    args_old = load_args(os.path.join("./train_log", args.name, "config.yaml"))
    args_new = vars(args)
    for key, value in args_old.items():
        if key not in args_new.keys():
            args_new[key] = value
    args = argparse.Namespace(**args_new)

    main(args)