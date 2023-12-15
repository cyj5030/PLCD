from glob import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import shutil

import sys
sys.path.append(os.getcwd())

# dataset
from core.model_utils import interp1d, interp2d
from core.model import LoopModule
from evaluation_scripts.eval_utils import *
from tools.tool_utils import traj_plot
from tools.tool_utils import norm_trajectory, resize
from evaluation_scripts.eval_visual import botw_loader, bow_loader, vgt_loader, plcd_loader, sequences

def main(args):
    needed_datasets = [
        "kitti", 
        # "kaist",
    ]

    visual_data = {}
    for dataset in needed_datasets:
        for seq in sequences[dataset]:
            bow = bow_loader(dataset, seq)
            # botw = botw_loader(dataset, seq)
            vgt = vgt_loader(dataset, seq)
            traj = torch.from_numpy(np.loadtxt(os.path.join("data", dataset, seq, "trajectory.txt"))).float()

            bow = interp2d(bow[None], traj.shape[0])[0]
            vgt = interp2d(vgt[None], traj.shape[0])[0]

            vgt_mask = torch.tril(torch.ones(vgt.shape[-2:]), diagonal=-50).bool()
            precision, recall, ths, ap = calc_pr_cuda((bow*vgt_mask).max(-1)[0], vgt.max(-1)[0], 0.4)
            p_wrt_f1, r_wrt_f1, f1, f_id = max_f1(precision, recall, True)
            threshold = ths[f_id][0]
            pred = (bow*vgt_mask).max(-1)[0] > threshold

            save_floder = os.path.join("train_log", args.name, "plots", "traj_bow", dataset)
            if not os.path.exists(save_floder):
                os.makedirs(save_floder)

            save_filename = os.path.join(save_floder, f"bow_{seq}_{f1:.4f}_{threshold:.2f}.png")
            fig, ax = plt.subplots()
            length = traj.shape[0]
            for i in range(length):
                if int(pred[i]) == 0:
                    ax.scatter(traj[i][0], traj[i][1], s=20, c="k")
            for i in range(length):
                if int(pred[i]) == 1:
                    ax.scatter(traj[i][0], traj[i][1], s=20, c="orangered")
            plt.axis('off')
            fig.savefig(save_filename, bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    """ environment """
    parser.add_argument('--name', default='attention_12_16_hidden_cell_mix_fbatt', help='name your experiment')
    parser.add_argument('--ckpt', default=100000, type=int, help='checkpoint to restore')
    parser.add_argument('--gpu_id', type=str, default="3", help="set gpu id if gpus is 1")
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