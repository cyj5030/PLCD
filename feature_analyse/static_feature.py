import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.stats as sci_stats
import cv2

import sys

sys.path.append(os.getcwd())
from core.model_utils import interp1d

from evaluation_scripts.eval_utils import *

from core.datasets.synthetic import Synthetic
from core.model import LoopModule

import pickle

def main(args):
    db = Synthetic(args.datapath, mode="train")

    # build dataloader
    train_loader = DataLoader(db, batch_size=1, num_workers=0, shuffle=False)

    model = LoopModule(args)
    model.cuda()
    model.eval()

    if args.ckpt is not None:
        ckpt_path = os.path.join("./train_log", args.name, "checkpoints", f"{args.ckpt:0>6d}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    # spatial feature
    features, position = [], []
    should_keep_testing, steps, total_steps = True, 0, 4000
    pbar = tqdm(total=total_steps)
    while should_keep_testing:
    # for i in tqdm(range(1000), desc='collecting features')):
        for i_batch, item in enumerate(train_loader):
            trajectory_input, trajectory_label, loop_label, loop_score = [x.to('cuda') for x in item]
            outputs = model.infer(trajectory_input)

            trajectory_label = interp1d(trajectory_label, args.size)
            features.append(outputs["feature"][0].cpu().numpy())
            position.append((trajectory_label[0]).cpu().numpy())
            
            steps += 1
            pbar.update(1)

            if steps == total_steps:
                should_keep_testing = False
                break

    features = np.stack(features, 0)
    position = np.stack(position, 0)
    save_path = os.path.join("./train_log", args.name, "feature_score", "features.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({"feature": features, "position": position}, f)

    # sq_length, feat_length = features.shape
    # bins = 40
    # save_path = os.path.join("./train_log", args.name, "features")
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # for f_id in tqdm(range(feat_length), desc='processing features'):
    #     ratemap = sci_stats.binned_statistic_2d(position[:, 0], position[:, 1], features[:, f_id], bins=bins, statistic='mean')[0]

    #     ratemap[np.isnan(ratemap)] = np.nanmean(ratemap)
    #     ratemap = cv2.GaussianBlur(ratemap, (3,3), sigmaX=1.0, sigmaY=0.0)

    #     save_name = 'feature_' + str(f_id).zfill(4) + '.png'
    #     np.save(os.path.join(save_path, save_name.replace(".png","")), ratemap)
    #     plt.imshow(ratemap, interpolation='none', cmap='jet') # bilinear none
    #     plt.axis('off')
    #     plt.savefig(os.path.join(save_path, save_name), bbox_inches="tight", pad_inches=0.0)
    #     # plt.show()
    #     plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    """ environment """
    parser.add_argument('--name', default='attention_12_16_hidden_cell_mix_fbatt', help='name your experiment')
    parser.add_argument('--ckpt', default=100000, type=int, help='checkpoint to restore')
    # parser.add_argument('--datapath', default='./data', help="path to dataset directory")
    parser.add_argument('--gpu_id', type=str, default="0", help="set gpu id if gpus is 1")
    parser.add_argument('--noise_model', type=str, default="None") # gaussian, mix, None
    # parser.add_argument('--size', type=int, default=1024, help="trajectory size")

    '''model setting'''
    # parser.add_argument('--sincos_scale', type=int, default=1, help="input scale for sin cos")
    # parser.add_argument('--hidden_dims', type=int, default=256, help="hidden_dims for model")
    # parser.add_argument('--half_pc_size', type=int, default=12)
    # parser.add_argument('--hd_size', type=int, default=12)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    args_old = load_args(os.path.join("./train_log", args.name, "config.yaml"))
    args_new = vars(args)
    for key, value in args_old.items():
        if key not in args_new.keys():
            args_new[key] = value
    args = argparse.Namespace(**args_new)

    main(args)