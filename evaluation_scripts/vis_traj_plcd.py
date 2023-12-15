import collections
import os
import numpy as np
import torch

import sys
sys.path.append(os.getcwd())

from core.model import LoopModule
from evaluation_scripts.eval_utils import *
from tools.tool_utils import norm_trajectory, resize

sequences = {}
sequences["kitti"] = ['00', '02', '05', '06', '07', '08', '09']
sequences["kaist"] = [
    'urban26-dongtan', 
    'urban27-dongtan', 
    'urban28-pankyo', 
    'urban30-gangnam', 
    'urban32-yeouido', 
    'urban33-yeouido', 
    'urban34-yeouido', 
    'urban38-pankyo', 
    'urban39-pankyo'
]

def dataloader(path, dataname, voname):
    seqs_name = sequences[dataname]
    vo_filenames = [os.path.join(path, "odometry", voname, dataname, f"{s}.txt") for s in seqs_name]
    label_filenames = [os.path.join(path, dataname, s, "trajectory.txt") for s in seqs_name]
    visual_label_names = sorted([os.path.join(path, "visual_label", dataname, s + ".npy") for s in seqs_name])
    
    outputs = {}
    for filename, vo_filename, vlabelname, seq_name in zip(label_filenames, vo_filenames, visual_label_names, seqs_name):
        gt_trajectory = torch.from_numpy(np.loadtxt(filename)).float()
        # label = torch.from_numpy(np.loadtxt(filename.replace("trajectory", "label"))).float()
        pscore = torch.from_numpy(np.load(filename.replace("trajectory.txt", "score_weight.npy"))).float()

        raw_vo = torch.from_numpy(np.loadtxt(vo_filename)).float()
        vo_trajectory = raw_vo[:, [1, 3]] if raw_vo.shape[1] == 8 else raw_vo[:, [3, 11]]
        vo_trajectory = norm_trajectory(vo_trajectory[None])
        vo_trajectory = resize(vo_trajectory, 1024)

        vscore = torch.from_numpy(np.load(vlabelname)).float()
        vscore_id = torch.from_numpy(np.loadtxt(vlabelname.replace(".npy", ".txt"))).long()

        outputs[dataname, seq_name, voname] = [vo_trajectory, gt_trajectory, pscore]
    return outputs

def prediction(data, network):
    size = network.size
    pred_outputs = {}
    for key, value in data.items():
        vo_trajectory, gt_trajectory, gt_pose_score = [x.cuda() for x in value]
        outputs = network.infer(vo_trajectory)

        pred = outputs["similarity"][0].max(-1)[0].cpu()
        gt = gt_pose_score.max(-1)[0]

        precision, recall, ths, ap = calc_pr_cuda(pred, gt, 0.4)
        p_wrt_f1, r_wrt_f1, f1, f_id = max_f1(precision, recall, True)
        threshold = ths[f_id][0]

        pred_outputs[key] = [gt_trajectory.cpu().numpy(), pred.numpy() >= threshold, f1, threshold]
    return pred_outputs

def main(args):
    torch.cuda.empty_cache()
    model = LoopModule(args)
    model.cuda()
    model.eval()

    if args.ckpt is not None:
        ckpt_path = os.path.join("./train_log", args.name, "checkpoints", f"{args.ckpt:0>6d}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    datasets_info=[
        ["kitti", "ORB-SLAM_mono"],
        ["kitti", "ORB-SLAM_stereo"],
        ["kitti", "RVO"],
        # ["kaist", "RVO"],
    ]

    for dataset, vo_name in datasets_info:
        input_data = dataloader("./data", dataset, vo_name)
        pred_data = prediction(input_data, model)

        save_filename = os.path.join("train_log", args.name, "plots", "traj_plcd", dataset)
        if not os.path.exists(save_filename):
            os.makedirs(save_filename)
        for key, value in pred_data.items():
            seq = key[1]
            th = value[3]
            f1 = value[2]
            plot_traj_gt(value[0], value[1]*3, os.path.join(save_filename, f"{vo_name}_{seq}_{f1:.4f}_{th:.2f}.png"))

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