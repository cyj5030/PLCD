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

def merge_algorithm(v_match, p_match, tmin, tmax):
    size = v_match.shape[-1]
    p_match = interp2d(p_match[None], size)[0]

    merge = v_match.clone()
    # t_mask = torch.masked_select(p_match, mask)
    # tmin = torch.quantile(t_mask, qmin)
    # tmax = torch.quantile(t_mask, qmax)

    merge[p_match <= tmin] = 0
    merge[p_match > tmax] = merge[p_match > tmax] * (1 + p_match[p_match > tmax])
    # merge[p_match > tmax] = merge[p_match > tmax] + p_match[p_match > tmax]
    # n = torch.sum(t_mask < tmin)
    mask_n = p_match <= tmin
    return merge, mask_n

def merge_predictor(visual_data, pose_data, qmin, qmax, dataset, vo_name, visual_methods):
    offset = -50
    predictions, gts = {}, {}
    for seq in sequences[dataset]:
        pscore = pose_data[dataset, seq, vo_name]
        gt_visual_mask = torch.tril(torch.ones(visual_data[dataset, seq, "vgt"].shape[-2:]), diagonal=offset).bool()
        gt_visual_mask_2 = torch.tril(torch.ones(visual_data[dataset, seq, "vgt"].shape[-2:]), diagonal=-1).bool()
        hw = visual_data[dataset, seq, "vgt"].shape[-2:]
        predict = {}
        for v_method in visual_methods:
            v_score = interp2d(visual_data[dataset, seq, v_method][None], hw[-1])[0]
            
            merge_score, discard_mask = merge_algorithm(v_score, pscore, qmin, qmax)
            n_total = int(torch.sum(gt_visual_mask_2))
            n_discard = int(torch.sum(discard_mask * gt_visual_mask_2))
            ratio = n_discard / n_total
            predict.update(
                {f"{vo_name}_{v_method}": [torch.masked_select(merge_score, gt_visual_mask), "vgt", (n_discard, n_total, ratio, qmin, qmax)]})

        gt = {
                "vgt": torch.masked_select(visual_data[dataset, seq, "vgt"], gt_visual_mask),
            }
        predictions[seq] = predict
        gts[seq] = gt
    return predictions, gts

def log(filename, string, isprint=True):
    with open(filename, "a") as f:
        f.write(string + "\n")
    if isprint:
        print(string)

def metrics_and_save(pred, label, threshold, eval_out_folder, qmin, qmax):
    preds, labels = {}, {}
    Precisions, Recalls, F1s, APs = {}, {}, {}, {}
    Ratio = {}
    out_filename = os.path.join(eval_out_folder, f"log_{threshold:.1f}_Qmin{qmin:.2f}_Qmax{qmax:.2f}.txt")
    log(out_filename, "{:<20s}{:<20s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}".format(
            "seq_name", "method", "precision", "recall", "F1", "discard", "total", "ratio", "t"))
    for seq_name in pred.keys():
        for method, values in pred[seq_name].items():
            P = pred[seq_name][method][0]
            label_key = pred[seq_name][method][1]
            L = label[seq_name][label_key]

            discard, total_num, ratio, tmin, tmax = pred[seq_name][method][2]
            Ratio[method] = [ratio] if method not in Ratio else Ratio[method] + [ratio]

            precision, recall, ths, ap = calc_pr_cuda(P, L, threshold)
            p_wrt_f1, r_wrt_f1, f1 = max_f1(precision, recall)
            log(out_filename, f"{seq_name:<20s}{method:<20s}{p_wrt_f1:<10.3f}{r_wrt_f1:<10.3f}{f1:<10.3f}" + \
                            f"{discard:<10d}{total_num:<10d}{tmin:<10.3f}{tmax:<10.3f}")
            
            pr_filename = os.path.join(eval_out_folder, f"pr_{seq_name}_{method}_{threshold:.1f}_Qmin{qmin}_Qmax{qmax}.txt")
            np.savetxt(pr_filename, np.stack([precision, recall], 1))

            preds[method] = [P] if method not in preds else preds[method] + [P]
            labels[method] = [L] if method not in labels else labels[method] + [L]

            Precisions[method] = [p_wrt_f1] if method not in Precisions else Precisions[method] + [p_wrt_f1]
            Recalls[method] = [r_wrt_f1] if method not in Recalls else Recalls[method] + [r_wrt_f1]
            F1s[method] = [f1] if method not in F1s else F1s[method] + [f1]
            APs[method] = [ap] if method not in APs else APs[method] + [ap]

    log(out_filename, "{:<20s}{:<20s}{:<10s}{:<10s}{:<10s}{:<10s}".format(
                        "seq_name", "method", "precision", "recall", "F1", "ratio"))
    for method, values in pred[seq_name].items():
        p_wrt_f1 = np.mean(Precisions[method])
        r_wrt_f1 = np.mean(Recalls[method])
        f1_ois = np.mean(F1s[method])
        ap = np.mean(APs[method])
        ratio = np.mean(Ratio[method])

        log(out_filename, "{:<20s}".format("OIS") + f"{method:<20s}{p_wrt_f1:<10.3f}{r_wrt_f1:<10.3f}{f1_ois:<10.3f}{ratio:<10.3f}")

    return f1_ois, out_filename

def main(args):
    torch.cuda.empty_cache()
    model = LoopModule(args)
    model.cuda()
    model.eval()

    if args.ckpt is not None:
        ckpt_path = os.path.join("./train_log", args.name, "checkpoints", f"{args.ckpt:0>6d}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    threshold = args.threshold
    datasets_info=[
        ["kitti", "ORB-SLAM_mono"],
        ["kitti", "ORB-SLAM_stereo"],
        ["kitti", "RVO"],
    ]
    needed_datasets = [
        "kitti", 
    ]

    visual_data = {}
    for dataset in needed_datasets:
        for seq in sequences[dataset]:
            visual_data[dataset, seq, 'bow'] = bow_loader(dataset, seq)
            # visual_data[dataset, seq, 'botw'] = botw_loader(dataset, seq)
            visual_data[dataset, seq, 'vgt'] = vgt_loader(dataset, seq)

    pose_data = {}
    for dataset, vo_name in datasets_info:
        for seq in sequences[dataset]:
            pose_data[dataset, seq, vo_name] = plcd_loader(model, dataset, seq, vo_name)
    

    Q = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    visual_methods = ['bow']

    for dataname, voname in datasets_info:
        eval_out_folder = os.path.join("./train_log", args.name, "Merge_evals_BoW", f"{dataname}_{voname}")
        mkdir(eval_out_folder)

        f1s = []
        for q in Q:
            merge_data, gts = merge_predictor(visual_data, pose_data, q, q, dataname, voname, visual_methods)
            f1_ois, out_filename = metrics_and_save(merge_data, gts, threshold, eval_out_folder, q, q)
            f1s.append(f1_ois)
        max_q = Q[np.argmax(np.array(f1s))]

        merge_data, gts = merge_predictor(visual_data, pose_data, max_q, max_q, dataname, voname, visual_methods)
        f1_ois, out_filename = metrics_and_save(merge_data, gts, threshold, eval_out_folder, max_q, max_q)
        shutil.copy(out_filename, os.path.join(os.path.dirname(out_filename), f"log_{threshold:.1f}.txt"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    """ environment """
    parser.add_argument('--name', default='attention_12_16_hidden_cell_mix_fbatt', help='name your experiment')
    parser.add_argument('--ckpt', default=100000, type=int, help='checkpoint to restore')
    # parser.add_argument('--dataname', default=["kitti", "kaist"], nargs='+', type=str) # ["kitti", "kaist"]
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