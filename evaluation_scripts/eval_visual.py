from glob import glob
import os
import numpy as np
import torch
from tqdm import tqdm


import sys
sys.path.append(os.getcwd())

# dataset
from core.model_utils import interp1d, interp2d
from core.model import LoopModule
from evaluation_scripts.eval_utils import *
from tools.tool_utils import traj_plot
from tools.tool_utils import norm_trajectory, resize

datasets = ['kitti', 'kaist']
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

bow_path="./data/BoW"
botw_path = "./data/BoTW"

def bow_loader(dataset, seq):
    npy_filename = os.path.join(bow_path, dataset, f"{seq}.npy")
    if os.path.exists(npy_filename):
        score = np.load(npy_filename)
    else:
        txt_raw = load_txt(npy_filename.replace(".npy", ".txt"))
        txt_raw = np.array([line.strip().split(' ') for line in txt_raw])
        r_id = txt_raw[:,0].astype("int")
        c_id = txt_raw[:,1].astype("int")
        value = txt_raw[:,2].astype("float")

        size = r_id[-1] + 1
        score = np.zeros((size, size))
        score[c_id, r_id] = value
        np.save(npy_filename, score)
    return torch.from_numpy(score).float()

def botw_loader(dataset, seq):
    from scipy.io import loadmat
    mat_filename = os.path.join(botw_path, dataset, f"loopClosureMatrix_{seq}.mat")
    score = loadmat(mat_filename)["loopClosureMatrix"]
    score = np.nan_to_num(score).astype("float64")
    # score = score / score.max()
    return torch.from_numpy(score).double()

def vgt_loader(dataset, seq):
    path = "./data"
    filename = os.path.join(path, "visual_label", dataset, seq + ".npy")
    gt_score = np.load(filename)
    return torch.from_numpy(gt_score).float()

def pgt_loader(dataset, seq):
    path = "./data"
    filename = os.path.join(path, dataset, seq, "score_weight.npy")
    gt_score = np.load(filename)
    return torch.from_numpy(gt_score).float()

def plcd_loader(network, dataset, seq, voname):
    path = "./data"
    vo_filename = os.path.join(path, "odometry", voname, dataset, f"{seq}.txt")
    
    raw_vo = torch.from_numpy(np.loadtxt(vo_filename)).float()
    vo_trajectory = raw_vo[:, [1, 3]] if raw_vo.shape[1] == 8 else raw_vo[:, [3, 11]]
    vo_trajectory = norm_trajectory(vo_trajectory[None])
    vo_trajectory = resize(vo_trajectory, 1024)

    outputs = network.infer(vo_trajectory.to("cuda"))
    p_score = outputs["similarity"][0].cpu()
    return p_score

def process_data(out_data, dataset, methods):
    offset = -50
    predictions, gts = {}, {}
    for seq in sequences[dataset]:
        hw = out_data[dataset, seq, "vgt"].shape[-2:]
        mask = torch.tril(torch.ones(hw), diagonal=offset).bool()

        for method in methods:
            out_data[dataset, seq, method] = interp2d(out_data[dataset, seq, method][None], hw[-1])[0]

        predict = {
            "botw": [torch.masked_select(out_data[dataset, seq, "botw"], mask), "vgt"],
            "bow": [torch.masked_select(out_data[dataset, seq, "bow"], mask), "vgt"],
            "RVO_v": [torch.masked_select(out_data[dataset, seq, "RVO"], mask), "vgt"],
            "RVO_p": [torch.masked_select(out_data[dataset, seq, "RVO"], mask), "pgt"],
            "pgt": [torch.masked_select(out_data[dataset, seq, "pgt"], mask), "vgt"],

            "d_botw": [(out_data[dataset, seq, "botw"] * mask).max(-1)[0], "d_vgt"],
            "d_bow": [(out_data[dataset, seq, "bow"] * mask).max(-1)[0], "d_vgt"],
            "d_RVO_v": [(out_data[dataset, seq, "RVO"] * mask).max(-1)[0], "d_vgt"],
            "d_RVO_p": [(out_data[dataset, seq, "RVO"] * mask).max(-1)[0], "d_pgt"],
            "d_pgt": [(out_data[dataset, seq, "pgt"] * mask).max(-1)[0], "d_vgt"],
        }

        gt = {
            "vgt": torch.masked_select(out_data[dataset, seq, "vgt"], mask),
            "pgt": torch.masked_select(out_data[dataset, seq, "pgt"], mask),

            "d_vgt": out_data[dataset, seq, "vgt"].max(-1)[0],
            "d_pgt": out_data[dataset, seq, "pgt"].max(-1)[0],
        }
        predictions[seq] = predict
        gts[seq] = gt

    return predictions, gts

def log(filename, string, isprint=True):
    with open(filename, "a") as f:
        f.write(string + "\n")
    if isprint:
        print(string)

def metrics_and_save(pred, label, threshold, eval_out_folder):
    preds, labels = {}, {}
    Precisions, Recalls, F1s, Rs = {}, {}, {}, {}
    out_filename = os.path.join(eval_out_folder, f"log_{threshold:.1f}.txt")
    for seq_name in pred.keys():
        log(out_filename, "{:<20s}{:<15s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}".format(
            "seq_name", "method", "precision", "recall", "F1", "threshold", "r", "r_threshold"))
        for method, values in pred[seq_name].items():
            P = pred[seq_name][method][0]
            label_key = pred[seq_name][method][1]
            L = label[seq_name][label_key]

            precision, recall, ths, ap = calc_pr_cuda(P, L, threshold)
            p_wrt_f1, r_wrt_f1, f1, f_id = max_f1(precision, recall, True)
            r_wrt_p, r_id = max_recall_wrt_precision(precision, recall, L, threshold)
            log(out_filename, f"{seq_name:<20s}{method:<15s}{p_wrt_f1:<10.4f}{r_wrt_f1:<10.4f}{f1:<10.4f}{ths[f_id][0]:<10.4f}" + \
                                f"{r_wrt_p:<10.4f}{ths[r_id][0]:<10.4f}")
            
            pr_filename = os.path.join(eval_out_folder, f"pr_{seq_name}_{method}_{threshold:.1f}.txt")
            np.savetxt(pr_filename, np.stack([precision, recall], 1))

            preds[method] = [P] if method not in preds else preds[method] + [P]
            labels[method] = [L] if method not in labels else labels[method] + [L]

            Precisions[method] = [p_wrt_f1] if method not in Precisions else Precisions[method] + [p_wrt_f1]
            Recalls[method] = [r_wrt_f1] if method not in Recalls else Recalls[method] + [r_wrt_f1]
            F1s[method] = [f1] if method not in F1s else F1s[method] + [f1]
            Rs[method] = [r_wrt_p] if method not in Rs else Rs[method] + [r_wrt_p]
        log(out_filename, "")

    log(out_filename, "{:<20s}{:<15s}{:<10s}{:<10s}{:<10s}{:<10s}".format(
        "seq_name", "method", "precision", "recall", "F1", "r"))
    for method, values in pred[seq_name].items():
        p_wrt_f1 = np.mean(Precisions[method])
        r_wrt_f1 = np.mean(Recalls[method])
        f1 = np.mean(F1s[method])
        r = np.mean(Rs[method])

        log(out_filename, "{:<20s}".format("OIS") + f"{method:<15s}{p_wrt_f1:<10.4f}{r_wrt_f1:<10.4f}{f1:<10.4f}{r:<10.4f}")

    return out_filename

def main(args):
    torch.cuda.empty_cache()

    model = LoopModule(args)
    model.cuda()
    model.eval()
    if args.ckpt is not None:
        ckpt_path = os.path.join("./train_log", args.name, "checkpoints", f"{args.ckpt:0>6d}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

    threshold = args.threshold

    out_data = {}
    needed_datasets = ["kitti"]

    for dataset in needed_datasets:
        for seq in sequences[dataset]:
            out_data[dataset, seq, 'bow'] = bow_loader(dataset, seq)
            out_data[dataset, seq, 'botw'] = botw_loader(dataset, seq)
            out_data[dataset, seq, 'vgt'] = vgt_loader(dataset, seq)
            out_data[dataset, seq, 'pgt'] = pgt_loader(dataset, seq)
            for vo in ["RVO"]:
                out_data[dataset, seq, vo] = plcd_loader(model, dataset, seq, vo)
    methods = ['bow', 'botw', 'vgt', 'pgt', 'RVO']

    for dataset in needed_datasets:
        eval_out_folder = os.path.join("./train_log", args.name, "VLCD_evals", f"{dataset}")
        mkdir(eval_out_folder)

        predictions, gts = process_data(out_data, dataset, methods)
        out_filename = metrics_and_save(predictions, gts, threshold, eval_out_folder)

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