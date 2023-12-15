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

def dataloader(path, dataname, voname):
    # seqs_name = [s.split('/')[-1][:-4] for s in vo_filenames]
    if dataname == "kitti":
        seqs_name = ['00', '02', '05', '06', '07', '08', '09']
    elif dataname == "kaist":
        seqs_name = [
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
    vo_filenames = [os.path.join(path, "odometry", voname, dataname, f"{s}.txt") for s in seqs_name]
    label_filenames = [os.path.join(path, dataname, s, "trajectory.txt") for s in seqs_name]
    visual_label_names = sorted([os.path.join(path, "visual_label", dataname, s + ".npy") for s in seqs_name])
    
    for filename, vo_filename, vlabelname, seq_name in zip(label_filenames, vo_filenames, visual_label_names, seqs_name):
        gt_trajectory = torch.from_numpy(np.loadtxt(filename)).float()
        # label = torch.from_numpy(np.loadtxt(filename.replace("trajectory", "label"))).float()
        pscore = torch.from_numpy(np.load(filename.replace("trajectory.txt", "score_weight.npy"))).float()

        raw_vo = torch.from_numpy(np.loadtxt(vo_filename)).float()
        vo_trajectory = raw_vo[:, [1, 3]] if raw_vo.shape[1] == 8 else raw_vo[:, [3, 11]]
        vo_trajectory = norm_trajectory(vo_trajectory[None])
        vo_trajectory = resize(vo_trajectory, 1024)

        yield vo_trajectory, gt_trajectory, pscore, seq_name

def dbow_score(dataset, seq):
    npy_filename = os.path.join(f"/home/cyj/code/DBoW3/results/{dataset}/{seq}.npy")
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
    # score = np.load(npy_filename)
    return score

def data_predictor(network, datapath, dataname, voname, eval_out_folder, plot=True):
    size = network.size
    offset = -1
    predictions, gts = {}, {}
    for i_batch, item in tqdm(enumerate(dataloader(datapath, dataname, voname))):
        vo_trajectory, gt_trajectory, gt_pose_score = [x for x in item[:-1]]
        seq_name = item[-1]

        # inference
        outputs = network.infer(vo_trajectory.to("cuda"))
        pred_matching_score = outputs["similarity"][0].cpu()
        pred_loop_score = outputs["similarity"][0].max(-1)[0].cpu()

        gt_pose_mask = torch.tril(torch.ones((size, size)), diagonal=offset).bool()

        if plot:
            traj_out_file = os.path.join(eval_out_folder, "plot_trajectory", f"{seq_name}.png")
            mkdir(os.path.dirname(traj_out_file))
            traj_plot(vo_trajectory.cpu().numpy()[0], 
                    loop_label=pred_loop_score.cpu().numpy(), 
                    filename=traj_out_file)
        
        # collect
        predict = {
            "pose_match": [torch.masked_select(pred_matching_score, gt_pose_mask), "pose_match"],
            "pose_detect": [pred_loop_score, "pose_detect"],
        }
        gt = {
            # "visual_match": torch.masked_select(gt_visual_score, gt_visual_mask),
            "pose_match": torch.masked_select(gt_pose_score, gt_pose_mask),
            "pose_detect": gt_pose_score.max(-1)[0],
        }
        predictions[seq_name] = predict
        gts[seq_name] = gt

    return predictions, gts

def log(filename, string, isprint=True):
    with open(filename, "a") as f:
        f.write(string + "\n")
    if isprint:
        print(string)

def metrics_and_save(pred, label, threshold, eval_out_folder):
    preds, labels = {}, {}
    Precisions, Recalls, F1s, APs = {}, {}, {}, {}
    out_filename = os.path.join(eval_out_folder, f"log_{threshold:.1f}.txt")
    for seq_name in pred.keys():
        log(out_filename, "{:<20s}{:<15s}{:<10s}{:<10s}{:<10s}{:<10s}".format("seq_name", "method", "precision", "recall", "F1", "ap"))
        for method, values in pred[seq_name].items():
            P = pred[seq_name][method][0]
            label_key = pred[seq_name][method][1]
            L = label[seq_name][label_key]

            precision, recall, ths, ap = calc_pr_cuda(P, L, threshold)
            p_wrt_f1, r_wrt_f1, f1 = max_f1(precision, recall)
            log(out_filename, f"{seq_name:<20s}{method:<15s}{p_wrt_f1:<10.4f}{r_wrt_f1:<10.4f}{f1:<10.4f}{ap:<10.4f}")
            
            pr_filename = os.path.join(eval_out_folder, f"pr_{seq_name}_{method}_{threshold:.1f}.txt")
            np.savetxt(pr_filename, np.stack([precision, recall], 1))

            preds[method] = [P] if method not in preds else preds[method] + [P]
            labels[method] = [L] if method not in labels else labels[method] + [L]

            Precisions[method] = [p_wrt_f1] if method not in Precisions else Precisions[method] + [p_wrt_f1]
            Recalls[method] = [r_wrt_f1] if method not in Recalls else Recalls[method] + [r_wrt_f1]
            F1s[method] = [f1] if method not in F1s else F1s[method] + [f1]
            APs[method] = [ap] if method not in APs else APs[method] + [ap]
        log(out_filename, "")

    log(out_filename, "{:<20s}{:<15s}{:<10s}{:<10s}{:<10s}{:<10s}".format("seq_name", "method", "precision", "recall", "F1", "ap"))
    for method, values in pred[seq_name].items():
        p_wrt_f1 = np.mean(Precisions[method])
        r_wrt_f1 = np.mean(Recalls[method])
        f1 = np.mean(F1s[method])
        ap = np.mean(APs[method])

        log(out_filename, "{:<20s}".format("OIS") + f"{method:<15s}{p_wrt_f1:<10.4f}{r_wrt_f1:<10.4f}{f1:<10.4f}{ap:<10.4f}")

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
        ["kaist", "RVO"],
    ]

    for dataname, voname in datasets_info:
        eval_out_folder = os.path.join("./train_log", args.name, "VO_evals", f"{dataname}_{voname}")
        mkdir(eval_out_folder)

        predictions, gts = data_predictor(model, args.datapath, dataname, voname, eval_out_folder)

        metrics_and_save(predictions, gts, threshold, eval_out_folder)


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