import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import scipy
import scipy.stats as sci_stats
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "feature_analyse", "grid_cell_analysis"))

from feature_analyse.grid_cell_analysis.PostSorting.open_field_head_direction import *
from evaluation_scripts.eval_visual import pgt_loader

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

def continue_gaussian_kernels(L, sigma):
    half = int(L/2)-1
    x = np.arange(-half, half+2, 1)
    y = np.exp( - (x**2) / (2*sigma*sigma))
    # y = y / y.sum()

    yy = []
    by = np.roll(y, -half)
    for i in range(L):
        yy.append(np.roll(by, i))
    return yy

def hd_score_coef(hd_hist, kernels):
    coeffs = []
    for kernel in kernels:
        pearson_coeff, p = sci_stats.pearsonr(kernel, hd_hist)
        coeffs.append( pearson_coeff )
    return coeffs

def get_hd_score_for_cluster(hd_hist):
    angles = np.linspace(-179, 180, 360)
    angles_rad = angles*np.pi/180
    dy = np.sin(angles_rad)
    dx = np.cos(angles_rad)

    totx = sum(dx * hd_hist)/sum(hd_hist)
    toty = sum(dy * hd_hist)/sum(hd_hist)
    r = np.sqrt(totx*totx + toty*toty)
    return r

def calc_hd_score(firing, angles):
    feat_length = firing.shape[-1]
    bins = 360
    angles = angles + np.pi

    hd_histogram, _ = np.histogram(angles, bins=bins)
    pre_angles = np.linspace(0, 360, bins)
    kernels = continue_gaussian_kernels(bins, 5)

    cells = []
    for f_id in tqdm(range(feat_length), desc='processing features'):
        hd_cells = {}
        
        ratemap = sci_stats.binned_statistic(angles, (firing[:, f_id]), bins=bins, statistic='mean')[0]
        # ratemap = scipy.ndimage.filters.gaussian_filter(ratemap, 3)
        ratemap = get_rolling_sum(ratemap, 21)
        ratemap = MinMaxScale(ratemap)

        max_firing_rate = np.max(ratemap)
        preferred = pre_angles[np.argmax(ratemap)]
        hd_coeffs = hd_score_coef(ratemap, kernels)
        max_hd_coef = max(hd_coeffs)

        hd_score = get_hd_score_for_cluster(ratemap)

        hd_cells["hd_histogram"] = ratemap
        # hd_cells["ref_histogram"] = kernels[np.argmax(np.array(hd_coeffs))]
        hd_cells["hd_coeffs"] = hd_coeffs
        hd_cells["max_firing_rate"] = max_firing_rate
        hd_cells["preferred"] = preferred
        hd_cells["hd_score"] = hd_score
        hd_cells["max_hd_coef"] = max_hd_coef

        cells.append(hd_cells)

    return cells

def plot_cells(cells, save_root):
    hd_score = get_score(cells, "hd_score")
    max_hd_coef = get_score(cells, "max_hd_coef")

    theta = np.linspace(0, 2*np.pi, 360)

    for id, prop in enumerate(cells):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta, prop["hd_histogram"], c='r', linewidth=2)
        # ax.plot(theta, prop["ref_histogram"], c='b', linewidth=2)
        ax.grid(True)

        save_folder = os.path.join(save_root, "hd_histogram")
        mkdir(save_folder)

        basename = f"{id:0>3d}_{hd_score[id]:.2f}_{max_hd_coef[id]:.2f}.png"
        fig.savefig(os.path.join(save_folder, basename), bbox_inches='tight', dpi=300)
        plt.close()

def MinMaxScale(x):
    maxim = np.max(x)
    minim = np.min(x)
    return (x - minim) / (maxim - minim)

def get_hdlike_score(cells):
    scores = []
    for c in cells:
        coeff = np.array(c["coefficients"])
        scores.append(np.max(coeff) - np.min(coeff))
    return np.array(scores)

def get_score(cells, name):
    score = np.array([c[name] for c in cells])
    return score

def get_shap_wrt_score(shap, score, p=2):
    score = np.around(score, p)
    u, indices, counts = np.unique(score, return_inverse=True, return_counts=True)
    yy, p75, p25 = [], [], []
    std = []
    for ids in range(u.shape[0]):
        c_ids = np.nonzero(indices == ids)[0]
        yy.append( np.mean(shap[c_ids]) )
        p75.append( np.percentile(shap[c_ids], 95) )
        p25.append( np.percentile(shap[c_ids], 5) )
        std.append(np.std(shap[c_ids]) )
    return yy, u, p75, p25, std

def plot_shap_score(score_names, shap_value, gt, save_filename):
    score, names = score_names
    mean_shaps = shap_value.mean(0)
    
    fig, ax = plt.subplots()

    color = ['blueviolet', 'saddlebrown']
    ax.scatter(score, mean_shaps, c=color[0], s=8, label='samples')
    # ax.scatter(score, shaps[0], c=color[0], s=8, label='positive labels')
    # ax.scatter(score, shaps[1], c=color[1], s=8, label='negtive labels')

    color = ['darkred', 'blue', 'black']
    # labels = []
    # for i, shap in enumerate(shaps[:2]):
    yy, u, pup, plow, pstd = get_shap_wrt_score(mean_shaps, score)
    yy = scipy.ndimage.filters.gaussian_filter(yy, 3)
    pup = scipy.ndimage.filters.gaussian_filter(pup, 3)
    plow = scipy.ndimage.filters.gaussian_filter(plow, 3)
    pstd = scipy.ndimage.filters.gaussian_filter(pstd, 3)

    ax.plot(u, yy, c=color[0], linewidth=2, label='average shap value')
    ax.fill_between(u, yy+pstd, yy-pstd, alpha=0.3)

    ax.set_xlabel(names, fontsize=18)
    ax.set_ylabel("Mean Absolute SHAP Value", fontsize=18)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)

    # ax.legend(loc='upper right')

    fig.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()

def save_piclke(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def position2matrix(position):
    x, y = position[:, :, 0], position[:, :, 1]
    theta = np.arctan2(y, x)
    return theta

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    """ environment """
    parser.add_argument('--name', default='attention_12_16_hidden_cell_mix_fbatt', help='name your experiment')
    parser.add_argument('--plot', action="store_true", default=False)
    parser.add_argument('--shap', action="store_true", default=False)
    args = parser.parse_args()

    hd_score_path = f"./train_log/{args.name}/feature_score/hd_cell/hd_score.pkl"
    if not os.path.exists(hd_score_path):
        mkdir(os.path.dirname(hd_score_path))
        features = pd.read_pickle(f"./train_log/{args.name}/feature_score/features.pkl")
        
        length, seq_length, nfeature = features["feature"].shape

        theta = position2matrix(features["position"])

        cells = calc_hd_score(features["feature"].reshape(-1, nfeature), theta.reshape(-1))
        save_piclke(hd_score_path, cells)

    cells = pd.read_pickle(hd_score_path)

    if args.plot:
        plot_cells(cells, os.path.dirname(hd_score_path))
    
    if args.shap:
        shap_path = f"./train_log/{args.name}/shap_values"
        methods=[
            ["kitti", "ORB-SLAM_mono"],
            # ["kitti", "ORB-SLAM_stereo"],
            # ["kitti", "RVO"],
            # ["kaist", "RVO"],
        ]

        x_data = {}
        x_data["hd_score"] = [get_score(cells, "hd_score"), "HD Score"]

        all_shap, all_gt = [], []
        for dataset, method in methods:
            dataset_shap, dataset_gt = [], []
            for seq in sequences[dataset]:
                shap_path = f"./train_log/{args.name}/shap_values/{dataset}/{method}/{seq}.pkl"
                shap_value = pd.read_pickle(shap_path)
                shap_value = np.concatenate(shap_value, 0)
                shap_value = np.abs(shap_value).sum(1)

                pgt = (pgt_loader(dataset, seq).max(-1)[0]).numpy() > 0.2
                dataset_shap.append(np.copy(shap_value))
                dataset_gt.append(np.copy(pgt))

                save_folder = os.path.join(os.path.dirname(hd_score_path), dataset, method)
                mkdir(save_folder)
                for key in x_data.keys():
                    save_filename = os.path.join(save_folder, f"{seq}_{key}.png")
                    plot_shap_score(x_data[key], shap_value, pgt, save_filename)
            
            all_shap.append( np.concatenate(dataset_shap, 0) )
            all_gt.append( np.concatenate(dataset_gt, 0) )
            for key in x_data.keys():
                save_filename = os.path.join(save_folder, f"{key}.png")
                plot_shap_score(x_data[key], all_shap[-1], all_gt[-1], save_filename)

        for key in x_data.keys():
            save_filename = os.path.join(os.path.dirname(hd_score_path), f"{key}.png")
            yy = np.concatenate(all_shap)
            gt = np.concatenate(all_gt)
            plot_shap_score(x_data[key], yy, gt, save_filename)