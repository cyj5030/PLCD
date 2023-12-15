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

from feature_analyse.grid_cell_analysis.PostSorting.open_field_grid_cells import *
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

def rotation_score(autocorr_map, field_properties, field_distances):
    correlation_coefficients = []
    for angle in range(30, 180, 30):
        autocorr_map_to_rotate = np.nan_to_num(autocorr_map)
        rotated_map = rotate(autocorr_map_to_rotate, angle, reshape=False)  # todo fix this
        rotated_map_binary = np.round(rotated_map)
        autocorr_map_ring = remove_inside_and_outside_of_grid_ring(autocorr_map, field_properties, field_distances)
        rotated_map_ring = remove_inside_and_outside_of_grid_ring(rotated_map_binary, field_properties, field_distances)
        autocorr_map_ring_to_correlate, rotated_map_ring_to_correlate = remove_nans(autocorr_map_ring, rotated_map_ring)
        pearson_coeff = np.corrcoef(autocorr_map_ring_to_correlate, rotated_map_ring_to_correlate)[0][1]
        correlation_coefficients.append(pearson_coeff)
    return correlation_coefficients

def firing_rate_map(firing, position):
    feat_length = firing.shape[-1]
    bins = 41

    cells = []
    for f_id in tqdm(range(feat_length), desc='processing features'):
        grid_cells = {}

        ratemap = sci_stats.binned_statistic_2d(position[:, 0], position[:, 1], firing[:, f_id], bins=bins, statistic='mean')[0]

        ratemap[np.isnan(ratemap)] = np.nanmean(ratemap)
        ratemap = cv2.GaussianBlur(ratemap, (3,3), sigmaX=1.0, sigmaY=0.0)
        ratemap = np.abs(ratemap)
        ratemap = ratemap / np.max(ratemap)
        grid_cells["ratemap"] = ratemap

        ratemap_autocorr = get_rate_map_autocorrelogram(ratemap)
        grid_cells["ratemap_autocorr"] = ratemap_autocorr
        copy_rc = np.copy(ratemap_autocorr)

        field_properties = find_autocorrelogram_peaks(copy_rc)
        grid_cells["peaks"] = len(field_properties)

        if len(field_properties) > 7:
            bin_size = 1  # cm
            field_distances_from_mid_point = find_field_distances_from_mid_point(copy_rc, field_properties)
            ring_distances = get_ring_distances(field_distances_from_mid_point)
            grid_spacing = calculate_grid_spacing(ring_distances, bin_size)
            field_size = calculate_field_size(field_properties, field_distances_from_mid_point, bin_size)
            # grid_score = calculate_grid_score(copy_rc, field_properties, field_distances_from_mid_point)
            coefficients = rotation_score(copy_rc, field_properties, field_distances_from_mid_point)
        else:
            grid_spacing = np.nan
            field_size = np.nan
            coefficients = np.nan

        grid_cells["spacing"] = grid_spacing
        grid_cells["field_sizes"] = field_size
        grid_cells["coefficients"] = coefficients
        grid_cells["grid_score"] = min(coefficients[i] for i in [1, 3]) - max(coefficients[i] for i in [0, 2, 4])
        cells.append(grid_cells)

    return cells

def plot_cells(cells, save_root):
    for_draw = ["ratemap_autocorr", "ratemap"]
    for id, prop in enumerate(cells): 
        prop["ratemap_autocorr"][:,-1] = np.nan
        prop["ratemap_autocorr"][-1,:] = np.nan

    grid_score = get_score(cells, "grid_score")
    grid_like_score = get_score(cells, "grid_like_score")

    for id, prop in enumerate(cells):
        for draw_id in for_draw:
            ax = sns.heatmap(prop[draw_id], cmap='jet', cbar=False, xticklabels=False, yticklabels=False)
            save_folder = os.path.join(save_root, draw_id)
            mkdir(save_folder)

            basename = f"{id:0>3d}_{grid_score[id]:.2f}_{grid_like_score[id]:.2f}.png"
            ax.get_figure().savefig(os.path.join(save_folder, basename), bbox_inches='tight', dpi=300)
            plt.close()

def MinMaxScale(x):
    maxim = np.max(x)
    minim = np.min(x)
    return (x - minim) / (maxim - minim)

def get_gridlike_score(cells):
    scores = []
    for c in cells:
        coeff = np.array(c["coefficients"])
        scores.append(np.max(coeff) - np.min(coeff))
    return np.array(scores)

def get_score(cells, name):
    if name == "grid_like_score":
        score = get_gridlike_score(cells)
    else:
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    """ environment """
    parser.add_argument('--name', default='attention_12_16_hidden_cell_mix_fbatt', help='name your experiment')
    parser.add_argument('--plot', action="store_true", default=False)
    parser.add_argument('--shap', action="store_true", default=False)
    args = parser.parse_args()

    grid_score_path = f"./train_log/{args.name}/feature_score/grid_cell/grid_score.pkl"
    
    if not os.path.exists(grid_score_path):
        os.makedirs(os.path.dirname(grid_score_path), exist_ok=True)
        features = pd.read_pickle(f"./train_log/{args.name}/feature_score/features.pkl")
        
        length, seq_length, nfeature = features["feature"].shape

        cells = firing_rate_map(features["feature"].reshape(-1, nfeature), features["position"].reshape(-1, 2))
        save_piclke(grid_score_path, cells)

    cells = pd.read_pickle(grid_score_path)

    if args.plot:
        plot_cells(cells, os.path.dirname(grid_score_path))

    if args.shap:
        shap_path = f"./train_log/{args.name}/shap_values"
        methods=[
            ["kitti", "ORB-SLAM_mono"],
            # ["kitti", "ORB-SLAM_stereo"],
            # ["kitti", "RVO"],
            # ["kaist", "RVO"],
        ]

        x_data = {}
        x_data["grid_score"] = [np.round(get_score(cells, "grid_score"), 2), "Gridness Score"]
        x_data["grid_like_score"] = [np.round(get_score(cells, "grid_like_score"), 2), "Grid-like Score"]
        x_data["peaks"] = [np.round(get_score(cells, "peaks"), 0), "Peak number"]
        x_data["field_sizes"] = [np.round(get_score(cells, "field_sizes"), 1), "Field Size"]
        x_data["spacing"] = [np.round(get_score(cells, "spacing"), 1), "Spacing"]

        all_shap, all_gt = [], []
        for dataset, method in methods:
            dataset_shap, dataset_gt = [], []
            for seq in sequences[dataset]:
                shap_path = f"./train_log/{args.name}/shap_values/{dataset}/{method}/{seq}.pkl"
                shap_value = pd.read_pickle(shap_path)
                shap_value = np.concatenate(shap_value, 0)
                shap_value = np.abs(shap_value).sum(1)
                # shap_value = shap_value.sum(1)

                pgt = (pgt_loader(dataset, seq).max(-1)[0]).numpy() > 0.2
                dataset_shap.append(np.copy(shap_value))
                dataset_gt.append(np.copy(pgt))

                save_folder = os.path.join(os.path.dirname(grid_score_path), dataset, method)
                # mkdir(save_folder)
                # for key in x_data.keys():
                #     save_filename = os.path.join(save_folder, f"{seq}_{key}.png")
                #     plot_shap_score(x_data[key], shap_value, pgt, save_filename)
            
            all_shap.append( np.concatenate(dataset_shap, 0) )
            all_gt.append( np.concatenate(dataset_gt, 0) )
            # for key in x_data.keys():
            #     save_filename = os.path.join(save_folder, f"{key}.png")
            #     plot_shap_score(x_data[key], all_shap[-1], all_gt[-1], save_filename)

        for key in x_data.keys():
            save_filename = os.path.join(os.path.dirname(grid_score_path), f"{key}.png")
            yy = np.concatenate(all_shap)
            gt = np.concatenate(all_gt)
            plot_shap_score(x_data[key], yy, gt, save_filename)