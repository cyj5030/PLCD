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

from pylab import mpl
fname = "/home/cyj/code/VO/debug/test-flow-PANet(full)-noise_Log/fzhei.ttf"
zhfont = mpl.font_manager.FontProperties(fname=fname, size=14)

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "feature_analyse", "grid_cell_analysis"))

from feature_analyse.grid_score import get_score

def logi(c1, c2):
    c1andc2 = np.logical_and(c1, c2)
    not_c1ac2 = np.logical_and(~c1, ~c2)
    c1only = np.logical_and(c1 == True, c2 == False)
    c2only = np.logical_and(c1 == False, c2 == True)
    return not_c1ac2, c1only, c2only, c1andc2

def save_id(ids, root, prefix):
    names = ["no", "onlyhd", "onlygrid", "both"]
    for i, item in enumerate(ids):
        np.savetxt(os.path.join(root, f"{prefix}_{names[i]}.txt"), np.nonzero(item)[0], fmt="%d")

def plot_grid_hd(grid_cells, hd_cells, save_folder):
    hd_score = get_score(hd_cells, 'hd_score')
    grid_score = get_score(grid_cells, 'grid_score')
    grid_like_score = get_score(grid_cells, 'grid_like_score')

    t_hd = 0.37
    t_grid = 0.37
    bin_hd = hd_score >= t_hd
    bin_grid = grid_score >= t_grid
    bin_gridlike = grid_like_score >= t_grid

    plot_info = {
        ("Unclassified", "HD", 'Gridness', "HD" + r" $\times$ " +"Gridness"): [hd_score, grid_score, logi(bin_hd, bin_grid)],
        ("Unclassified", "HD", 'Grid-like', "HD" + r" $\times$ " +"Grid-like"): [hd_score, grid_like_score, logi(bin_hd, bin_gridlike)],
    }
    for key, value in plot_info.items():
        fig, ax = plt.subplots()
        index = value[-1]
        color = ['black', 'blue', 'olivedrab', 'darkred']
        for i, iid in enumerate(index):
            ax.scatter(value[0][iid], value[1][iid], c=color[i], s=12, label=key[i])

        save_id(index, save_folder, key[2])

        x_std = np.std(value[0])
        y_std = np.std(value[1])
        xMinMax = [value[0].min() - x_std, value[0].max() + x_std]
        yMinMax = [value[1].min() - y_std, value[1].max() + y_std]
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.plot([t_hd]*2, np.linspace(*y_lim, 2), c='black', linestyle='--', linewidth=1.5)
        ax.plot(np.linspace(*x_lim, 2), [t_grid]*2, c='black', linestyle='--',linewidth=1.5)

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)

        ax.set_xlabel(f"{key[1]} Score", fontsize=18, fontproperties=zhfont)
        ax.set_ylabel(f"{key[2]} Score", fontsize=18, fontproperties=zhfont)

        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        font2 = {
            'family': 'serif',
            'weight': 'normal',
            'size': 14,
        }
        ax.legend(loc='upper left', prop=zhfont)

        save_filename = os.path.join(save_folder, f"{key[-1]}.png")
        fig.savefig(save_filename, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    hd_score_path = "./train_log/attention_12_16_hidden_cell_mix_fbatt/feature_score/hd_cell/hd_score.pkl"
    grid_score_path = "./train_log/attention_12_16_hidden_cell_mix_fbatt/feature_score/grid_cell/grid_score.pkl"
    
    grid_cells = pd.read_pickle(grid_score_path)
    hd_cells = pd.read_pickle(hd_score_path)
    
    save_root = os.path.dirname(os.path.dirname(grid_score_path))
    plot_grid_hd(grid_cells, hd_cells, save_root)