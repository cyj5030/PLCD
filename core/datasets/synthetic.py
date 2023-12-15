import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import glob

import sys
sys.path.append(os.getcwd())
from core.datasets.utils import *


class Synthetic(torch.utils.data.Dataset):
    def __init__(self, path, mode="train"):
        self.path = path
        self.Aug = DataTransform(seed=1234)
        self.filenames = np.loadtxt(os.path.join(path, mode + ".txt"), dtype="str").tolist()
        
        trajectories, labels, scores = [], [], []
        for filename in sorted(self.filenames):
            trajectories.append(np.loadtxt(filename))
            labels.append(np.loadtxt(filename.replace("trajectory", "label")))
            scores.append(np.load(filename.replace("trajectory.txt", "score_weight.npy")))

        self.dataset = {
            'trajectory': trajectories,
            'label': labels,
            'score': scores,
        }
        self.length = len(trajectories)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        trajectory_label = self.dataset["trajectory"][idx]
        loop_label = self.dataset["label"][idx]
        loop_score = self.dataset['score'][idx]
        
        # aug
        # if np.random.rand() > 0.5:
        #     trajectory_label = self.Aug._xy_noise(trajectory_label)
        trajectory_label = self.Aug._rotation(trajectory_label)
        trajectory_label = self.Aug._zoom(trajectory_label)
        trajectory_label = self.Aug._norm(trajectory_label)

        # to tensor
        trajectory_input = torch.from_numpy(trajectory_label).float()
        trajectory_label = torch.from_numpy(trajectory_label).float()
        loop_label = torch.from_numpy(loop_label).float()
        loop_score = torch.from_numpy(loop_score).float()
        return trajectory_input, trajectory_label, loop_label, loop_score

if __name__ == '__main__':
    kitti = Synthetic('./data')
    trajectory_input, trajectory_label, loop_label, loop_score = kitti.__getitem__(5)

    import matplotlib.pylab as plt
    from tools.tool_utils import *
    out_dir = "./train_log/collection/ttt"
    mkdir(out_dir)
    # traj_plot(trajectory_input, loop_label, os.path.join(out_dir, "input.png"))
    traj_plot(trajectory_label.numpy(), loop_label, os.path.join(out_dir, "label.png"))

    # for i in range(100):
    #     trajectory_input, trajectory_label, loop_label, loop_score = kitti.__getitem__(i)
    #     fig, ax = plt.subplots()

    #     ax.spy(loop_score.numpy())
    #     plt.savefig(os.path.join(out_dir, f"{i:0>4d}_score.png"), bbox_inches="tight", pad_inches=0.0)
    #     plt.show()
    #     plt.close()