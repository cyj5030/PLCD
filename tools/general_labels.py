import os
import glob
import numpy as np
import json
from tqdm import tqdm
import torch

import sys
sys.path.append(os.getcwd())
from tools.tool_utils import *

def json_dump(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

def norm(trajectory):
    trajectory = torch.from_numpy(trajectory)[None]
    return norm_trajectory(trajectory)[0].numpy()

if __name__ == "__main__":
    root = "./data"
    datasets = [
        "kitti",
        "kaist",
        "synthetic",
    ]

    sigma= 0.1
    th = 0.2

    for dataset in datasets:
        filenames = sorted(glob.glob(os.path.join(root, dataset, "*", "trajectory.txt")))
        for filename in filenames:
            traj = np.loadtxt(filename)
            traj = norm(traj)
            isloop, weight, score_weight, score, dist = matching(traj, sigma, th)
            if isloop.sum() > 0:
                print(f"{filename} exist loop closure.")
            else:
                print(f"{filename} no loop !!!")
            np.savetxt(filename.replace("trajectory", "label"), weight)
            np.save(filename.replace("trajectory.txt", "score_weight"), score_weight)