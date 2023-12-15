import torch
import torch.nn.functional as F
import os
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def traj_plot(points, loop_label=None, filename=None):
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='jet')
    if loop_label is None:
        loop_label = np.ones_like(points[:, 0])
    lc.set_array(loop_label)
    lc.set_linewidth(2)

    fig, ax = plt.subplots()
    font = {
        'weight' : 'normal',
        'size'   : 18,
    }
    plt.xlabel('x (m)', font)
    plt.ylabel('y (m)', font)

    line = ax.add_collection(lc)
    ax.scatter(points[-1, 0], points[-1, 1], s=18, c='black', label='End position')
    ax.scatter(points[0, 0], points[0, 1], s=18, c='r', label='Start position')
    # ax.legend(loc='upper left')

    # ax.set_xlim(-64, 64)
    # ax.set_ylim(-64, 64)
    plt.axis('off')
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()

def plot_score(score, out_file):
    fig, ax = plt.subplots()
    ax.matshow(score)

    mkdir(os.path.dirname(out_file))
    plt.savefig(out_file, bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()

def matching_time_indices(stamps_1: np.ndarray, stamps_2: np.ndarray,
                          max_diff: float = 0.01,
                          offset_2: float = 0.0):
    matching_indices_1 = []
    matching_indices_2 = []
    stamps_2 = copy.deepcopy(stamps_2)
    stamps_2 += offset_2
    for index_1, stamp_1 in enumerate(stamps_1):
        diffs = np.abs(stamps_2 - stamp_1)
        index_2 = int(np.argmin(diffs))
        if diffs[index_2] <= max_diff:
            matching_indices_1.append(index_1)
            matching_indices_2.append(index_2)
    return matching_indices_1, matching_indices_2

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def norm_trajectory(poses):
    maxim = torch.max(poses, 1, True)[0]
    minim = torch.min(poses, 1, True)[0]
    scale = (maxim - minim) / 2
    pmean = (maxim + minim) / 2
    poses -= pmean
    poses /= torch.max(scale, 2, True)[0]
    return poses

def resize(poses, out_size):
    return F.interpolate(poses.transpose(1,2), out_size, mode="linear", align_corners=True).transpose(1,2)

def max_distance(src, tgt, sigma=0.01):
    dist = []
    for _src in src:
        dist.append( np.exp(-np.sum((_src[None] - tgt) ** 2, 1) / (2*sigma*sigma) ) )
    dist = np.stack(dist, 0)
    return dist

def get_mask(dist, th):
    mask = []
    for i, _dist in enumerate(dist):
        _mask = (_dist > th).astype("float") # set 1 if distance < threshold
        _mask[i:] = 0 # only compute the later points
        ids = np.nonzero(_mask[1:] - _mask[:-1] == 1)[0] # distinct left contour
        if ids.size == 0: # 
            mask.append( np.zeros_like(_mask) )
        else: # if exist left contour, set the last to zeros
            _mask[ids[-1]:] = 0
            mask.append( _mask )
    mask = np.stack(mask, 0)
    return mask

def accuracy_index(x, score):
    start_ids = np.nonzero(np.logical_and(x[:-1] == False, x[1:] == True))[0] + 1
    end_ids = np.nonzero(np.logical_and(x[:-1] == True, x[1:] == False))[0] + 1
    if start_ids.size < end_ids.size:
        start_ids = np.pad(start_ids, (1, 0))
    index, value = [], []
    for sid, eid in zip(start_ids, end_ids):
        value.append( np.max(score[sid:eid]) )
        index.append( np.argmax(score[sid:eid]) + sid )
    return index, value

def nms(score, index, w=5):
    length = score.size
    padw = int((w - 1) / 2)
    score[~index] = 0
    score = np.pad(score, (padw, padw))
    for i in range(length):
        cid = i + padw
        if score[cid] != np.max(score[cid-padw: cid+padw]):
            score[cid] = 0
    return score[padw:-padw]

def score2predict(dist, th1=0.1):
    dist = dist.numpy()
    mask = get_mask(dist, th1)

    size = dist.shape[0]
    indexs = np.arange(size)
    weight = np.zeros((size, size))
    for i, _dist in enumerate(dist):
        # filter condition
        # conditions = np.logical_and(_dist > th1, mask[i] )
        conditions = np.logical_and(_dist > th1, mask[i] )
        
        prob_index = np.nonzero(conditions)[0]

        if prob_index.size > 0:
            weight[i][prob_index] = _dist[prob_index]

    weight = torch.from_numpy(weight) 
    score = weight * torch.tril(torch.ones_like(weight))
    return score

def matching(traj, sigma=0.05, th1=0.01):
    dist = max_distance(traj, traj, sigma=sigma)
    mask = get_mask(dist, th1)

    size = dist.shape[0]
    score_weight = np.zeros((size, size))
    score = np.zeros((size, size))
    isloop = np.zeros(size)
    weight = np.zeros(size)
    for i, _dist in enumerate(dist):
        # filter condition
        conditions = np.logical_and(_dist > th1, mask[i] )
        prob_index = np.nonzero(conditions)[0]

        if prob_index.size > 0:
            isloop[i] = 1
            weight[i] = np.max(_dist * mask[i])

            score_weight[i][prob_index] = _dist[prob_index]
            acc_index, acc_value = accuracy_index(conditions, _dist) # convert one-to-many to one-to-one
            score[i][acc_index] = _dist[acc_index]

    # score += score.T - np.diag(score.diagonal())
    # score_weight += score_weight.T - np.diag(score_weight.diagonal())

    return isloop, weight, score_weight, score, dist