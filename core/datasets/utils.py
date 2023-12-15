import os
import yaml
import pickle
import json
import numpy as np
import torch

class DataTransform:
    def __init__(self, seed=1234):
        self.translation = [-0.5, 1.0] # min range
        self.zoom = [0.5, 1.0]
        self.velocity_noise = 0.03
        self.orientation_noise = 0.01
        self.xy_noise = 0.02

        # np.random.seed(seed)

    def _rotation(self, trajectory):
        theta = np.random.rand() * 2 * np.pi
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        trajectory = np.matmul(trajectory[:,None,:], rot_matrix[None])
        return np.squeeze(trajectory, 1)

    def _translation(self, trajectory):
        trans = np.random.rand(1, 2) * self.translation[1] + self.translation[0]
        trajectory = trajectory - trans
        return trajectory

    def _norm(self, trajectory):
        maxim = np.max(trajectory, 0)
        minim = np.min(trajectory, 0)
        scale = (maxim - minim) / 2
        pmean = (maxim + minim) / 2
        trajectory -= pmean[None]
        trajectory /= np.max(scale)
        return trajectory

    def _zoom(self, trajectory):
        zoom = np.random.rand(1, 2) * self.zoom[1] + self.zoom[0]
        trajectory = trajectory * zoom
        return trajectory
    
    def _velocity_noise(self, trajectory, intensity, mode="gauss"):
        polar = xy2polar(trajectory)
        velo = polar[1:] - polar[:-1]
        size = velo.shape[0]
        
        noise = np.random.randn(size, 2).clip(-3, 3) if mode == "gauss" else np.random.rand(size, 2)
        intensity = np.abs(velo).mean(0) * np.array([1.5, 0.25])
        velo = velo + noise * intensity.reshape(1, 2)
        velo[:,1] = (velo[:,1] + np.pi) % (2*np.pi) - np.pi
        return polar[:1], velo
    
    def _polar_noise(self, trajectory, mode="gauss"):
        size = trajectory.shape[0]
        polar = xy2polar(trajectory)
        noise = np.random.randn(size, 2).clip(-3, 3) if mode == "gauss" else np.random.rand(size, 2)

        noise[:, 0] *= self.velocity_noise
        noise[:, 1] *= self.orientation_noise
        polar = polar + noise
        return polar2xy(polar)

    def _xy_noise(self, trajectory, mode="gauss"):
        size = trajectory.shape[0]
        scale = trajectory.max(0) - trajectory.min(0)
        noise = np.random.randn(size, 2).clip(-3, 3) if mode == "gauss" else np.random.rand(size, 2)
        trajectory = trajectory + noise * scale.reshape(1, 2) / 100
        return trajectory

def xy2polar(xy, split=False):
    x, y = xy[:, 0], xy[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if split:
        return r, theta
    else:
        return np.stack([r, theta], 1)

def polar2xy(ploar):
    r, theta = ploar[:, 0], ploar[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], 1)

def score2matrix(score):
    size = len(score)
    matrix = np.zeros((size, size))
    for i, item_score in enumerate(score):
        if len(item_score["index"]) > 0:
            matrix[i][item_score["index"]] = np.array(item_score["score"])
    return matrix

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def yaml_dump(data, file):
    with open(file, "w") as f:
        yaml.dump(data, f)

def json_dump(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

def json_load(file):
    with open(file, "r") as f:
        data = json.load(f, )
    return data

def pickle_dump(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)

def pickle_load(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data