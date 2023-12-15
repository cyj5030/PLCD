import torch
import torch.nn.functional as F
import os

import numpy as np

class MixedNoise:
    def __init__(self, p, gauss_params, uniform_params, sigma, length=1023, isSmooth=True):
        self.p = p
        self.mu = gauss_params[0]
        self.sigma = gauss_params[1]
        
        self.minim = uniform_params[0]
        self.maxim = uniform_params[1]

        self.smooth_sigma = sigma
        self.kernel = self.gaussian_kernel_1d(sigma)
        self.length = length
        self.isSmooth = isSmooth
        
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # random.seed(seed)
    
    def gaussian_kernel_1d(self, sigma):
        W = int(sigma * 3)
        x = torch.arange(-W, W+1, 1)
        kernel = torch.exp( - x.pow(2) / (2*sigma*sigma))
        kernel /= kernel.sum()
        return kernel
    
    def gaussian_filter1d(self, x, device):
        pad = int(self.smooth_sigma * 3)
        x = F.conv1d(x.reshape(1, 1, -1), self.kernel.reshape(1, 1, -1).to(device), stride=1, padding=pad, bias=None)
        x = x.view(-1)
        return x

    def sample(self, n_samples, device):
        nn_samples = n_samples * self.length

        # multinomial
        self.p = torch.rand(2, device=device) * (1-self.p)
        self.p[0] = 1 - self.p[1]
        p_choice = torch.multinomial(self.p, num_samples=nn_samples, replacement=True).to(device)
        mask0 = p_choice == 0
        mask1 = p_choice == 1
        
        # gauss
        if isinstance(self.mu, list):
            gauss_mu = torch.rand(1, device=device) * (self.mu[1] - self.mu[0]) + self.mu[0]
            gauss_mu = torch.randn(mask0.sum(), device=device) * self.sigma + gauss_mu
        else:
            gauss_mu = torch.randn(mask0.sum(), device=device) * self.sigma + self.mu
        
        # uniform
        uniform_mu = torch.rand(mask1.sum(), device=device) * (self.maxim - self.minim) + self.minim
        # uniform_p = torch.rand(2, device=device)
        # uniform_p[1] = 1 - uniform_p[0]
        # p_choice2 = torch.multinomial(uniform_p, num_samples=mask1.sum(), replacement=True).to(device)
        # uniform_mu[p_choice2==0] *= -1

        # total
        y_samples = torch.zeros(nn_samples, device=device)
        y_samples[mask0] = gauss_mu
        y_samples[mask1] = uniform_mu
        if self.isSmooth:
            y_samples = self.gaussian_filter1d(y_samples, device)
        y_samples = y_samples.reshape(n_samples, self.length)

        return y_samples

class GausNoise:
    def __init__(self, mu, sigma, length=1023):
        self.mu = mu
        self.sigma = sigma
        self.length = length
    
    def sample(self, n_sample, device):
        y_samples = torch.randn((n_sample, self.length), device=device) * self.sigma + self.mu
        return y_samples

class NoiseModel:
    def __init__(self, length, normal=False, 
                 tsigma_g=1, tsigma_u=2,
                 rsigma_g=1e-3, rsigma_u=1e-2,):
        self.length = length
        self.normal = normal

        if normal:
            self.NoiseModel_Rot = GausNoise(0, 1e-3, length)
            # self.NoiseModel_Tx = GausNoise(-3e-4, 2e-3, length)
            self.NoiseModel_Tx = GausNoise(0, 1, length)
            self.NoiseModel_Ty = GausNoise(0, 1, length)
        else:
            # rot params
            p = 0.99 # [0.99, 0.01] rot, 
            gauss_params = [0, 1e-3] # [0.0, 0.01-0.1] rot, 
            uniform_params = [-1e-2, 1e-2] # [-2-20, 2-20] rot, 
            sigma = 2 # 2-4 rot, 
            self.NoiseModel_Rot = MixedNoise(p, gauss_params, uniform_params, sigma, length=length, isSmooth=False)
            # self.NoiseModel_Rot = GausNoise(0, 1e-3, length)

            p = 0.8 # [0.95, 0.00-0.05]
            gauss_params = [0, 1] # [0.0, 0.01-0.2] rot, 
            uniform_params = [-2, 2] # [-2, 2] rot, 
            sigma = 2 # 2-4 rot, 
            self.NoiseModel_Tx = MixedNoise(p, gauss_params, uniform_params, sigma, length=length)
            self.NoiseModel_Ty = MixedNoise(p, gauss_params, uniform_params, sigma, length=length)
    
    def sample(self, n_sample, device):
        # if torch.rand(1) < 0.9:
        # if self.normal:
        #     y_samples = self.NoiseModel.sample(n_sample, device)
        # else:
        y_samples_rot = self.NoiseModel_Rot.sample(n_sample, device)
        y_samples_Tx = self.NoiseModel_Tx.sample(n_sample, device)
        y_samples_Ty = self.NoiseModel_Ty.sample(n_sample, device)
        y_samples = torch.stack([y_samples_Tx, y_samples_Ty, y_samples_rot], 1)
        y_samples = y_samples * torch.rand(1, device=device)
        # else:
        #     y_samples = torch.zeros((n_sample, 2, self.length), device=device)
        return y_samples

import os
os.sys.path.append(os.getcwd())
from evaluation_scripts.eval_utils import plot_save_lines
if __name__ == "__main__":
    Noise = NoiseModel(1023, False)
    y = Noise.sample(1, "cpu").reshape(-1, 1023)
    x = torch.linspace(0, 1, 1023)

    trans_label = "Translational error (m)"
    rot_label = r"Rotational error ($\degree$)"
    xytlabel = ["Percentage of trajectory (%)", trans_label, ""] 
    save_filename = os.path.join("./train_log/collection/noise_model", f"gp_fit_test.pdf")

    plot_save_lines(y, ["fff"]*y.shape[0], save_filename, xytlabel, x=x, legend="upper right", linewidth=0.5)