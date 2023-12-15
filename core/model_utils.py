import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

PI = np.pi

def interp1d(x, size):
    return F.interpolate(x.transpose(1,2), size, mode="linear", align_corners=True).transpose(1,2)
def interp2d(x, size):
    return F.interpolate(x.unsqueeze(1), (size, size), mode="bilinear", align_corners=True).squeeze(1)

def pc_piror(minmax, dim_num):
    # size = minmax[1] - minmax[0]
    # dim_num = int(size / (sigma * 6))
    one_dim = torch.linspace(*minmax, dim_num)
    grid_x, grid_y = torch.meshgrid(one_dim, one_dim, indexing='xy')
    piror = torch.stack([grid_x, -grid_y], 0).view(2, -1)
    return piror, dim_num**2

def encode_pc(x, piror, sigma): #=1e-5
    x = rearrange(x, 'b l d -> b l 1 d')
    piror = rearrange(piror, 'd c -> 1 1 c d').to(x.device)
    logp = -0.5 * (x - piror).pow(2).sum(-1, False) / (2*sigma*sigma)
    activations = logp.softmax(2)
    return activations

def hd_piror(size):
    # piror = torch.randint(0, 360, (1, 1, size)) * PI / 180
    piror = torch.linspace(0, 2*PI, size + 1)[:-1].view(1, 1, size)
    return piror

def encode_hd(x, piror, k=20):
    # piror = rearrange(piror, 'c -> 1 1 c').cuda()

    logp = torch.cos(x + PI - piror.to(x.device)) * k
    activations = F.softmax(logp, 2)
    return activations

def polar2xy(ploar):
    r, theta = ploar[:, :, 0], ploar[:, :, 1]
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], 2)

def xy2polar_v2(xy):
    rxy = xy[:, 1:] - xy[:, :-1]
    base_xy = torch.cat([torch.zeros_like(rxy[:, :1, :]), rxy], 1)
    x, y = base_xy[:, :, 0], base_xy[:, :, 1]
    r = torch.sqrt(x**2 + y**2)
    theta = torch.arctan2(y, x)
    return r, theta, xy - base_xy

def xy2polar(xy):
    x, y = xy[:, :, 0], xy[:, :, 1]
    r = torch.sqrt(x**2 + y**2)
    theta = torch.arctan2(y, x)
    return r, theta

def toangle(a1, a2):
    v = a1 - a2
    cond1 = v < -np.pi
    cond2 = v >  np.pi
    v[cond1] += 2*np.pi
    v[cond2] -= 2*np.pi
    return v

def xy2T(xy):
    x, y = xy[:, :, 0], xy[:, :, 1]
    theta = torch.arctan2(y, x)
    T = polar2T(x, y, theta)
    return T

def polar2T(x, y, theta):
    batch, length = x.shape
    # x, y, theta = polar[:, :, 0], polar[:, :, 1], polar[:, :, 2]
    T = torch.zeros((batch, length, 3, 3), device=x.device)
    T[:, :, 0, 0] = torch.cos(theta)
    T[:, :, 0, 1] = -torch.sin(theta)
    T[:, :, 1, 0] = torch.sin(theta)
    T[:, :, 1, 1] = torch.cos(theta)

    T[:, :, 0, 2] = x
    T[:, :, 1, 2] = y
    T[:, :, 2, 2] = torch.ones_like(y)
    return T

def integ(T, Tv):
    T_integ = T.clone()
    for i in range(Tv.shape[1]):
        T_integ[:, i+1] = Tv[:, i] @ T_integ[:, i]
    return T_integ

def T2theta(T):
    a = T[:, :, 0, 0]
    b = T[:, :, 1, 1]
    d = 0.5*(a+b-1.0)
    theta = torch.arccos(d.clamp(-1.0, 1.0))
    return theta

def traj2velo(traj_input, noise_model):
    batch, size, _ = traj_input.shape
    device = traj_input.device

    T = xy2T(traj_input)
    T_v = torch.matmul(T[:, 1:], T[:, :-1].inverse())

    if noise_model is not None:
        noise = noise_model.sample(batch, device).transpose(1, 2)

        if noise_model.normal is False:
            mean_x = T_v[:,:,0,2].mean(-1, True)
            mean_y = T_v[:,:,1,2].mean(-1, True)
            noise[:, :, 0] *= mean_x
            noise[:, :, 1] *= mean_y

            trans = torch.sqrt(T_v[:,:,0,2]**2 + T_v[:,:,1,2]**2)
            sig_t = torch.sqrt(mean_x**2 + mean_y**2)
            t_weight = torch.exp(-0.5*(trans**2) / (sig_t**2))
            noise[:, :, 2] *= t_weight

        T_noise = polar2T(noise[:, :, 0], noise[:, :, 1], noise[:, :, 2])
        T_v_noise =  T_noise @ T_v
    else:
        T_v_noise = T_v

    velo_out = torch.cat([T_v_noise[:, :, :2, 2], T2theta(T_v_noise).unsqueeze(-1), ], -1)
    velo_out = torch.cat([torch.zeros_like(velo_out[:, :1, :]), velo_out], 1)

    T_noise = integ(T, T_v_noise)
    position = T_noise[:, :, :2, 2]
    direction = T2theta(T_noise).unsqueeze(-1)
    # if noise_model is not None:
    #     T_noise = integ(T, T_v_noise)
    #     position = T_noise[:, :, :2, 2]
    #     direction = T2theta(T_noise).unsqueeze(-1)
    # else:
    #     position = T[:, :, :2, 2]
    #     direction = T2theta(T).unsqueeze(-1)
    
    return velo_out, position, direction

def traj2velo_old(traj_input, noise_model, isnoise=True):
    batch, size, _ = traj_input.shape
    device = traj_input.device

    r, theta, diff_xy = xy2polar_v2(traj_input)
    traj_polar = torch.stack([r, theta], 2)
    velo = toangle(traj_polar[:, 1:], traj_polar[:, :-1])

    if isnoise:
        noise = noise_model.sample(1, device).transpose(1, 2)
        sigr = np.pi / 9
        sigt = 0.001
        weight1 = torch.exp(-0.5*(velo[:,:,1]**2) / (sigr**2))
        weight2 = torch.exp(-0.5*(velo[:,:,0]**2) / (sigt**2))
        noise = noise.repeat(10, 1, 1)
        noise[:,:,0] *= (0.7*weight1 + 0.3*weight2)
        velo = velo + noise
    # velo[:,1] = (velo[:,1] + PI) % (2*PI) - PI

    polar_noise = torch.cumsum(torch.cat([traj_polar[:, :1, :], velo], 1), 1)
    position = polar2xy(polar_noise) + diff_xy
    # plot_for_debug(traj_input, "label")
    # plot_for_debug(position, "noise")
    direction = (polar_noise[:,:,1:2] + PI) % (2*PI) - PI
    velo_out = torch.cat([torch.zeros_like(traj_input[:, :1, :]), velo], 1)
    return velo_out, position, direction

def velo2input_old(velo, scale=1):
    velo_theta = velo[:, :, 1:2]
    r = velo[:, :, 0:1]
    sin_velo = [torch.sin(2**i * velo_theta) for i in range(scale)]
    cos_velo = [torch.cos(2**i * velo_theta) for i in range(scale)]
    u = [r * _v for _v in cos_velo]
    v = [r * _v for _v in sin_velo]
    velo_input = torch.cat(u + v + sin_velo + cos_velo, 2)
    return velo_input

def velo2input(velo, scale=1):
    velo_theta = velo[:, :, 2:3]
    x = velo[:, :, 0:1]
    y = velo[:, :, 1:2]
    sin_velo = [torch.sin(2**i * velo_theta) for i in range(scale)]
    cos_velo = [torch.cos(2**i * velo_theta) for i in range(scale)]
    velo_input = torch.cat([x] + [y] + sin_velo + cos_velo, 2)
    return velo_input

def make_mask(size, sigma):
    ids = torch.linspace(0, 1, size)
    mask = torch.zeros((size, size))
    for i, iid in enumerate(ids):
        mask[i] = np.exp(-torch.abs(iid - ids)**2 / (2*sigma*sigma) )
    return 1 - mask

from tools.tool_utils import traj_plot
def plot_for_debug(traj, prefix):
    traj = traj.detach().cpu().numpy()
    for i, _traj in enumerate(traj):
        traj_plot(_traj, filename=f"./train_log/plots/{prefix}_{i:0>3d}.png")