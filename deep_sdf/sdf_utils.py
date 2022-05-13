import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from transforms3d.quaternions import *
from transforms3d.axangles import *
import torch.nn.functional as F
import time


def read_sdf(sdf_file):
    with open(sdf_file, "r") as file:
        lines = file.readlines()
        nx, ny, nz = map(int, lines[0].split(' '))
        x0, y0, z0 = map(float, lines[1].split(' '))
        delta = float(lines[2].strip())
        data = np.zeros([nx, ny, nz])
        for i, line in enumerate(lines[3:]):
            idx = i % nx
            idy = int(i / nx) % ny
            idz = int(i / (nx * ny))
            val = float(line.strip())
            data[idx, idy, idz] = val
    return (data, np.array([x0, y0, z0]), delta)


def load_sdf(sdf_file, into_gpu=True):

    assert sdf_file[-3:] == 'sdf' or sdf_file[-3:] == 'pth', "cannot load this type of data"

    print(' start loading sdf from {} ... '.format(sdf_file))

    if sdf_file[-3:] == 'sdf':
        sdf_info = read_sdf(sdf_file)
        sdf = sdf_info[0]
        min_coords = sdf_info[1]
        delta = sdf_info[2]
        max_coords = min_coords + delta * np.array(sdf.shape)
        xmin, ymin, zmin = min_coords
        xmax, ymax, zmax = max_coords
        sdf_torch = torch.from_numpy(sdf).float()
    elif sdf_file[-3:] == 'pth':
        sdf_info = torch.load(sdf_file)
        min_coords = sdf_info['min_coords']
        max_coords = sdf_info['max_coords']
        xmin, ymin, zmin = min_coords
        xmax, ymax, zmax = max_coords
        sdf_torch = sdf_info['sdf_torch'][0, 0].permute(1, 0, 2)

    sdf_limits = torch.tensor([xmin, ymin, zmin, xmax, ymax, zmax], dtype=torch.float32, requires_grad=False)

    if into_gpu:
        sdf_torch = sdf_torch.cuda()
        sdf_limits = sdf_limits.cuda()

    print('     sdf size = {}x{}x{}'.format(sdf_torch.size(0), sdf_torch.size(1), sdf_torch.size(2)))
    print('     minimal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm'.format(xmin * 100, ymin * 100, zmin * 100))
    print('     maximal coordinate = ({:.4f}, {:.4f}, {:.4f}) cm'.format(xmax * 100, ymax * 100, zmax * 100))
    print(' finished loading sdf ! ')

    return sdf_torch, sdf_limits

def skew(w, gpu=False):
    if gpu:
        wc = torch.stack((torch.tensor(0, dtype=torch.float32).cuda(), -w[2], w[1],
                          w[2], torch.tensor(0, dtype=torch.float32).cuda(), -w[0],
                          -w[1], w[0], torch.tensor(0, dtype=torch.float32).cuda()
                          )).view(3, 3)
    else:
        wc = torch.stack((torch.tensor(0, dtype=torch.float32), -w[2], w[1],
                          w[2], torch.tensor(0, dtype=torch.float32), -w[0],
                          -w[1], w[0], torch.tensor(0, dtype=torch.float32)
                         )).view(3, 3)

    return wc


def Exp(dq, gpu):

    if gpu:
        I = torch.eye(3, dtype=torch.float32).cuda()
    else:
        I = torch.eye(3, dtype=torch.float32)

    dphi = torch.norm(dq, p=2, dim=0)

    u = 1/dphi * dq

    ux = skew(u, gpu)

    if gpu:
        dR = I + torch.sin(dphi) * ux + (torch.tensor(1, dtype=torch.float32).cuda() - torch.cos(dphi)) * torch.mm(ux, ux)
    else:
        dR = I + torch.sin(dphi) * ux + (torch.tensor(1, dtype=torch.float32) - torch.cos(dphi)) * torch.mm(ux, ux)

    return dR


def Oplus(T, v, gpu=False):

    dR = Exp(v[3:], gpu)
    dt = v[:3]

    if gpu:
        dT = torch.stack((dR[0, 0], dR[0, 1], dR[0, 2], dt[0],
                          dR[1, 0], dR[1, 1], dR[1, 2], dt[1],
                          dR[2, 0], dR[2, 1], dR[2, 2], dt[2],
                          torch.tensor(0, dtype=torch.float32).cuda(),
                          torch.tensor(0, dtype=torch.float32).cuda(),
                          torch.tensor(0, dtype=torch.float32).cuda(),
                          torch.tensor(1, dtype=torch.float32).cuda())).view(4, 4)
    else:
        dT = torch.stack((dR[0, 0], dR[0, 1], dR[0, 2], dt[0],
                          dR[1, 0], dR[1, 1], dR[1, 2], dt[1],
                          dR[2, 0], dR[2, 1], dR[2, 2], dt[2],
                          torch.tensor(0, dtype=torch.float32),
                          torch.tensor(0, dtype=torch.float32),
                          torch.tensor(0, dtype=torch.float32),
                          torch.tensor(1, dtype=torch.float32))).view(4, 4)

    return torch.mm(T, dT)

def Exp_v(v, gpu=False):
    dR = Exp(v[3:], gpu)
    dt = v[:3]

    if gpu:
        dT = torch.stack((dR[0, 0], dR[0, 1], dR[0, 2], dt[0],
                          dR[1, 0], dR[1, 1], dR[1, 2], dt[1],
                          dR[2, 0], dR[2, 1], dR[2, 2], dt[2],
                          torch.tensor(0, dtype=torch.float32).cuda(),
                          torch.tensor(0, dtype=torch.float32).cuda(),
                          torch.tensor(0, dtype=torch.float32).cuda(),
                          torch.tensor(1, dtype=torch.float32).cuda())).view(4, 4)
    else:
        dT = torch.stack((dR[0, 0], dR[0, 1], dR[0, 2], dt[0],
                          dR[1, 0], dR[1, 1], dR[1, 2], dt[1],
                          dR[2, 0], dR[2, 1], dR[2, 2], dt[2],
                          torch.tensor(0, dtype=torch.float32),
                          torch.tensor(0, dtype=torch.float32),
                          torch.tensor(0, dtype=torch.float32),
                          torch.tensor(1, dtype=torch.float32))).view(4, 4)

    return dT

