# ipython magic
# %matplotlib
# %load_ext autoreload
# %autoreload 2

# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy

from models import Model

# cuda
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using {device}')

model = Model(resolution=32,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=(1,2,2,2),
                num_res_blocks=2,
                attn_resolutions=(16,),
                dropout=0.1).to(device)
T = 1000
beta = torch.linspace(0.0001, 0.02, T) # same beta as in DDPM
alpha = torch.cat([torch.tensor([1.]), torch.cumprod(1.-beta,0)]).to(device)

# Load weights
# model.load_state_dict(torch.load("cifar_weights.pt", map_location=device))
# Use weights from https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/
model.load_state_dict(torch.load("cifar-790000_weights.ckpt", map_location=device))

## Sample every 100th step
tau = list(range(0,T+1,100))
alpha_tau = alpha[tau]
n_tau = len(tau) - 1
XT = torch.randn((64,3,32,32), device=device)
with torch.no_grad():
    Xt = XT
    for i in range(n_tau, 0, -1):
        t = tau[i]
        print(t)
        t_tensor = torch.full((Xt.shape[0],), t, device=device)
        eps = model(Xt, t_tensor)
        Xt = (alpha_tau[i-1]/alpha_tau[i]).sqrt() * (Xt - (1-alpha_tau[i]).sqrt() * eps) + (1-alpha_tau[i-1]).sqrt() * eps


X0_clip = (torch.clip(Xt, -1., 1.) + 1.) / 2.
torchvision.utils.save_image(X0_clip, "assets/cifar.png")