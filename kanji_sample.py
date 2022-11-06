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

model = Model(resolution=64,
                in_channels=1,
                out_ch=1,
                ch=128,
                ch_mult=(1,1,2,2,4),
                num_res_blocks=2,
                attn_resolutions=(16,),
                dropout=0.05).to(device)
T = 1000
beta = torch.linspace(0.0001, 0.02, T) # same beta as in DDPM
alpha = torch.cat([torch.tensor([1.]), torch.cumprod(1.-beta,0)]).to(device)

# Load weights
model.load_state_dict(torch.load("kanji_weights.pt", map_location=device))
model.eval()

## Sample every 10th step
tau = list(range(0,T+1,10))
alpha_tau = alpha[tau]
n_tau = len(tau) - 1
XT = torch.randn((8,1,64,64), device=device)
denoise = []
pred = []
with torch.no_grad():
    Xt = XT
    for i in range(n_tau, 0, -1):
        t = tau[i]
        print(t)
        t_tensor = torch.full((Xt.shape[0],), t, device=device)
        eps = model(Xt, t_tensor)
        X0 = (Xt - (1-alpha_tau[i]).sqrt() * eps) / alpha_tau[i].sqrt()
        if i%10 == 0:
            denoise.append(Xt.clone())
            pred.append(X0.clone())

        Xt = alpha_tau[i-1].sqrt() * X0 + (1-alpha_tau[i-1]).sqrt() * eps

denoise.append(Xt.clone())
pred.append(Xt.clone())

denoise_clip = (torch.clip(torch.cat(denoise), -1, 1) + 1) / 2
pred_clip = (torch.clip(torch.cat(pred), -1, 1) + 1) / 2

torchvision.utils.save_image(denoise_clip, "assets/kanji_denoise.png")
torchvision.utils.save_image(pred_clip, "assets/kanji_predict.png")