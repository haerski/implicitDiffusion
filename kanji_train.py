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

# Load dataset
dataset = torchvision.datasets.ImageFolder(root = "./kkanji2",
                                    transform = transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=50,
                                        shuffle=True,
                                        num_workers=4)

# Load and configure model
model = Model(resolution=64,
                in_channels=1,
                out_ch=1,
                ch=128,
                ch_mult=(1,1,2,2,4),
                num_res_blocks=2,
                attn_resolutions=(16,),
                dropout=0.05).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
loss_fcn = nn.MSELoss()
T = 1000
beta = torch.linspace(0.0001, 0.02, T) # same beta as in DDPM
alpha = torch.cat([torch.tensor([1.]), torch.cumprod(1.-beta,0)]).to(device)

epochs = 30
## Train
model.train()
for epoch in range(epochs):
    for XX, y in dataloader:
        X = (XX[:,[0]] * 2. - 1.).to(device)
        t = (torch.randint(T, (X.shape[0],)) + 1).to(device)
        noise = torch.randn(X.shape, device = device)
        Xt = alpha[t,None,None,None].sqrt() * X + (1-alpha[t,None,None,None]).sqrt() * noise
        loss = loss_fcn(model(Xt, t), noise)
        print(f'epoch {epoch}, loss: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    chkpt = deepcopy(model.state_dict())
    torch.save(chkpt, "kanji_checkpoint.pt")
torch.save(model.state_dict(), "kanji_weights.pt")