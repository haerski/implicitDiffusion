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
dataset = torchvision.datasets.MNIST(root = ".",
                                    download = True,
                                    transform = transforms.Compose([
                                        # transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()]))
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=500,
                                        shuffle=True,
                                        num_workers=4)

XX, y = next(iter(dataloader))
plt.imshow(TF.to_pil_image(torchvision.utils.make_grid(XX)))
plt.savefig("bar.png")


# Load and configure model
model = Model(resolution=28,
                in_channels=1,
                out_ch=1,
                ch=128,
                ch_mult=(1,2,2),
                num_res_blocks=2,
                attn_resolutions=(14,),
                dropout=0.1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
loss_fcn = nn.MSELoss()
n_times = 1000
beta = torch.linspace(0.0001, 0.02, n_times) # same beta as in DDPM
alpha = torch.cat([torch.tensor([1.]), torch.cumprod(1.-beta,0)]).to(device)

epoch = 1
## Train
model.train()
while True:
    for XX, y in dataloader:
        X = (XX * 2. - 1.).to(device)
        t = (torch.randint(n_times, (X.shape[0],)) + 1).to(device)
        noise = torch.randn(X.shape, device = device)
        Xt = alpha[t,None,None,None].sqrt() * X + (1-alpha[t,None,None,None]).sqrt() * noise
        loss = loss_fcn(model(Xt, t), noise)
        print(f'epoch {epoch}, loss: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch += 1
    chkpt = deepcopy(model.state_dict())
    torch.save(chkpt, "checkpoint.pt")
chkpt = deepcopy(model.state_dict())
torch.save(chkpt, "checkpoint.pt")
model.load_state_dict(torch.load("checkpoint.pt"))



model.eval()
## Normal sample
XT = torch.randn((64,1,28,28), device=device)
with torch.no_grad():
    Xt = XT
    for t in range(n_times,0,-1):
        print(t)
        t_tensor = torch.full((Xt.shape[0],), t, device=device)
        eps = model(Xt, t_tensor)
        Xt = (alpha[t-1]/alpha[t]).sqrt() * (Xt - (1-alpha[t]).sqrt() * eps) + (1-alpha[t-1]).sqrt() * eps

X0_clip = (torch.clip(Xt, -1., 1.) + 1.) / 2.
torchvision.utils.save_image(X0_clip, "slow.png")

## Fast sample, every 5th step
tau = list(range(0,n_times+1,5))
alpha_tau = alpha[tau]
n_tau = len(tau) - 1
with torch.no_grad():
    Xt = XT
    for i in range(n_tau, 0, -1):
        t = tau[i]
        print(t)
        t_tensor = torch.full((Xt.shape[0],), t, device=device)
        eps = model(Xt, t_tensor)
        Xt = (alpha_tau[i-1]/alpha_tau[i]).sqrt() * (Xt - (1-alpha_tau[i]).sqrt() * eps) + (1-alpha_tau[i-1]).sqrt() * eps


X0_clip = (torch.clip(Xt, -1., 1.) + 1.) / 2.
torchvision.utils.save_image(X0_clip, "fast.png")

## Faster sample, every 10th step
tau = list(range(0,n_times+1,10))
alpha_tau = alpha[tau]
n_tau = len(tau) - 1
with torch.no_grad():
    Xt = XT
    for i in range(n_tau, 0, -1):
        t = tau[i]
        print(t)
        t_tensor = torch.full((Xt.shape[0],), t, device=device)
        eps = model(Xt, t_tensor)
        Xt = (alpha_tau[i-1]/alpha_tau[i]).sqrt() * (Xt - (1-alpha_tau[i]).sqrt() * eps) + (1-alpha_tau[i-1]).sqrt() * eps


X0_clip = (torch.clip(Xt, -1., 1.) + 1.) / 2.
torchvision.utils.save_image(X0_clip, "faster.png")


## Fastest sample, every 100th step
tau = list(range(0,n_times+1,100))
alpha_tau = alpha[tau]
n_tau = len(tau) - 1
with torch.no_grad():
    Xt = XT
    for i in range(n_tau, 0, -1):
        t = tau[i]
        print(t)
        t_tensor = torch.full((Xt.shape[0],), t, device=device)
        eps = model(Xt, t_tensor)
        Xt = (alpha_tau[i-1]/alpha_tau[i]).sqrt() * (Xt - (1-alpha_tau[i]).sqrt() * eps) + (1-alpha_tau[i-1]).sqrt() * eps


X0_clip = (torch.clip(Xt, -1., 1.) + 1.) / 2.
torchvision.utils.save_image(X0_clip, "fastest.png")