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
dataset = torchvision.datasets.CIFAR10(root = ".",
                                    download = True,
                                    transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()]))
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=300,
                                        shuffle=True,
                                        num_workers=4)

XX, y = next(iter(dataloader))
plt.imshow(TF.to_pil_image(torchvision.utils.make_grid(XX)))
plt.savefig("bar.png")


# Load and configure model
model = Model(resolution=32,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=(1,2,2,2),
                num_res_blocks=2,
                attn_resolutions=(16,),
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
model.load_state_dict(torch.load("model-790000.ckpt"))
model.load_state_dict(torch.load("checkpoint.pt"))



model.eval()
with torch.no_grad():
    ## Sample
    XT = torch.randn((64,3,32,32), device=device)
    for t in range(n_times,0,-1):
        print(t)
        t_tensor = torch.full((XT.shape[0],), t, device=device)
        eps = model(XT, t_tensor)
        XT = (alpha[t-1]/alpha[t]).sqrt() * (XT - (1-alpha[t]).sqrt() * eps) + (1-alpha[t-1]).sqrt() * eps
        # plt.imshow(TF.to_pil_image(torchvision.utils.make_grid(XT)))
        # plt.pause(0.001)

# clip
XT_clip = (torch.clip(XT, -1., 1.) + 1.) / 2.

for i, X in enumerate(XT_clip.cpu()):
    print(i)
    torchvision.utils.save_image(X, f'img/{i}.png')    