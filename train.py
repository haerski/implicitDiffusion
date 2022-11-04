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

# Load dataset
dataset = torchvision.datasets.MNIST(root = ".",
                                    download = True,
                                    transform = transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=512,
                                        shuffle=True,
                                        num_workers=4)

X, y = next(iter(dataloader))
plt.imshow(TF.to_pil_image(torchvision.utils.make_grid(X)))



# Load and configure model
model = Model(resolution=28,
                in_channels=1,
                out_ch=1,
                ch=128,
                ch_mult=(1,2,2),
                num_res_blocks=2,
                attn_resolutions=(16,),
                dropout=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
loss_fcn = nn.MSELoss()
n_times = 1000
beta = torch.linspace(1e-4, 0.02, n_times) # same beta as in DDPM
alpha = torch.cat([torch.tensor([1.]), torch.cumprod(1.-beta,0)])


## Visualize
XT = model(X,torch.tensor([2]))
plt.imshow(TF.to_pil_image(torchvision.utils.make_grid(XT)))


epoch = 1
## Train
while True:
    for X, y in dataloader:
        X = X * 2. - 1.
        t = torch.randint(n_times, (X.shape[0],)) + 1
        noise = torch.randn(X.shape)
        Xt = alpha[t,None,None,None].sqrt() * X + (1-alpha[t,None,None,None]).sqrt() * noise
        loss = loss_fcn(model(Xt, t), noise)
        print(f'epoch {epoch}, loss: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    chkpt = deepcopy(model.state_dict())
    torch.save(chkpt, "checkpoint.pt")





## Sample
XT = torch.randn((8,1,28,28))
for t in range(n_times,0,-1):
    print(t)
    XT = (alpha[t-1]/alpha[t]).sqrt() * (XT - (1-alpha[t]).sqrt() * model(XT, torch.tensor([t]))) + (1-alpha[t-1]).sqrt() * model(XT, torch.full((XT.shape[0],), t))
    # plt.imshow(TF.to_pil_image(torchvision.utils.make_grid(XT)))
    # plt.pause(0.001)

plt.imshow(TF.to_pil_image(torchvision.utils.make_grid((XT + 1.)/2.)))
plt.savefig("foo.png")