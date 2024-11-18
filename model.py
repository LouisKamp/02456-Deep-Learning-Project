#%%

import matplotlib.pyplot as plt
import torch
import torch.optim.adam
import torch.optim.sgd
from neuralop.models import FNO

if torch.backends.mps.is_available():
    torch.set_default_device("mps")
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

from dataset import FunctionDataset

# %%

training_data = FunctionDataset("data/solutions.pt", "data/laplacians.pt")
gen = torch.Generator(device=device)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True, generator=gen)

#%%

class MyFNO(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fno = FNO(n_modes=(5, 5), n_layers=3, hidden_channels=5, in_channels=2, out_channels=1)
        self.loss_fn = torch.nn.MSELoss()
    def forward(self,X):
        # ensure that the shape is (batch size, 2, n, n)
        # here the dimension of 2 encodes 1: the laplacians, 2: bouldery conditions
        Y_hat = self.fno(X)
        return Y_hat

    def criterion(self,Y, Y_hat):
        laplacian_kernel = torch.tensor([
            [0.,  1., 0.],
            [1., -4., 1.],
            [0.,  1., 0.]]
        ).unsqueeze(0).unsqueeze(0)

        laplacian_hat = torch.conv2d(Y_hat, laplacian_kernel, padding=1)
        laplacian = torch.conv2d(Y, laplacian_kernel, padding=1)

        return self.loss_fn(laplacian_hat, laplacian)


# %%
model = MyFNO()
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.00001)

#%%
def train():
    losses = []
    try:
        for i in range(10000):
            X, Y = next(iter(train_dataloader))
            optimizer.zero_grad()
            Y_hat = model.forward(X)
            loss = model.criterion(Y, Y_hat)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu())

            if i % 100 == 0:
                print(losses[-1])
                plt.semilogy(losses)
                plt.show()
    except KeyboardInterrupt:
        return losses


# %%
losses = train()
# %%
n = 10
X, Y = next(iter(train_dataloader))
Y_hat = model.forward(X[0].reshape((-1,2,n,n)))

fig, axs = plt.subplots(1,2)

axs[0].matshow(Y_hat[0,0].detach().cpu())
axs[1].matshow(Y[0,0].cpu())
# %%
