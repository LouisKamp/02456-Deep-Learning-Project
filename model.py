#%%

import matplotlib.pyplot as plt
import torch
import torch.optim.adam
import torch.optim.sgd
from neuralop.models import FNO

device = 'cpu'
if torch.backends.mps.is_available():
    torch.set_default_device("mps")
    device = 'mps'

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    device = 'cuda'

from dataset import FunctionDataset

# %%

training_data = FunctionDataset("data/solutions.pt", "data/laplacians.pt")
gen = torch.Generator(device=device)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True, generator=gen)

#%%

class MyFNO(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.fno = FNO(n_modes=(5, 5), n_layers=3, hidden_channels=5, in_channels=2, out_channels=1)
        # Fourier neural operator
        self.fno = FNO(n_modes=(16, 16), n_layers=16, hidden_channels=32, in_channels=2, out_channels=1)
        self.loss_fn = torch.nn.MSELoss()
    def forward(self,X):
        # ensure that the shape is (batch size, 2, n, n)
        # here the dimension of 2 encodes 1: the laplacians, 2: bouldery conditions
        Y_hat = self.fno(X) #+ X[:,1,:,:].reshape((-1,1,16,16))
        return Y_hat

    def criterion(self,Y, Y_hat):
        laplacian_kernel = torch.tensor([
            [0.,  1., 0.],
            [1., -4., 1.],
            [0.,  1., 0.]]
        ).unsqueeze(0).unsqueeze(0)

        laplacian_hat = torch.conv2d(Y_hat, laplacian_kernel, padding=1)
        laplacian = torch.conv2d(Y, laplacian_kernel, padding=1)

        return self.loss_fn(Y_hat, Y) + self.loss_fn(laplacian_hat, laplacian)


# %%
model = MyFNO()

#%%
def train(lr: float, epoch: int):
    losses = []
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    try:
        for i in range(epoch):
            X, Y = next(iter(train_dataloader))
            optimizer.zero_grad()
            Y_hat = model.forward(X)
            loss = model.criterion(Y, Y_hat)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                losses.append(loss.detach().cpu())
                print(losses[-1])
                plt.semilogy(losses)
                plt.show()
    except InterruptedError:
        print("Interrupted")

    return torch.tensor(losses)


# %%
losses = train(lr=0.000001,epoch=10_000)
torch.save(losses,"./losses.pt")
torch.save(model, "./model.pt")

# %%
n = 16
X, Y = next(iter(train_dataloader))
Y_hat = model.forward(X[0].reshape((-1,2,n,n)))

fig, axs = plt.subplots(1,3)

axs[0].matshow(Y_hat[0,0].detach().cpu())
axs[1].matshow(Y[0,0].cpu())
axs[2].matshow(X[0,0].cpu())
# %%
