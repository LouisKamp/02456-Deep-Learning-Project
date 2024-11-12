#%%

import torch
from neuralop.models import FNO
import matplotlib.pyplot as plt
import torch.optim.adam
import torch.optim.sgd

#%%

if torch.backends.mps.is_available():
    torch.set_default_device("mps")

#%%
# shape (10,10)
solutions = torch.load("./data/solutions.pt")
laplacians = torch.load("./data/laplacians.pt")

n = solutions.shape[1]
# %%

class MyFNO(torch.nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.fno = FNO(n_modes=(5, 5), hidden_channels=n, in_channels=2, out_channels=1)
        self.loss_fn = torch.nn.MSELoss()
    def forward(self,X):
        # ensure that the shape is (batch size, 2, n, n)
        # here the dimension of 2 encodes 1: the laplacians, 2: bouldery conditions
        Y = self.fno(X)
        return Y

    def criterion(self,Y, Y_hat):
        return self.loss_fn(Y_hat, Y) # TODO + laplacian error


# %%

model = MyFNO(n)
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.0001)
# %%
X = torch.zeros((len(solutions)), 2, n,n)
Y = torch.zeros((len(solutions)), 1, n,n)


X[:,0] = laplacians
X[:,1] = solutions
X[:,1,1:-1,1:-1] = 0

Y[:,0] = solutions

def train():
    losses = []
    for i in range(100000):
        optimizer.zero_grad()
        Y_hat = model.forward(X)
        loss = model.criterion(Y, Y_hat)
        loss.backward()
        optimizer.step()
        print(loss)
        losses.append(loss.detach())



# %%
train()
# %%
