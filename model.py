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

training_data = FunctionDataset("data/solutions_16.pt", "data/laplacians_16.pt")
gen = torch.Generator(device=device)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True, generator=gen)

#%%

class MyFNO(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.fno = FNO(n_modes=(5, 5), n_layers=3, hidden_channels=5, in_channels=2, out_channels=1)
        # Fourier neural operator
        self.fno = FNO(n_modes=(2, 2), n_layers=2, hidden_channels=64, in_channels=2, out_channels=1)
        self.loss_fn = torch.nn.MSELoss()
    def forward(self,X):
        # ensure that the shape is (batch size, 2, n, n)
        # here the dimension of 2 encodes 1: the laplacians, 2: bouldery conditions
        Y_hat = self.fno(X) #+ X[:,1,:,:].reshape((-1,1,16,16))
        return Y_hat
    
    def laplacian(self, A):
        
        laplacian_kernel = torch.tensor([
            [0., 0., 1., 0., .0],
            [0., 2., -8., 2., .0],
            [1., -8., 20., -8., .0],
            [0., 2., -8., 2., .0],
            [0., 0., 1., 0., .0],
        ]).unsqueeze(0).unsqueeze(0)

        A_laplacian = torch.conv2d(A, laplacian_kernel)

        return A_laplacian

    def criterion(self, Y, Y_hat):

        X = self.laplacian(Y)
        X_hat = self.laplacian(Y_hat)
        e1 = torch.mean((Y - Y_hat)**2)
        e2 = torch.mean((X - X_hat)**2)
        # print(f"e1: {e1}, e2: {e2}")
        return e1 + e2


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
                # plt.semilogy(losses)
                # plt.show()
    except InterruptedError:
        print("InterruptedError")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    return torch.tensor(losses)


# %%
losses = train(lr=0.00001,epoch=10_000)
# torch.save(losses,"./losses.pt")
# torch.save(model, "./model.pt")

# %%
test_data = FunctionDataset("data/solutions_64.pt", "data/laplacians_64.pt")
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, generator=gen)
n_test = 64
#%%
X, Y = next(iter(test_dataloader))
Y_hat = model.forward(X[0].reshape((-1,2,n_test,n_test)))

fig, axs = plt.subplots(1,2)

axs[0].matshow(Y_hat[0,0].detach().cpu())
axs[0].set_title("Predicted")

axs[1].matshow(Y[0,0].detach().cpu())
axs[1].set_title("True")
# %%