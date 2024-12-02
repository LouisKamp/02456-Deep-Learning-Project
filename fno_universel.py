#%%

import matplotlib.pyplot as plt
import torch
import torch.optim.adam
import torch.optim.sgd
from neuralop.models import FNO
from softadapt import NormalizedSoftAdapt

device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'

if torch.cuda.is_available():
    device = 'cuda'

from dataset import FunctionDataset

# %%
validation_data = FunctionDataset("data/solutions_validate_16.pt", "data/laplacians_validate_16.pt", device)
training_data = FunctionDataset("data/solutions_16.pt", "data/laplacians_16.pt", device)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=100, shuffle=True)

#%%

class MyFNO(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fno = FNO(n_modes=(1, 1), n_layers=1, hidden_channels=10, in_channels=1, out_channels=1)
        self.loss_fn = torch.nn.MSELoss()
        self.laplacian_kernel = torch.tensor([
            [0., 0., 1., 0., .0],
            [0., 2., -8., 2., .0],
            [1., -8., 20., -8., 1.],
            [0., 2., -8., 2., .0],
            [0., 0., 1., 0., .0],
        ]).unsqueeze(0).unsqueeze(0).to(device)


    def forward(self,X):
        Y_hat = self.fno(X)
        return Y_hat
    
    def laplacian(self, A):
        A_laplacian = torch.conv2d(A, self.laplacian_kernel)

        return A_laplacian
    

    def losses(self, Y, Y_hat):
        X = self.laplacian(Y)
        X_hat = self.laplacian(Y_hat)

        loss_solution = torch.mean((Y - Y_hat)**2)
        loss_laplacian = torch.mean((X - X_hat)**2)

        return loss_solution, loss_laplacian

    def criterion(self, losses, weights):
        return sum(losses[i]*weights[i] for i in range(len(losses)))
    
    @staticmethod
    def train_loop(lr: float, training_epochs: int, epochs_to_make_updates: int = 4):
        weights = [1,1]

        softadapt_object = NormalizedSoftAdapt(beta=0.01)
        
        train_losses_idx = torch.tensor([0]) 
        validation_losses_idx = torch.tensor([])

        train_losses = torch.zeros((0,2))
        validation_losses = torch.zeros((0,2))

        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
        try:
            for current_epoch in range(training_epochs):
                for batch_idx, data in enumerate(train_dataloader):
                    train_losses_idx = torch.cat((train_losses_idx, (train_losses_idx[-1] + 1).reshape(1)))
                    X,Y = data
                    optimizer.zero_grad()
                    Y_hat = model.forward(X)
                    train_loss = model.losses(Y, Y_hat)
                    train_loss_sum = model.criterion(train_loss, weights)
                    train_loss_sum.backward()
                    optimizer.step()

                    train_losses = torch.cat((train_losses, torch.tensor(train_loss).detach().cpu().reshape((1,2))))
                
                model.eval()

                validation_losses_idx = torch.cat((validation_losses_idx, train_losses_idx[-1].reshape(1)))

                X_validate, Y_validate = next(iter(validation_dataloader))
                Y_validate_hat = model.forward(X_validate)
                validation_loss =  model.losses(Y_validate, Y_validate_hat)

                validation_losses = torch.cat((validation_losses, torch.tensor(validation_loss).detach().cpu().reshape((1,2))))

                # plt.semilogy(train_losses_idx[:-1], torch.sum(train_losses,dim=1), label="Train loss")
                plt.semilogy(validation_losses_idx, torch.sum(validation_losses, dim=1), label="Validation loss")
                plt.semilogy(validation_losses_idx, validation_losses[:,0], label="Validation MSE")
                plt.legend()
                plt.show()

                print(f"epoch: {current_epoch}, train MSE: {validation_losses[-1,0]}, train loss: {torch.sum(validation_losses[-1])}")

                if current_epoch % epochs_to_make_updates == 0 and current_epoch != 0:
                    
                    weights = softadapt_object.get_component_weights(validation_losses[-epochs_to_make_updates-1:,0], validation_losses[-epochs_to_make_updates-1:,1]).float().to(device)
                    print(f"New weights: {weights}")

                model.train()
        except InterruptedError:
            print("InterruptedError")
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

        return torch.tensor(train_losses)
    
model = MyFNO().to(device)


# %%
losses = model.train_loop(lr=0.0001,training_epochs=20)
# torch.save(losses,"./losses.pt")
# torch.save(model, "./model.pt")

# %%
test_data = FunctionDataset("data/solutions_64.pt", "data/laplacians_64.pt", device)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
n_test = 64
#%%
X, Y = next(iter(test_dataloader))
Y_hat = model.forward(X[0].reshape((-1,1,n_test,n_test)))

fig, axs = plt.subplots(1,2)

axs[0].matshow(Y_hat[0,0].detach().cpu())
axs[0].set_title("Predicted")

axs[1].matshow(Y[0,0].detach().cpu())
axs[1].set_title("True")

# %% TEST
from sympy import *
x, y = symbols("x, y")
sol = sin(4 * pi * (x + y)) + cos(4 * pi * x * y)

s = lambdify((x,y), sol)
f = lambdify((x,y), diff(sol, x,x) + diff(sol, y,y))

# %%
grid_width = 16
X, Y = torch.meshgrid(torch.linspace(0,1,grid_width).cpu(), torch.linspace(0,1,grid_width).cpu())
F = f(X,Y)
# %%
R = model(F.reshape((1,1,grid_width,grid_width)).to(device)).cpu().detach()
# %%
plt.contourf(R[0,0])
plt.colorbar()
plt.title("Predicted")
# %%
plt.contourf(s(X,Y))
plt.colorbar()
plt.title("True")
# %%
