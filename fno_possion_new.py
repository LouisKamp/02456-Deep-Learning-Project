#%%

import matplotlib.pyplot as plt
import torch
import torch.optim.adam
import torch.optim.sgd
from neuralop.models import FNO
from softadapt import NormalizedSoftAdapt
from sympy import *

device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'

if torch.cuda.is_available():
    device = 'cuda'

#%%

class MyFNO(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fno = FNO(n_modes=(5, 5), n_layers=1, hidden_channels=32, in_channels=1, out_channels=1)
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
    
    def get_boundary(self, matrix):
        # Top row
        top_row = matrix[0, :]
        # Bottom row
        bottom_row = matrix[-1, :]
        # Left column excluding corners
        left_column = matrix[1:-1, 0]
        # Right column excluding corners
        right_column = matrix[1:-1, -1]
        
        # Combine boundaries
        boundary = torch.cat([top_row, right_column, bottom_row.flip(0), left_column.flip(0)])
        return boundary
    

    def losses(self, Y, Y_hat):
        X = self.laplacian(Y)
        X_hat = self.laplacian(Y_hat)

        loss_boundary = torch.mean((self.get_boundary(Y) - self.get_boundary(Y_hat))**2)
        loss_laplacian = torch.mean((X - X_hat)**2)

        return loss_boundary, loss_laplacian

    def criterion(self, losses, weights):
        return sum(losses[i]*weights[i] for i in range(len(losses)))
    
    @staticmethod
    def train_loop(lr: float, training_epochs: int, epochs_to_make_updates: int = 100):
        
        weights = [1,0.1]
        softadapt_object = NormalizedSoftAdapt(beta=0.01)
        train_losses = torch.zeros((0,2))

        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)


        x, y = symbols("x, y")
        sol = sin(4 * pi * (x + y)) + cos(4 * pi * x * y)

        s = lambdify((x,y), sol)
        f = lambdify((x,y), diff(sol, x,x) + diff(sol, y,y))

        grid_width = 100
        X, Y = torch.meshgrid(torch.linspace(0,1,grid_width).cpu(), torch.linspace(0,1,grid_width).cpu())
        X = X.reshape((1,1,grid_width,grid_width))
        Y = Y.reshape((1,1,grid_width,grid_width))

        try:
            for current_epoch in range(training_epochs):
                # Solution
                S = s(X,Y).to(device)

                # Laplacian
                F = f(X,Y).to(device)
                optimizer.zero_grad()
                S_hat = model.forward(F)
                losses = model.losses(S, S_hat)
                loss_sum = model.criterion(losses, weights)
                loss_sum.backward()
                optimizer.step()

                train_losses = torch.cat((train_losses, torch.tensor(losses).detach().cpu().reshape((1,2))))
                print(torch.sum(train_losses[-1]))

                if current_epoch % epochs_to_make_updates == 0 and current_epoch != 0:
                    
                    weights = softadapt_object.get_component_weights(train_losses[-epochs_to_make_updates-1:,0], train_losses[-epochs_to_make_updates-1:,1]).float().to(device)
                    print(f"New weights: {weights}")

        except InterruptedError:
            print("InterruptedError")
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

        return torch.tensor(train_losses)
    
model = MyFNO().to(device)

# %%
losses = model.train_loop(lr=0.0001,training_epochs=2000)
# torch.save(losses,"./losses.pt")
# torch.save(model, "./model.pt")
# %%

plt.semilogy(losses)
