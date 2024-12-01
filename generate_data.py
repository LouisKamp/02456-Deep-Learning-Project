#%%
import numpy as np
import torch
from sympy import *

# %%

x,y = symbols("x, y")

phis = []
fs = []

# number of functions
N = 100

# number of blobs
blobs = 10

for i in range(N):
    p = 0
    for j in range(blobs):
        pos_x = np.random.rand()
        pos_y = np.random.rand()

        width = 0.2

        p += 1 * exp(-(
            (
                ((x - pos_x)**2) / (2 * width**2) +
                ((y - pos_y)**2) / (2 * width**2)
            )))
    phis.append(lambdify((x,y), p))
    fs.append(lambdify((x,y), diff(p, x,x) + diff(p, y,y)))
# %%

grid_width = 64
X, Y = torch.meshgrid(torch.linspace(0,1,grid_width), torch.linspace(0,1,grid_width))

solutions = torch.zeros((len(phis), grid_width, grid_width))
laplacians = torch.zeros((len(phis), grid_width, grid_width))

for i, (phi, f) in enumerate(zip(phis, fs)):
    solutions[i] = phi(X,Y)
    laplacians[i] = f(X,Y)

torch.save(solutions,"./data/solutions_validate_64.pt")
torch.save(laplacians,"./data/laplacians_validate_64.pt")


# %%
grid_width = 16
X, Y = torch.meshgrid(torch.linspace(0,1,grid_width), torch.linspace(0,1,grid_width))

solutions = torch.zeros((len(phis), grid_width, grid_width))
laplacians = torch.zeros((len(phis), grid_width, grid_width))

for i, (phi, f) in enumerate(zip(phis, fs)):
    solutions[i] = phi(X,Y)
    laplacians[i] = f(X,Y)

torch.save(solutions,"./data/solutions_validate_16.pt")
torch.save(laplacians,"./data/laplacians_validate_16.pt")
# %%
