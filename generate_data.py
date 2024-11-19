#%%
import numpy as np
import torch
from matplotlib import pyplot as plt
from sympy import *

# %%

x,y = symbols("x, y")

phis = []
fs = []

# number of functions
N = 1000

# number of blobs
n = 10

for i in range(N):
    p = 0
    for j in range(n):
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
n = 32

X, Y = torch.meshgrid(torch.linspace(0,1,n), torch.linspace(0,1,n))

#%%

solutions = torch.zeros((len(phis), n, n))
laplacians = torch.zeros((len(phis), n, n))

for i, (phi, f) in enumerate(zip(phis, fs)):
    solutions[i] = phi(X,Y)
    laplacians[i] = f(X,Y)

torch.save(solutions,"./data/solutions.pt")
torch.save(laplacians,"./data/laplacians.pt")
# %%

plt.matshow(laplacians[0])

# %%
plt.matshow(solutions[0])
# %%
