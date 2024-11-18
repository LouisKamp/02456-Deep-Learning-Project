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
N = 1_000

# number of blobs
n = 5

for i in range(N):
    p = 1
    for j in range(n):
        p += 1 * exp(-(((x - np.random.rand())**2 + (y - np.random.rand())**2) / (2 * 0.2**2)))
    phis.append(lambdify((x,y), p))
    fs.append(lambdify((x,y), diff(p, x) + diff(p, y)))
# %%
n = 10

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

plt.matshow(solutions[90])

# %%
