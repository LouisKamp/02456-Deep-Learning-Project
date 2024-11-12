#%%
from sympy import *
import itertools
import torch
# %%


func_dict = {
    sin,
    cos,
    lambda x: x**2,
    lambda x: x**3,
}

operator_dict = {
    lambda x,y: x*y,
    lambda x, y: x + y,
}

x,y = symbols("x, y")

phis = []
fs = []

for func_1, operator_1 in itertools.product(func_dict, operator_dict):

    phi = func_1(operator_1(x, y))
    phis.append(lambdify((x,y), phi))
    fs.append(lambdify((x,y),diff(phi, x,x) + diff(phi,y,y)))

    for func_2, operator_2 in itertools.product(func_dict, operator_dict):
        phi = operator_1(func_1(operator_1(x, y)),func_2(operator_2(x,y)))
        phis.append(lambdify((x,y), phi))
        fs.append(lambdify((x,y),diff(phi, x,x) + diff(phi,y,y)))
        
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
