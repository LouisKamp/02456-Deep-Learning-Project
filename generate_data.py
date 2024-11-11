#%%
from sympy import *
import itertools
import matplotlib.pyplot as plt
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

fig, axs = plt.subplots(1,2)

i = 30

axs[0].matshow(fs[i](X,Y))
axs[0].set_title("$\phi$")
axs[1].matshow(phis[i](X,Y))
axs[1].set_title("$f$")
# %%
