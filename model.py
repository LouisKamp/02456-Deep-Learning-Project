#%%

import torch
from neuralop.models import FNO
import matplotlib.pyplot as plt
#%%
# shape (10,10)
solutions = torch.load("./data/solutions.pt")
laplacians = torch.load("./data/laplacians.pt")

n = solutions.shape[1]

#%%

operator = FNO(n_modes=(5, 5), hidden_channels=10, in_channels=1, out_channels=1)

# %%
res = operator(solutions[0].reshape((1,1,10,10)))
#%%

plt.matshow(solutions[0])

# %%
plt.matshow(res.reshape((10,10)).detach())
# %%



