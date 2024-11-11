# 02456 Deep Learning Project

In this project, we aim to solve boundary value problems for partial differential equations using deep learning methods. We want to explore and compare the methods:

- Deep Fourier or Laplacian Neural Operators,
- PINN,
- Pure data-driven approach.

Using these methods we want to compare accuracy, invariance, and speedup to more traditional methods of solving PDE's. 

The data we will be training on is composed of known solutions to the Poisson equation.

That is, our end product should be able to take in the boundary conditions for one particular solution and output the corresponding complete solution. Papers such as xx and yy claim to be able to solve PDE problems both faster but also with the benefit of dimensionality invariance. This means that the deep neural operators are able to make good approximations at fine resolutions even though they were only trained on coarse resolutions. We want to replicate this.


*Milestones:*
- Week 10: Working example of the Neural Operators
- Week 11: Working example of PINN 
- Week 12: Choosing which equation to focus on
- Week 13: Displaying advantages of different models through examples