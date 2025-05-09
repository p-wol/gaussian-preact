# gaussian-preact
Computations related to the paper [Gaussian Pre-Activations in Neural Networks: Myth or Reality?](https://openreview.net/forum?id=goe6fv6iSh), Pierre Wolinski and Julyan Arbel, published in [Transactions on Machine Learning Research](https://jmlr.org/tmlr/), April 2025.

This package computes the functions $\phi_{\theta}$. One can try the main features of this package by using `Example01_Full_computation.ipynb`.

Notebooks:
* `Example01_Full_computation.ipynb`: full computation of the odd activation functions $\phi_{\theta}^{\text{o}}$;
* `Example02_positive_phi.ipynb`: computation of the positive activation functions $\phi_{\theta}^{\text{p}}$;
* `Example03_Gaussianity.ipynb`: tests of the gaussianity of the pre-activations when using $\phi_{\theta}^{\text{o/p}}$.

## Reminder

We want to find a pair $(\mathrm{P}, \phi)$ such that:

$$W \sim \mathrm{P}, X \sim \mathcal{N}(0, 1) \Rightarrow W \phi(X) \sim \mathcal{N}(0, 1).$$

We have chosen $W \sim \mathcal{W}(\theta, 1)$, the symmetric Weibull distribution of parameters $(\theta, 1)$, that is, the CDF of $W$ is:

$$F_W(t) = \frac{1}{2} + \frac{1}{2}\mathrm{sgn}(t) \exp\left(-|t|^{\theta} \right).$$

Thus, given $\theta$, we have to compute the corresponding odd activation function $\phi_{\theta}$ such that:

$$W \sim \mathrm{P}\_{\theta} = \mathcal{W}(\theta, 1), X \sim \mathcal{N}(0, 1) \Rightarrow W \phi\_{\theta}(X) \sim \mathcal{N}(0, 1).$$

Two steps:
 1. find a symmetric distribution $\mathrm{Q}_{\theta}$ such that: 

$$W \sim \mathrm{P}\_{\theta}, Y \sim \mathrm{Q}\_{\theta} \Rightarrow W Y \sim \mathcal{N}(0, 1);$$

 2. find an odd function $\phi_{\theta}$ such that:

$$X \sim \mathcal{N}(0, 1) \Rightarrow Y := \phi\_{\theta}(X) \sim \mathrm{Q}\_{\theta}.$$

## First step

We approximate by $g_{\Lambda}$ (where $\Lambda$ represent the parameters of $g$) the theoretical density $f_Y$ of $\mathrm{Q}\_{\theta}$. 
Let $Y$ be a random variable sampled from to $g_{\Lambda}$. Let G := W Y. 
We have then the CDF of $|G|$:

$$F\_{\Lambda}(z) = \int\_{0}^{\infty} F\_{|W|}\left(\frac{z}{t}\right) g\_{\Lambda}(t) \mathrm{d} t.$$

Therefore, we only have to optimize the following loss according to $\Lambda$:

$$\ell(\Lambda) := || \hat{F}\_{\Lambda} - F\_{|G|} ||\_{\infty}.$$

For that, we:
1. use `integration.ParameterizedFunction` as function $g_{\Lambda}$;
2. perform the integration with `integration.Integrand`;
3. compute the loss;
4. backpropagate the gradient of the loss through the computation of the integral to compute $\frac{\partial \ell}{\partial \Lambda}$;
5. make a gradient step to train $\Lambda$.

This optimization process is coded in `optimization.find_density`.

## Second step

Once $g_{\Lambda}$ has been optimized, let us denote by $F_{\mathrm{Q}}$ the related CDF: $F_{\mathrm{Q}}' = g_{\Lambda}$ with $F_{\mathrm{Q}}(-\infty) = 0$ and $F_{\mathrm{Q}}(\infty) = 1$. 
We then have:

$$\phi\_{\theta}(x) := F\_{\mathrm{Q}}^{-1}(F\_X(x)),$$

where $F_X$ is the CDF of $X \sim \mathcal{N}(0, 1)$.

To compute $\phi_{\theta}$, we perform:
 1. an interpolation: make a numerical computation of $\phi_{\theta}(x)$ for several $x$;
 2. an approximation: build a function that approximates well the "interpolation" (or "graph") computed at step 1:  
  a. propose a family of parameterized functions `activation.ActivationFunction` that would fit well the interpolation for any $\theta$,  
  b. for a given $\theta$, opzimize the parameters of `activation.ActivationFunction` by gradient descent, so that the graph of the final function would be close to the interpolation.

This optimization process is coded in `optimization.find_activation`.
