import torch
import torch.nn as nn
import numpy as np
from .integration import ParameterizedFunction, Integrand

def surv_Gaussian_abs(x):
    """
    Survival function of the absolute value of Normal(0, 1).
    """
    return 1 - torch.erf(x/np.sqrt(2))

def build_surv_kernel_Weib(theta, lambd):
    """
    Let S_W(x) = exp(-(x/lambd)**theta) be the survival function of Weibull(theta, lambd).
    This function builds the function: (z, y) -> S_W(z/y)
    """
    def surv_kernel_Weib(z, y):
        return np.exp(-np.power((z/y)/lambd, theta))
    return surv_kernel_Weib

def compute_Linf_error(y_targets, y_samples):
    """
    Given two vectors y_targets and y_samples, computes the L-infinity norm between the two.
    """
    return (y_targets - y_samples).abs().max()

def find_density(theta, surv_kernel, density, integrand, 
        inputs, targets, lr = .001, epochs = 100, scipy_update_period = 10,
        theta_conj_phases = 2):
    """
    We want to make the random variable W * Y Gaussian, provided that:
        * the survival function S_W of W is: x -> surv_kernel(x*y, y), for all y;
        * the density f_Y of Y is: x -> density(x).
    For that, the function find_density optimizes the parameters of 'density', in order
        to minimize the L-infinity distance between the CDF of Normal(0, 1) and the CDF
        of W * Y.
    """

    # Set the final theta_conj
    theta_conj = 1/(1/2 - 1/theta)

    # Build schedule for theta_conj
    if theta_conj_phases == 2:
        epochs1 = epochs // 2
        lst_thetas_conj1 = np.linspace(2., theta_conj, epochs1)
        lst_thetas_conj2 = np.array([theta_conj] * (epochs - epochs1))
        lst_thetas_conj = np.concatenate((lst_thetas_conj1, lst_thetas_conj2))
        epochs2 = -1
    elif theta_conj_phases == 3:
        epochs1 = epochs // 3
        epochs2 = 2 * epochs // 3
        lst_thetas_conj1 = np.array([2.] * epochs1)
        lst_thetas_conj2 = np.linspace(2., theta_conj, epochs2 - epochs1)
        lst_thetas_conj3 = np.array([theta_conj] * (epochs - epochs2))
        lst_thetas_conj = np.concatenate((lst_thetas_conj1, lst_thetas_conj2, lst_thetas_conj3))
    else:
        raise ValueError('theta_conj_phases should be equal to 2 or 3, found {}'.format(theta_conj_phases))
    
    # Optimizer and lr scheduler
    optimizer = torch.optim.Adam(density.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
        factor = np.power(.1, 1/3), patience = 20, threshold = .01, cooldown = 20)
    
    # Training
    lst_intervals = [None] * len(inputs)
    lst_dxs = [None] * len(inputs)
    for epoch in range(epochs):
        # Update integration intervals
        if epoch % scipy_update_period == 0:
            for i, z in enumerate(inputs):
                lst_intervals[i], lst_dxs[i], _ = integrand.compute_intervals(z, 0, 10, 50)
        
        # theta_conj schedule
        if epoch == epochs1 or epoch == epochs2:
            optimizer = torch.optim.Adam(density.parameters(), lr = lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                factor = np.power(.1, 1/3), patience = 20, threshold = .01, cooldown = 20) 
        density.theta_conj.data = torch.tensor(lst_thetas_conj[epoch])
        
        # Set the gradients to zero
        optimizer.zero_grad()

        # Compute the loss
        surv_samples = torch.empty_like(inputs)
        for i, z in enumerate(inputs):
            if i == 0:
                surv_samples[i] = 1
            else:
                surv_samples[i] = integrand.integrate(z, lst_intervals[i], lst_dxs[i])
        loss = compute_Linf_error(targets, surv_samples)

        # Compute the gradients
        loss.backward()
        for n, t in density.named_parameters():
            if t.grad is not None and torch.isnan(t.grad):
                t.grad.zero_()
        
        # Optimization step
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            print('    epoch {} ; loss: {:.6f}'.format(epoch, loss.item()))
        #print('lst_y = {}'.format(lst_y))
        """for n, t in density.named_parameters():
            print('{} = {:.6f}'.format(n, t.item()))"""
        #print('epoch: ', epoch, '; loss: ', loss.item())
    
    final_surv = torch.empty_like(inputs)
    for i, z in enumerate(inputs):
        final_surv[i] = integrand.integrate_scipy(z, 0, np.inf)
    loss = compute_Linf_error(targets, final_surv)
    print('Final loss: {:.6f}'.format(loss))
        
    return density, loss


def find_activation(theta, activation, act_inter_x, act_inter_y, inputs,
                    lr = .01, epochs = 100):
    # Build targets
    # 1 - Build functional interpolation on a symmetric interval around 0
    f_interp = lambda x: np.interp(x, torch.concat((-act_inter_x.flip(0), act_inter_x)),
                                   torch.concat((-act_inter_y.flip(0), act_inter_y)))

    # 2 - Build targets
    targets = torch.tensor(f_interp(inputs))

    # Set up the optimizer
    optimizer = torch.optim.Adam(activation.parameters(), lr = .01)

    # Optimization
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = activation(inputs)
        loss = (targets - outputs).pow(2).mean()
        """
        Note: minimizing a L2 loss does not provide any guarantee on the Gaussianity of W * Y.
        The L2 loss has been chosen by convenience.
        """

        loss.backward()
        optimizer.step()

    error = np.sqrt(loss.item())

    return activation, error
