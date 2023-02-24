import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy import special, integrate


class ParameterizedFunction(torch.nn.Module):
    """
    theta: parameter of the Weibull distribution
    1/2 = 1/theta + 1/theta_conj
    f(x) = gamma * alpha / lambda_1^alpha x^(alpha - 1) exp(-(x / lambda_1^)alpha) + sqrt(2/pi) 1/Gamma(1 - 1/theta) exp(-(x/lambda_2)^theta_conj)
    """
    def __init__(self, theta, theta_conj, alpha, lambd_1, lambd_2, gamma):
        super(ParameterizedFunction, self).__init__()
        dtype = torch.get_default_dtype()

        self.register_buffer('theta', torch.tensor(theta, dtype = dtype))
        self.register_buffer('theta_conj', torch.tensor(theta_conj, dtype = dtype))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype = dtype))
        self.lambd_1 = nn.Parameter(torch.tensor(lambd_1, dtype = dtype))
        self.lambd_2 = nn.Parameter(torch.tensor(lambd_2, dtype = dtype))
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype = dtype))

    def term_peak(self, x):
        a1 = self.gamma * (self.alpha / torch.pow(self.lambd_1, self.alpha)) \
                * torch.pow(x, self.alpha - 1) * torch.exp(-torch.pow(x / self.lambd_1, self.alpha))
        return a1

    def term_bounds(self, x):
        return np.sqrt(2/np.pi) * torch.exp(- torch.pow(x / self.lambd_2, self.theta_conj)) \
                / special.gamma(1 - 1/self.theta)

    def forward(self, x):
        return self.term_peak(x) + self.term_bounds(x)

class Integrand(torch.nn.Module):
    """
    Class used to compute the survival function of the random variable W * Y, where:
        * W is a Weibull random variable of survival function S_W;
        * Y is a candidate random variable of density f_Y.

    Goal: tune f_Y in order to have W * Y distributed according to Normal(0, 1).

    This goal is achieved by:
        1) computing the CDF of W * Y, denoted by F;
        2) computing the L-infinity distance between F and the CDF of Normal(0, 1);
        3) backpropagating this distance to the parameters of f_Y;
        4) doing a gradient descent on these parameters.

    Parameters:
        * self.surv_kernel: function: (z, x) -> S_W(z/x);
        * self.density: density of Y.

    Reminder: the CDF F of W * Y can be computed in the following way:
        F(z) = \int_0^{\infty} S_W(z/x) f_Y(x) dx        (1)

    Computation of the integral:
        a) the computation of the integral (1) is a critical step of the whole process,
            since we must be able to backpropagate the gradients through it;
        b) the integration scheme must work for both z close to 0 and z slightly greater than 1.

    To solve b), we use torch.trapezoid. However, this function needs a uniform partitioning of
        the integration interval.
    To solve a) and find adapted partitions of the integration interval for each z, we use
        the advanced information outputted by scipy.integrate.quad, specifically, the partition
        that this function computes.

    Note: scipy.integrate.quad is much slower than torch.trapezoid, so it is preferable to call
        it every few steps (so, the same partition of the integration interval is kept for 
        several steps, which is supposed to not damage training).
    """
    def __init__(self, density, surv_kernel):
        super(Integrand, self).__init__()

        self.density = density
        self.surv_kernel = surv_kernel

    def get(self, z):
        """
        Returns the function to integrate, given z.
        """
        return lambda x: self.surv_kernel(z, x) * self.density(x)
        
    def compute_intervals(self, z, a, b, num_intervals):
        """
        Compute the integral (1) by using SciPy integration functions, then returns some features of the
            integration process (inspired by the partition computed by SciPy), along with the result.

        Arguments:
            * z: scalar parameter of the integrand;
            * a, b: respectively left and right bounds of the integral;
            * num_intervals: once SciPy has returned a first partition of [a, b] of size K, we
                build a uniform sub-partition of size num_intervals of each of the initial K intervals.
                For instance, if SciPy returns a partition of size 7 and num_intervals = 20, then
                the total number of sub-intervals we build is equal to 7*20 = 140. Each of the initial
                7 intervals is uniformly partitioned.

        Return values:
            * intervals: list of size K (size of the partition proposed by SciPy), where each element is
                a torch.tensor of size num_intervals+1 (which represents a sub-partition of size 
                num_intervals);
            * dxs: list of K scalars; each dxs[i] contains the size of the sub-intervals contained in 
                intervals[i];
            * value of the integral computed by SciPy.
        """
        scipy_result = integrate.quad(self.get(z), a, b, epsabs = 1e-4, epsrel = 1e-4, full_output = 1)
        scipy_num_intervals = scipy_result[2]['last']
        left_pts = sorted(scipy_result[2]['alist'][:scipy_num_intervals])
        right_pts = sorted(scipy_result[2]['blist'][:scipy_num_intervals])

        intervals = [torch.linspace(l, r, num_intervals) for l, r in zip(left_pts, right_pts)]
        dxs = [inter[1] - inter[0] for inter in intervals]

        return intervals, dxs, scipy_result[0]

    def integrate(self, tens_z, intervals, dxs):
        """
        Integrate the integrand for each parameter z in tens_z, by using the partition
            intervals and integration steps dxs. The result of this integration process can be
            differentiated according to the parameters of self.density.

        Arguments:
            * tens_z: scalar, list or tensor of points z at which the integral will be computed
            * intervals: list of K tensors, representing a K-partition of the integration domain;
                - each intervals[i] is a uniform sub-partition of the i-th interval,
                - dxs[i] is the distance between any (intervals[i][j], intervals[i][j + 1];
            * dxs: list of K scalars representing the fineness of tbe subdivision of each intervals[i].

        Return value:
            * result: tensor with the same shape as tens_z; each result[i] contains the result of
                integration of the integrand with tens_z[i].
        """
        # Preprocess tens_z if it is a scalar or a non-torch object
        if isinstance(tens_z, (int, float, list)):
            tens_z = torch.tensor(tens_z, dtype = torch.get_default_dtype())        

        to_squeeze = False
        if tens_z.dim() == 0:
            tens_z = tens_z.unsqueeze(0)
            to_squeeze = True

        # Compute the integral
        result = torch.zeros_like(tens_z)
        for i, z in enumerate(tens_z):
            f = self.get(z)
            for inter, dx in zip(intervals, dxs):
                result[i] += torch.trapezoid(f(inter), dx = dx)

        # Format and return the result
        if to_squeeze:
            return result.squeeze()
        else:
            return result
    
    def integrate_scipy(self, z, a, b):
        return integrate.quad(self.get(z), a, b)[0]

    def forward(self, z, x):
        return self.get(z)(x)
