import torch
import torch.nn as nn


class ParameterizedFunction(torch.nn.Module):
    """
    theta: parameter of the Weibull distribution
    1/2 = 1/theta + 1/theta_conj
    f(x) = gamma * alpha / lambda_1^alpha x^(alpha - 1) exp(-(x / lambda_1^)alpha) + sqrt(2/pi) 1/Gamma(1 - 1/theta) exp(-(x/lambda_2)^theta_conj)
    """
    def __init__(self, theta, theta_conj, alpha, lambd_1, lambd_2, gamma):
        super(ParameterizedFunction, self).__init__()
        self.register_buffer('theta', torch.tensor(theta))
        self.register_buffer('theta_conj', torch.tensor(theta_conj))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.lambd_1 = nn.Parameter(torch.tensor(lambd_1))
        self.lambd_2 = nn.Parameter(torch.tensor(lambd_2))
        self.gamma = nn.Parameter(torch.tensor(gamma))

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
        * self.tail_kernel: function: (z, x) -> S_W(z/x);
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
    def __init__(self, density, tail_kernel):
        super(Integrand, self).__init__()

        self.density = density
        self.tail_kernel = tail_kernel

    def get(self, z):
        return lambda x: self.tail_kernel(z, x) * self.density(x)
        
    def compute_intervals(self, z, a, b, num_intervals):
        scipy_result = integrate.quad(self.get(z), a, b, epsabs = 1e-4, epsrel = 1e-4, full_output = 1)
        scipy_num_intervals = scipy_result[2]['last']
        left_pts = sorted(scipy_result[2]['alist'][:scipy_num_intervals])
        right_pts = sorted(scipy_result[2]['blist'][:scipy_num_intervals])

        intervals = [torch.linspace(l, r, num_intervals) for l, r in zip(left_pts, right_pts)]
        dxs = [inter[1] - inter[0] for inter in intervals]

        return intervals, dxs

    def integrate(self, z, intervals, dxs):
        f = self.get(z)

        result = 0
        for inter, dx in zip(intervals, dxs):
            result += torch.trapezoid(f(inter), dx = dx)

        return result
    
    def integrate_scipy(self, z, a, b):
        return integrate.quad(self.get(z), a, b)[0]

    def forward(self, x):
        if isinstance(x, (int, float, list)):
            x = torch.tensor(x, dtype = torch.float)

        return self.approx_mod(x) * self.kernel(self.z, x)
