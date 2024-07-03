import torch

def sampler_normal(size, device = torch.device('cpu'), dtype = torch.float32):
    return torch.randn(size, device = device, dtype = dtype)

def build_sampler_weibull(theta, lambd = None, normalized = False, device = 'cpu'):
    if normalized:
        if lambd is None:
            lambd = 1.
        else:
            raise ValueError('Error: a fixed value for lambda and normalized == True are incompatible.')
    if not normalized and lambd is None:
        lambd = 1.

    theta = torch.tensor(theta, device = device)
    lambd = torch.tensor(lambd, device = device)
    p = torch.tensor(.5, device = device)

    weib = torch.distributions.weibull.Weibull(lambd, theta)
    bern = torch.distributions.bernoulli.Bernoulli(p)

    if not normalized:
        cst_norm = 1.
    else:
        cst_norm = (weib.variance + weib.mean.pow(2)).sqrt()

    def sampler_weibull(size, device = None, dtype = None):
        y = weib.sample(size)
        y.mul_((bern.sample(size) * 2 - 1) / cst_norm)
        return y

    return sampler_weibull
