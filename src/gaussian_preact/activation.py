import torch
import torch.nn as nn
from importlib.resources import files

def build_act_function(act_type, theta, with_np = False):
    dirname = files('gaussian_preact').joinpath('../../objects/')
    if act_type == 'odd':
        dct = torch.load(dirname.joinpath('ActFunction_theta-{:.2f}.pkl'.format(theta)))
    elif act_type == 'pos':
        dct = torch.load(dirname.joinpath('ActFunctionPos_theta-{:.2f}.pkl'.format(theta)))
    else:
        raise ValueError(f'Unknown value for "act_type". Must be "odd" or "pos", found "{act_type}".')

    return dct['activation'].to_function(with_np)

def build_act_function_p(act_type, theta, with_np = False):
    dirname = files('gaussian_preact').joinpath('../../objects/')
    if act_type == 'odd':
        dct = torch.load(dirname.joinpath('ActFunction_theta-{:.2f}.pkl'.format(theta)))
    elif act_type == 'pos':
        dct = torch.load(dirname.joinpath('ActFunctionPos_theta-{:.2f}.pkl'.format(theta)))
    else:
        raise ValueError(f'Unknown value for "act_type". Must be "odd" or "pos", found "{act_type}".')

    return dct['activation'].to_function_p(with_np)

def act_function(x, alpha, a, b, c, d, f, g):
    """
    f(x) = a * tanh(b * x) * ((|x| + c)^alpha + d) * (log(|x| + f) + g)
    """
    return a * torch.tanh(b * x) * (torch.pow(torch.abs(x) + c, alpha) + d) * \
                (torch.log(torch.abs(x) + f) + g)

def act_function_p(x, alpha, a, b, c, d, f, g):
    x = torch.abs(x)
    A = torch.tanh(x * b)
    B = (torch.pow(torch.abs(x) + c, alpha) + d)
    C = (torch.log(torch.abs(x) + f) + g)

    ret = b * (1 - torch.pow(torch.tanh(b * x), 2)) * B * C
    ret = ret + A * alpha * torch.pow(x + c, alpha - 1) * C
    ret = ret + A * B / (x + f)

    return a * ret

class ActivationFunctionTraining(torch.nn.Module):
    """
    Function representing the activation function, built in such a way it is easy to compute
    and backpropagate for pytorch.

    f(x) = a * tanh(b * x) * ((|x| + c)^alpha + d) * (log(|x| + f) + g)
    """
    def __init__(self, theta, a, b, c, d, f, g):
        super(ActivationFunctionTraining, self).__init__()
        dtype = torch.get_default_dtype()

        alpha = 1 - 2/theta
        
        self.register_buffer('alpha', torch.tensor(alpha))
        self.a = nn.Parameter(torch.tensor(a, dtype = dtype))
        self.b = nn.Parameter(torch.tensor(b, dtype = dtype))
        self.c = nn.Parameter(torch.tensor(c, dtype = dtype))
        self.d = nn.Parameter(torch.tensor(d, dtype = dtype))
        
        self.f = nn.Parameter(torch.tensor(f, dtype = dtype))
        self.g = nn.Parameter(torch.tensor(g, dtype = dtype))
        
    def forward(self, x):
        if self.train:
            return act_function(x, self.alpha, self.a, self.b, self.c, self.d, self.f, self.g)
        else:
            with torch.no_grad():
                return act_function(x, self.alpha, self.a, self.b, self.c, self.d, self.f, self.g)

    def forward_ng(self, x):
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, device = self.a.device)
            elif x.device != self.a.device:
                x = x.to(device = self.a.device)

            return self.forward(x)
        
    def derivative(self, x):
        if self.to_train:
            x = torch.abs(x)
            A = torch.tanh(x * self.b)
            B = (torch.pow(torch.abs(x) + self.c, self.alpha) + self.d)
            C = (torch.log(torch.abs(x) + self.f) + self.g)

            ret = self.b * (1 - torch.pow(torch.tanh(self.b * x), 2)) * B * C
            ret = ret + A * self.alpha * torch.pow(x + self.c, self.alpha - 1) * C
            ret = ret + A * B / (x + self.f)

            return self.a * ret
        else:
            with torch.no_grad():
                x = torch.abs(x)
                A = torch.tanh(x * self.b)
                B = (torch.pow(torch.abs(x) + self.c, self.alpha) + self.d)
                C = (torch.log(torch.abs(x) + self.f) + self.g)

                ret = self.b * (1 - torch.pow(torch.tanh(self.b * x), 2)) * B * C
                ret = ret + A * self.alpha * torch.pow(x + self.c, self.alpha - 1) * C
                ret = ret + A * B / (x + self.f)

                return self.a * ret

    def derivative_ng(self, x):
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, device = self.a.device)
            elif x.device != self.a.device:
                x = x.to(device = self.a.device)

            return self.derivative(x)

    def derivative2(self, x):
        x = torch.abs(x)

        tanh_bx = torch.tanh(x * self.b)

        A = tanh_bx
        B = (torch.pow(torch.abs(x) + self.c, self.alpha) + self.d)
        C = (torch.log(torch.abs(x) + self.f) + self.g)

        Ap = self.b * (1 - torch.pow(tanh_bx, 2))
        Bp = self.alpha * torch.pow(x + self.c, self.alpha - 1)
        Cp = 1 / (x + self.f)

        App = -2 * torch.pow(self.b, 2) * tanh_bx * (1 - torch.pow(tanh_bx, 2))
        Bpp = self.alpha * (self.alpha - 1) * torch.pow(x + self.c, self.alpha - 2)
        Cpp = - 1 / torch.pow(x + self.f, 2)

        ret = App * B * C + A * Bpp * C + A * B * Cpp
        ret = ret + 2 * (Ap * Bp * C + Ap * B * Cp + A * Bp * Cp)

        return self.a * ret

    def derivative2_ng(self, x):
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, device = self.a.device)
            elif x.device != self.a.device:
                x = x.to(device = self.a.device)

            return self.derivative2(x)

class ActivationFunction(torch.nn.Module):
    """
    Function representing the activation function, built in such a way it is easy to compute
    and backpropagate for pytorch.

    f(x) = a * tanh(b * x) * ((|x| + c)^alpha + d) * (log(|x| + f) + g)
    """
    def __init__(self, act_function):
        super(ActivationFunction, self).__init__()

        self.register_buffer('alpha', act_function.alpha.clone().detach())
        
        self.register_buffer('a', act_function.a.clone().detach())
        self.register_buffer('b', act_function.b.clone().detach())
        self.register_buffer('c', act_function.c.clone().detach())
        self.register_buffer('d', act_function.d.clone().detach())
        
        self.register_buffer('f', act_function.f.clone().detach())
        self.register_buffer('g', act_function.g.clone().detach())
        
    def forward(self, x):
        if self.train:
            return act_function(x, self.alpha, self.a, self.b, self.c, self.d, self.f, self.g)
        else:
            with torch.no_grad():
                return act_function(x, self.alpha, self.a, self.b, self.c, self.d, self.f, self.g)

    def forward_ng(self, x):
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, device = self.a.device)
            elif x.device != self.a.device:
                x = x.to(device = self.a.device)

            return self.forward(x)

    def to_function(self, with_np = False):
        alpha = self.alpha.item()

        a = self.a.item()
        b = self.b.item()
        c = self.c.item()
        d = self.d.item()

        f = self.f.item()
        g = self.g.item()

        if with_np:
            def func(x):
                x = torch.tensor(x)
                return act_function(x, alpha, a, b, c, d, f, g)
        else:
            def func(x):
                return act_function(x, alpha, a, b, c, d, f, g)

        return func

    def to_function_p(self, with_np = False):
        alpha = self.alpha.item()

        a = self.a.item()
        b = self.b.item()
        c = self.c.item()
        d = self.d.item()

        f = self.f.item()
        g = self.g.item()

        if with_np:
            def func(x):
                x = torch.tensor(x)
                return act_function_p(x, alpha, a, b, c, d, f, g)
        else:
            def func(x):
                return act_function_p(x, alpha, a, b, c, d, f, g)

        return func

def act_function_pos(x, alpha, a, b, c, d, f, g, h):
    """
    f(x) = a * sigmoid(b * x + h) * ((softplus(x) + c)^alpha + d) * (log(softplus(x) + f) + g)
    """
    return a * torch.sigmoid(b * x + h) * (torch.pow(torch.nn.functional.softplus(x) + c, alpha) + d) * \
        (torch.log(torch.nn.functional.softplus(x) + f) + g)

def act_function_pos_p(x, alpha, a, b, c, d, f, g, h):
    s = torch.sigmoid(b * x + h)
    A = a * s
    Ap = a * s * (1 - s) * b
    t = torch.nn.functional.softplus(x)
    u = torch.sigmoid(x)
    B = torch.pow(t + c, alpha) + d
    Bp = alpha * torch.pow(t + c, alpha - 1) * u
    C = torch.log(t + f) + g
    Cp = (1 / (t + f)) * u
    return Ap * B * C + A * Bp * C + A * B * Cp

class ActivationFunctionPosTraining(torch.nn.Module):
    """
    Function representing the activation function, built in such a way it is easy to compute
    and backpropagate for pytorch.

    f(x) = a * sigmoid(b * x + h) * ((softplus(x) + c)^alpha + d) * (log(softplus(x) + f) + g)
    """
    def __init__(self, theta, a, b, c, d, f, g, h):
        super(ActivationFunctionPosTraining, self).__init__()
        dtype = torch.get_default_dtype()

        alpha = 1 - 2/theta
        
        self.register_buffer('alpha', torch.tensor(alpha))
        self.a = nn.Parameter(torch.tensor(a, dtype = dtype))
        self.b = nn.Parameter(torch.tensor(b, dtype = dtype))
        self.h = nn.Parameter(torch.tensor(h, dtype = dtype))
        self.c = nn.Parameter(torch.tensor(c, dtype = dtype))
        self.d = nn.Parameter(torch.tensor(d, dtype = dtype))
        
        self.f = nn.Parameter(torch.tensor(f, dtype = dtype))
        self.g = nn.Parameter(torch.tensor(g, dtype = dtype))
        
    def forward(self, x):
        return act_function_pos(x, self.alpha, self.a, self.b, self.c, self.d, self.f, self.g, self.h)

    def forward_ng(self, x):
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, device = self.a.device)
            elif x.device != self.a.device:
                x = x.to(device = self.a.device)

            return self.forward(x)

class ActivationFunctionPos(torch.nn.Module):
    """
    Function representing the activation function, built in such a way it is easy to compute
    and backpropagate for pytorch.

    f(x) = a * sigmoid(b * x + h) * ((softplus(x) + c)^alpha + d) * (log(softplus(x) + f) + g)
    """
    def __init__(self, act_function):
        super(ActivationFunctionPos, self).__init__()
        
        self.register_buffer('alpha', act_function.alpha.clone().detach())
        
        self.register_buffer('a', act_function.a.clone().detach())
        self.register_buffer('b', act_function.b.clone().detach())
        self.register_buffer('h', act_function.h.clone().detach())
        self.register_buffer('c', act_function.c.clone().detach())
        self.register_buffer('d', act_function.d.clone().detach())
        
        self.register_buffer('f', act_function.f.clone().detach())
        self.register_buffer('g', act_function.g.clone().detach())
        
    def forward(self, x):
        return act_function_pos(x, self.alpha, self.a, self.b, self.c, self.d, self.f, self.g, self.h)

    def forward_ng(self, x):
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, device = self.a.device)
            elif x.device != self.a.device:
                x = x.to(device = self.a.device)

            return self.forward(x)

    def to_function(self, with_np = False):
        alpha = self.alpha.item()

        a = self.a.item()
        b = self.b.item()
        h = self.h.item()
        c = self.c.item()
        d = self.d.item()

        f = self.f.item()
        g = self.g.item()

        if with_np:
            def func(x):
                x = torch.tensor(x)
                return act_function_pos(x, alpha, a, b, c, d, f, g, h)
        else:
            def func(x):
                return act_function_pos(x, alpha, a, b, c, d, f, g, h)

        return func

    def to_function_p(self, with_np = False):
        alpha = self.alpha.item()

        a = self.a.item()
        b = self.b.item()
        h = self.h.item()
        c = self.c.item()
        d = self.d.item()

        f = self.f.item()
        g = self.g.item()

        if with_np:
            def func(x):
                x = torch.tensor(x)
                return act_function_pos_p(x, alpha, a, b, c, d, f, g, h)
        else:
            def func(x):
                return act_function_pos_p(x, alpha, a, b, c, d, f, g, h)

        return func
