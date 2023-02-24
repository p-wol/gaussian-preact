import torch
import torch.nn as nn
import numpy as np

class ActivationFunction(torch.nn.Module):
    """
    Function representing the activation function, built in such a way it is easy to compute
    and backpropagate for pytorch.

    f(x) = a * tanh(b * x) * ((|x| + c)^alpha + d) * (log(|x| + f) + g)
    """
    def __init__(self, theta, a, b, c, d, f, g):
        super(ActivationFunction, self).__init__()
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
            return self.a * torch.tanh(self.b * x) * (torch.pow(torch.abs(x) + self.c, self.alpha) + self.d) * \
                (torch.log(torch.abs(x) + self.f) + self.g)
        else:
            with torch.no_grad():
                return self.a * torch.tanh(self.b * x) * (torch.pow(torch.abs(x) + self.c, self.alpha) + self.d) * \
                    (torch.log(torch.abs(x) + self.f) + self.g)

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
        Bpp = self.alpha * (self.alpha - 1) * torch.pow(x + self.c, alpha - 2)
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
