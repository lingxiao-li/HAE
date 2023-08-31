"""
Network definitions from https://github.com/ferrine/hyrnn
"""

from audioop import bias
import geoopt
import geoopt.manifolds.stereographic.math as gmath
import math

import torch
import torch.nn.init as init
import numpy as np
import torch.nn
import torch.nn.functional
from torch.cuda.amp import autocast

def _tensor_dot(x, y):
    res = torch.einsum("ij,kj->ik", (x, y))
    return res


def _mobius_addition_batch(x, y, c):
    xy = _tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res


def hyperbolic_softmax(X, A, P, c):
    lambda_pkc = 2 / (1 - c * P.pow(2).sum(dim=1))
    k = lambda_pkc * torch.norm(A, dim=1) / torch.sqrt(c)
    mob_add = _mobius_addition_batch(-P, X, c)
    num = 2 * torch.sqrt(c) * torch.sum(mob_add * A.unsqueeze(1), dim=-1)
    denom = torch.norm(A, dim=1, keepdim=True) * (1 - c * mob_add.pow(2).sum(dim=2))
    logit = k.unsqueeze(1) * gmath.arsinh(num / denom)
    return logit.permute(1, 0)

def mobius_linear(
        input,
        weight,
        bias=None,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        k=-1.0,
):
    k = torch.tensor(k)
    if hyperbolic_input:
        output = mobius_matvec(weight, input, k=k)
    else:
        output = torch.nn.functional.linear(input, weight)
        #print(weight)
        #print(output)
        output = gmath.expmap0(output, k=k)
        #output = gmath.expmap0(input, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = gmath.expmap0(bias, k=k)
        output = gmath.mobius_add(output, bias.unsqueeze(0).expand_as(output), k=k)
    if nonlin is not None:
        output = gmath.mobius_fn_apply(nonlin, output, k=k)
    output = gmath.project(output, k=k)
    return output


def mobius_matvec(m: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _mobius_matvec(m, x, k, dim=dim)


def _mobius_matvec(m: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Möbius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    if dim != -1 or m.dim() == 2:
        # mx = torch.tensordot(x, m, [dim], [1])
        mx = torch.matmul(m, x.transpose(1, 0)).transpose(1, 0)
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = gmath.tan_k(mx_norm / x_norm * gmath.artan_k(x_norm, k), k) * (mx / mx_norm)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res

class HyperbolicMLR(torch.nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """

    def __init__(self, ball_dim, n_classes, c):
        super(HyperbolicMLR, self).__init__()
        self.a_vals = torch.nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = torch.nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.reset_parameters()

    def forward(self, x, c=None):
        if c is None:
            c = torch.as_tensor(self.c).type_as(x)
        else:
            c = torch.as_tensor(c).type_as(x)
        p_vals_poincare = gmath.expmap0(self.p_vals, k=torch.tensor(-1.0))
        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor
        logits = hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits

    def extra_repr(self):
        return "Poincare ball dim={}, n_classes={}, c={}".format(
            self.ball_dim, self.n_classes, self.c
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))

class MobiusLinear(torch.nn.Linear):
    def __init__(
            self,
            *args,
            hyperbolic_input=True,
            hyperbolic_bias=True,
            nonlin=None,
            k=-1.0,
            fp64_hyper=False,
            **kwargs
    ):
        k = torch.tensor(k)
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=k.abs())
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(gmath.expmap0(self.bias.normal_() / 4, k=k))
                    #self.bias.set_(gmath.expmap0(self.bias.normal_() / 400, k=k))
        with torch.no_grad():
            # 1e-2 was the original value in the code. The updated one is from HNN++
            #std = 1 / np.sqrt(2 * self.weight.shape[0] * self.weight.shape[1])
            std = 1e-2
            # Actually, we divide that by 100 so that it starts really small and far from the border
            #std = std / 100
            self.weight.normal_(std=std)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin
        self.k = k
        self.fp64_hyper = fp64_hyper

    def forward(self, input):
        if self.fp64_hyper:
            input = input.double()
        else:
            input = input.float()
        with autocast(enabled=False):  # Do not use fp16
            return mobius_linear(
                input,
                weight=self.weight,
                bias=self.bias,
                hyperbolic_input=self.hyperbolic_input,
                nonlin=self.nonlin,
                hyperbolic_bias=self.hyperbolic_bias,
                k=self.k,
            )
            
    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info

    


class MobiusDist2Hyperplane(torch.nn.Module):
    def __init__(self, in_features, out_features, k=-1.0, fp64_hyper=True):
        k = torch.tensor(k)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball = geoopt.PoincareBall(c=k.abs())
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = torch.nn.Parameter(torch.zeros(out_features))
        point = torch.randn(out_features, in_features) / 4
        point = gmath.expmap0(point, k=k)
        tangent = torch.randn(out_features, in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
        self.fp64_hyper = fp64_hyper
        with torch.no_grad():
            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()

    def forward(self, input):
        if self.fp64_hyper:
            input = input.double()
        else:
            input = input.float()
        with autocast(enabled=False):  # Do not use fp16
            input = input.unsqueeze(-2)
            distance = gmath.dist2plane(
                x=input, p=self.point, a=self.tangent, k=self.ball.c, signed=True
            )
            return distance * self.scale.exp()

    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}"
            #             "c={ball.c}".format(
            #                 **self.__dict__
            #             )
        )
