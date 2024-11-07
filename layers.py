import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _single
from torch.nn.common_types import _size_1_t
from typing import Optional, Tuple
import numpy as np

import torch.nn as nn

class ConstructKernel1d(Module):
    def __init__(
        self,
        out_channels,
        in_channels,
        kernel_count,
        dilated_kernel_size,
        version,
    ):
        super().__init__()
        self.version = version
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.dilated_kernel_size = dilated_kernel_size
        self.kernel_count = kernel_count
        self.IDX = None
        self.lim = None

    def __init_tmp_variables__(self, device):
        if self.IDX is None or self.lim is None:
            I = Parameter(
                torch.arange(0, self.dilated_kernel_size[0]), requires_grad=False
            ).to(device)
            IDX = I.unsqueeze(0)
            IDX = IDX.expand(
                self.out_channels,
                self.in_channels,
                self.kernel_count,
                -1,
                -1,
            ).permute(4, 3, 0, 1, 2)
            self.IDX = IDX
            lim = torch.tensor(self.dilated_kernel_size).to(device)
            self.lim = lim.expand(
                self.out_channels,
                self.in_channels,
                self.kernel_count,
                -1,
            ).permute(3, 0, 1, 2)
        else:
            pass


    def forward_vmax(self, W, P, SIG):
        P = P + self.lim // 2
        SIG = SIG.abs() + 1.0
        X = self.IDX - P
        X = ((SIG - X.abs()).relu()).prod(1)
        X = X / (X.sum(0) + 1e-7)  # normalization
        K = (X * W).sum(-1)
        K = K.permute(1, 2, 0)
        return K

    def forward_vgauss(self, W, P, SIG):
        P = P + self.lim // 2
        SIG = SIG.abs() + 0.27
        X = ((self.IDX - P) / SIG).norm(2, dim=1)
        X = (-0.5 * X**2).exp()
        X = X / (X.sum(0) + 1e-7)  # normalization
        K = (X * W).sum(-1)
        K = K.permute(1, 2, 0)
        return K

    def forward(self, W, P, SIG):
        self.__init_tmp_variables__(W.device)
        if self.version == "max":
            return self.forward_vmax(W, P, SIG)
        elif self.version == "gauss":
            return self.forward_vgauss(W, P, SIG)
        else:
            raise

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_count={kernel_count}, version={version}"
        if self.dilated_kernel_size:
            s += ", dilated_kernel_size={dilated_kernel_size}"
        return s.format(**self.__dict__)


class Dcls1d(Module):
    __constants__ = [

        "dilated_kernel_size",
        "in_channels",
        "out_channels",
        "kernel_count",
        "version",
        "dynamic",
        "dalean"
    ]
    _in_channels: int
    out_channels: int
    kernel_count: int
    dilated_kernel_size: Tuple[int, ...]
    weight: Tensor
    dynamic: bool
    dalean: bool
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_count: int,
        dilated_kernel_size: _size_1_t = 1,
        version: str = "gauss",
        dynamic: bool = True,
        dalean: bool = True
    ):
        
        dilated_kernel_size = _single(dilated_kernel_size)
        super(Dcls1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_count = kernel_count
        self.dilated_kernel_size = dilated_kernel_size
        self.version = version
        self.dynamic = dynamic
        self.weight = Parameter(
            torch.Tensor(out_channels, in_channels, kernel_count)
        )
        if dynamic:
            if dalean:
                self.sign = Parameter(torch.tensor(torch.broadcast_to(torch.sign(torch.from_numpy(np.random.randn(in_channels, kernel_count))), (out_channels, in_channels, kernel_count))))
            else:
                self.sign = Parameter(torch.tensor(torch.sign(torch.from_numpy(np.random.randn(out_channels, in_channels, kernel_count)))))
        self.P = Parameter(
            torch.Tensor(
                len(dilated_kernel_size),
                out_channels,
                in_channels,
                kernel_count,
            )
        )
        self.SIG = Parameter(
            torch.Tensor(
                len(dilated_kernel_size),
                out_channels,
                in_channels,
                kernel_count,
            )
        )

        self.reset_parameters()
        self.DCK = ConstructKernel1d(
            self.out_channels,
            self.in_channels,
            self.kernel_count,
            self.dilated_kernel_size,
            self.version,
        )

    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            for i in range(len(self.dilated_kernel_size)):
                lim = self.dilated_kernel_size[i] // 2
                init.normal_(self.P.select(0, i), 0, 0.5).clamp_(-lim, lim)
            if self.version == "gauss":
                init.constant_(self.SIG, 0.23)
            else:
                init.constant_(self.SIG, 0.0)
    
    def clamp_parameters(self) -> None:
        for i in range(len(self.dilated_kernel_size)):
            with torch.no_grad():
                lim = self.dilated_kernel_size[i] // 2
                self.P.select(0, i).clamp_(-lim, lim)
    
    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_count={kernel_count} (previous kernel_size)"
            ", version={version}"
        )
        if self.dilated_kernel_size != (1,) * len(self.dilated_kernel_size):
            s += ", dilated_kernel_size={dilated_kernel_size} (learnable)"
        return s.format(**self.__dict__)
    
    def __setstate__(self, state):
        super(Dcls1d, self).__setstate__(state)

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        P: Tensor,
        SIG: Optional[Tensor],
    ):
        return F.conv1d(  
            input,
            self.DCK(weight, P, SIG),
            None,
            1,
            0,
            _single(1),
            1,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.dynamic:
            return self._conv_forward(input, self.weight * self.sign.to(self.weight), self.P, self.SIG)
        return self._conv_forward(input, self.weight, self.P, self.SIG)
    

class SurrGradSpike(torch.autograd.Function):
    
    alpha = 5 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return SurrGradSpike.alpha / 2 / (1 + (math.pi / 2 * SurrGradSpike.alpha * input).pow_(2)) * grad_input
    
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SurrGradSpike.apply

class Dropout_Seq(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def create_mask(self, x: Tensor):
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x_seq: Tensor):
        if self.training:
            self.create_mask(x_seq[0])

            return x_seq * self.mask
        else:
            return x_seq

class LIF(nn.Module):
    def __init__(self, tau, v_threshold, detach_reset):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
    def forward(self, inputs: Tensor):
        spk_rec = []
        syn = torch.zeros((inputs.shape[1],inputs.shape[2]), device=inputs.device)
        mem = torch.zeros((inputs.shape[1],inputs.shape[2]), device=inputs.device)
        for t in range(inputs.shape[0]):
            h1 = inputs[t]
            mthr = mem-self.v_threshold
            out = spike_fn(mthr)
            if self.detach_reset:
                rst = out.detach() # We do not want to backprop through the reset

            new_syn = h1
            new_mem =((1-(1/self.tau))*mem +syn)*(1.0-rst)

            spk_rec.append(out)

            mem = new_mem
            syn = new_syn
        return torch.stack(spk_rec,dim=1).permute(1,0,2)

class LI(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
    def forward(self, inputs: Tensor):
        flt = torch.zeros((inputs.shape[1],inputs.shape[2]), device=inputs.device)
        mem = torch.zeros((inputs.shape[1],inputs.shape[2]), device=inputs.device)
        mem_rec = []
        for t in range(inputs.shape[0]):
            new_flt = inputs[t]
            new_mem = (1-(1/self.tau))*mem +flt

            flt = new_flt
            mem = new_mem

            mem_rec.append(mem)
        return torch.stack(mem_rec,dim=1).permute(1,0,2)  
