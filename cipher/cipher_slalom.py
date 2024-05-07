import numpy as np
import cupy as cp
import torch
import cipher.tensor_conversion as tc
import time
from collections import OrderedDict


class SlalomLinear:
    def __init__(self):
        self.weight_T = None
        self.bias = None
        self.r_cache = OrderedDict()
        self.u_cache = OrderedDict()  # blind factor

    def __run(self, activation: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, alpha: float, beta: float):
        # Generate zero tensor that matches the dimension of x
        if self.weight_T is None:
            self.weight_T = weight.transpose(0, 1).to(
                torch.device('cuda:0')).to(torch.int32)
            # either int8 or float32
            self.bias = bias.to(torch.device('cuda:0'))
            assert self.weight_T.device == torch.device('cuda:0')
            assert self.bias.device == torch.device('cuda:0')

        assert activation.device == torch.device('cpu')

        shape = tuple(activation.shape)
        if shape not in self.r_cache:
            self.r_cache[shape] = torch.zeros_like(activation, dtype=torch.int32) 
        act_enc = activation + self.r_cache[shape]

        act_enc_gpu_cp = tc.FromNumpyToCupy(tc.FromTorchToNumpy(act_enc))
        weight_gpu_cp = tc.FromTorchToCupy(self.weight_T)

        z_enc_gpu_cp = cp.matmul(act_enc_gpu_cp, weight_gpu_cp)

        z_enc_gpu = tc.FromCupyToTorch(z_enc_gpu_cp)

        z_enc_gpu = z_enc_gpu.to(torch.float32) * alpha
        z_enc_gpu += self.bias * beta

        z_enc = z_enc_gpu.to(torch.device('cpu'))

        shape = tuple(z_enc.shape)
        if shape not in self.u_cache:
            self.u_cache[shape] = torch.zeros_like(z_enc, dtype=torch.int32)

        z = z_enc - self.u_cache[shape]

        return z.to(self.bias.dtype)

    def __call__(self, activation: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, alpha: float, beta: float):
        return self.__run(activation, weight, bias, alpha, beta)
