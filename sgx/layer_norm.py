import numpy
import torch
from ctypes import *
from ctypes import POINTER
import random

TRUSTED_LIB = "App/enclave_bridge.so"


class SGXLIB(object):

    def __init__(self, num_enclaves=1):
        self.lib = cdll.LoadLibrary(TRUSTED_LIB)
        self.lib.initialize_enclave.restype = c_ulong
        self.eid = []
        for i in range(num_enclaves):
            self.eid.append(self.lib.initialize_enclave())
        self.lib.LayerNorm.argtypes = [c_ulong,POINTER(c_float), POINTER(
        c_float), POINTER(c_float), c_float, c_int, c_int, c_int]

    def destroy(self):
        if self.eid is not None:
            self.lib.destroy_enclave.argtypes = [c_ulong]
            for eid in self.eid:
                self.lib.destroy_enclave(eid)
            self.eid = None

    def LayerNorm(self,x,gamma,beta,eps,B,M,N):
        self.lib.LayerNorm(self.eid[0],cast(x.data_ptr(), POINTER(c_float)),
                cast(gamma.data_ptr(), POINTER(c_float)),
                cast(beta.data_ptr(), POINTER(c_float)),
                eps,B,M,N)

# LayerNorm unittest
B = random.randint(1, 1000)
M = random.randint(1, 1000)
N = random.randint(1, 1000)

eps = 1e-5

# Unittest with numpy
x_np = numpy.random.randn(B, M, N).astype(numpy.float32)
gamma_np = numpy.random.randn(N).astype(numpy.float32)
beta_np = numpy.random.randn(N).astype(numpy.float32)

mean_np = numpy.mean(x_np, axis=2, keepdims=True)
var_np = numpy.var(x_np, axis=2, keepdims=True)
div_np = numpy.sqrt(var_np + eps)
y_np = (x_np - mean_np) / div_np * gamma_np + beta_np
###################################################
# Unittest with torch

x_torch = torch.from_numpy(x_np)
gamma_torch = torch.from_numpy(gamma_np)
beta_torch = torch.from_numpy(beta_np)
y_torch = torch.nn.functional.layer_norm(x_torch, [N], gamma_torch, beta_torch, eps)

are_equal = torch.allclose(y_torch, torch.from_numpy(y_np), atol=1e-5)
print("Are equal: ", are_equal)
assert are_equal

lib = SGXLIB()
lib.LayerNorm(x_torch,gamma_torch,beta_torch,eps,B,M,N)
are_equal = torch.allclose(y_torch, x_torch, atol=1e-5)
print("Are equal: ", are_equal)
lib.destroy()
