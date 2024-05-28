import torch
import cupy
import time

from ctypes import *

import smoothquant.layer_struct_c as lsc
import smoothquant.opt

lsc = lsc.LayerStructC()

CIPHER_CPP_LIB_PATH = "./smoothquant/build/libcipher_cpp.so"
cipher_cpp_lib = cdll.LoadLibrary(CIPHER_CPP_LIB_PATH)
cipher_cpp_lib.GetCPRNG.argtypes = [POINTER(c_ubyte), c_int]

class BMM_S8X_S8Y_FP32Z_Mixed:
  def __init__(self, torch_int_nn_bmm, privacy_on):
    self.privacy_on = privacy_on
    self.alpha = torch.tensor(torch_int_nn_bmm.a.item(), dtype=torch.float32)
    self.unblind_factor_id_u = lsc.GetBlindFactorID()
    self.unblind_factor_id_v = lsc.GetBlindFactorID()

    self.bmm_id = lsc.SetBmmParams(torch_int_nn_bmm)

  def __run(self, x, y):
    assert x.device == torch.device('cpu')
    assert y.device == torch.device('cpu')
      
    y = y.transpose(-1, -2).contiguous()

    x = x.to(torch.int32)
    y = y.to(torch.int32)

    # Encrypt if needed
    if self.privacy_on:
      lsc.BlindInputOp2_I8I8(x, y, self.unblind_factor_id_u, self.unblind_factor_id_v)

      # blind_factor_u = torch.empty((x.shape[0], 1, x.shape[2]), dtype=torch.int32)
      # blind_factor_v = torch.empty((y.shape[0], y.shape[1], 1), dtype=torch.int32)

      # cipher_cpp_lib.GetCPRNG(cast(blind_factor_u.data_ptr(), POINTER(c_ubyte)), blind_factor_u.numel() * 4)
      # cipher_cpp_lib.GetCPRNG(cast(blind_factor_v.data_ptr(), POINTER(c_ubyte)), blind_factor_v.numel() * 4)

      # unblind_factor_xv = torch.matmul(x, blind_factor_v)
      # unblind_factor_uy = torch.matmul(blind_factor_u, y)

      # unblind_factor_uv = torch.matmul(blind_factor_u, blind_factor_v)

      # x += blind_factor_u
      # y += blind_factor_v

    # Main computation
    x = x.to(torch.device('cuda:0'))
    y = y.to(torch.device('cuda:0'))
    x = cupy.from_dlpack(x)
    y = cupy.from_dlpack(y)
    z = cupy.matmul(x, y)
    z = torch.from_dlpack(z)
    z = z.to(torch.device('cpu'))

    # Decrypt if needed
    if self.privacy_on:
      lsc.UnblindOutputOp2_I8I8(z, self.unblind_factor_id_u, self.unblind_factor_id_v)
      # z -= unblind_factor_xv
      # z -= unblind_factor_uy 
      # z -= unblind_factor_uv

    z = z.to(torch.float32)
    # Compute Epilogue
    if smoothquant.opt.my_exec_mode.value >= smoothquant.opt.ExecMode.Mode4.value:
      lsc.ComputeEpilogue_BMM_I8I8(z, self.bmm_id)
    else:
      z *= self.alpha
    assert z.dtype == torch.float32
    return z
  
  def __call__(self, x, y):
    start_time = time.perf_counter_ns()
    y = self.__run(x, y)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt

class BMM_S8X_S8Y_S8Z_Mixed(BMM_S8X_S8Y_FP32Z_Mixed):
  def __init__(self, torch_int_nn_bmm, privacy_on):
    super().__init__(torch_int_nn_bmm, privacy_on)

  def __run(self, x, y):
    return super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y).to(torch.int8)
  
  def __call__(self, x, y):
    start_time = time.perf_counter_ns()
    y = self.__run(x, y)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt

class BMM_S8X_S8Y_FP32Z_GPU:
  def __init__(self, torch_int_nn_bmm):
    self.alpha = cupy.array(torch_int_nn_bmm.a.item(), dtype=cupy.float32)

  def __run(self, x, y):
    assert x.device == torch.device('cuda:0')
    assert y.device == torch.device('cuda:0')
    y = y.transpose(-1, -2)

    x = x.to(torch.int32)
    y = y.to(torch.int32)
    x = cupy.from_dlpack(x)
    y = cupy.from_dlpack(y)
    z = cupy.matmul(x, y)
    z = z.astype(cupy.float32)
    z *= self.alpha
    z = torch.from_dlpack(z)
    assert z.dtype == torch.float32
    return z
  
  def __call__(self, x, y):
    start_time = time.perf_counter_ns()
    y = self.__run(x, y)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt
  
class BMM_S8X_S8Y_S8Z_GPU(BMM_S8X_S8Y_FP32Z_GPU):
  def __init__(self, torch_int_nn_bmm):
    super().__init__(torch_int_nn_bmm)

  def __run(self, x, y):
    return super()._BMM_S8X_S8Y_FP32Z_GPU__run(x, y).to(torch.int8)
  
  def __call__(self, x, y):
    start_time = time.perf_counter_ns()
    y = self.__run(x, y)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt