import torch
import cupy
import time
import smoothquant.opt

from ctypes import *

import smoothquant.layer_struct_c as lsc

lsc = lsc.LayerStructC()

CIPHER_CPP_LIB_PATH = "./smoothquant/build/libcipher_cpp.so"
cipher_cpp_lib = cdll.LoadLibrary(CIPHER_CPP_LIB_PATH)
cipher_cpp_lib.GetCPRNG.argtypes = [POINTER(c_ubyte), c_int]

class Linear_S8W_S8A_S8B_FP32O_Mixed:
  # Only perform matmul in GPU with cupy
  # Other operations done in CPU with torch
  def __init__(self, torch_int_nn_linear, privacy_on):
    self.privacy_on = privacy_on

    # weight 2D
    # bias 1D

    self.weight_cpu = torch_int_nn_linear.weight.to(torch.int32).transpose(-2, -1).contiguous()
    self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0'))) # Send to GPU
    self.bias = torch_int_nn_linear.bias # stay in CPU
    assert self.bias.device == torch.device('cpu')

    self.alpha = torch.tensor(torch_int_nn_linear.a.item(), dtype=torch.float32)
    self.beta = torch.tensor(torch_int_nn_linear.b.item(), dtype=torch.float32)
    assert self.alpha.dtype == torch.float32
    assert self.beta.dtype == torch.float32
    assert self.alpha.device == torch.device('cpu')
    assert self.beta.device == torch.device('cpu')

    if smoothquant.opt.my_exec_mode.value >= smoothquant.opt.ExecMode.Mode4.value:
      self.linear_id = lsc.SetLinearParams_I8I8I8(torch_int_nn_linear)
      self.blind_factor_id = lsc.GetBlindFactorID()

  def __run(self, x):
    assert x.device == torch.device('cpu')
    x = x.to(torch.int32)

    # Encrypt if needed
    if self.privacy_on:
      # How to generate torch.empty_like of dimension (B, 1, K) if x's dimension is (B, M, K) for a tensor name 'blind_factor'??
      # print("x before: ", x)
      lsc.BlindInputOp1_I8I8I8(x, self.blind_factor_id)

      # blind_factor = torch.empty((x.shape[0], 1, x.shape[2]), dtype=torch.int32)
      # cipher_cpp_lib.GetCPRNG(cast(blind_factor.data_ptr(), POINTER(c_ubyte)), blind_factor.numel() * 4)
      # x += blind_factor

    # Main computation
    x = x.to(torch.device('cuda:0'))
    x = cupy.from_dlpack(x)
    y = cupy.matmul(x, self.weight)
    y = torch.from_dlpack(y)
    y = y.to(torch.device('cpu'))

    # Decrypt if needed
    if self.privacy_on:
      # pass
      # unblind_factor = torch.matmul(blind_factor, self.weight_cpu)
      # y -= unblind_factor
      # print("before y: ", y)
      lsc.UnblindOutputOp1_I8I8I8(y, self.blind_factor_id, self.linear_id)
      # print("after y: ", y)

    # Compute Epilogue
    y = y.to(torch.float32)
    if smoothquant.opt.my_exec_mode.value >= smoothquant.opt.ExecMode.Mode4.value:
      lsc.ComputeEpilogue_I8I8I8(y, self.linear_id)
    else:
      y *= self.alpha
      y += self.beta * self.bias
    assert y.dtype == torch.float32
    return y
  
  def __call__(self, x):
    start_time = time.perf_counter_ns()
    y = self.__run(x)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt

class Linear_S8W_S8A_S8B_S8O_Mixed(Linear_S8W_S8A_S8B_FP32O_Mixed):
  def __init__(self, torch_int_nn_linear, privacy_on):
    super().__init__(torch_int_nn_linear, privacy_on)

  def __run(self, x):
    return super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x).to(torch.int8)
  
  def __call__(self, x):
    start_time = time.perf_counter_ns()
    y = self.__run(x)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt
  
class Linear_S8W_S8A_FP32B_FP32O_Mixed:
  def __init__(self, torch_int_nn_linear, privacy_on):
    self.privacy_on = privacy_on

    self.weight_cpu = torch_int_nn_linear.weight.to(torch.int32).transpose(-2, -1).contiguous()
    self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
    self.bias = torch_int_nn_linear.bias
    assert self.bias.device == torch.device('cpu')
    self.alpha = torch.tensor(torch_int_nn_linear.a.item(), dtype=torch.float32)
    assert self.alpha.dtype == torch.float32
    assert self.alpha.device == torch.device('cpu')


    if smoothquant.opt.my_exec_mode.value >= smoothquant.opt.ExecMode.Mode4.value:
      self.linear_id = lsc.SetLinearParams_I8I8FP32FP32(torch_int_nn_linear)
      self.blind_factor_id = lsc.GetBlindFactorID()


  def __run(self, x):
    assert x.device == torch.device('cpu')
    # Preprocess
    x = x.to(torch.int32)

    # Encrypt if needed
    if self.privacy_on:
      lsc.BlindInputOp1_I8FP32FP32(x, self.blind_factor_id)
      # blind_factor = torch.empty((x.shape[0], 1, x.shape[2]), dtype=torch.int32)
      # cipher_cpp_lib.GetCPRNG(cast(blind_factor.data_ptr(), POINTER(c_ubyte)), blind_factor.numel() * 4)
      # x += blind_factor

    # Main computation
    x = x.to(torch.device('cuda:0'))
    x = cupy.from_dlpack(x)
    y = cupy.matmul(x, self.weight)
    y = torch.from_dlpack(y)
    y = y.to(torch.device('cpu'))

    # Decrypt if needed
    if self.privacy_on:
      lsc.UnblindOutputOp1_I8FP32FP32(y, self.blind_factor_id, self.linear_id)
      # unblind_factor = torch.matmul(blind_factor, self.weight_cpu)
      # y -= unblind_factor

    # Compute Epilogue

    y = y.to(torch.float32)
    if smoothquant.opt.my_exec_mode.value >= smoothquant.opt.ExecMode.Mode4.value:
      lsc.ComputeEpilogue_I8FP32FP32(y, self.linear_id)
    else:
      y *= self.alpha
      y += self.bias
    assert y.dtype == torch.float32
    return y
  
  def __call__(self, x):
    start_time = time.perf_counter_ns()
    y = self.__run(x)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt

class Linear_S8W_S8A_S8B_FP32O_GPU:
  def __init__(self, torch_int_nn_linear):
    ndim = torch_int_nn_linear.weight.ndim
    self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.transpose(ndim - 2, ndim - 1).contiguous().to(torch.device('cuda:0')))
    self.bias = cupy.from_dlpack(torch_int_nn_linear.bias.to(torch.device('cuda:0')))
    self.alpha = cupy.array(torch_int_nn_linear.a.item(), dtype=cupy.float32)
    self.beta = cupy.array(torch_int_nn_linear.b.item(), dtype=cupy.float32)

  def __run(self, x):
    assert x.device == torch.device('cuda:0')
    x = x.to(torch.int32)
    x = cupy.from_dlpack(x)
    y = cupy.matmul(x, self.weight)
    y = y.astype(cupy.float32)
    y *= self.alpha
    y += self.beta * self.bias
    y = torch.from_dlpack(y)
    return y
  
  def __call__(self, x):
    start_time = time.perf_counter_ns()
    y = self.__run(x)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt
  
class Linear_S8W_S8A_S8B_S8O_GPU(Linear_S8W_S8A_S8B_FP32O_GPU):
  def __init__(self, torch_int_nn_linear):
    super().__init__(torch_int_nn_linear)

  def __run(self, x):
    return super()._Linear_S8W_S8A_S8B_FP32O_GPU__run(x).to(torch.int8)
  
  def __call__(self, x):
    start_time = time.perf_counter_ns()
    y = self.__run(x)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt

class Linear_S8W_S8A_FP32B_FP32O_GPU:
  def __init__(self, torch_int_nn_linear):
    ndim = torch_int_nn_linear.weight.ndim
    self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.transpose(ndim - 2, ndim - 1).contiguous().to(torch.device('cuda:0')))
    self.bias = cupy.from_dlpack(torch_int_nn_linear.bias.to(torch.device('cuda:0')))
    assert self.bias.dtype == cupy.float32
    self.alpha = cupy.array(torch_int_nn_linear.a.item(), dtype=cupy.float32)

  def __run(self, x):
    assert x.device == torch.device('cuda:0')
    x = x.to(torch.int32)
    x = cupy.from_dlpack(x)
    y = cupy.matmul(x, self.weight)
    y = y.astype(cupy.float32)
    y *= self.alpha
    y += self.bias
    y = torch.from_dlpack(y)
    assert y.dtype == torch.float32
    return y
  
  def __call__(self, x):
    start_time = time.perf_counter_ns()
    y = self.__run(x)
    end_time = time.perf_counter_ns()
    dt = (end_time - start_time) / 1e9
    return y, dt

    