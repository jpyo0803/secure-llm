import torch
import cupy

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
    return self.__run(x)
  
class Linear_S8W_S8A_S8B_S8O_GPU(Linear_S8W_S8A_S8B_FP32O_GPU):
  def __init__(self, torch_int_nn_linear):
    super().__init__(torch_int_nn_linear)

  def __run(self, x):
    return super()._Linear_S8W_S8A_S8B_FP32O_GPU__run(x).to(torch.int8)
  
  def __call__(self, x):
    return self.__run(x)

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
    return self.__run(x)
    