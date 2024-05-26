import torch
import cupy

class Linear_S8W_S8A_S8B_FP32O_Mixed:
  # Only perform matmul in GPU with cupy
  # Other operations done in CPU with torch
  def __init__(self, torch_int_nn_linear):
    self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0'))) # Send to GPU
    self.bias = torch_int_nn_linear.bias # stay in CPU
    assert self.bias.device == torch.device('cpu')

    self.alpha = torch.tensor(torch_int_nn_linear.a.item(), dtype=torch.float32)
    self.beta = torch.tensor(torch_int_nn_linear.b.item(), dtype=torch.float32)
    assert self.alpha.dtype == torch.float32
    assert self.beta.dtype == torch.float32
    assert self.alpha.device == torch.device('cpu')
    assert self.beta.device == torch.device('cpu')

  def __run(self, x):
    assert x.device == torch.device('cpu')
    x = x.to(torch.int32)

    # Encrypt if needed

    # Main computation
    x = x.to(torch.device('cuda:0'))
    x = cupy.from_dlpack(x)
    y = cupy.matmul(x, self.weight)
    y = torch.from_dlpack(y)
    y = y.to(torch.device('cpu'))

    # Decrypt if needed


    # Compute Epilogue
    y = y.to(torch.float32)
    y *= self.alpha
    y += self.beta * self.bias
    assert y.dtype == torch.float32
    return y
  
  def __call__(self, x):
    return self.__run(x)

class Linear_S8W_S8A_S8B_S8O_Mixed(Linear_S8W_S8A_S8B_FP32O_Mixed):
  def __init__(self, torch_int_nn_linear):
    super().__init__(torch_int_nn_linear)

  def __run(self, x):
    return super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x).to(torch.int8)
  
  def __call__(self, x):
    return self.__run(x)
  
class Linear_S8W_S8A_FP32B_FP32O_Mixed:
  def __init__(self, torch_int_nn_linear):
    self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
    self.bias = torch_int_nn_linear.bias
    assert self.bias.device == torch.device('cpu')
    self.alpha = torch.tensor(torch_int_nn_linear.a.item(), dtype=torch.float32)
    assert self.alpha.dtype == torch.float32
    assert self.alpha.device == torch.device('cpu')

  def __run(self, x):
    assert x.device == torch.device('cpu')
    x = x.to(torch.int32)

    # Encrypt if needed

    # Main computation
    x = x.to(torch.device('cuda:0'))
    x = cupy.from_dlpack(x)
    y = cupy.matmul(x, self.weight)
    y = torch.from_dlpack(y)
    y = y.to(torch.device('cpu'))

    # Decrypt if needed

    # Compute Epilogue
    y = y.to(torch.float32)
    y *= self.alpha
    y += self.bias
    assert y.dtype == torch.float32
    return y
  
  def __call__(self, x):
    return self.__run(x)

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
    