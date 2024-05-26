import torch
import cupy
import time

class BMM_S8X_S8Y_FP32Z_Mixed:
  def __init__(self, torch_int_nn_bmm):
    self.alpha = torch.tensor(torch_int_nn_bmm.a.item(), dtype=torch.float32)

  def __run(self, x, y):
    assert x.device == torch.device('cpu')
    assert y.device == torch.device('cpu')
    y = y.transpose(-1, -2)

    x = x.to(torch.int32)
    y = y.to(torch.int32)

    # Encrypt if needed

    x = x.to(torch.device('cuda:0'))
    y = y.to(torch.device('cuda:0'))
    x = cupy.from_dlpack(x)
    y = cupy.from_dlpack(y)
    z = cupy.matmul(x, y)
    z = torch.from_dlpack(z)
    z = z.to(torch.device('cpu'))

    # Decrypt if needed

    # Compute Epilogue
    z = z.to(torch.float32)
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
  def __init__(self, torch_int_nn_bmm):
    super().__init__(torch_int_nn_bmm)

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