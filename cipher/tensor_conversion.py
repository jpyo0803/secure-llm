import cupy
import torch
import numpy


def FromNumpyToCupy(x: numpy.ndarray) -> cupy.ndarray:
    '''
      tensor is transferred from CPU to GPU
    '''
    return cupy.asarray(x)


def FromCupyToNumpy(x: cupy.ndarray) -> numpy.ndarray:
    '''
      tensor is transferred from CPU to GPU
    '''
    return cupy.asnumpy(x)


def FromTorchToCupy(x: torch.Tensor) -> cupy.ndarray:
    '''
      Notice a tensor is always in the GPU memory
    '''
    return cupy.from_dlpack(x)


def FromCupyToTorch(x: cupy.ndarray) -> torch.Tensor:
    '''
      Notice a tensor is always in the GPU memory
    '''
    return torch.from_dlpack(x)


def FromTorchToNumpy(x: torch.Tensor) -> numpy.ndarray:
    '''
      Notice a tensor is always in the CPU memory
    '''
    return x.numpy()


def FromNumpyToTorch(x: numpy.ndarray) -> torch.Tensor:
    '''
      Notice a tensor is always in the CPU memory
    '''
    return torch.from_numpy(x)


def FromCpuToGPUTorch(x: torch.Tensor) -> torch.Tensor:
    '''
      Notice a tensor is always in the CPU memory
    '''
    return x.to(torch.device('cuda:0'))


def FromGpuToCpuTorch(x: torch.Tensor) -> torch.Tensor:
    '''
      Notice a tensor is always in the CPU memory
    '''
    return x.to(torch.device('cpu'))
