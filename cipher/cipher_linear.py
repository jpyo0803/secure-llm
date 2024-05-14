import cipher.cipher
import torch
import numpy
import cupy
import time
from torch_int.nn.linear import W8A8B8O8LinearReLU, W8A8B8O8Linear, W8A8B32O32Linear, W8A8BFP32OFP32Linear
import cipher_cpp

'''
    GEMM
    C = alpha * A * B + beta * C
'''

'''
    Linear
    Y = alpha * X * W + beta * bias
'''


class Linear_S8W_S8A_S8B_S8O_Slalom:
    def __init__(self, torch_int_nn_linear):
        ndim = torch_int_nn_linear.weight.ndim

        # In slalom, weight and bias are not protected
        self.weight = cupy.from_dlpack(
            torch_int_nn_linear.weight.transpose(ndim - 2, ndim - 1).to(torch.device('cuda:0')))
        self.bias = torch_int_nn_linear.bias.to(torch.device('cuda:0'))
        self.alpha = torch.tensor(
            torch_int_nn_linear.a.item(), dtype=torch.float32)
        self.beta = torch.tensor(
            torch_int_nn_linear.b.item(), dtype=torch.float32)

        self.r_cache = dict()  # random matrix
        self.u_cache = dict()  # blind factor

    def __run(self, x):
        assert x.device == torch.device('cpu')
        # input x must be prodected, so it starts in CPU

        # st =  time.perf_counter_ns()
        x = x.to(torch.int32)
        # et =  time.perf_counter_ns()
        # print(f"int8 to int32: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        shape = tuple(x.shape)
        if shape not in self.r_cache:
            self.r_cache[shape] = torch.zeros_like(x, dtype=torch.int32)
        # et =  time.perf_counter_ns()
        # print(f"r cache search: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        x += self.r_cache[shape]  # blind input
        # et =  time.perf_counter_ns()
        # print(f"act add: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        x = x.to(torch.device('cuda:0'))  # transfer to GPU
        # et =  time.perf_counter_ns()
        # print(f"cpu to gpu: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        x = cupy.from_dlpack(x)
        # et =  time.perf_counter_ns()
        # print(f"torch2cupy: {(et - st) / 1e9}")
        # st =  time.perf_counter_ns()
        # print(x.shape, self.weight.shape)

        y = cupy.matmul(x, self.weight)
        #cupy.cuda.Stream.null.synchronize()
        # et =  time.perf_counter_ns()
        # print(f"matmul: {(et - st) / 1e9}")
        # st =  time.perf_counter_ns()
        y = torch.from_dlpack(y)
        # et =  time.perf_counter_ns()
        # print(f"cupy2torch: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y = y.to(torch.float32)
        # et =  time.perf_counter_ns()
        # print(f"int32 to float32: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y *= self.alpha
        # et =  time.perf_counter_ns()
        # print(f"alpha mul: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y += self.beta * self.bias
        # et =  time.perf_counter_ns()
        # print(f"bias add: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y = y.to(torch.device('cpu'))  # transfer to CPU
        # et =  time.perf_counter_ns()
        # print(f"gpu to cpu: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        shape = tuple(y.shape)
        if shape not in self.u_cache:
            self.u_cache[shape] = torch.zeros_like(y, dtype=torch.int32)
        # et =  time.perf_counter_ns()
        # print(f"u cache search: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y -= self.u_cache[shape]  # unblind output, alpha * r * W
        # et =  time.perf_counter_ns()
        # print(f"unblind: {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y = y.to(torch.int8)
        # et =  time.perf_counter_ns()
        # print(f"float32 to int8: {(et - st) / 1e9}")

        assert y.device == torch.device('cpu')

        return y

    def __call__(self, x):
        return self.__run(x)


class Linear_S8W_S8A_S8B_S8O_Relu_Slalom:
    def __init__(self, torch_int_nn_linear):
        ndim = torch_int_nn_linear.weight.ndim

        # In slalom, weight and bias are not protected
        self.weight = cupy.from_dlpack(
            torch_int_nn_linear.weight.transpose(ndim - 2, ndim - 1).to(torch.device('cuda:0')))
        self.bias = torch_int_nn_linear.bias.to(torch.device('cuda:0'))
        self.alpha = torch.tensor(
            torch_int_nn_linear.a.item(), dtype=torch.float32)
        self.beta = torch.tensor(
            torch_int_nn_linear.b.item(), dtype=torch.float32)

        self.r_cache = dict()  # random matrix
        self.u_cache = dict()  # blind factor

        self.relu = torch.nn.ReLU()

    def __run(self, x):
        assert x.device == torch.device('cpu')
        # input x must be prodected, so it starts in CPU

        x = x.to(torch.int32)

        shape = tuple(x.shape)
        if shape not in self.r_cache:
            self.r_cache[shape] = torch.zeros_like(x, dtype=torch.int32)

        x += self.r_cache[shape]  # blind input

        x = x.to(torch.device('cuda:0'))  # transfer to GPU

        x = cupy.from_dlpack(x)
        y = cupy.matmul(x, self.weight)
        y = torch.from_dlpack(y)

        y = y.to(torch.float32)
        y *= self.alpha
        y += self.beta * self.bias

        y = y.to(torch.device('cpu'))  # transfer to CPU

        shape = tuple(y.shape)
        if shape not in self.u_cache:
            self.u_cache[shape] = torch.zeros_like(y, dtype=torch.int32)

        y -= self.u_cache[shape]  # unblind output, alpha * r * W

        y = self.relu(y)

        y = y.to(torch.int8)

        assert y.device == torch.device('cpu')

        return y

    def __call__(self, x):
        return self.__run(x)


class Linear_S8W_S8A_F32B_F32O_Slalom:
    def __init__(self, torch_int_nn_linear):
        ndim = torch_int_nn_linear.weight.ndim
        self.weight = cupy.from_dlpack(
            torch_int_nn_linear.weight.transpose(ndim - 2, ndim - 1).to(torch.device('cuda:0')))
        self.bias = torch_int_nn_linear.bias.to(torch.device('cuda:0'))
        self.alpha = torch.tensor(
            torch_int_nn_linear.a.item(), dtype=torch.float32)

        self.r_cache = dict()  # random matrix
        self.u_cache = dict()  # blind factor

    def __run(self, x):
        assert x.device == torch.device('cpu')

        x = x.to(torch.int32)

        shape = tuple(x.shape)
        if shape not in self.r_cache:
            self.r_cache[shape] = torch.zeros_like(x, dtype=torch.int32)

        x += self.r_cache[shape]  # blind input

        x = x.to(torch.device('cuda:0'))  # transfer to GPU

        x = cupy.from_dlpack(x)
        y = cupy.matmul(x, self.weight)
        y = torch.from_dlpack(y)

        y = y.to(torch.float32)
        y *= self.alpha
        y += self.bias

        y = y.to(torch.device('cpu'))  # transfer to CPU

        shape = tuple(y.shape)
        if shape not in self.u_cache:
            self.u_cache[shape] = torch.zeros_like(y, dtype=torch.int32)

        y -= self.u_cache[shape]  # unblind output, alpha * r * W
        assert y.dtype == torch.float32

        assert y.device == torch.device('cpu')

        return y

    def __call__(self, x):
        return self.__run(x)


class Linear_S8W_S8A_S8B_S8O_Unsecure_Gpu:
    def __init__(self, torch_int_nn_linear):
        ndim = torch_int_nn_linear.weight.ndim
        self.weight = cupy.from_dlpack(
            torch_int_nn_linear.weight.transpose(ndim - 2, ndim - 1).to(torch.device('cuda:0')))
        self.bias = torch_int_nn_linear.bias.to(torch.device('cuda:0'))
        self.alpha = torch.tensor(
            torch_int_nn_linear.a.item(), dtype=torch.float32)
        self.beta = torch.tensor(
            torch_int_nn_linear.b.item(), dtype=torch.float32)

    def __run(self, x):
        assert x.device == torch.device('cuda:0')
        # st =  time.perf_counter_ns()
        x = x.to(torch.int32)
        # et =  time.perf_counter_ns()
        # print(f"int8 to int32 (Unsecure): {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        x = cupy.from_dlpack(x)
        # et =  time.perf_counter_ns()
        # print(f"torch2cupy (Unsecure): {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y = cupy.matmul(x, self.weight)
        cupy.cuda.Stream.null.synchronize()
        # et =  time.perf_counter_ns()
        # print(f"matmul (Unsecure): {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y = torch.from_dlpack(y)
        # et =  time.perf_counter_ns()
        # print(f"cupy2torch (Unsecure): {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y = y.to(torch.float32)
        # et =  time.perf_counter_ns()
        # print(f"int32 to float32 (Unsecure): {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y *= self.alpha
        # et =  time.perf_counter_ns()
        # print(f"alpha mul (Unsecure): {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y += self.beta * self.bias
        # et =  time.perf_counter_ns()
        # print(f"bias add (Unsecure): {(et - st) / 1e9}")

        # st =  time.perf_counter_ns()
        y = y.to(torch.int8)
        # et =  time.perf_counter_ns()
        # print(f"float32 to int8 (Unsecure): {(et - st) / 1e9}")
        return y

    def __call__(self, x):
        return self.__run(x)


class Linear_S8W_S8A_S8B_S8O_Relu_Unsecure_Gpu:
    def __init__(self, torch_int_nn_linear):
        ndim = torch_int_nn_linear.weight.ndim
        self.weight = cupy.from_dlpack(
            torch_int_nn_linear.weight.transpose(ndim - 2, ndim - 1).to(torch.device('cuda:0')))
        self.bias = torch_int_nn_linear.bias.to(torch.device('cuda:0'))
        self.alpha = torch.tensor(
            torch_int_nn_linear.a.item(), dtype=torch.float32)
        self.beta = torch.tensor(
            torch_int_nn_linear.b.item(), dtype=torch.float32)
        self.relu = torch.nn.ReLU()

    def __run(self, x):
        assert x.device == torch.device('cuda:0')
        x = x.to(torch.int32)
        x = cupy.from_dlpack(x)
        y = cupy.matmul(x, self.weight)
        y = torch.from_dlpack(y)
        y = y.to(torch.float32)
        y *= self.alpha
        y += self.beta * self.bias
        #y = self.relu(y)
        y_np = y.cpu().numpy()
        cipher_cpp.ReluTensor3D(y_np)
        y = torch.from_numpy(y_np)  
        # y before :  torch.Size([1, 512, 3072]) torch.float32
        # y after :  torch.Size([1, 512, 3072]) torch.float32
        y = y.to(torch.int8)
        return y

    def __call__(self, x):
        return self.__run(x)


class Linear_S8W_S8A_F32B_F32O_Unsecure_Gpu:
    def __init__(self, torch_int_nn_linear):
        ndim = torch_int_nn_linear.weight.ndim
        self.weight = cupy.from_dlpack(
            torch_int_nn_linear.weight.transpose(ndim - 2, ndim - 1).to(torch.device('cuda:0')))
        self.bias = torch_int_nn_linear.bias.to(torch.device('cuda:0'))
        self.alpha = torch.tensor(
            torch_int_nn_linear.a.item(), dtype=torch.float32)

    def __run(self, x):
        assert x.device == torch.device('cuda:0')
        x = x.to(torch.int32)
        x = cupy.from_dlpack(x)
        y = cupy.matmul(x, self.weight)
        y = torch.from_dlpack(y)
        y = y.to(torch.float32)
        y *= self.alpha
        y += self.bias
        return y

    def __call__(self, x):
        return self.__run(x)


class Linear_S8W_S8A_S8B_S8O_Unsecure_Cpu:
    def __init__(self, torch_int_nn_linear):
        ndim = torch_int_nn_linear.weight.ndim
        self.weight = torch_int_nn_linear.weight.transpose(
            ndim - 2, ndim - 1).to(torch.device('cpu')).numpy()
        self.bias = torch_int_nn_linear.bias.to(torch.device('cpu'))
        self.alpha = torch.tensor(
            torch_int_nn_linear.a.item(), dtype=torch.float32)
        self.beta = torch.tensor(
            torch_int_nn_linear.b.item(), dtype=torch.float32)

    def __run(self, x):
        assert x.device == torch.device('cpu')
        x = x.to(torch.int32)
        x = x.numpy()
        y = numpy.matmul(x, self.weight)
        y = torch.from_dlpack(y)
        y = y.to(torch.float32)
        y *= self.alpha
        y += self.beta * self.bias
        y = y.to(torch.int8)
        return y

    def __call__(self, x):
        return self.__run(x)


class Linear_S8W_S8A_S8B_S8O_Relu_Unsecure_Cpu:
    def __init__(self, torch_int_nn_linear):
        ndim = torch_int_nn_linear.weight.ndim
        self.weight = torch_int_nn_linear.weight.transpose(
            ndim - 2, ndim - 1).to(torch.device('cpu')).numpy()
        self.bias = torch_int_nn_linear.bias.to(torch.device('cpu'))
        self.alpha = torch.tensor(
            torch_int_nn_linear.a.item(), dtype=torch.float32)
        self.beta = torch.tensor(
            torch_int_nn_linear.b.item(), dtype=torch.float32)
        self.relu = torch.nn.ReLU()

    def __run(self, x):
        assert x.device == torch.device('cpu')
        x = x.to(torch.int32)
        x = x.numpy()
        y = numpy.matmul(x, self.weight)
        y = torch.from_dlpack(y)
        y = y.to(torch.float32)
        y *= self.alpha
        y += self.beta * self.bias
        y = self.relu(y)
        y = y.to(torch.int8)
        return y

    def __call__(self, x):
        return self.__run(x)


class Linear_S8W_S8A_F32B_F32O_Unsecure_Cpu:
    def __init__(self, torch_int_nn_linear):
        ndim = torch_int_nn_linear.weight.ndim
        self.weight = torch_int_nn_linear.weight.transpose(
            ndim - 2, ndim - 1).to(torch.device('cpu')).numpy()
        self.bias = torch_int_nn_linear.bias.to(torch.device('cpu'))
        self.alpha = torch.tensor(
            torch_int_nn_linear.a.item(), dtype=torch.float32)

    def __run(self, x):
        assert x.device == torch.device('cpu')
        x = x.to(torch.int32)
        x = x.numpy()
        y = numpy.matmul(x, self.weight)
        y = torch.from_dlpack(y)
        y = y.to(torch.float32)
        y *= self.alpha
        y += self.bias
        return y

    def __call__(self, x):
        return self.__run(x)
