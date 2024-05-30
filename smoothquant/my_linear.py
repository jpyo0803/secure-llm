import torch
import cupy
import time
import smoothquant.opt

from ctypes import *

import smoothquant.layer_struct_c as lsc

lsc = lsc.LayerStructC()

import sgx.sgx_layer_struct as sgx_lsc
sgx_lsc = sgx_lsc.SgxLayerStructC()


class Linear_S8W_S8A_S8B_FP32O_Mixed:
    # Only perform matmul in GPU with cupy
    # Other operations done in CPU with torch
    def __init__(self, torch_int_nn_linear, privacy_on):
        self.privacy_on = privacy_on

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            self.weight_cpu = torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous()
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))  # Send to GPU
            self.bias = torch_int_nn_linear.bias  # stay in CPU
            self.alpha = torch.tensor(
                torch_int_nn_linear.a.item(), dtype=torch.float32)
            self.beta = torch.tensor(
                torch_int_nn_linear.b.item(), dtype=torch.float32)
            assert self.bias.device == torch.device('cpu')
            assert self.alpha.dtype == torch.float32
            assert self.beta.dtype == torch.float32
            assert self.alpha.device == torch.device('cpu')
            assert self.beta.device == torch.device('cpu')
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))  # Send to GPU
            # weight 2D
            # bias 1D
            self.linear_layer_id = lsc.Set_Linear_Param_WS8BS8(
                torch_int_nn_linear)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))  # Send to GPU
            # weight 2D
            # bias 1D
            self.linear_layer_id = lsc.Set_Linear_Param_WS8BS8(
                torch_int_nn_linear)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))  # Send to GPU
            # weight 2D
            # bias 1D
            self.linear_layer_id = sgx_lsc.Set_Linear_Param_WS8BS8(
                torch_int_nn_linear)
        else:
            assert False

    def __run(self, x):
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            assert x.device == torch.device('cpu')
            x = x.to(torch.int32)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            x = lsc.Cast_From_Int8_To_Int32(x)
            x = lsc.Get_Tensor_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x = lsc.Cast_From_Int8_To_Int32(x)
            x, blind_factor_id = lsc.Get_Encrypted_Tensor_Opr1_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            x = sgx_lsc.Cast_From_Int8_To_Int32(x)
            x, blind_factor_id = sgx_lsc.Get_Encrypted_Tensor_Opr1_Int32(x)
        else:
            assert False

        # Main computation
        x = x.to(torch.device('cuda:0'))
        x = cupy.from_dlpack(x)
        y = cupy.matmul(x, self.weight)
        y = torch.from_dlpack(y)
        y = y.to(torch.device('cpu'))

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            y = y.to(torch.float32)
            y *= self.alpha
            y += self.beta * self.bias
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            y = lsc.Set_Tensor_Int32(y)
            y = lsc.Compute_Epilogue_WS8BS8(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            y = lsc.Set_Decrypted_Tensor_Opr1_Int32(
                y, blind_factor_id, self.linear_layer_id)
            y = lsc.Compute_Epilogue_WS8BS8(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            y = sgx_lsc.Set_Decrypted_Tensor_Opr1_Int32(
                y, blind_factor_id, self.linear_layer_id)
            y = sgx_lsc.Compute_Epilogue_WS8BS8(y, self.linear_layer_id)
        else:
            assert False

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
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            return super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x).to(torch.int8)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            return lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            return lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            return sgx_lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))

    def __call__(self, x):
        start_time = time.perf_counter_ns()
        y = self.__run(x)
        end_time = time.perf_counter_ns()
        dt = (end_time - start_time) / 1e9
        return y, dt


class Linear_S8W_S8A_FP32B_FP32O_Mixed:
    def __init__(self, torch_int_nn_linear, privacy_on):
        self.privacy_on = privacy_on

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            self.weight_cpu = torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous()
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
            self.bias = torch_int_nn_linear.bias
            self.alpha = torch.tensor(
                torch_int_nn_linear.a.item(), dtype=torch.float32)
            assert self.bias.device == torch.device('cpu')
            assert self.alpha.dtype == torch.float32
            assert self.alpha.device == torch.device('cpu')
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
            self.linear_layer_id = lsc.Set_Linear_Param_WS8BFP32(
                torch_int_nn_linear)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
            self.linear_layer_id = lsc.Set_Linear_Param_WS8BFP32(
                torch_int_nn_linear)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
            self.linear_layer_id = sgx_lsc.Set_Linear_Param_WS8BFP32(
                torch_int_nn_linear)

    def __run(self, x):
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            assert x.device == torch.device('cpu')
            x = x.to(torch.int32)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            x = lsc.Cast_From_Int8_To_Int32(x)
            x = lsc.Get_Tensor_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x = lsc.Cast_From_Int8_To_Int32(x)
            x, blind_factor_id = lsc.Get_Encrypted_Tensor_Opr1_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            x = sgx_lsc.Cast_From_Int8_To_Int32(x)
            x, blind_factor_id = sgx_lsc.Get_Encrypted_Tensor_Opr1_Int32(x)

        # Main computation
        x = x.to(torch.device('cuda:0'))
        x = cupy.from_dlpack(x)
        y = cupy.matmul(x, self.weight)
        y = torch.from_dlpack(y)
        y = y.to(torch.device('cpu'))

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            y = y.to(torch.float32)
            y *= self.alpha
            y += self.bias
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            y = lsc.Set_Tensor_Int32(y)
            y = lsc.Compute_Epilogue_WS8BFP32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            y = lsc.Set_Decrypted_Tensor_Opr1_Int32(
                y, blind_factor_id, self.linear_layer_id)
            y = lsc.Compute_Epilogue_WS8BFP32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            y = sgx_lsc.Set_Decrypted_Tensor_Opr1_Int32(
                y, blind_factor_id, self.linear_layer_id)
            y = sgx_lsc.Compute_Epilogue_WS8BFP32(y, self.linear_layer_id)
        return y

    def __call__(self, x):
        start_time = time.perf_counter_ns()
        y = self.__run(x)
        end_time = time.perf_counter_ns()
        dt = (end_time - start_time) / 1e9
        return y, dt
