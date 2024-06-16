import torch
import cupy
import time
import smoothquant.opt

from ctypes import *

import secure_llm.secure_llm as lsc
import secure_llm.secure_llm_sgx as sgx_lsc

import singleton_timer as st

timer = st.SingletonTimer()


class Linear_S8W_S8A_S8B_FP32O_Mixed:
    # Only perform matmul in GPU with cupy
    # Other operations done in CPU with torch
    def __init__(self, torch_int_nn_linear, privacy_on, module_name=None):
        self.privacy_on = privacy_on
        self.module_name = module_name
        # print(f'{module_name} is created')

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            self.sgx_lsc = sgx_lsc.SgxSecureLLM()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            self.sgx_lsc = sgx_lsc.SgxSecureLLM()
        else:
            assert False

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
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))  # Send to GPU
            # weight 2D
            # bias 1D
            self.linear_layer_id = self.lsc.Set_Linear_Param_WS8BS8(
                torch_int_nn_linear)
            del torch_int_nn_linear
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))  # Send to GPU
            # weight 2D
            # bias 1D
            self.linear_layer_id = self.lsc.Set_Linear_Param_WS8BS8(
                torch_int_nn_linear)
            del torch_int_nn_linear
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))  # Send to GPU
            # weight 2D
            # bias 1D
            self.linear_layer_id = self.sgx_lsc.Set_Linear_Param_WS8BS8(
                torch_int_nn_linear)
            del torch_int_nn_linear
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))  # Send to GPU
            # weight 2D
            # bias 1D
            self.linear_layer_id = self.lsc.Set_Linear_Param_WS8BS8(
                torch_int_nn_linear)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
            self.linear_layer_id = self.sgx_lsc.Set_Linear_Param_WS8BS8(
                torch_int_nn_linear)
            del torch_int_nn_linear
        else:
            assert False

    def __run(self, x):
        state = 'Prefill' if smoothquant.opt.is_prefill else 'Generation'

        # Cast from 8 to int32 for convenience, dont include time measure
        t = timer.start(tag=f'{self.module_name}, Cast From Int8 To Int32 ({state})', category=f'{self.module_name}, Cast From Int8 To Int32 ({state})')
        timer.end(t)
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            assert x.device == torch.device('cpu')
            x = x.to(torch.int32)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            x = self.lsc.Cast_From_Int8_To_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x = self.lsc.Cast_From_Int8_To_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            x = self.sgx_lsc.Cast_From_Int8_To_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            x = self.lsc.Cast_From_Int8_To_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            x = self.sgx_lsc.Cast_From_Int8_To_Int32(x)
        else:
            assert False

        t = timer.start(tag=f'{self.module_name}, Process Input Tensor Before Offload ({state})', category=f'{self.module_name}, Process Input Tensor Before Offload ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            x = self.lsc.Get_Tensor_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x = self.lsc.Get_Encrypted_Tensor_Opr1_Int32(x, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            x = self.sgx_lsc.Get_Encrypted_Tensor_Opr1_Int32(x, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            x = self.lsc.Get_Encrypted_Tensor_Opr1_Int32(x, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            x = self.sgx_lsc.Get_Encrypted_Tensor_Opr1_Int32(x, self.linear_layer_id)
        else:
            assert False
        timer.end(t)

        t = timer.start(tag=f'{self.module_name}, Generate Decryption Key ({state})', category=f'{self.module_name}, Generate Decryption Key ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            pass
            # decryption_key_id = self.lsc.Generate_Decryption_Key_Opr1_Int32(blind_factor_id, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            pass
            # decryption_key_id = self.sgx_lsc.Generate_Decryption_Key_Opr1_Int32(blind_factor_id, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            pass
        else:
            assert False
        timer.end(t)

        # Main computation
        torch.cuda.synchronize() # wait for all previous computation
        t = timer.start(tag=f'{self.module_name}, Host to Device ({state})', category=f'{self.module_name}, Host to Device ({state})')
        x = x.to(torch.device('cuda:0'))
        torch.cuda.synchronize()
        timer.end(t)

        x = cupy.from_dlpack(x) # torch to cupy

        cupy.cuda.Stream.null.synchronize() 
        t = timer.start(tag=f'{self.module_name}, Main Computation ({state})', category=f'{self.module_name}, Main Computation ({state})')
        y = cupy.matmul(x, self.weight)
        cupy.cuda.Stream.null.synchronize() 
        timer.end(t)

        y = torch.from_dlpack(y) # cupy to torch

        torch.cuda.synchronize()
        t = timer.start(tag=f'{self.module_name}, Device to Host ({state})', category=f'{self.module_name}, Device to Host ({state})')
        y = y.to(torch.device('cpu'))
        torch.cuda.synchronize()
        timer.end(t)


        t = timer.start(tag=f'{self.module_name}, Process Output Tensor After Offload ({state})', category=f'{self.module_name}, Process Output Tensor After Offload ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            y = self.lsc.Set_Tensor_Int32(y)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            y = self.lsc.Set_Decrypted_Tensor_Opr1_Int32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            y = self.sgx_lsc.Set_Decrypted_Tensor_Opr1_Int32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            y = self.lsc.Set_Decrypted_Tensor_Opr1_Int32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            y = self.sgx_lsc.Set_Decrypted_Tensor_Opr1_Int32(y, self.linear_layer_id)
        else:
            assert False
        timer.end(t)

        if smoothquant.opt.ENABLE_MATMUL_OUTPUT_SUM:
            if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
                y_test = self.lsc.Get_Tensor_Int32(y)
            else:
                y_test = self.sgx_lsc.Get_Tensor_Int32(y)
            smoothquant.opt.check_sum += torch.sum(y_test)

        t = timer.start(tag=f'{self.module_name}, Compute Epilogue ({state})', category=f'{self.module_name}, Compute Epilogue ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            y = y.to(torch.float32)
            y *= self.alpha
            y += self.beta * self.bias
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            y = self.lsc.Compute_Epilogue_WS8BS8(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            y = self.lsc.Compute_Epilogue_WS8BS8(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            y = self.sgx_lsc.Compute_Epilogue_WS8BS8(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            y = self.lsc.Compute_Epilogue_WS8BS8(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            y = self.sgx_lsc.Compute_Epilogue_WS8BS8(y, self.linear_layer_id)
        else:
            assert False
        timer.end(t)

        return y

    def __call__(self, x):
        start_time = time.perf_counter_ns()
        y = self.__run(x)
        end_time = time.perf_counter_ns()
        dt = (end_time - start_time) / 1e9
        return y, dt


class Linear_S8W_S8A_S8B_S8O_Mixed(Linear_S8W_S8A_S8B_FP32O_Mixed):
    def __init__(self, torch_int_nn_linear, privacy_on, module_name=None):
        super().__init__(torch_int_nn_linear, privacy_on, module_name)

    def __run(self, x):
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            return super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x).to(torch.int8)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            return self.lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            return self.lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            return self.sgx_lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            return self.lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            return self.sgx_lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
        else:
            assert False

    def __call__(self, x):
        start_time = time.perf_counter_ns()
        y = self.__run(x)
        end_time = time.perf_counter_ns()
        dt = (end_time - start_time) / 1e9
        return y, dt


class Linear_S8W_S8A_FP32B_FP32O_Mixed:
    def __init__(self, torch_int_nn_linear, privacy_on, module_name=None):
        self.privacy_on = privacy_on
        self.module_name = module_name
        # print(f'{module_name} is created')

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            self.sgx_lsc = sgx_lsc.SgxSecureLLM()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            self.sgx_lsc = sgx_lsc.SgxSecureLLM()
        else:
            assert False

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
            self.linear_layer_id = self.lsc.Set_Linear_Param_WS8BFP32(
                torch_int_nn_linear)
            del torch_int_nn_linear
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
            self.linear_layer_id = self.lsc.Set_Linear_Param_WS8BFP32(
                torch_int_nn_linear)
            del torch_int_nn_linear
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
            self.linear_layer_id = self.sgx_lsc.Set_Linear_Param_WS8BFP32(
                torch_int_nn_linear)
            del torch_int_nn_linear
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
            self.linear_layer_id = self.lsc.Set_Linear_Param_WS8BFP32(
                torch_int_nn_linear)
            del torch_int_nn_linear
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            self.weight = cupy.from_dlpack(torch_int_nn_linear.weight.to(
                torch.int32).transpose(-2, -1).contiguous().to(torch.device('cuda:0')))
            self.linear_layer_id = self.sgx_lsc.Set_Linear_Param_WS8BFP32(
                torch_int_nn_linear)
            del torch_int_nn_linear

    def __run(self, x):
        state = 'Prefill' if smoothquant.opt.is_prefill else 'Generation'

        t = timer.start(tag=f'{self.module_name}, Cast From Int8 To Int32 ({state})', category=f'{self.module_name}, Cast From Int8 To Int32 ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            assert x.device == torch.device('cpu')
            x = x.to(torch.int32)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            x = self.lsc.Cast_From_Int8_To_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x = self.lsc.Cast_From_Int8_To_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            x = self.sgx_lsc.Cast_From_Int8_To_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            x = self.lsc.Cast_From_Int8_To_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            x = self.sgx_lsc.Cast_From_Int8_To_Int32(x)
        else:
            assert False
        timer.end(t)

        t = timer.start(f'{self.module_name}, Process Input Tensor Before Offload ({state})', category=f'{self.module_name}, Process Input Tensor Before Offload ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            x = self.lsc.Get_Tensor_Int32(x)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x = self.lsc.Get_Encrypted_Tensor_Opr1_Int32(x, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            x = self.sgx_lsc.Get_Encrypted_Tensor_Opr1_Int32(x, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            x = self.lsc.Get_Encrypted_Tensor_Opr1_Int32(x, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            x = self.sgx_lsc.Get_Encrypted_Tensor_Opr1_Int32(x, self.linear_layer_id)
        else:
            assert False
        timer.end(t)

        t = timer.start(tag=f'{self.module_name}, Generate Decryption Key ({state})', category=f'{self.module_name}, Generate Decryption Key ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            pass
            # decryption_key_id = self.lsc.Generate_Decryption_Key_Opr1_Int32(blind_factor_id, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            pass
            # decryption_key_id = self.sgx_lsc.Generate_Decryption_Key_Opr1_Int32(blind_factor_id, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            pass
        else:
            assert False
        timer.end(t)

        
        # Main computation

        torch.cuda.synchronize() # wait for all previous computation
        t = timer.start(tag=f'{self.module_name}, Host to Device ({state})', category=f'{self.module_name}, Host to Device ({state})')
        x = x.to(torch.device('cuda:0'))
        torch.cuda.synchronize()
        timer.end(t)

        x = cupy.from_dlpack(x)

        cupy.cuda.Stream.null.synchronize()
        t = timer.start(tag=f'{self.module_name}, Main Computation ({state})', category=f'{self.module_name}, Main Computation ({state})')
        y = cupy.matmul(x, self.weight)
        cupy.cuda.Stream.null.synchronize()
        timer.end(t)

        y = torch.from_dlpack(y)

        torch.cuda.synchronize()
        t = timer.start(tag=f'{self.module_name}, Device to Host ({state})', category=f'{self.module_name}, Device to Host ({state})')
        y = y.to(torch.device('cpu'))
        torch.cuda.synchronize()
        timer.end(t)

        t = timer.start(tag=f'{self.module_name}, Process Output Tensor After Offload ({state})', category=f'{self.module_name}, Process Output Tensor After Offload ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            y = self.lsc.Set_Tensor_Int32(y)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            y = self.lsc.Set_Decrypted_Tensor_Opr1_Int32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            y = self.sgx_lsc.Set_Decrypted_Tensor_Opr1_Int32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            y = self.lsc.Set_Decrypted_Tensor_Opr1_Int32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            y = self.sgx_lsc.Set_Decrypted_Tensor_Opr1_Int32(y, self.linear_layer_id)
        else:
            assert False
        timer.end(t)

        if smoothquant.opt.ENABLE_MATMUL_OUTPUT_SUM:
            if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
                y_test = self.lsc.Get_Tensor_Int32(y)
            else:
                y_test = self.sgx_lsc.Get_Tensor_Int32(y)
            smoothquant.opt.check_sum += torch.sum(y_test)

        t = timer.start(tag=f'{self.module_name}, Compute Epilogue ({state})', category=f'{self.module_name}, Compute Epilogue ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            y = y.to(torch.float32)
            y *= self.alpha
            y += self.bias
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            y = self.lsc.Compute_Epilogue_WS8BFP32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            y = self.lsc.Compute_Epilogue_WS8BFP32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            y = self.sgx_lsc.Compute_Epilogue_WS8BFP32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            y = self.lsc.Compute_Epilogue_WS8BFP32(y, self.linear_layer_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            y = self.sgx_lsc.Compute_Epilogue_WS8BFP32(y, self.linear_layer_id)
        else:
            assert False
        timer.end(t)

        return y

    def __call__(self, x):
        start_time = time.perf_counter_ns()
        y = self.__run(x)
        end_time = time.perf_counter_ns()
        dt = (end_time - start_time) / 1e9
        return y, dt
