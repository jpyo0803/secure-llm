import torch
import cupy
import time
import smoothquant.opt

from ctypes import *

import secure_llm.secure_llm_no_sgx2 as non_sgx_lsc
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

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            self.lsc = sgx_lsc.SgxSecureLLM()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            self.lsc =non_sgx_lsc.NonSgxSecureLLM()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            self.lsc = sgx_lsc.SgxSecureLLM()
        else:
            assert False

        self.weight_last_dim = torch_int_nn_linear.weight.shape[-1]
        self.linear_layer_id = self.lsc.Set_Linear_Param_WS8BS8(
                torch_int_nn_linear)
        del torch_int_nn_linear

    def __run(self, x):
        x = self.lsc.Get_Tensor_Int8(x)
        y = torch.rand((x.shape[0], x.shape[1], self.weight_last_dim), dtype=torch.float32)
        y = self.lsc.Set_Tensor_Float(y)
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
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            return self.lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            return self.lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            return self.lsc.Cast_From_Float_To_Int8(super()._Linear_S8W_S8A_S8B_FP32O_Mixed__run(x))
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

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
            self.lsc = sgx_lsc.SgxSecureLLM()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            self.lsc =non_sgx_lsc.NonSgxSecureLLM()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            self.lsc = sgx_lsc.SgxSecureLLM()
        else:
            assert False

        
        self.weight_last_dim = torch_int_nn_linear.weight.shape[-1]
        self.linear_layer_id = self.lsc.Set_Linear_Param_WS8BFP32(
                torch_int_nn_linear)
        del torch_int_nn_linear

    def __run(self, x):
        x = self.lsc.Get_Tensor_Int8(x)
        y = torch.rand((x.shape[0], x.shape[1], self.weight_last_dim), dtype=torch.float32)
        y = self.lsc.Set_Tensor_Float(y)
        return y

    def __call__(self, x):
        start_time = time.perf_counter_ns()
        y = self.__run(x)
        end_time = time.perf_counter_ns()
        dt = (end_time - start_time) / 1e9
        return y, dt
