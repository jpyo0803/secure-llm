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

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            self.alpha = torch.tensor(
                torch_int_nn_bmm.a.item(), dtype=torch.float32)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            self.bmm_id = lsc.Set_Bmm_Param(torch_int_nn_bmm)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.bmm_id = lsc.Set_Bmm_Param(torch_int_nn_bmm)

    def __run(self, x, y):
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            assert x.device == torch.device('cpu')
            assert y.device == torch.device('cpu')
            y = y.transpose(-1, -2).contiguous()

            x = x.to(torch.int32)
            y = y.to(torch.int32)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            pass

        # Main computation
        x = x.to(torch.device('cuda:0'))
        y = y.to(torch.device('cuda:0'))
        x = cupy.from_dlpack(x)
        y = cupy.from_dlpack(y)
        z = cupy.matmul(x, y)
        z = torch.from_dlpack(z)
        z = z.to(torch.device('cpu'))

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            z = z.to(torch.float32)
            z *= self.alpha
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            pass

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
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            return super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y).to(torch.int8)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            return lsc.Cast_From_Float_To_Int8(super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            return lsc.Cast_From_Float_To_Int8(super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y))

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
