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
            x = lsc.Cast_From_Int8_To_Int32(x)
            y = lsc.Cast_From_Int8_To_Int32(y)
            x = lsc.Get_Tensor_Int32(x)
            y = lsc.Get_Tensor_Int32(y)

            y = y.transpose(-1, -2).contiguous()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x = lsc.Cast_From_Int8_To_Int32(x)
            y = lsc.Cast_From_Int8_To_Int32(y)
            x, y, unblind_factor_id = lsc.Get_Encrypted_Tensor_Opr2_Int32(x, y)
            # y is transposed inside

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
            z = lsc.Set_Tensor_Int32(z)
            z = lsc.Compute_Epilogue_BMM(z, self.bmm_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            z = lsc.Set_Decrypted_Tensor_Opr2_Int32(z, unblind_factor_id)
            z = lsc.Compute_Epilogue_BMM(z, self.bmm_id)

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
