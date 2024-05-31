import torch
import cupy
import time

from ctypes import *

import smoothquant.opt


import smoothquant.layer_struct_c as lsc
import sgx.sgx_layer_struct as sgx_lsc

import singleton_timer as st

timer = st.SingletonTimer()

CIPHER_CPP_LIB_PATH = "./smoothquant/build/libcipher_cpp.so"
cipher_cpp_lib = cdll.LoadLibrary(CIPHER_CPP_LIB_PATH)
cipher_cpp_lib.GetCPRNG.argtypes = [POINTER(c_ubyte), c_int]


class BMM_S8X_S8Y_FP32Z_Mixed:
    def __init__(self, torch_int_nn_bmm, privacy_on, module_name=None):
        self.privacy_on = privacy_on
        self.module_name = module_name
        # print(f'{module_name} is created')

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            self.sgx_lsc = sgx_lsc.SgxLayerStructC()

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            self.alpha = torch.tensor(
                torch_int_nn_bmm.a.item(), dtype=torch.float32)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            self.bmm_id = self.lsc.Set_Bmm_Param(torch_int_nn_bmm)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.bmm_id = self.lsc.Set_Bmm_Param(torch_int_nn_bmm)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            self.bmm_id = self.sgx_lsc.Set_Bmm_Param(torch_int_nn_bmm)

    def __run(self, x, y):
        state = 'Prefill' if smoothquant.opt.is_prefill else 'Generation'

        t = timer.start(tag=f'{self.module_name}, Cast From Int8 To Int32 ({state})', category=f'{self.module_name}, Cast From Int8 To Int32 ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            assert x.device == torch.device('cpu')
            assert y.device == torch.device('cpu')

            x = x.to(torch.int32)
            y = y.to(torch.int32)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            x = self.lsc.Cast_From_Int8_To_Int32(x)
            y = self.lsc.Cast_From_Int8_To_Int32(y)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x = self.lsc.Cast_From_Int8_To_Int32(x)
            y = self.lsc.Cast_From_Int8_To_Int32(y)
            # y is transposed inside
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            x = self.sgx_lsc.Cast_From_Int8_To_Int32(x)
            y = self.sgx_lsc.Cast_From_Int8_To_Int32(y)
            # y is transposed inside
        else:
            assert False
        timer.end(t)

        t = timer.start(tag=f'{self.module_name}, Process Input Tensors Before Offload ({state})', category=f'{self.module_name}, Process Input Tensors Before Offload ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            y = y.transpose(-1, -2).contiguous()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            x = self.lsc.Get_Tensor_Int32(x)
            y = self.lsc.Get_Tensor_Int32(y)
            y = y.transpose(-1, -2).contiguous()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x_copy = x # copy it so that can be passed to Decryption Key generation
            y_copy = y # copy it so that can be passed to Decryption Key generation
            x, y, blind_factor_u_id, blind_factor_v_id = self.lsc.Get_Encrypted_Tensor_Opr2_Int32(x, y)
            y = y.transpose(-1, -2).contiguous()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            x_copy = x # copy it so that can be passed to Decryption Key generation
            y_copy = y # copy it so that can be passed to Decryption Key generation
            x, y, blind_factor_u_id, blind_factor_v_id = self.sgx_lsc.Get_Encrypted_Tensor_Opr2_Int32(x, y)
            y = y.transpose(-1, -2).contiguous() 
        else:
            assert False
        timer.end(t)

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            decryption_key_id = self.lsc.Generate_Decryption_Key_Opr2_Int32(x_copy, y_copy, blind_factor_u_id, blind_factor_v_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            decryption_key_id = self.sgx_lsc.Generate_Decryption_Key_Opr2_Int32(x_copy, y_copy, blind_factor_u_id, blind_factor_v_id)
        else:
            assert False

        # Main computation

        torch.cuda.synchronize()
        t = timer.start(tag=f'{self.module_name}, Host to Device ({state})', category=f'{self.module_name}, Host to Device ({state})')
        x = x.to(torch.device('cuda:0'))
        y = y.to(torch.device('cuda:0'))
        torch.cuda.synchronize()
        timer.end(t)

        x = cupy.from_dlpack(x)
        y = cupy.from_dlpack(y)

        cupy.cuda.Stream.null.synchronize()
        t = timer.start(tag=f'{self.module_name}, GPU Computation ({state})', category=f'{self.module_name}, GPU Computation ({state})')
        z = cupy.matmul(x, y)
        cupy.cuda.Stream.null.synchronize()
        timer.end(t)

        z = torch.from_dlpack(z)

        torch.cuda.synchronize()
        t = timer.start(tag=f'{self.module_name}, Device to Host ({state})', category=f'{self.module_name}, Device to Host ({state})')
        z = z.to(torch.device('cpu'))
        torch.cuda.synchronize()
        timer.end(t)

        t = timer.start(tag=f'{self.module_name}, Process Output Tensors After Offload ({state})', category=f'{self.module_name}, Process Output Tensors After Offload ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            z = self.lsc.Set_Tensor_Int32(z)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            z = self.lsc.Set_Decrypted_Tensor_Opr2_Int32(z, decryption_key_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            z = self.sgx_lsc.Set_Decrypted_Tensor_Opr2_Int32(z, decryption_key_id)
        else:
            assert False
        timer.end(t)

        t = timer.start(tag=f'{self.module_name}, Compute Epilogue ({state})', category=f'{self.module_name}, Compute Epilogue ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            z = z.to(torch.float32)
            z *= self.alpha
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            z = self.lsc.Compute_Epilogue_BMM(z, self.bmm_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            z = self.lsc.Compute_Epilogue_BMM(z, self.bmm_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            z = self.sgx_lsc.Compute_Epilogue_BMM(z, self.bmm_id)
        else:
            assert False
        timer.end(t)

        return z

    def __call__(self, x, y):
        start_time = time.perf_counter_ns()
        y = self.__run(x, y)
        end_time = time.perf_counter_ns()
        dt = (end_time - start_time) / 1e9
        return y, dt


class BMM_S8X_S8Y_S8Z_Mixed(BMM_S8X_S8Y_FP32Z_Mixed):
    def __init__(self, torch_int_nn_bmm, privacy_on, module_name = None):
        super().__init__(torch_int_nn_bmm, privacy_on, module_name)

    def __run(self, x, y):
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            return super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y).to(torch.int8)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            return self.lsc.Cast_From_Float_To_Int8(super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            return self.lsc.Cast_From_Float_To_Int8(super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            return self.sgx_lsc.Cast_From_Float_To_Int8(super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y))

    def __call__(self, x, y):
        start_time = time.perf_counter_ns()
        y = self.__run(x, y)
        end_time = time.perf_counter_ns()
        dt = (end_time - start_time) / 1e9
        return y, dt
