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
        self.is_pv_bmm = True if module_name == "PV BMM" else False
        # print(f'{module_name} is created')

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            self.sgx_lsc = sgx_lsc.SgxLayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            self.lsc = lsc.LayerStructC()
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
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
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            self.bmm_id = self.lsc.Set_Bmm_Param(torch_int_nn_bmm)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            self.bmm_id = self.sgx_lsc.Set_Bmm_Param(torch_int_nn_bmm)

        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            self.cache = None
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            self.cache = None


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
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            x = self.lsc.Cast_From_Int8_To_Int32(x)
            y = self.lsc.Cast_From_Int8_To_Int32(y)
            # y is transposed inside
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            x = self.sgx_lsc.Cast_From_Int8_To_Int32(x)
            y = self.sgx_lsc.Cast_From_Int8_To_Int32(y)
            # y is transposed inside
        else:
            assert False
        timer.end(t)
        
        t = timer.start(tag=f'{self.module_name}, Process Input Tensors Before Offload ({state})', category=f'{self.module_name}, Process Input Tensors Before Offload ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            x = self.lsc.Get_Tensor_Int32(x)
            y = self.lsc.Get_Tensor_Int32(y)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            x_copy = x # copy it so that can be passed to Decryption Key generation
            y_copy = y # copy it so that can be passed to Decryption Key generation
            if self.is_pv_bmm:
                x, y, blind_factor_u_id, blind_factor_v_id = self.lsc.Get_Encrypted_Tensor_PV_Int32(x, y)
            else:
                x, y, blind_factor_u_id, blind_factor_v_id = self.lsc.Get_Encrypted_Tensor_QK_Int32(x, y)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            x_copy = x # copy it so that can be passed to Decryption Key generation
            y_copy = y # copy it so that can be passed to Decryption Key generation
            x, y, blind_factor_u_id, blind_factor_v_id = self.sgx_lsc.Get_Encrypted_Tensor_Opr2_Int32(x, y)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            x_copy = x
            y_copy = y

            if self.is_pv_bmm:
                x, y = self.lsc.Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(x, y, self.bmm_id)
            else:
                x, y = self.lsc.Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(x, y, self.bmm_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            x_copy = x
            y_copy = y

            if self.is_pv_bmm:
                x, y = self.sgx_lsc.Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(x, y, self.bmm_id)
            else:
                x, y = self.sgx_lsc.Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(x, y, self.bmm_id)
        else:
            assert False
        timer.end(t)

        t = timer.start(tag=f'{self.module_name}, Generate Decryption Key ({state})', category=f'{self.module_name}, Generate Decryption Key ({state})')
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode3:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode4:
            pass
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            if self.is_pv_bmm:
                decryption_key_id = self.lsc.Generate_Decryption_Key_PV_Int32(x_copy, y_copy, blind_factor_u_id, blind_factor_v_id)
            else:
                decryption_key_id = self.lsc.Generate_Decryption_Key_QK_Int32(x_copy, y_copy, blind_factor_u_id, blind_factor_v_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            decryption_key_id = self.sgx_lsc.Generate_Decryption_Key_Opr2_Int32(x_copy, y_copy, blind_factor_u_id, blind_factor_v_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            if self.is_pv_bmm:
                decryption_key_id = self.lsc.Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(x_copy, y_copy, self.bmm_id)
            else:
                decryption_key_id = self.lsc.Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(x_copy, y_copy, self.bmm_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            if self.is_pv_bmm:
                decryption_key_id = self.sgx_lsc.Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(x_copy, y_copy, self.bmm_id)
            else:
                decryption_key_id = self.sgx_lsc.Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(x_copy, y_copy, self.bmm_id)
        else:
            assert False

        timer.end(t)
        # Main computation
        torch.cuda.synchronize()
        t = timer.start(tag=f'{self.module_name}, Host to Device ({state})', category=f'{self.module_name}, Host to Device ({state})')
        x = x.to(torch.device('cuda:0'))
        y = y.to(torch.device('cuda:0')) # already transposed
        torch.cuda.synchronize()
        timer.end(t)

        torch.cuda.synchronize()
        t = timer.start(tag=f'{self.module_name}, Manage Y ({state})', category=f'{self.module_name}, Manage Y ({state})')

        y = y.transpose(1, 2).contiguous()
        if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            if self.cache is None:
                self.cache = y
            else:
                # print("cache shape vs y shape", self.cache.shape, y.shape)
                if self.is_pv_bmm:
                    self.cache = torch.cat([self.cache, y], dim=1)
                else:
                    self.cache = torch.cat([self.cache, y], dim=2)
            y = self.cache
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            if self.cache is None:
                self.cache = y
            else:
                # print("cache shape vs y shape", self.cache.shape, y.shape)
                if self.is_pv_bmm:
                    self.cache = torch.cat([self.cache, y], dim=1)
                else:
                    self.cache = torch.cat([self.cache, y], dim=2)
            y = self.cache
        
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
            z_test = self.lsc.Get_Tensor_Int32(z)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode5:
            if self.is_pv_bmm:
                z = self.lsc.Set_Decrypted_Tensor_PV_Int32(z, decryption_key_id)
            else: 
                z = self.lsc.Set_Decrypted_Tensor_QK_Int32(z, decryption_key_id)
            z_test = self.lsc.Get_Tensor_Int32(z)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
            z = self.sgx_lsc.Set_Decrypted_Tensor_Opr2_Int32(z, decryption_key_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            if self.is_pv_bmm:
                z = self.lsc.Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(z, decryption_key_id)
            else:
                z = self.lsc.Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(z, decryption_key_id)
            z_test = self.lsc.Get_Tensor_Int32(z)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            if self.is_pv_bmm:
                z = self.sgx_lsc.Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(z, decryption_key_id)
            else:
                z = self.sgx_lsc.Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(z, decryption_key_id)
            z_test = self.sgx_lsc.Get_Tensor_Int32(z)
        else:
            assert False
        timer.end(t)

        # Checksum
        print(f"BMM ID={self.bmm_id}, {self.is_pv_bmm}, sum = {torch.sum(z_test)}, state={state}")
        
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
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            z = self.lsc.Compute_Epilogue_BMM(z, self.bmm_id)
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
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
    
    def cache_len(self):
        return self.cache.shape[2] if self.cache is not None else 0


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
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode7:
            return self.lsc.Cast_From_Float_To_Int8(super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y))
        elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
            return self.sgx_lsc.Cast_From_Float_To_Int8(super()._BMM_S8X_S8Y_FP32Z_Mixed__run(x, y))

    def __call__(self, x, y):
        start_time = time.perf_counter_ns()
        y = self.__run(x, y)
        end_time = time.perf_counter_ns()
        dt = (end_time - start_time) / 1e9
        return y, dt
