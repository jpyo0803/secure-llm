from ctypes import *
import torch
import time

LAYER_STRUCT_C_LIB_PATH = "./smoothquant/build/libcipher_cpp.so"


class LayerStructC:
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            print("LayerStructC Instance Created")
            cls._instance = super().__new__(cls)

            cls.lib = cdll.LoadLibrary(LAYER_STRUCT_C_LIB_PATH)

        return cls._instance

    def __init__(self):
        cls = type(self)
        if not hasattr(cls, "__init"):
            cls.__init = True

    @classmethod
    def Set_Hidden_States(cls, hidden_states):
        return cls.lib.Ex_Set_Hidden_States(cast(hidden_states.data_ptr(), POINTER(c_float)), hidden_states.size(0), hidden_states.size(1), hidden_states.size(2))

    @classmethod
    def Copy_Hidden_States(cls, src_id):
        return cls.lib.Ex_Copy_Hidden_States(src_id)

    @classmethod
    def Set_Layer_Norm_Param(cls, layer_norm):
        return cls.lib.Ex_Set_Layer_Norm_Param(cast(layer_norm.weight.data_ptr(), POINTER(c_float)), cast(layer_norm.bias.data_ptr(), POINTER(c_float)),  layer_norm.weight.size(0), layer_norm.weight.size(0))

    @classmethod
    def Layer_Norm_Q(cls, src_id, layer_norm_param_id):
        return cls.lib.Ex_Layer_Norm_Q(src_id, layer_norm_param_id)

    @classmethod
    def Set_Linear_Param_WS8BS8(cls, linear):
        return cls.lib.Ex_Set_Linear_Param_WS8BS8(cast(linear.weight.data_ptr(), POINTER(c_int8)), cast(linear.bias.data_ptr(), POINTER(c_int8)), linear.weight.size(0), linear.weight.size(1), c_float(linear.a.item()), c_float(linear.b.item()))

    @classmethod
    def Set_Linear_Param_WS8BFP32(cls, linear):
        return cls.lib.Ex_Set_Linear_Param_WS8BFP32(cast(linear.weight.data_ptr(), POINTER(c_int8)), cast(linear.bias.data_ptr(), POINTER(c_float)), linear.weight.size(0), linear.weight.size(1), c_float(linear.a.item()))

    @classmethod
    def Get_Tensor_Dim_Int32(cls, src_id):
        dim = torch.empty(3, dtype=torch.int32)
        cls.lib.Ex_Get_Tensor_Dim_Int32(
            src_id, cast(dim.data_ptr(), POINTER(c_int32)))
        return (dim[0], dim[1], dim[2])

    @classmethod
    def Get_Tensor_Int32(cls, src_id):
        B, M, N = cls.Get_Tensor_Dim_Int32(src_id)
        out = torch.empty((B, M, N), dtype=torch.int32)
        cls.lib.Ex_Get_Tensor_Int32(
            src_id, cast(out.data_ptr(), POINTER(c_int32)))
        return out

    @classmethod
    def Set_Tensor_Int32(cls, src):
        return cls.lib.Ex_Set_Tensor_Int32(cast(src.data_ptr(), POINTER(c_int32)), src.size(0), src.size(1), src.size(2))

    @classmethod
    def Get_Encrypted_Tensor_Opr1_Int32(cls, src_id, linear_param_id):
        B, M, N = cls.Get_Tensor_Dim_Int32(src_id)
        out = torch.empty((B, M, N), dtype=torch.int32)
        cls.lib.Ex_Get_Encrypted_Tensor_Opr1_Int32(
            src_id, linear_param_id, cast(out.data_ptr(), POINTER(c_int32)))
        return out

    @classmethod
    def Generate_Decryption_Key_Opr1_Int32(cls, blind_factor_id, linear_param_id):
        return cls.lib.Ex_Generate_Decryption_Key_Opr1_Int32(blind_factor_id, linear_param_id)

    @classmethod
    def Set_Decrypted_Tensor_Opr1_Int32(cls, src, linear_param_id):
        return cls.lib.Ex_Set_Decrypted_Tensor_Opr1_Int32(cast(src.data_ptr(), POINTER(c_int32)),
                                                          src.size(0), src.size(
                                                              1), src.size(2), linear_param_id)

    @classmethod
    def Get_Encrypted_Tensor_Opr2_Int32(cls, src_id1, src_id2):
        B, M, K = cls.Get_Tensor_Dim_Int32(src_id1)
        _, N, K = cls.Get_Tensor_Dim_Int32(src_id2)

        enc_x = torch.empty((B, M, K), dtype=torch.int32)
        enc_y = torch.empty((B, N, K), dtype=torch.int32) 

        blind_factor_ids = torch.empty(2, dtype=torch.int32)

        start_time = time.perf_counter_ns()
        cls.lib.Ex_Get_Encrypted_Tensor_Opr2_Int32(src_id1, src_id2, cast(
            enc_x.data_ptr(), POINTER(c_int32)), cast(enc_y.data_ptr(), POINTER(c_int32)), cast(blind_factor_ids.data_ptr(), POINTER(c_int32)))
        print(f"Ex_Get_Encrypted_Tensor_Opr2_Int32 End-to-end Latency: {(time.perf_counter_ns() - start_time)/1e9:0.6f} s")
        return enc_x, enc_y, blind_factor_ids[0], blind_factor_ids[1]

    @classmethod
    def Generate_Decryption_Key_Opr2_Int32(cls, src_id1, src_id2, blind_factor_u_id, blind_factor_v_id):
        return cls.lib.Ex_Generate_Decryption_Key_Opr2_Int32(src_id1, src_id2, c_int32(blind_factor_u_id), c_int32(blind_factor_v_id))

    @classmethod
    def Set_Decrypted_Tensor_Opr2_Int32(cls, src, decryption_key_id):
        start_time = time.perf_counter_ns()
        ret_id = cls.lib.Ex_Set_Decrypted_Tensor_Opr2_Int32(cast(src.data_ptr(), POINTER(c_int32)),
                                                          src.size(0), src.size(
                                                              1), src.size(2),
                                                          decryption_key_id)
        print(f"Ex_Set_Decrypted_Tensor_Opr2_Int32 End-to-end Latency: {(time.perf_counter_ns() - start_time)/1e9:0.6f} s")
        return ret_id

    @classmethod
    def Compute_Epilogue_WS8BS8(cls, src_id, linear_param_id):
        return cls.lib.Ex_Compute_Epilogue_WS8BS8(src_id, linear_param_id)

    @classmethod
    def Compute_Epilogue_WS8BFP32(cls, src_id, linear_param_id):
        return cls.lib.Ex_Compute_Epilogue_WS8BFP32(src_id, linear_param_id)

    @classmethod
    def Compute_Epilogue_BMM(cls, src_id, bmm_param_id):
        return cls.lib.Ex_Compute_Epilogue_BMM(src_id, bmm_param_id)

    @classmethod
    def ReLU(cls, src_id):
        return cls.lib.Ex_ReLU(src_id)

    @classmethod
    def Softmax(cls, src_id):
        return cls.lib.Ex_Softmax(src_id)

    @classmethod
    def Quantize_Post_Softmax(cls, src_id):
        return cls.lib.Ex_Quantize_Post_Softmax(src_id)

    @classmethod
    def Cast_From_Float_To_Int8(cls, src_id):
        return cls.lib.Ex_Cast_From_Float_To_Int8(src_id)

    @classmethod
    def Cast_From_Float_To_Int32(cls, src_id):
        return cls.lib.Ex_Cast_From_Float_To_Int32(src_id)

    @classmethod
    def Cast_From_Int8_To_Int32(cls, src_id):
        return cls.lib.Ex_Cast_From_Int8_To_Int32(src_id)

    @ classmethod
    def Get_Tensor_Dim_Int8(cls, src_id):
        dim = torch.empty(3, dtype=torch.int32)
        cls.lib.Ex_Get_Tensor_Dim_Int8(
            src_id, cast(dim.data_ptr(), POINTER(c_int32)))
        return (dim[0], dim[1], dim[2])

    @ classmethod
    def Get_Tensor_Int8(cls, src_id):
        B, M, N = cls.Get_Tensor_Dim_Int8(src_id)
        out = torch.empty((B, M, N), dtype=torch.int8)
        cls.lib.Ex_Get_Tensor_Int8(
            src_id, cast(out.data_ptr(), POINTER(c_int8)))
        return out

    @ classmethod
    def Set_Tensor_Int8(cls, src):
        return cls.lib.Ex_Set_Tensor_Int8(cast(src.data_ptr(), POINTER(c_int8)), src.size(0), src.size(1), src.size(2))

    @classmethod
    def Get_Tensor_Dim_Float(cls, src_id):
        dim = torch.empty(3, dtype=torch.int32)
        cls.lib.Ex_Get_Tensor_Dim_Float(
            src_id, cast(dim.data_ptr(), POINTER(c_int32)))
        return (dim[0], dim[1], dim[2])

    @classmethod
    def Get_Tensor_Float(cls, src_id):
        B, M, N = cls.Get_Tensor_Dim_Float(src_id)
        out = torch.empty((B, M, N), dtype=torch.float32)
        cls.lib.Ex_Get_Tensor_Float(
            src_id, cast(out.data_ptr(), POINTER(c_float)))
        return out

    @classmethod
    def Set_Tensor_Float(cls, src):
        return cls.lib.Ex_Set_Tensor_Float(cast(src.data_ptr(), POINTER(c_float)), src.size(0), src.size(1), src.size(2))

    @ classmethod
    def Set_Bmm_Param(cls, bmm):
        return cls.lib.Ex_Set_Bmm_Param(c_float(bmm.a.item()))

    @classmethod
    def Residual_Add(cls, src_id_1, src_id_2):
        return cls.lib.Ex_Residual_Add(src_id_1, src_id_2)
