from ctypes import *

LAYER_STRUCT_C_LIB_PATH = "./smoothquant/build/libcipher_cpp.so"

class LayerStructC:
    def __new__(cls):
        if not hasattr(cls, "_instance"):
          print("LayerStructC Instance Created")
          cls._instance = super().__new__(cls)

          cls.lib = cdll.LoadLibrary(LAYER_STRUCT_C_LIB_PATH)
          cls.layer_id = 0
          cls.linear_id_i8i8i8 = 0
          cls.linear_id_i8i8fp32fp32 = 0
          cls.bmm_id = 0

          cls.blind_factor_id = 0
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
        weight_transpose = linear.weight.transpose(-2, -1).contiguous()
        return cls.lib.Ex_Set_Linear_Param_WS8BS8(cast(weight_transpose.data_ptr(), POINTER(c_int8)), cast(linear.bias.data_ptr(), POINTER(c_int8)), weight_transpose.size(0), weight_transpose.size(1), c_float(linear.a.item()), c_float(linear.b.item()))
    
    @classmethod
    def Get_Tensor_Dim_Int32(cls, src_id):
        B = M = K = -1
        cls.lib.Ex_Get_Tensor_Dim_Int32(src_id, byref(B), byref(M), byref(K))
        return (B, M, K)
    
    @classmethod
    def Get_Tensor_Int32(cls, src_id, out):
        cls.lib.Ex_Get_Tensor_Int32(src_id, cast(out.data_ptr(), POINTER(c_int32)))

    @classmethod
    def Get_Encrpyted_Tensor_Opr1_Int32(cls, src_id, out):
        return cls.lib.Ex_Get_Encrpyted_Tensor_Int32(src_id, cast(out.data_ptr(), POINTER(c_int32)))

    