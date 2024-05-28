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

          cls.blind_factor_id = 0
        return cls._instance

    def __init__(self):
        cls = type(self)
        if not hasattr(cls, "__init"):
            cls.__init = True

    @classmethod
    def GetBlindFactorID(cls):
        ret = cls.blind_factor_id
        cls.blind_factor_id += 1
        return ret

    @classmethod
    def SetLayerNormParams(cls, layer_norm):
        layer_id = cls.layer_id
        cls.lib.LS_SetLayerNormParams(cast(layer_norm.weight.data_ptr(), POINTER(c_float)), cast(layer_norm.bias.data_ptr(), POINTER(c_float)), layer_norm.weight.size(0), c_float(layer_norm.eps))
        cls.layer_id += 1
        return layer_id
    
    @classmethod
    def SetLinearParams_I8I8I8(cls, linear):
        linear_id = cls.linear_id_i8i8i8
        weight_transpose = linear.weight.transpose(-2, -1).contiguous()
        cls.lib.LS_SetLinearParams_I8I8I8(cast(weight_transpose.data_ptr(), POINTER(c_char)), cast(linear.bias.data_ptr(), POINTER(c_char)), 
                                            weight_transpose.size(0), weight_transpose.size(1), c_float(linear.a.item()), c_float(linear.b.item()))
        cls.linear_id_i8i8i8 += 1
        return linear_id


    @classmethod
    def SetLinearParams_I8I8FP32FP32(cls, linear):
        linear_id = cls.linear_id_i8i8fp32fp32
        weight_transpose = linear.weight.transpose(-2, -1).contiguous()
        cls.lib.LS_SetLinearParams_I8I8FP32FP32(cast(weight_transpose.data_ptr(), POINTER(c_char)), cast(linear.bias.data_ptr(), POINTER(c_char)), 
                                                weight_transpose.size(0), weight_transpose.size(1), c_float(linear.a.item()), c_float(1.0))
        cls.linear_id_i8i8fp32fp32 += 1
        return linear_id

    @classmethod
    def LayerNorm(cls, x, layer_id):
        cls.lib.LS_LayerNorm(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2), layer_id)

    @classmethod
    def ReLU(cls, x):
        cls.lib.LS_ReLU(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2))
    
    @classmethod
    def Softmax(cls, x):
        cls.lib.LS_Softmax(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2))

    @classmethod
    def ResidualAdd(cls, x, y):
        # Check if dimensions of x and y are equal
        assert x.size() == y.size()
        cls.lib.LS_ResidualAdd(cast(x.data_ptr(), POINTER(c_float)), cast(y.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2))

    @classmethod
    def BlindInputOp1_I8I8I8(cls, x, blind_factor_id):
        cls.lib.LS_Blind_Input_Op1_I8I8I8(cast(x.data_ptr(), POINTER(c_int32)), x.size(0), x.size(1), x.size(2), blind_factor_id)

    @classmethod
    def UnblindOutputOp1_I8I8I8(cls, x, blind_factor_id, linear_id):
        cls.lib.LS_Unblind_Output_Op1_I8I8I8(cast(x.data_ptr(), POINTER(c_int32)), x.size(0), x.size(1), x.size(2), blind_factor_id, linear_id)

    @classmethod
    def BlindInputOp1_I8FP32FP32(cls, x, blind_factor_id):
        cls.lib.LS_Blind_Input_Op1_I8FP32FP32(cast(x.data_ptr(), POINTER(c_int32)), x.size(0), x.size(1), x.size(2), blind_factor_id)

    @classmethod
    def UnblindOutputOp1_I8FP32FP32(cls, x, blind_factor_id, linear_id):
        cls.lib.LS_Unblind_Output_Op1_I8FP32FP32(cast(x.data_ptr(), POINTER(c_int32)), x.size(0), x.size(1), x.size(2), blind_factor_id, linear_id)

    @classmethod
    def SetHiddenStates_Internal(cls, x):
        cls.lib.LS_SetHiddenStatesInternal(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2))
    
    @classmethod
    def CopyResidual1_Internal(cls):
        cls.lib.LS_CopyResidual1Internal()

    @classmethod
    def SelfAttnLayerNormQ_Internal(cls, layer_id):
        cls.lib.LS_SelfAttnLayerNormQInternal(layer_id)

    @classmethod
    def GetSelfAttnLayerNormQ_Internal(cls, x):
        cls.lib.LS_GetSelfAttnLayerNormQInternal(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2))

    @classmethod
    def GetResidual1_Internal(cls, x):
        cls.lib.LS_GetResidual1Internal(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2))

    @classmethod
    def EncryptHiddenStates(cls, x):
        cls.lib.LS_EncryptHiddenStates(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2))