from ctypes import *
from ctypes import POINTER

import torch

TRUSTED_LIB = "sgx/App/enclave_bridge.so"

class SgxLayerStructC:
  def __new__(cls, num_enclaves=1):
    if not hasattr(cls, "_instance"):
      print("SgxLayerStructC Instance Created")
      cls._instance = super().__new__(cls)
      cls.lib = cdll.LoadLibrary(TRUSTED_LIB)
      cls.lib.initialize_enclave.restype = c_ulong
      cls.eid = []
      for _ in range(num_enclaves):
        eid = cls.lib.initialize_enclave()
        cls.eid.append(eid)
    return cls._instance

  def __init__(self):
    cls = type(self)
    if not hasattr(cls, "__init"):
      cls.__init = True
  
  @classmethod
  def Destroy(cls):
    if cls.eid is not None:
      cls.lib.destroy_enclave.argtypes = [c_ulong]
      for eid in cls.eid:
        cls.lib.destroy_enclave(eid)
      cls.eid = None

  @classmethod
  def Set_Hidden_States(cls, hidden_states):
    return cls.lib.Sgx_Set_Hidden_States(cls.eid[0], cast(hidden_states.data_ptr(), POINTER(c_float)), hidden_states.size(0), hidden_states.size(1), hidden_states.size(2))

  @classmethod
  def Copy_Hidden_States(cls, src_id):
    return cls.lib.Sgx_Copy_Hidden_States(cls.eid[0],src_id)
  
  @classmethod
  def Set_Layer_Norm_Param(cls, layer_norm):
    return cls.lib.Sgx_Set_Layer_Norm_Param(cls.eid[0],cast(layer_norm.weight.data_ptr(), POINTER(c_float)), cast(layer_norm.bias.data_ptr(), POINTER(c_float)), layer_norm.weight.size(0), layer_norm.weight.size(0))
  
  @classmethod
  def Layer_Norm_Q(cls, src_id, layer_norm_param_id):
    return cls.lib.Sgx_Layer_Norm_Q(cls.eid[0],src_id, layer_norm_param_id)
  
  @classmethod
  def Set_Linear_Param_WS8BS8(cls, linear):
    return cls.lib.Sgx_Set_Linear_Param_WS8BS8(cls.eid[0],cast(linear.weight.data_ptr(), POINTER(c_int8)), cast(linear.bias.data_ptr(), POINTER(c_int8)), linear.weight.size(0), linear.weight.size(1), c_float(linear.a.item()), c_float(linear.b.item()))
  
  @classmethod
  def Set_Linear_Param_WS8BFP32(cls, linear):
    return cls.lib.Sgx_Set_Linear_Param_WS8BFP32(cls.eid[0],cast(linear.weight.data_ptr(), POINTER(c_int8)), cast(linear.bias.data_ptr(), POINTER(c_float)), linear.weight.size(0), linear.weight.size(1), c_float(linear.a.item()))
  
  @classmethod
  def Get_Tensor_Dim_Int32(cls, src_id):
    dim = torch.empty(3, dtype=torch.int32)
    cls.lib.Sgx_Get_Tensor_Dim_Int32(cls.eid[0],src_id, cast(dim.data_ptr(), POINTER(c_int32)))
    return (dim[0], dim[1], dim[2])

  @classmethod
  def Get_Tensor_Int32(cls, src_id):
    B, M, N = cls.Get_Tensor_Dim_Int32(src_id)
    out = torch.empty((B, M, N), dtype=torch.int32)
    cls.lib.Sgx_Get_Tensor_Int32(cls.eid[0],src_id, cast(out.data_ptr(), POINTER(c_int32)))
    return out
  
  @classmethod
  def Set_Tensor_Int32(cls, src):
    return cls.lib.Sgx_Set_Tensor_Int32(cls.eid[0],cast(src.data_ptr(), POINTER(c_int32)), src.size(0), src.size(1), src.size(2))
  
  @classmethod
  def Get_Encrypted_Tensor_Opr1_Int32(cls, src_id, linear_param_id):
      B, M, N = cls.Get_Tensor_Dim_Int32(src_id)
      out = torch.empty((B, M, N), dtype=torch.int32)
      cls.lib.Sgx_Get_Encrypted_Tensor_Opr1_Int32(cls.eid[0],
          src_id, linear_param_id, cast(out.data_ptr(), POINTER(c_int32)))
      return out

  @classmethod
  def Generate_Decryption_Key_Opr1_Int32(cls, blind_factor_id, linear_param_id):
      return cls.lib.Sgx_Generate_Decryption_Key_Opr1_Int32(cls.eid[0],blind_factor_id, linear_param_id)

  @classmethod
  def Set_Decrypted_Tensor_Opr1_Int32(cls, src, linear_param_id):
      return cls.lib.Sgx_Set_Decrypted_Tensor_Opr1_Int32(cls.eid[0],cast(src.data_ptr(), POINTER(c_int32)),
                                                        src.size(0), src.size(
                                                            1), src.size(2), linear_param_id)
  @classmethod
  def Compute_Epilogue_WS8BS8(cls, src_id, linear_param_id):
    return cls.lib.Sgx_Compute_Epilogue_WS8BS8(cls.eid[0],src_id, linear_param_id)
  
  @classmethod
  def Compute_Epilogue_WS8BFP32(cls, src_id, linear_param_id):
    return cls.lib.Sgx_Compute_Epilogue_WS8BFP32(cls.eid[0],src_id, linear_param_id)
  
  @classmethod
  def Compute_Epilogue_BMM(cls, src_id, bmm_id):
    return cls.lib.Sgx_Compute_Epilogue_BMM(cls.eid[0],src_id, bmm_id)
  
  @classmethod
  def ReLU(cls, src_id):
    return cls.lib.Sgx_ReLU(cls.eid[0],src_id)
  
  @classmethod
  def Softmax(cls, src_id):
    return cls.lib.Sgx_Softmax(cls.eid[0],src_id)
  
  @classmethod
  def Quantize_Post_Softmax(cls, src_id):
    return cls.lib.Sgx_Quantize_Post_Softmax(cls.eid[0],src_id)
  
  @classmethod
  def Cast_From_Float_To_Int8(cls, src_id):
    return cls.lib.Sgx_Cast_From_Float_To_Int8(cls.eid[0],src_id)
  
  @classmethod
  def Cast_From_Float_To_Int32(cls, src_id):
    return cls.lib.Sgx_Cast_From_Float_To_Int32(cls.eid[0],src_id)
  
  @classmethod
  def Cast_From_Int8_To_Int32(cls, src_id):
    return cls.lib.Sgx_Cast_From_Int8_To_Int32(cls.eid[0],src_id)
  
  @classmethod
  def Get_Tensor_Dim_Int8(cls, src_id):
    dim = torch.empty(3, dtype=torch.int32)
    cls.lib.Sgx_Get_Tensor_Dim_Int8(cls.eid[0],src_id, cast(dim.data_ptr(), POINTER(c_int32)))
    return (dim[0], dim[1], dim[2])
  
  @classmethod
  def Get_Tensor_Int8(cls, src_id):
    B, M, N = cls.Get_Tensor_Dim_Int8(src_id)
    out = torch.empty((B, M, N), dtype=torch.int8)
    cls.lib.Sgx_Get_Tensor_Int8(cls.eid[0],src_id, cast(out.data_ptr(), POINTER(c_int8)))
    return out
  
  @classmethod
  def Set_Tensor_Int8(cls, src):
    return cls.lib.Sgx_Set_Tensor_Int8(cls.eid[0],cast(src.data_ptr(), POINTER(c_int8)), src.size(0), src.size(1), src.size(2))
  
  @classmethod
  def Get_Tensor_Dim_Float(cls, src_id):
    dim = torch.empty(3, dtype=torch.int32)
    cls.lib.Sgx_Get_Tensor_Dim_Float(cls.eid[0],src_id, cast(dim.data_ptr(), POINTER(c_int32)))
    return (dim[0], dim[1], dim[2])
  
  @classmethod
  def Get_Tensor_Float(cls, src_id):
    B, M, N = cls.Get_Tensor_Dim_Float(src_id)
    out = torch.empty((B, M, N), dtype=torch.float32)
    cls.lib.Sgx_Get_Tensor_Float(cls.eid[0],src_id, cast(out.data_ptr(), POINTER(c_float)))
    return out
  
  @classmethod
  def Set_Tensor_Float(cls, src):
    return cls.lib.Sgx_Set_Tensor_Float(cls.eid[0],cast(src.data_ptr(), POINTER(c_float)), src.size(0), src.size(1), src.size(2))
  
  @classmethod
  def Set_Bmm_Param(cls, bmm):
    return cls.lib.Sgx_Set_Bmm_Param(cls.eid[0],c_float(bmm.a.item()))
  
  @classmethod
  def Residual_Add(cls, src_id1, src_id2):
    return cls.lib.Sgx_Residual_Add(cls.eid[0],src_id1, src_id2)
  
  @classmethod
  def Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(cls, src_id1, src_id2, layer_id):
    B, M, K = cls.Get_Tensor_Dim_Int32(src_id1)
    _, N, K2 = cls.Get_Tensor_Dim_Int32(src_id2)

    enc_x = torch.empty((B, M, K), dtype=torch.int32)
    enc_y = torch.empty((B, N, K2), dtype=torch.int32)

    cls.lib.Sgx_Get_Encrypted_Tensor_QK_Int32_KV_Cache_Opt(cls.eid[0],src_id1, src_id2, cast(enc_x.data_ptr(), POINTER(c_int32)), cast(enc_y.data_ptr(), POINTER(c_int32)), layer_id)
    return enc_x, enc_y
  
  @classmethod
  def Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(cls, src_id1, src_id2, layer_id):
    return cls.lib.Sgx_Generate_Decryption_Key_QK_Int32_KV_Cache_Opt(cls.eid[0],src_id1, src_id2, layer_id)
  
  @classmethod
  def Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(cls, src, decryption_key_id):
    return cls.lib.Sgx_Set_Decrypted_Tensor_QK_Int32_KV_Cache_Opt(cls.eid[0],cast(src.data_ptr(), POINTER(c_int32)), src.size(0), src.size(1), src.size(2), decryption_key_id)

  @classmethod
  def Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(cls, src_id1, src_id2, layer_id):
    B, M, K = cls.Get_Tensor_Dim_Int32(src_id1)
    _, N, K2 = cls.Get_Tensor_Dim_Int32(src_id2)

    enc_x = torch.empty((B, M, K), dtype=torch.int32)
    enc_y = torch.empty((B, N, K2), dtype=torch.int32)

    cls.lib.Sgx_Get_Encrypted_Tensor_PV_Int32_KV_Cache_Opt(cls.eid[0],src_id1, src_id2, cast(enc_x.data_ptr(), POINTER(c_int32)), cast(enc_y.data_ptr(), POINTER(c_int32)), layer_id)
    return enc_x, enc_y
  
  @classmethod
  def Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(cls, src_id1, src_id2, layer_id):
    return cls.lib.Sgx_Generate_Decryption_Key_PV_Int32_KV_Cache_Opt(cls.eid[0],src_id1, src_id2, layer_id)
  
  @classmethod
  def Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(cls, src, decryption_key_id):
    return cls.lib.Sgx_Set_Decrypted_Tensor_PV_Int32_KV_Cache_Opt(cls.eid[0],cast(src.data_ptr(), POINTER(c_int32)), src.size(0), src.size(1), src.size(2), decryption_key_id)

  @classmethod
  def Pre_Init(cls):
    cls.lib.Sgx_Pre_Init(cls.eid[0])

if __name__ == "__main__":
  sgx = SgxLayerStructC()
  print(sgx.eid)
  N = 23
  x = torch.randint(-10, 10, (23, 4, 3), dtype=torch.int8)
  x = sgx.Set_Tensor_Int8(x)
  print("before")
  A, B, C = sgx.Get_Tensor_Dim_Int8(x)
  print(A, B, C)
  # y = sgx.Get_Tensor_Float(x)
  # print("after: ", y)
  sgx.Destroy()