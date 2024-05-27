from ctypes import *

LAYER_STRUCT_C_LIB_PATH = "./smoothquant/build/liblayer_struct_c.so"

class LayerStructC:
    def __init__(self):
        self.lib = cdll.LoadLibrary(LAYER_STRUCT_C_LIB_PATH)
        self.layer_id = 0

    def SetLayerNormParams(self, layer_norm):
        layer_id = self.layer_id
        self.lib.LS_SetLayerNormParams(cast(layer_norm.weight.data_ptr(), POINTER(c_float)), cast(layer_norm.bias.data_ptr(), POINTER(c_float)), layer_id, c_float(layer_norm.eps))
        self.layer_id += 1
        return layer_id
    
    def LayerNorm(self, x, layer_id):
        self.lib.LS_LayerNorm(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2), layer_id)

    def ReLU(self, x):
        self.lib.LS_ReLU(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2))
    
    def Softmax(self, x):
        self.lib.LS_Softmax(cast(x.data_ptr(), POINTER(c_float)), x.size(0), x.size(1), x.size(2))
