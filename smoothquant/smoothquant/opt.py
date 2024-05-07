import torch
from torch import nn
from transformers.models.opt.modeling_opt import (
    OPTConfig,
    OPTForCausalLM,
    OPTModel,
    OPTPreTrainedModel,
    OPTLearnedPositionalEmbedding,
    OPTAttention,
    OPTDecoderLayer,
    OPTDecoder,
    BaseModelOutputWithPast,
)
from typing import Optional, Tuple, List
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from torch_int.nn.fused import LayerNormQ
from transformers.utils import logging
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T

import cupy as cp
import numpy as np
import time
import cipher.tensor_conversion as tc
import cipher.cipher_slalom as cs

from enum import Enum


class ExecutionMode(Enum):
    Mode1 = 0  # GPU-only, torch-int
    Mode2 = 1  # GPU-only, Cupy
    Mode3 = 2  # CPU-only, Numpy
    Mode4 = 3  # Mixed, regular torch + torch-int
    Mode5 = 4  # Mixed, Numpy + Cupy
    Mode6 = 5  # Mixed, Mine (Secure)


my_exec_mode = None

logger = logging.get_logger(__name__)


class Int8OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.attention_weight_scale = 1.0

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.k_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.v_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.q_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.out_proj = W8A8BFP32OFP32Linear(embed_dim, embed_dim)

        global my_exec_mode

        if my_exec_mode == ExecutionMode.Mode2:
            self.q_proj_weight_mode2 = None
            self.k_proj_weight_mode2 = None
            self.k_proj_bias_mode2 = None
            self.v_proj_weight_mode2 = None
            self.v_proj_bias_mode2 = None
            self.out_proj_weight_mode2 = None
        elif my_exec_mode == ExecutionMode.Mode3:
            self.q_proj_weight_mode3 = None
            self.k_proj_weight_mode3 = None
            self.k_proj_bias_mode3 = None
            self.v_proj_weight_mode3 = None
            self.v_proj_bias_mode3 = None
            self.out_proj_weight_mode3 = None
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            self.q_proj_weight_mode5 = None
            self.k_proj_weight_mode5 = None
            self.k_proj_bias_mode5 = None
            self.v_proj_weight_mode5 = None
            self.v_proj_bias_mode5 = None
            self.out_proj_weight_mode5 = None
        elif my_exec_mode == ExecutionMode.Mode6:
            self.q_proj_weight_mode6 = None
            self.k_proj_weight_mode6 = None
            self.k_proj_bias_mode6 = None
            self.v_proj_weight_mode6 = None
            self.v_proj_bias_mode6 = None
            self.out_proj_weight_mode6 = None

            self.q_proj_slalom = cs.SlalomLinear()
            self.k_proj_slalom = cs.SlalomLinear()
            self.v_proj_slalom = cs.SlalomLinear()
            self.out_proj_slalom = cs.SlalomLinear()

    @staticmethod
    @torch.no_grad()
    def from_float(
        module: OPTAttention,
        input_scale: float,
        q_output_scale: float,
        k_output_scale: float,
        v_output_scale: float,
        out_input_scale: float,
    ):
        int8_module = Int8OPTAttention(module.embed_dim, module.num_heads)
        # Fuse the scaling into the q_proj output scale
        q_output_scale = q_output_scale * module.scaling
        module.q_proj.weight *= module.scaling
        module.q_proj.bias *= module.scaling

        int8_module.q_proj = W8A8B8O8Linear.from_float(
            module.q_proj, input_scale, q_output_scale
        )
        int8_module.k_proj = W8A8B8O8Linear.from_float(
            module.k_proj, input_scale, k_output_scale
        )
        int8_module.v_proj = W8A8B8O8Linear.from_float(
            module.v_proj, input_scale, v_output_scale
        )
        int8_module.out_proj = W8A8BFP32OFP32Linear.from_float(
            module.out_proj, out_input_scale
        )
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            q_output_scale, k_output_scale)

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, v_output_scale, out_input_scale
        )
        return int8_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        global my_exec_mode
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        if my_exec_mode == ExecutionMode.Mode1:
            query_states = self.q_proj(hidden_states)
        elif my_exec_mode == ExecutionMode.Mode2:
            '''
                GEMM computes C = alpha A * B + beta C, where A, B, and C are matrices.
                A is an M-by-K matrix, B is a K-by-N matrix, and C is an M-by-N matrix
            '''
            if self.q_proj_weight_mode2 is None:
                self.q_proj_weight_mode2 = tc.FromTorchToCupy(
                    self.q_proj.weight.transpose(0, 1).to(torch.int32))

            hidden_states_cp = tc.FromTorchToCupy(
                hidden_states.to(torch.int32))
            query_states = cp.matmul(
                hidden_states_cp, self.q_proj_weight_mode2)
            query_states = tc.FromCupyToTorch(query_states).to(
                torch.float32) * float(self.q_proj.a.item())
            query_states += self.q_proj.bias * float(self.q_proj.b.item())
            query_states = query_states.to(torch.int8)
        elif my_exec_mode == ExecutionMode.Mode3:
            if self.q_proj_weight_mode3 is None:
                self.q_proj_weight_mode3 = tc.FromTorchToNumpy(
                    self.q_proj.weight.transpose(0, 1).to(torch.int32))

            hidden_states_np = tc.FromTorchToNumpy(
                hidden_states.to(torch.int32))
            query_states = np.matmul(
                hidden_states_np, self.q_proj_weight_mode3)
            query_states = tc.FromNumpyToTorch(query_states).to(
                torch.float32) * float(self.q_proj.a.item())
            query_states += self.q_proj.bias * float(self.q_proj.b.item())
            query_states = query_states.to(torch.int8)
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            '''
                Here what it has to do is similar to the mode 2 but the necessary data must 
                be transferred from CPU to GPU and the result from GPU to CPU 
            '''
            if self.q_proj_weight_mode5 is None:
                self.q_proj_weight_mode5 = tc.FromNumpyToCupy(
                    tc.FromTorchToNumpy(self.q_proj.weight.transpose(0, 1))).astype(cp.int32)
                self.q_proj.bias = tc.FromCpuToGPUTorch(self.q_proj.bias)

            assert hidden_states.device == torch.device('cpu')
            hidden_states_cp = tc.FromNumpyToCupy(
                tc.FromTorchToNumpy(hidden_states)).astype(cp.int32)
            query_states = cp.matmul(
                hidden_states_cp, self.q_proj_weight_mode5)
            query_states = tc.FromCupyToTorch(query_states).to(
                torch.float32) * float(self.q_proj.a.item())
            query_states += self.q_proj.bias * float(self.q_proj.b.item())
            query_states = query_states.to(torch.int8)
            query_states = query_states.to(torch.device('cpu'))
            assert query_states.device == torch.device('cpu')
        elif my_exec_mode == ExecutionMode.Mode6:
            hidden_states_i32 = hidden_states.to(torch.int32)
            query_states = self.q_proj_slalom(
                hidden_states_i32, self.q_proj.weight, self.q_proj.bias, self.q_proj.a.item(), self.q_proj.b.item())
            assert query_states.device == torch.device('cpu')
        else:
            assert False

            # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            assert False
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            assert False
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            if my_exec_mode == ExecutionMode.Mode1:
                key_states = self.k_proj(hidden_states, -1, bsz)
                value_states = self.v_proj(hidden_states, -1, bsz)
            elif my_exec_mode == ExecutionMode.Mode2:
                if self.k_proj_weight_mode2 is None:
                    self.k_proj_weight_mode2 = tc.FromTorchToCupy(
                        self.k_proj.weight.transpose(0, 1).to(torch.int32))
                    self.v_proj_weight_mode2 = tc.FromTorchToCupy(
                        self.v_proj.weight.transpose(0, 1).to(torch.int32))
                    self.k_proj_bias_mode2 = self.k_proj.bias.to(torch.float32)
                    self.v_proj_bias_mode2 = self.v_proj.bias.to(torch.float32)
                key_states = cp.matmul(
                    hidden_states_cp, self.k_proj_weight_mode2)
                key_states = tc.FromCupyToTorch(key_states).to(
                    torch.float32) * float(self.k_proj.a.item())
                key_states += self.k_proj_bias_mode2 * \
                    float(self.k_proj.b.item())
                key_states = key_states.to(torch.int8)

                value_states = cp.matmul(
                    hidden_states_cp, self.v_proj_weight_mode2)
                value_states = tc.FromCupyToTorch(value_states).to(
                    torch.float32) * float(self.v_proj.a.item())
                value_states += self.v_proj_bias_mode2 * \
                    float(self.v_proj.b.item())
                value_states = value_states.to(torch.int8)

            elif my_exec_mode == ExecutionMode.Mode3:
                if self.k_proj_weight_mode3 is None:
                    self.k_proj_weight_mode3 = tc.FromTorchToNumpy(
                        self.k_proj.weight.transpose(0, 1).to(torch.int32))
                    self.v_proj_weight_mode3 = tc.FromTorchToNumpy(
                        self.v_proj.weight.transpose(0, 1).to(torch.int32))
                    # self.k_proj_bias_mode3 = self.k_proj.bias.to(torch.float32)
                    # self.v_proj_bias_mode3 = self.v_proj.bias.to(torch.float32)
                key_states = np.matmul(
                    hidden_states_np, self.k_proj_weight_mode3)
                key_states = tc.FromNumpyToTorch(key_states).to(
                    torch.float32) * float(self.k_proj.a.item())
                key_states += self.k_proj.bias * \
                    float(self.k_proj.b.item())
                key_states = key_states.to(torch.int8)

                value_states = np.matmul(
                    hidden_states_np, self.v_proj_weight_mode3)
                value_states = tc.FromNumpyToTorch(value_states).to(
                    torch.float32) * float(self.v_proj.a.item())
                value_states += self.v_proj.bias * \
                    float(self.v_proj.b.item())
                value_states = value_states.to(torch.int8)

            elif my_exec_mode == ExecutionMode.Mode4:
                pass
            elif my_exec_mode == ExecutionMode.Mode5:
                if self.k_proj_weight_mode5 is None:
                    assert False, "Should be reach here"
                    self.k_proj_weight_mode5 = tc.FromNumpyToCupy(
                        tc.FromTorchToNumpy(self.k_proj.weight.transpose(0, 1))).astype(cp.int32)
                    self.v_proj_weight_mode5 = tc.FromNumpyToCupy(
                        tc.FromTorchToNumpy(self.v_proj.weight.transpose(0, 1))).astype(cp.int32)
                    self.k_proj.bias = tc.FromCpuToGPUTorch(self.k_proj.bias)
                    self.v_proj.bias = tc.FromCpuToGPUTorch(self.v_proj.bias)

                key_states = cp.matmul(
                    hidden_states_cp, self.k_proj_weight_mode5)
                key_states = tc.FromCupyToTorch(key_states).to(
                    torch.float32) * float(self.k_proj.a.item())
                key_states += self.k_proj.bias * float(self.k_proj.b.item())
                key_states = key_states.to(torch.int8)
                # key_states = key_states.to(torch.device('cpu'))

                value_states = cp.matmul(
                    hidden_states_cp, self.v_proj_weight_mode5)
                value_states = tc.FromCupyToTorch(value_states).to(
                    torch.float32) * float(self.v_proj.a.item())
                value_states += self.v_proj.bias * float(self.v_proj.b.item())
                value_states = value_states.to(torch.int8)
                # value_states = value_states.to(torch.device('cpu'))
            elif my_exec_mode == ExecutionMode.Mode6:
                if self.k_proj_weight_mode6 is None:
                    assert False, "Should be reach here"
                    self.k_proj_weight_mode6 = tc.FromNumpyToCupy(
                        tc.FromTorchToNumpy(self.k_proj.weight.transpose(0, 1))).astype(cp.int32)
                    self.v_proj_weight_mode6 = tc.FromNumpyToCupy(
                        tc.FromTorchToNumpy(self.v_proj.weight.transpose(0, 1))).astype(cp.int32)
                    self.k_proj.bias = tc.FromCpuToGPUTorch(self.k_proj.bias)
                    self.v_proj.bias = tc.FromCpuToGPUTorch(self.v_proj.bias)

                key_states = self.k_proj_slalom(
                    hidden_states_i32, self.k_proj.weight, self.k_proj.bias, self.k_proj.a.item(), self.k_proj.b.item())
                value_states = self.v_proj_slalom(
                    hidden_states_i32, self.v_proj.weight, self.v_proj.bias, self.v_proj.a.item(), self.v_proj.b.item())
                assert key_states.device == torch.device('cpu')
                assert value_states.device == torch.device('cpu')

            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            if my_exec_mode == ExecutionMode.Mode1:
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
            elif my_exec_mode == ExecutionMode.Mode2:
                if self.k_proj_weight_mode2 is None:
                    self.k_proj_weight_mode2 = tc.FromTorchToCupy(
                        self.k_proj.weight.transpose(0, 1).to(torch.int32))
                    self.v_proj_weight_mode2 = tc.FromTorchToCupy(
                        self.v_proj.weight.transpose(0, 1).to(torch.int32))
                    self.k_proj_bias_mode2 = self.k_proj.bias.to(torch.float32)
                    self.v_proj_bias_mode2 = self.v_proj.bias.to(torch.float32)
                key_states = cp.matmul(
                    hidden_states_cp, self.k_proj_weight_mode2)
                key_states = tc.FromCupyToTorch(key_states).to(
                    torch.float32) * float(self.k_proj.a.item())
                key_states += self.k_proj_bias_mode2 * \
                    float(self.k_proj.b.item())
                key_states = key_states.to(torch.int8)

                value_states = cp.matmul(
                    hidden_states_cp, self.v_proj_weight_mode2)
                value_states = tc.FromCupyToTorch(value_states).to(
                    torch.float32) * float(self.v_proj.a.item())
                value_states += self.v_proj_bias_mode2 * \
                    float(self.v_proj.b.item())
                value_states = value_states.to(torch.int8)
            elif my_exec_mode == ExecutionMode.Mode3:
                if self.k_proj_weight_mode3 is None:
                    self.k_proj_weight_mode3 = tc.FromTorchToNumpy(
                        self.k_proj.weight.transpose(0, 1).to(torch.int32))
                    self.v_proj_weight_mode3 = tc.FromTorchToNumpy(
                        self.v_proj.weight.transpose(0, 1).to(torch.int32))
                    # self.k_proj_bias_mode3 = self.k_proj.bias.to(torch.float32)
                    # self.v_proj_bias_mode3 = self.v_proj.bias.to(torch.float32)
                key_states = np.matmul(
                    hidden_states_np, self.k_proj_weight_mode3)
                key_states = tc.FromNumpyToTorch(key_states).to(
                    torch.float32) * float(self.k_proj.a.item())
                key_states += self.k_proj.bias * \
                    float(self.k_proj.b.item())

                value_states = np.matmul(
                    hidden_states_np, self.v_proj_weight_mode3)
                value_states = tc.FromNumpyToTorch(value_states).to(
                    torch.float32) * float(self.v_proj.a.item())
                value_states += self.v_proj.bias * \
                    float(self.v_proj.b.item())
                value_states = value_states.to(torch.int8)

            elif my_exec_mode == ExecutionMode.Mode4:
                pass
            elif my_exec_mode == ExecutionMode.Mode5:
                if self.k_proj_weight_mode5 is None:
                    self.k_proj_weight_mode5 = tc.FromNumpyToCupy(
                        tc.FromTorchToNumpy(self.k_proj.weight.transpose(0, 1))).astype(cp.int32)
                    self.v_proj_weight_mode5 = tc.FromNumpyToCupy(
                        tc.FromTorchToNumpy(self.v_proj.weight.transpose(0, 1))).astype(cp.int32)
                    self.k_proj.bias = tc.FromCpuToGPUTorch(self.k_proj.bias)
                    self.v_proj.bias = tc.FromCpuToGPUTorch(self.v_proj.bias)

                key_states = cp.matmul(
                    hidden_states_cp, self.k_proj_weight_mode5)
                key_states = tc.FromCupyToTorch(key_states).to(
                    torch.float32) * float(self.k_proj.a.item())
                key_states += self.k_proj.bias * float(self.k_proj.b.item())
                key_states = key_states.to(torch.int8)

                value_states = cp.matmul(
                    hidden_states_cp, self.v_proj_weight_mode5)
                value_states = tc.FromCupyToTorch(value_states).to(
                    torch.float32) * float(self.v_proj.a.item())
                value_states += self.v_proj.bias * float(self.v_proj.b.item())
                value_states = value_states.to(torch.int8)
            elif my_exec_mode == ExecutionMode.Mode6:
                if self.k_proj_weight_mode6 is None:
                    self.k_proj_weight_mode6 = tc.FromNumpyToCupy(
                        tc.FromTorchToNumpy(self.k_proj.weight.transpose(0, 1))).astype(cp.int32)
                    self.v_proj_weight_mode6 = tc.FromNumpyToCupy(
                        tc.FromTorchToNumpy(self.v_proj.weight.transpose(0, 1))).astype(cp.int32)
                    self.k_proj.bias = tc.FromCpuToGPUTorch(self.k_proj.bias)
                    self.v_proj.bias = tc.FromCpuToGPUTorch(self.v_proj.bias)

                key_states = self.k_proj_slalom(
                    hidden_states_i32, self.k_proj.weight, self.k_proj.bias, self.k_proj.a.item(), self.k_proj.b.item())
                value_states = self.v_proj_slalom(
                    hidden_states_i32, self.v_proj.weight, self.v_proj.bias, self.v_proj.a.item(), self.v_proj.b.item())
                assert key_states.device == torch.device('cpu')
                assert value_states.device == torch.device('cpu')

            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)

        '''
            'past_key_value' 
            Mode2: cuda:0
            Mode3: cpu
            Mode5: cuda:0
        '''
        past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(
            query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        '''
            QK BMM
            Input: query_states, key_states are torch.int8
            Output: attn_weights is torch.float32
        '''
        if my_exec_mode == ExecutionMode.Mode1:
            attn_weights = self.qk_bmm(query_states, key_states)
        elif my_exec_mode == ExecutionMode.Mode2:
            query_states_cp = tc.FromTorchToCupy(query_states.to(torch.int32))
            key_states_cp = tc.FromTorchToCupy(
                key_states.to(torch.int32)).transpose(0, 2, 1)
            attn_weights = cp.matmul(query_states_cp, key_states_cp)
            attn_weights = tc.FromCupyToTorch(attn_weights).to(torch.float32)
            attn_weights *= float(self.qk_bmm.a.item())
            assert attn_weights.dtype == torch.float32
        elif my_exec_mode == ExecutionMode.Mode3:
            query_states_np = tc.FromTorchToNumpy(query_states.to(torch.int32))
            key_states_np = tc.FromTorchToNumpy(
                key_states.to(torch.int32)).transpose(0, 2, 1)
            attn_weights = np.matmul(query_states_np, key_states_np)
            attn_weights = tc.FromNumpyToTorch(attn_weights).to(torch.float32)
            attn_weights *= float(self.qk_bmm.a.item())
            assert attn_weights.dtype == torch.float32
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            query_states_cp = tc.FromTorchToCupy(
                tc.FromCpuToGPUTorch(query_states)).astype(cp.int32)
            key_states_cp = tc.FromTorchToCupy(
                key_states.to(torch.int32)).transpose(0, 2, 1)
            attn_weights = cp.matmul(query_states_cp, key_states_cp)
            attn_weights = tc.FromCupyToTorch(attn_weights).to(torch.float32)
            attn_weights *= float(self.qk_bmm.a.item())
            attn_weights = tc.FromGpuToCpuTorch(attn_weights)
            assert attn_weights.dtype == torch.float32
            assert attn_weights.device == torch.device('cpu')
        elif my_exec_mode == ExecutionMode.Mode6:
            query_states_cp = tc.FromTorchToCupy(
                tc.FromCpuToGPUTorch(query_states)).astype(cp.int32)
            key_states_cp = tc.FromNumpyToCupy(
                tc.FromTorchToNumpy(key_states.transpose(1, 2)))
            attn_weights = cp.matmul(query_states_cp, key_states_cp)
            attn_weights = tc.FromCupyToTorch(attn_weights).to(torch.float32)
            attn_weights *= float(self.qk_bmm.a.item())
            attn_weights = tc.FromGpuToCpuTorch(attn_weights)
            assert attn_weights.dtype == torch.float32
            assert attn_weights.device == torch.device('cpu')

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(1, -1, 1, 1) * attn_probs.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_probs = attn_probs.view(
                bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_probs_reshaped = attn_probs.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_probs_reshaped = None

        # (A_row V_row)_row = (A_row V_col ^T)_row
        attn_probs.mul_(127).round_()
        attn_probs = attn_probs.to(torch.int8)

        value_states = value_states.transpose(1, 2).contiguous()

        if my_exec_mode == ExecutionMode.Mode1:
            attn_output = self.pv_bmm(attn_probs, value_states)
        elif my_exec_mode == ExecutionMode.Mode2:
            attn_probs_cp = tc.FromTorchToCupy(attn_probs.to(torch.int32))
            value_states_cp = tc.FromTorchToCupy(
                value_states.to(torch.int32)).transpose(0, 2, 1)
            attn_output = cp.matmul(attn_probs_cp, value_states_cp)
            attn_output = tc.FromCupyToTorch(attn_output).to(
                torch.float32) * float(self.pv_bmm.a.item())
            attn_output = attn_output.to(torch.int8)
        elif my_exec_mode == ExecutionMode.Mode3:
            attn_probs_np = tc.FromTorchToNumpy(attn_probs.to(torch.int32))
            value_states_np = tc.FromTorchToNumpy(
                value_states.to(torch.int32)).transpose(0, 2, 1)
            attn_output = np.matmul(attn_probs_np, value_states_np)
            attn_output = tc.FromNumpyToTorch(attn_output).to(
                torch.float32) * float(self.pv_bmm.a.item())
            attn_output = attn_output.to(torch.int8)
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            attn_probs_cp = tc.FromTorchToCupy(
                tc.FromCpuToGPUTorch(attn_probs)).astype(cp.int32)
            value_states_cp = tc.FromTorchToCupy(
                value_states.to(torch.int32)).transpose(0, 2, 1)
            attn_output = cp.matmul(attn_probs_cp, value_states_cp)
            attn_output = tc.FromCupyToTorch(attn_output).to(
                torch.float32) * float(self.pv_bmm.a.item())
            attn_output = attn_output.to(torch.int8)
            attn_output = attn_output.to(torch.device('cpu'))
            assert attn_output.device == torch.device('cpu')
        elif my_exec_mode == ExecutionMode.Mode6:
            attn_probs_cp = tc.FromTorchToCupy(
                tc.FromCpuToGPUTorch(attn_probs)).astype(cp.int32)
            value_states_cp = tc.FromNumpyToCupy(
                tc.FromTorchToNumpy(value_states.transpose(1, 2)))
            attn_output = cp.matmul(attn_probs_cp, value_states_cp)
            attn_output = tc.FromCupyToTorch(attn_output).to(
                torch.float32) * float(self.pv_bmm.a.item())
            attn_output = attn_output.to(torch.int8)
            attn_output = attn_output.to(torch.device('cpu'))
            assert attn_output.device == torch.device('cpu')

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(
            bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(
            bsz, tgt_len, self.embed_dim).contiguous()

        if my_exec_mode == ExecutionMode.Mode1:
            attn_output = self.out_proj(attn_output)
        elif my_exec_mode == ExecutionMode.Mode2:
            if self.out_proj_weight_mode2 is None:
                self.out_proj_weight_mode2 = tc.FromTorchToCupy(
                    self.out_proj.weight.transpose(0, 1).to(torch.int32))
            attn_output_cp = tc.FromTorchToCupy(attn_output.to(torch.int32))
            attn_output = cp.matmul(attn_output_cp, self.out_proj_weight_mode2)
            attn_output = tc.FromCupyToTorch(attn_output).to(
                torch.float32) * float(self.out_proj.a.item())
            attn_output += self.out_proj.bias
            assert attn_output.dtype == torch.float32
        elif my_exec_mode == ExecutionMode.Mode3:
            if self.out_proj_weight_mode3 is None:
                self.out_proj_weight_mode3 = tc.FromTorchToNumpy(
                    self.out_proj.weight.transpose(0, 1).to(torch.int32))
            attn_output_np = tc.FromTorchToNumpy(attn_output.to(torch.int32))
            attn_output = np.matmul(attn_output_np, self.out_proj_weight_mode3)
            attn_output = tc.FromNumpyToTorch(attn_output).to(
                torch.float32) * float(self.out_proj.a.item())
            attn_output += self.out_proj.bias
            assert attn_output.dtype == torch.float32
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            if self.out_proj_weight_mode5 is None:
                self.out_proj_weight_mode5 = tc.FromNumpyToCupy(
                    tc.FromTorchToNumpy(self.out_proj.weight.transpose(0, 1))).astype(cp.int32)
                self.out_proj.bias = tc.FromCpuToGPUTorch(self.out_proj.bias)

            attn_output_cp = tc.FromNumpyToCupy(
                tc.FromTorchToNumpy(attn_output)).astype(cp.int32)
            attn_output = cp.matmul(attn_output_cp, self.out_proj_weight_mode5)
            attn_output = tc.FromCupyToTorch(attn_output).to(
                torch.float32) * float(self.out_proj.a.item())
            attn_output += self.out_proj.bias
            attn_output = attn_output.to(torch.device('cpu'))
            assert attn_output.device == torch.device('cpu')
            assert attn_output.dtype == torch.float32
        elif my_exec_mode == ExecutionMode.Mode6:
            attn_output = self.out_proj_slalom(
                attn_output, self.out_proj.weight, self.out_proj.bias, self.out_proj.a.item(), 1.0)

            assert attn_output.device == torch.device('cpu')
            assert attn_output.dtype == torch.float32

        return attn_output, attn_probs_reshaped, past_key_value


class Int8OPTDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, ffn_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = Int8OPTAttention(
            embed_dim=self.embed_dim, num_heads=num_attention_heads
        )

        self.self_attn_layer_norm = LayerNormQ(self.embed_dim)
        self.fc1 = W8A8B8O8LinearReLU(self.embed_dim, ffn_dim)
        self.fc2 = W8A8BFP32OFP32Linear(ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNormQ(self.embed_dim)

        self.fc1_relu = nn.ReLU()

        self.fc1_weight_mode2 = None
        self.fc1_bias_mode2 = None
        self.fc2_weight_mode2 = None
        self.fc2_bias_mode2 = None

        self.fc1_weight_mode3 = None
        self.fc1_bias_mode3 = None
        self.fc2_weight_mode3 = None
        self.fc2_bias_mode3 = None

        self.fc1_weight_mode5 = None
        self.fc1_bias_mode5 = None
        self.fc2_weight_mode5 = None
        self.fc2_bias_mode5 = None

        self.fc1_weight_mode6 = None
        self.fc1_bias_mode6 = None
        self.fc2_weight_mode6 = None
        self.fc2_bias_mode6 = None

        self.fc1_slalom = cs.SlalomLinear()
        self.fc2_slalom = cs.SlalomLinear()

    @staticmethod
    def from_float(
        module: OPTDecoderLayer,
        attn_input_scale: float,
        q_output_scale: float,
        k_output_scale: float,
        v_output_scale: float,
        out_input_scale: float,
        fc1_input_scale: float,
        fc2_input_scale: float,
    ):
        int8_module = Int8OPTDecoderLayer(
            module.embed_dim, module.self_attn.num_heads, module.fc1.out_features
        )
        int8_module.self_attn_layer_norm = LayerNormQ.from_float(
            module.self_attn_layer_norm, attn_input_scale
        )
        int8_module.self_attn = Int8OPTAttention.from_float(
            module.self_attn,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            out_input_scale,
        )
        int8_module.final_layer_norm = LayerNormQ.from_float(
            module.final_layer_norm, fc1_input_scale
        )
        int8_module.fc1 = W8A8B8O8LinearReLU.from_float(
            module.fc1, fc1_input_scale, fc2_input_scale
        )
        int8_module.fc2 = W8A8BFP32OFP32Linear.from_float(
            module.fc2, fc2_input_scale)
        return int8_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                          torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.Int8Tensor`): the output of previous layer's layernorm in INT8
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        residual.add_(hidden_states.to(residual.dtype))

        hidden_states = self.final_layer_norm(residual)

        if my_exec_mode == ExecutionMode.Mode1:
            hidden_states = self.fc1(hidden_states)
        elif my_exec_mode == ExecutionMode.Mode2:
            if self.fc1_weight_mode2 is None:
                self.fc1_weight_mode2 = tc.FromTorchToCupy(
                    self.fc1.weight.transpose(0, 1).to(torch.int32))
                self.fc1_bias_mode2 = self.fc1.bias.to(torch.float32)
            hidden_states_cp = tc.FromTorchToCupy(
                hidden_states.to(torch.int32))
            hidden_states_cp = cp.matmul(
                hidden_states_cp, self.fc1_weight_mode2)
            hidden_states = tc.FromCupyToTorch(hidden_states_cp).to(
                torch.float32) * float(self.fc1.a.item())
            hidden_states += self.fc1_bias_mode2 * float(self.fc1.b.item())
            hidden_states = self.fc1_relu(hidden_states)
            hidden_states = hidden_states.to(torch.int8)
        elif my_exec_mode == ExecutionMode.Mode3:
            if self.fc1_weight_mode3 is None:
                self.fc1_weight_mode3 = tc.FromTorchToNumpy(
                    self.fc1.weight.transpose(0, 1).to(torch.int32))
                self.fc1_bias_mode3 = self.fc1.bias.to(torch.float32)
            hidden_states_np = tc.FromTorchToNumpy(
                hidden_states.to(torch.int32))
            hidden_states_np = np.matmul(
                hidden_states_np, self.fc1_weight_mode3)
            hidden_states = tc.FromNumpyToTorch(hidden_states_np).to(
                torch.float32) * float(self.fc1.a.item())
            hidden_states += self.fc1_bias_mode3 * float(self.fc1.b.item())
            hidden_states = self.fc1_relu(hidden_states)
            hidden_states = hidden_states.to(torch.int8)
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            if self.fc1_weight_mode5 is None:
                self.fc1_weight_mode5 = tc.FromNumpyToCupy(
                    tc.FromTorchToNumpy(self.fc1.weight.transpose(0, 1))).astype(cp.int32)
                self.fc1_bias_mode5 = tc.FromCpuToGPUTorch(
                    self.fc1.bias).to(torch.float32)

            assert hidden_states.device == torch.device('cpu')
            hidden_states_cp = tc.FromNumpyToCupy(
                tc.FromTorchToNumpy(hidden_states)).astype(cp.int32)
            hidden_states_cp = cp.matmul(
                hidden_states_cp, self.fc1_weight_mode5)
            hidden_states = tc.FromCupyToTorch(hidden_states_cp).to(
                torch.float32) * float(self.fc1.a.item())
            hidden_states += self.fc1_bias_mode5 * float(self.fc1.b.item())
            hidden_states = self.fc1_relu(hidden_states)
            hidden_states = hidden_states.to(torch.int8)
            hidden_states = hidden_states.to(torch.device('cpu'))
            assert hidden_states.device == torch.device('cpu')
        elif my_exec_mode == ExecutionMode.Mode6:
            hidden_states = self.fc1_slalom(
                hidden_states, self.fc1.weight, self.fc1.bias, self.fc1.a.item(), self.fc1.b.item())
            assert hidden_states.device == torch.device('cpu')

        if my_exec_mode == ExecutionMode.Mode1:
            hidden_states = self.fc2(hidden_states)
        elif my_exec_mode == ExecutionMode.Mode2:
            if self.fc2_weight_mode2 is None:
                self.fc2_weight_mode2 = tc.FromTorchToCupy(
                    self.fc2.weight.transpose(0, 1).to(torch.int32))
                self.fc2_bias_mode2 = self.fc2.bias.to(torch.float32)
            hidden_states_cp = tc.FromTorchToCupy(
                hidden_states.to(torch.int32))
            hidden_states_cp = cp.matmul(
                hidden_states_cp, self.fc2_weight_mode2)
            hidden_states = tc.FromCupyToTorch(hidden_states_cp).to(
                torch.float32) * float(self.fc2.a.item())
            hidden_states += self.fc2_bias_mode2
        elif my_exec_mode == ExecutionMode.Mode3:
            if self.fc2_weight_mode3 is None:
                self.fc2_weight_mode3 = tc.FromTorchToNumpy(
                    self.fc2.weight.transpose(0, 1).to(torch.int32))
                self.fc2_bias_mode3 = self.fc2.bias.to(torch.float32)
            hidden_states_np = tc.FromTorchToNumpy(
                hidden_states.to(torch.int32))
            hidden_states_np = np.matmul(
                hidden_states_np, self.fc2_weight_mode3)
            hidden_states = tc.FromNumpyToTorch(hidden_states_np).to(
                torch.float32) * float(self.fc2.a.item())
            hidden_states += self.fc2_bias_mode3
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            if self.fc2_weight_mode5 is None:
                self.fc2_weight_mode5 = tc.FromNumpyToCupy(
                    tc.FromTorchToNumpy(self.fc2.weight.transpose(0, 1))).astype(cp.int32)
                self.fc2_bias_mode5 = tc.FromCpuToGPUTorch(
                    self.fc2.bias).to(torch.float32)

            assert hidden_states.device == torch.device('cpu')
            hidden_states_cp = tc.FromNumpyToCupy(
                tc.FromTorchToNumpy(hidden_states)).astype(cp.int32)
            hidden_states_cp = cp.matmul(
                hidden_states_cp, self.fc2_weight_mode5)
            hidden_states = tc.FromCupyToTorch(hidden_states_cp).to(
                torch.float32) * float(self.fc2.a.item())
            hidden_states += self.fc2_bias_mode5
            hidden_states = hidden_states.to(torch.device('cpu'))
            assert hidden_states.device == torch.device('cpu')
        elif my_exec_mode == ExecutionMode.Mode6:
            hidden_states = self.fc2_slalom(
                hidden_states, self.fc2.weight, self.fc2.bias, self.fc2.a.item(), 1.0)
            assert hidden_states.device == torch.device('cpu')

        residual.add_(hidden_states.to(residual.dtype))

        outputs = (residual,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Int8OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Int8OPTDecoderLayer`]

    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [
                Int8OPTDecoderLayer(
                    config.hidden_size, config.num_attention_heads, config.ffn_dim
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = OPTDecoder.get_input_embeddings
    set_input_embeddings = OPTDecoder.set_input_embeddings
    _prepare_decoder_attention_mask = OPTDecoder._prepare_decoder_attention_mask
    old_forward = OPTDecoder.forward

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8OPTDecoder(module.config)
        int8_module.embed_tokens = module.embed_tokens
        int8_module.embed_positions = module.embed_positions
        int8_module.project_out = module.project_out
        int8_module.final_layer_norm = module.final_layer_norm
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8OPTDecoderLayer.from_float(
                layer, **decoder_layer_scales[i]
            )
        return int8_module

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        # pad the input to the multiple of 16

        input_len = input_ids.shape[1]
        from torch.nn.functional import pad

        if input_len % 16 != 0:
            pass
            # <pad> is 1
            '''
            padding_len = 16 - input_len % 16
            input_ids = pad(input_ids, (0, padding_len), value=1)
            print("atten mask before : ", attention_mask.size(), input_len)
            if attention_mask is not None:
                attention_mask = pad(attention_mask, (0, padding_len), value=0)
            print("atten mask after : ", attention_mask.size(), input_len)
            '''

        output = self.old_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # slice the output to the original length
        if input_len % 16 != 0:
            output.last_hidden_state = output.last_hidden_state[:,
                                                                :input_len, :]

        return output


class Int8OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = Int8OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = OPTModel.get_input_embeddings
    set_input_embeddings = OPTModel.set_input_embeddings
    get_decoder = OPTModel.get_decoder
    forward = OPTModel.forward

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8OPTModel(module.config)
        int8_module.decoder = Int8OPTDecoder.from_float(
            module.decoder, decoder_layer_scales
        )
        return int8_module


class Int8OPTForCausalLM(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Int8OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8OPTForCausalLM(module.config)
        int8_module.model = Int8OPTModel.from_float(
            module.model, decoder_layer_scales)
        int8_module.lm_head = module.lm_head
        return int8_module

    get_input_embeddings = OPTForCausalLM.get_input_embeddings
    set_input_embeddings = OPTForCausalLM.set_input_embeddings
    get_output_embeddings = OPTForCausalLM.get_output_embeddings
    set_output_embeddings = OPTForCausalLM.set_output_embeddings
    set_decoder = OPTForCausalLM.set_decoder
    get_decoder = OPTForCausalLM.get_decoder
    forward = OPTForCausalLM.forward
    prepare_inputs_for_generation = OPTForCausalLM.prepare_inputs_for_generation
    _reorder_cache = OPTForCausalLM._reorder_cache

    @staticmethod
    def set_exec_mode(exec_mode):
        global my_exec_mode
        my_exec_mode = exec_mode
