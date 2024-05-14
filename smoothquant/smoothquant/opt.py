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
import cipher.cipher_light as cipher
import cipher.cipher_linear as cl
import singleton_timer as st
import cipher_cpp

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

is_prefill = True


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

        self.my_q_proj = None
        self.my_k_proj = None
        self.my_v_proj = None
        self.my_out_proj = None

        if my_exec_mode == ExecutionMode.Mode6:
            self.secure_qk_bmm = cipher.SBMM_Light_Experimental()
            self.secure_pv_bmm = cipher.SBMM_Light_Experimental()

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
        global my_exec_mode, is_prefill

        timer = st.SingletonTimer(False)
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        if my_exec_mode == ExecutionMode.Mode1:
            t = timer.start(tag=f'Q Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Q Proj ({"Prefill" if is_prefill else "Decode"})')
            query_states = self.q_proj(hidden_states) # WQ
            timer.end(t)
        elif my_exec_mode == ExecutionMode.Mode2:
            '''
                GEMM computes C = alpha A * B + beta C, where A, B, and C are matrices.
                A is an M-by-K matrix, B is a K-by-N matrix, and C is an M-by-N matrix
            '''
            if self.my_q_proj is None:
                self.my_q_proj = cl.Linear_S8W_S8A_S8B_S8O_Unsecure_Gpu(
                    self.q_proj)

            t = timer.start(tag=f'Q Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Q Proj ({"Prefill" if is_prefill else "Decode"})')
            query_states = self.my_q_proj(hidden_states)
            timer.end(t)
        elif my_exec_mode == ExecutionMode.Mode3:
            if self.my_q_proj is None:
                self.my_q_proj = cl.Linear_S8W_S8A_S8B_S8O_Unsecure_Cpu(
                    self.q_proj)

            t = timer.start(tag=f'Q Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Q Proj ({"Prefill" if is_prefill else "Decode"})')
            query_states = self.my_q_proj(hidden_states)
            timer.end(t)
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            '''
                Here what it has to do is similar to the mode 2 but the necessary data must 
                be transferred from CPU to GPU and the result from GPU to CPU 
            '''
            if self.my_q_proj is None:
                self.my_q_proj = cl.Linear_S8W_S8A_S8B_S8O_Unsecure_Gpu(
                    self.q_proj)
            assert hidden_states.device == torch.device('cpu')
            t = timer.start(tag=f'Q Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Q Proj ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = hidden_states.to(torch.device('cuda:0'))
            query_states = self.my_q_proj(hidden_states)
            query_states = query_states.to(torch.device('cpu'))
            timer.end(t)

        elif my_exec_mode == ExecutionMode.Mode6:
            hidden_states_i32 = hidden_states.to(torch.int32)

            if self.my_q_proj is None:
                self.my_q_proj = cl.Linear_S8W_S8A_S8B_S8O_Slalom(self.q_proj)

            t = timer.start(tag=f'Q Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Q Proj ({"Prefill" if is_prefill else "Decode"})')
            query_states = self.my_q_proj(hidden_states_i32)
            timer.end(t)

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
                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                key_states = self.k_proj(hidden_states, -1, bsz) # WK
                value_states = self.v_proj(hidden_states, -1, bsz) # WV
                timer.end(t)
            elif my_exec_mode == ExecutionMode.Mode2:

                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                key_states = self.my_k_proj(hidden_states)
                value_states = self.my_v_proj(hidden_states)
                timer.end(t)

            elif my_exec_mode == ExecutionMode.Mode3:

                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                key_states = self.my_k_proj(hidden_states)
                value_states = self.my_v_proj(hidden_states)
                timer.end(t)

            elif my_exec_mode == ExecutionMode.Mode4:
                pass
            elif my_exec_mode == ExecutionMode.Mode5:
                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                assert hidden_states.device == torch.device('cuda:0')
                key_states = self.my_k_proj(hidden_states)
                key_states = key_states.to(torch.device('cpu'))
                value_states = self.my_v_proj(hidden_states)
                value_states = value_states.to(torch.device('cpu'))
                timer.end(t)
                assert key_states.device == torch.device('cpu')
                assert value_states.device == torch.device('cpu')
                # New KV cache stays in GPU
            elif my_exec_mode == ExecutionMode.Mode6:

                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                key_states = self.my_k_proj(hidden_states_i32)
                value_states = self.my_v_proj(hidden_states_i32)
                timer.end(t)

                assert key_states.device == torch.device('cpu')
                assert value_states.device == torch.device('cpu')

            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # Prefill phase
            if my_exec_mode == ExecutionMode.Mode1:
                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
                timer.end(t)
            elif my_exec_mode == ExecutionMode.Mode2:
                if self.my_k_proj is None:
                    self.my_k_proj = cl.Linear_S8W_S8A_S8B_S8O_Unsecure_Gpu(
                        self.k_proj)
                    self.my_v_proj = cl.Linear_S8W_S8A_S8B_S8O_Unsecure_Gpu(
                        self.v_proj)

                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                key_states = self.my_k_proj(hidden_states)
                value_states = self.my_v_proj(hidden_states)
                timer.end(t)
            elif my_exec_mode == ExecutionMode.Mode3:
                if self.my_k_proj is None:
                    self.my_k_proj = cl.Linear_S8W_S8A_S8B_S8O_Unsecure_Cpu(
                        self.k_proj)
                    self.my_v_proj = cl.Linear_S8W_S8A_S8B_S8O_Unsecure_Cpu(
                        self.v_proj)

                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                key_states = self.my_k_proj(hidden_states)
                value_states = self.my_v_proj(hidden_states)
                timer.end(t)

            elif my_exec_mode == ExecutionMode.Mode4:
                pass
            elif my_exec_mode == ExecutionMode.Mode5:
                if self.my_k_proj is None:
                    self.my_k_proj = cl.Linear_S8W_S8A_S8B_S8O_Unsecure_Gpu(
                        self.k_proj)
                    self.my_v_proj = cl.Linear_S8W_S8A_S8B_S8O_Unsecure_Gpu(
                        self.v_proj)

                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                assert hidden_states.device == torch.device('cuda:0')
                # hidden states was moved to cuda for q proj
                key_states = self.my_k_proj(hidden_states)
                key_states = key_states.to(torch.device('cpu'))
                value_states = self.my_v_proj(hidden_states)
                value_states = value_states.to(torch.device('cpu'))
                timer.end(t)
                assert key_states.device == torch.device('cpu')
                assert value_states.device == torch.device('cpu')

                # k, v states stay in CPU
            elif my_exec_mode == ExecutionMode.Mode6:
                if self.my_k_proj is None:
                    self.my_k_proj = cl.Linear_S8W_S8A_S8B_S8O_Slalom(
                        self.k_proj)
                    self.my_v_proj = cl.Linear_S8W_S8A_S8B_S8O_Slalom(
                        self.v_proj)

                t = timer.start(tag=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})',
                                category=f'K, V Proj ({"Prefill" if is_prefill else "Decode"})')
                key_states = self.my_k_proj(hidden_states_i32)
                value_states = self.my_v_proj(hidden_states_i32)
                timer.end(t)

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

        t = timer.start(tag=f'Q, K, V reshape ({"Prefill" if is_prefill else "Decode"})',
                        category=f'Q, K, V reshape ({"Prefill" if is_prefill else "Decode"})')
        past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(
            query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)

        timer.end(t)
        '''
            QK BMM
            Input: query_states, key_states are torch.int8
            Output: attn_weights is torch.float32
        '''

        t = timer.start(tag=f'QK^T ({"Prefill" if is_prefill else "Decode"})',
                        category=f'QK^T ({"Prefill" if is_prefill else "Decode"})')
        if my_exec_mode == ExecutionMode.Mode1:
            attn_weights = self.qk_bmm(query_states, key_states) # QKT
        elif my_exec_mode == ExecutionMode.Mode2:
            query_states_cp = tc.FromTorchToCupy(query_states.to(torch.int32))
            key_states_cp = tc.FromTorchToCupy(
                key_states.to(torch.int32)).transpose(0, 2, 1)
            attn_weights = cp.matmul(query_states_cp, key_states_cp)
            attn_weights = tc.FromCupyToTorch(attn_weights).to(torch.float32)
            attn_weights *= torch.tensor(self.qk_bmm.a.item(),
                                         dtype=torch.float32)
            assert attn_weights.dtype == torch.float32
        elif my_exec_mode == ExecutionMode.Mode3:
            query_states_np = tc.FromTorchToNumpy(query_states.to(torch.int32))
            key_states_np = tc.FromTorchToNumpy(
                key_states.to(torch.int32)).transpose(0, 2, 1)
            attn_weights = np.matmul(query_states_np, key_states_np)
            attn_weights = tc.FromNumpyToTorch(attn_weights).to(torch.float32)
            attn_weights *= torch.tensor(self.qk_bmm.a.item(),
                                         dtype=torch.float32)
            assert attn_weights.dtype == torch.float32
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            query_states_cp = tc.FromTorchToCupy(
                tc.FromCpuToGPUTorch(query_states)).astype(cp.int32)
            key_states_cp = tc.FromTorchToCupy(tc.FromCpuToGPUTorch(
                key_states).to(torch.int32)).transpose(0, 2, 1)
            attn_weights = cp.matmul(query_states_cp, key_states_cp)
            attn_weights = tc.FromCupyToTorch(attn_weights).to(torch.float32)
            attn_weights *= torch.tensor(self.qk_bmm.a.item(),
                                         dtype=torch.float32)
            attn_weights = tc.FromGpuToCpuTorch(attn_weights)
            assert attn_weights.dtype == torch.float32
            assert attn_weights.device == torch.device('cpu')
        elif my_exec_mode == ExecutionMode.Mode6:
            query_states_np = query_states.numpy()
            key_states_np = key_states.transpose(1, 2).numpy()
            assert query_states_np.dtype == np.int8
            assert key_states_np.dtype == np.int8

            attn_weights = self.secure_qk_bmm(query_states_np, key_states_np)

            attn_weights = tc.FromNumpyToTorch(attn_weights).to(torch.float32)
            # done in CPU
            attn_weights *= torch.tensor(self.qk_bmm.a.item(),
                                         dtype=torch.float32)
           # query_states_cp = tc.FromTorchToCupy(
            #    tc.FromCpuToGPUTorch(query_states)).astype(cp.int32)
            # key_states_cp = tc.FromTorchToCupy(tc.FromCpuToGPUTorch(
            #    key_states).to(torch.int32)).transpose(0, 2, 1)
            # attn_weights2 = cp.matmul(query_states_cp, key_states_cp)
            # attn_weights2 = tc.FromCupyToTorch(attn_weights2).to(torch.float32)
            # attn_weights2 *= torch.tensor(self.qk_bmm.a.item(),
            #                              dtype=torch.float32)
            # attn_weights2 = tc.FromGpuToCpuTorch(attn_weights2)
            # print(attn_weights)
            # print(attn_weights2)
            # assert torch.allclose(attn_weights, attn_weights2, atol=1e-3)
            assert attn_weights.dtype == torch.float32
            assert attn_weights.device == torch.device('cpu')
        timer.end(t)

        t = timer.start(tag=f'Attn Mask & Softmax ({"Prefill" if is_prefill else "Decode"})',
                        category=f'Attn Mask & Softmax ({"Prefill" if is_prefill else "Decode"})')
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

        #print("my softmax")
        attn_weights_np = attn_weights.numpy()
        cipher_cpp.SoftmaxTensor3D(attn_weights_np)
        attn_probs = torch.from_numpy(attn_weights_np)

        #attn_probs = nn.functional.softmax(attn_weights, dim=-1)

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

        timer.end(t)

        t = timer.start(tag=f'PV ({"Prefill" if is_prefill else "Decode"})',
                        category=f'PV ({"Prefill" if is_prefill else "Decode"})')
        if my_exec_mode == ExecutionMode.Mode1:
            attn_output = self.pv_bmm(attn_probs, value_states) # PV
        elif my_exec_mode == ExecutionMode.Mode2:
            attn_probs_cp = tc.FromTorchToCupy(attn_probs.to(torch.int32))
            value_states_cp = tc.FromTorchToCupy(
                value_states.to(torch.int32)).transpose(0, 2, 1)
            attn_output = cp.matmul(attn_probs_cp, value_states_cp)
            attn_output = tc.FromCupyToTorch(attn_output).to(
                torch.float32) * torch.tensor(self.pv_bmm.a.item(), dtype=torch.float32)
            attn_output = attn_output.to(torch.int8)
        elif my_exec_mode == ExecutionMode.Mode3:
            attn_probs_np = tc.FromTorchToNumpy(attn_probs.to(torch.int32))
            value_states_np = tc.FromTorchToNumpy(
                value_states.to(torch.int32)).transpose(0, 2, 1)
            attn_output = np.matmul(attn_probs_np, value_states_np)
            attn_output = tc.FromNumpyToTorch(attn_output).to(
                torch.float32) * torch.tensor(self.pv_bmm.a.item(), dtype=torch.float32)
            attn_output = attn_output.to(torch.int8)
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            attn_probs_cp = tc.FromTorchToCupy(
                tc.FromCpuToGPUTorch(attn_probs)).astype(cp.int32)
            value_states_cp = tc.FromTorchToCupy(tc.FromCpuToGPUTorch(
                value_states).to(torch.int32)).transpose(0, 2, 1)
            attn_output = cp.matmul(attn_probs_cp, value_states_cp)
            attn_output = tc.FromCupyToTorch(attn_output).to(
                torch.float32) * torch.tensor(self.pv_bmm.a.item(), dtype=torch.float32)
            attn_output = attn_output.to(torch.int8)
            attn_output = attn_output.to(torch.device('cpu'))
            assert attn_output.device == torch.device('cpu')
        elif my_exec_mode == ExecutionMode.Mode6:
            attn_probs_np = attn_probs.numpy()
            value_states_np = value_states.transpose(1, 2).numpy()

            attn_output = self.secure_pv_bmm(attn_probs_np, value_states_np)
            attn_output = tc.FromNumpyToTorch(attn_output).to(
                torch.float32)
            attn_output *= torch.tensor(self.pv_bmm.a.item(),
                                        dtype=torch.float32)

           # attn_probs_cp = tc.FromTorchToCupy(
           #     tc.FromCpuToGPUTorch(attn_probs)).astype(cp.int32)
           # value_states_cp = tc.FromTorchToCupy(tc.FromCpuToGPUTorch(
           #     value_states).to(torch.int32)).transpose(0, 2, 1)
          #  attn_output = cp.matmul(attn_probs_cp, value_states_cp)
           # attn_output = tc.FromCupyToTorch(attn_output).to(
           #     torch.float32) * torch.tensor(self.pv_bmm.a.item(), dtype=torch.float32)
          #  attn_output = attn_output.to(torch.int8)
           # attn_output = attn_output.to(torch.device('cpu'))
          #  assert attn_output.device == torch.device('cpu')
            # attn_probs_np = tc.FromTorchToNumpy(attn_probs)
            # value_states_np = tc.FromTorchToNumpy(
            #     value_states.transpose(1, 2))

            # attn_output = self.secure_pv_bmm(attn_probs_np, value_states_np)
            # attn_output = tc.FromNumpyToTorch(attn_output).to(
            #     torch.float32) * torch.tensor(self.pv_bmm.a.item(), dtype=torch.float32)

            # attn_output = attn_output.to(torch.int8)
            # attn_output = attn_output.to(torch.device('cpu'))
            # assert attn_output.device == torch.device('cpu')
        timer.end(t)

        t = timer.start(tag=f'Attn shape ({"Prefill" if is_prefill else "Decode"})',
                        category=f'Attn shape ({"Prefill" if is_prefill else "Decode"})')
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

        timer.end(t)

        if my_exec_mode == ExecutionMode.Mode1:
            t = timer.start(tag=f'Out Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Out Proj ({"Prefill" if is_prefill else "Decode"})')
            attn_output = self.out_proj(attn_output) # WO
            timer.end(t)
        elif my_exec_mode == ExecutionMode.Mode2:
            if self.my_out_proj is None:
                self.my_out_proj = cl.Linear_S8W_S8A_F32B_F32O_Unsecure_Gpu(
                    self.out_proj)

            t = timer.start(tag=f'Out Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Out Proj ({"Prefill" if is_prefill else "Decode"})')
            attn_output = self.my_out_proj(attn_output)
            timer.end(t)
            assert attn_output.dtype == torch.float32
        elif my_exec_mode == ExecutionMode.Mode3:
            if self.my_out_proj is None:
                self.my_out_proj = cl.Linear_S8W_S8A_F32B_F32O_Unsecure_Cpu(
                    self.out_proj)

            t = timer.start(tag=f'Out Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Out Proj ({"Prefill" if is_prefill else "Decode"})')
            attn_output = self.my_out_proj(attn_output)
            timer.end(t)
            assert attn_output.dtype == torch.float32
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            if self.my_out_proj is None:
                self.my_out_proj = cl.Linear_S8W_S8A_F32B_F32O_Unsecure_Gpu(
                    self.out_proj)

            assert attn_output.device == torch.device('cpu')
            t = timer.start(tag=f'Out Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Out Proj ({"Prefill" if is_prefill else "Decode"})')
            attn_output = attn_output.to(torch.device('cuda:0'))
            attn_output = self.my_out_proj(attn_output)
            attn_output = attn_output.to(torch.device('cpu'))
            timer.end(t)
            assert attn_output.dtype == torch.float32
            assert attn_output.device == torch.device('cpu')

        elif my_exec_mode == ExecutionMode.Mode6:
            if self.my_out_proj is None:
                self.my_out_proj = cl.Linear_S8W_S8A_F32B_F32O_Slalom(
                    self.out_proj)

            t = timer.start(tag=f'Out Proj ({"Prefill" if is_prefill else "Decode"})',
                            category=f'Out Proj ({"Prefill" if is_prefill else "Decode"})')
            attn_output = self.my_out_proj(attn_output)
            timer.end(t)

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
        global my_exec_mode

        self.self_attn_layer_norm = LayerNormQ(self.embed_dim)
        self.fc1 = W8A8B8O8LinearReLU(self.embed_dim, ffn_dim)
        self.fc2 = W8A8BFP32OFP32Linear(ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNormQ(self.embed_dim)

        self.fc1_relu = nn.ReLU()

        self.my_fc1 = None
        self.my_fc2 = None

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
        timer = st.SingletonTimer()
        global is_prefill
        if is_prefill:
            if past_key_value is not None:
                is_prefill = False

        t = timer.start(tag=f'1st layer norm ({"Prefill" if is_prefill else "Decode"})',
                        category=f'1st layer norm ({"Prefill" if is_prefill else "Decode"})')
        residual = hidden_states

       # print(f'hidden_states: {hidden_states.shape}, {hidden_states.dtype}')
       # print(f'weight shape: {self.self_attn_layer_norm.weight.shape}')
        # print(f'bias shape: {self.self_attn_layer_norm.bias.shape}')
        # print(f'eps : {self.self_attn_layer_norm.eps}')

        #hidden_states = self.self_attn_layer_norm(
        #    hidden_states)  # shape = 3, torch, float16
        hidden_states2 = self.self_attn_layer_norm(
            hidden_states)  # shape = 3, torch, float16
        #hidden_states = hidden_states.to(
        #    self.self_attn_layer_norm.weight.dtype)
        hidden_states_np = hidden_states.numpy()
        # print("my later norm")
        cipher_cpp.LayerNormTensor3D(
            hidden_states_np, self.self_attn_layer_norm.weight, self.self_attn_layer_norm.bias, self.self_attn_layer_norm.eps)
        hidden_states = torch.from_numpy(hidden_states_np)
        hidden_states_copy = hidden_states.clone().detach()
        hidden_states = hidden_states.round().clamp(-128, 127).to(torch.int8)
        timer.end(t)

        print(hidden_states)
        print(hidden_states2)
        is_eq = torch.equal(hidden_states, hidden_states2)
        mask = (hidden_states != hidden_states2)
        print(hidden_states_copy[mask])

        print(hidden_states[mask])
        print(hidden_states2[mask])
        print("is eq :" , is_eq)
        # assert is_eq
        # assert False

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        t = timer.start(tag=f'1st residual add ({"Prefill" if is_prefill else "Decode"})',
                        category=f'1st residual add ({"Prefill" if is_prefill else "Decode"})')
        residual.add_(hidden_states.to(residual.dtype)) # residual 1 (SGX)
        timer.end(t)

        t = timer.start(tag=f'2nd layer norm ({"Prefill" if is_prefill else "Decode"})',
                        category=f'2nd layer norm ({"Prefill" if is_prefill else "Decode"})')

        hidden_states = self.final_layer_norm(
            residual)  # shape = (1, 512, 768)
        timer.end(t)

        if my_exec_mode == ExecutionMode.Mode1:
            t = timer.start(tag=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = self.fc1(hidden_states) # FC1 + RELU (SGX)
            timer.end(t)
        elif my_exec_mode == ExecutionMode.Mode2:
            if self.my_fc1 is None:
                self.my_fc1 = cl.Linear_S8W_S8A_S8B_S8O_Relu_Unsecure_Gpu(
                    self.fc1)

            t = timer.start(tag=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = self.my_fc1(hidden_states)
            timer.end(t)

        elif my_exec_mode == ExecutionMode.Mode3:
            if self.my_fc1 is None:
                self.my_fc1 = cl.Linear_S8W_S8A_S8B_S8O_Relu_Unsecure_Cpu(
                    self.fc1)

            t = timer.start(tag=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = self.my_fc1(hidden_states)
            timer.end(t)
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            if self.my_fc1 is None:
                self.my_fc1 = cl.Linear_S8W_S8A_S8B_S8O_Relu_Unsecure_Gpu(
                    self.fc1)

            assert hidden_states.device == torch.device('cpu')
            t = timer.start(tag=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = hidden_states.to(torch.device('cuda:0'))
            hidden_states = self.my_fc1(hidden_states)
            hidden_states = hidden_states.to(torch.device('cpu'))
            timer.end(t)

            assert hidden_states.device == torch.device('cpu')
        elif my_exec_mode == ExecutionMode.Mode6:
            if self.my_fc1 is None:
                self.my_fc1 = cl.Linear_S8W_S8A_S8B_S8O_Relu_Slalom(
                    self.fc1)

            t = timer.start(tag=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN1 + Relu ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = self.my_fc1(hidden_states)
            timer.end(t)
            assert hidden_states.device == torch.device('cpu')

        if my_exec_mode == ExecutionMode.Mode1:
            t = timer.start(tag=f'FFN2 ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN2 ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = self.fc2(hidden_states)
            timer.end(t)
        elif my_exec_mode == ExecutionMode.Mode2:
            if self.my_fc2 is None:
                self.my_fc2 = cl.Linear_S8W_S8A_F32B_F32O_Unsecure_Gpu(
                    self.fc2)

            t = timer.start(tag=f'FFN2 ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN2 ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = self.my_fc2(hidden_states)
            timer.end(t)
        elif my_exec_mode == ExecutionMode.Mode3:
            if self.my_fc2 is None:
                self.my_fc2 = cl.Linear_S8W_S8A_F32B_F32O_Unsecure_Cpu(
                    self.fc2)
            t = timer.start(tag=f'FFN2 ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN2 ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = self.my_fc2(hidden_states)
            timer.end(t)
        elif my_exec_mode == ExecutionMode.Mode4:
            pass
        elif my_exec_mode == ExecutionMode.Mode5:
            if self.my_fc2 is None:
                self.my_fc2 = cl.Linear_S8W_S8A_F32B_F32O_Unsecure_Gpu(
                    self.fc2)

            assert hidden_states.device == torch.device('cpu')
            t = timer.start(tag=f'FFN2 ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN2 ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = hidden_states.to(torch.device('cuda:0'))
            hidden_states = self.my_fc2(hidden_states)
            hidden_states = hidden_states.to(torch.device('cpu'))
            timer.end(t)

            assert hidden_states.device == torch.device('cpu')
        elif my_exec_mode == ExecutionMode.Mode6:
            if self.my_fc2 is None:
                self.my_fc2 = cl.Linear_S8W_S8A_F32B_F32O_Slalom(
                    self.fc2)

            t = timer.start(tag=f'FFN2 ({"Prefill" if is_prefill else "Decode"})',
                            category=f'FFN2 ({"Prefill" if is_prefill else "Decode"})')
            hidden_states = self.my_fc2(hidden_states)
            timer.end(t)
            assert hidden_states.device == torch.device('cpu')

        t = timer.start(tag=f'2nd residual add ({"Prefill" if is_prefill else "Decode"})',
                        category=f'2nd residual add ({"Prefill" if is_prefill else "Decode"})')
        residual.add_(hidden_states.to(residual.dtype)) # Last residual (SGX)
        timer.end(t)

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
                    config.hiden_size, config.num_attention_heads, config.ffn_dim
                )
                for _ in range(config.num_hdidden_layers)
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
