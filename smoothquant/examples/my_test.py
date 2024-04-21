from smoothquant.opt2 import Int8OPTAttention
import torch

attn = Int8OPTAttention(512, 32)

hidden_states = torch.randint(0, 10, (1, 512, 512), dtype=torch.int8)
out = attn.forward(hidden_states)

print("Q proj Weight / bias: ", attn.q_proj.weight.dtype, " / ", attn.q_proj.bias.dtype)
print("K proj Weight / bias: ", attn.k_proj.weight.dtype, " / ", attn.k_proj.bias.dtype)
print("V proj Weight / bias: ", attn.v_proj.weight.dtype, " / ", attn.v_proj.bias.dtype)

# print(attn.qk_bmm.weight.dtype)
