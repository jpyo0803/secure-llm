import torch
from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM
import smoothquant.opt
import os
from torch.nn.functional import pad
import time

import singleton_timer as st

import sgx.sgx_layer_struct as sgx_lsc


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode5
model_size = '125m'
target_input_token_len = 128
target_output_token_len = 256
num_batches = 1


print("Mode: ", smoothquant.opt.my_exec_mode)

start_gpu = (smoothquant.opt.my_exec_mode ==
             smoothquant.opt.ExecMode.Mode1) or (smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode2)

# Load the model and tokenizer
model_smoothquant = Int8OPTForCausalLM.from_pretrained(
    f'mit-han-lab/opt-{model_size}-smoothquant', torch_dtype=torch.float32)
model_smoothquant.eval()

tokenizer = GPT2Tokenizer.from_pretrained(f'facebook/opt-{model_size}')
# Define the input prompt
# input_prompt = 'her pay for the evening was almost double that of the wait staff and although that might not seem like a lot to some people , it was a small fortune to claire . after loading her final tray for a server , claire went to the restroom to freshen up and begin preparations for being loaded into the cake . pam had a couple of young men from college who assisted her into the cake . brian and max were a lot of fun and always made her laugh as they hoisted her up to the top of the cake'

model_smoothquant.to('cuda:0' if start_gpu else 'cpu')

prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
          "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
          "there?")

prompts = [prompt] * num_batches


model_inputs = tokenizer(prompts, return_tensors='pt').to(
    'cuda:0' if start_gpu else 'cpu')


# print input token length
print(f"Input token length: {model_inputs['input_ids'].shape[1]}")
assert model_inputs['input_ids'].shape[1] == target_input_token_len

smoothquant.opt.is_prefill = True

start_time = time.perf_counter_ns()
generated_ids = model_smoothquant.generate(
    **model_inputs, min_length=target_output_token_len, max_length=target_output_token_len, do_sample=False)
end_time = time.perf_counter_ns()
print(f"End-to-end Latency: {(end_time - start_time)/1e9:0.6f} s")

for i in range(num_batches):
    print(tokenizer.decode(generated_ids[i]))

if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
    sgx_lsc.SgxLayerStructC().Destroy()