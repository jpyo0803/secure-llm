import torch
from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM
import smoothquant.opt
import os
from torch.nn.functional import pad
import time
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode4

'''
    NOTE(jpyo0803): Set execution mode

    Mode1 = Smoothquant original (GPU only)
    Mode2 = GPU Only (Unsecure)
    Mode3 = CPU (torch native) + GPU (Unsecure), Flexgen style
    Mode4 = CPU (custom cpp) + GPU (Unsecure), Flexgen style, KV cache managed in CPU
    Mode5 = CPU (custom cpp) + GPU (Secure), Flexgen style, KV cache manged in CPU
    Mode6 = CPU (custom cpp on SGX) + GPU (Secure), Flexgen style, KV cache managed in CPU
    Mode7 = CPU (custom cpp on SGX) + GPU (Secure), Flexgen style, KV cache managed in GPU

    From Mode 3 to 4 show torch native vs. custom cpp performance
    From Mode 4 to 5 show unsecure vs. secure performance regarding addtive cipher
    From Mode 5 to 6 show the effect of SGX on performance
    From Mode 6 to 7 show the effect of placing KV cache in GPU
'''



print("Mode: ", smoothquant.opt.my_exec_mode)

start_gpu = (smoothquant.opt.my_exec_mode ==
             smoothquant.opt.ExecMode.Mode1) or (smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode2)

# Load the model and tokenizer
model_smoothquant = Int8OPTForCausalLM.from_pretrained(
    'mit-han-lab/opt-125m-smoothquant', torch_dtype=torch.float32)
model_smoothquant.eval()

tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-125m')
# Define the input prompt
# input_prompt = 'her pay for the evening was almost double that of the wait staff and although that might not seem like a lot to some people , it was a small fortune to claire . after loading her final tray for a server , claire went to the restroom to freshen up and begin preparations for being loaded into the cake . pam had a couple of young men from college who assisted her into the cake . brian and max were a lot of fun and always made her laugh as they hoisted her up to the top of the cake'

model_smoothquant.to('cuda:0' if start_gpu else 'cpu')
model_smoothquant.pre_init()

'''
    NOTE(jpyo0803): Warmup
'''
dummy_prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the ")
dummy_model_inputs = tokenizer([dummy_prompt], return_tensors='pt').to('cuda:0' if start_gpu else 'cpu') 

dummy = model_smoothquant.generate(
    **dummy_model_inputs, max_new_tokens=128, do_sample=False)

'''
    NOTE(jpyo0803): Above is warmup
'''

prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
          "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
          "there?")

model_inputs = tokenizer([prompt], return_tensors='pt').to(
    'cuda:0' if start_gpu else 'cpu')

target_input_token_len = 512

pad_len = target_input_token_len - model_inputs['input_ids'].shape[1]

model_inputs['input_ids'] = pad(model_inputs['input_ids'], (0, pad_len), value=77)
model_inputs['attention_mask'] = pad(model_inputs['attention_mask'], (0, pad_len), value=1)

# print input token length
print(f"Input token length: {model_inputs['input_ids'].shape[1]}")
assert model_inputs['input_ids'].shape[1] == target_input_token_len


smoothquant.opt.is_prefill = True
smoothquant.opt.time_stats.on()

target_output_token_len = 1024
start_time = time.perf_counter_ns()
generated_ids = model_smoothquant.generate(
    **model_inputs, min_length=target_output_token_len, max_length=target_output_token_len, do_sample=False)
end_time = time.perf_counter_ns()
print(f"End-to-end Latency: {(end_time - start_time)/1e9:0.6f} s")
smoothquant.opt.time_stats.print_summary()
# print Output token length
print(f"Output token length: {generated_ids.shape[1]}")
assert generated_ids.shape[1] == target_output_token_len
# print(tokenizer.batch_decode(generated_ids)[0])