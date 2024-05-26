import torch
from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM
import smoothquant.opt
import os
from torch.nn.functional import pad
import time
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode3

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
prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
          "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
          "there?")

model_inputs = tokenizer([prompt], return_tensors='pt').to(
    'cuda:0' if start_gpu else 'cpu')

model_smoothquant.to('cuda:0' if start_gpu else 'cpu')

dummy = model_smoothquant.generate(
    **model_inputs, max_new_tokens=1, do_sample=False)

target_num_tokens = 128
start_time = time.perf_counter_ns()
generated_ids = model_smoothquant.generate(
    **model_inputs, min_length=target_num_tokens, max_length=target_num_tokens, do_sample=False)
end_time = time.perf_counter_ns()
print(f"End-to-end Latency: {(end_time - start_time)/1e9:0.6f} s")
print(tokenizer.batch_decode(generated_ids)[0])