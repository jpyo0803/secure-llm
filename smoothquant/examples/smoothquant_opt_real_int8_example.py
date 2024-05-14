import singleton_timer as st
import torch
from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM
import smoothquant.opt
import os
from torch.nn.functional import pad
import time
import singleton_timer as st
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

Int8OPTForCausalLM.set_exec_mode(smoothquant.opt.ExecutionMode.Mode3)

print("Mode: ", smoothquant.opt.my_exec_mode)

timer = st.SingletonTimer(False)

start_gpu = (smoothquant.opt.my_exec_mode ==
             smoothquant.opt.ExecutionMode.Mode1) or (smoothquant.opt.my_exec_mode == smoothquant.opt.ExecutionMode.Mode2)

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

timer.disable()
dummy = model_smoothquant.generate(
    **model_inputs, max_new_tokens=1, do_sample=False)

timer.enable()
start_time = time.perf_counter_ns()
generated_ids = model_smoothquant.generate(
    **model_inputs, max_new_tokens=128, do_sample=False)
end_time = time.perf_counter_ns()
print(f"End-to-end Latency: {(end_time - start_time)/1e9:0.6f} s")
print(tokenizer.batch_decode(generated_ids)[0])

raw_data = timer.display_summary()

data = []

partial_data = ['Data']
partial_data.append(raw_data['1st layer norm (Prefill)'])
partial_data.append(raw_data['Q Proj (Prefill)'])
partial_data.append(raw_data['K, V Proj (Prefill)'])
partial_data.append(raw_data['QK^T (Prefill)'])
partial_data.append(raw_data['Attn Mask & Softmax (Prefill)'])
partial_data.append(raw_data['PV (Prefill)'])
partial_data.append(raw_data['Out Proj (Prefill)'])
partial_data.append(raw_data['1st residual add (Prefill)'])
partial_data.append(raw_data['2nd layer norm (Prefill)'])
partial_data.append(raw_data['FFN1 + Relu (Prefill)'])
partial_data.append(raw_data['FFN2 (Prefill)'])
partial_data.append(raw_data['2nd residual add (Prefill)'])
partial_data.append(raw_data['1st layer norm (Decode)'])
partial_data.append(raw_data['Q Proj (Decode)'])
partial_data.append(raw_data['K, V Proj (Decode)'])
partial_data.append(raw_data['QK^T (Decode)'])
partial_data.append(raw_data['Attn Mask & Softmax (Decode)'])
partial_data.append(raw_data['PV (Decode)'])
partial_data.append(raw_data['Out Proj (Decode)'])
partial_data.append(raw_data['1st residual add (Decode)'])
partial_data.append(raw_data['2nd layer norm (Decode)'])
partial_data.append(raw_data['FFN1 + Relu (Decode)'])
partial_data.append(raw_data['FFN2 (Decode)'])
partial_data.append(raw_data['2nd residual add (Decode)'])

data.append(partial_data)

f = open(f'end_to_end_cpu.csv', 'w')
writer = csv.writer(f)
writer.writerows(data)
f.close()
