import torch
from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM
import smoothquant.opt
import os
from torch.nn.functional import pad
import time
import csv
import sys

import singleton_timer as st
import accuracy_measure_tools as amt
import smoothquant.my_bmm as my_bmm

amt.set_clock_speed()

timer = st.SingletonTimer()

import secure_llm.secure_llm_sgx as sgx_lsc

timer.disable()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

assert len(sys.argv) == 5
if sys.argv[1] == 'mode5':
    smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode5
elif sys.argv[1] == 'mode6':
    smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode6
elif sys.argv[1] == 'mode7':
    smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode7
elif sys.argv[1] == 'mode8':
    smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode8
elif sys.argv[1] == 'mode9':
    smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode9
else:
    assert False

model_size = '125m'

target_input_token_len = int(sys.argv[2])
target_output_token_len = int(sys.argv[3])
num_batches = int(sys.argv[4])

print("Model: ", model_size)
print("Mode: ", smoothquant.opt.my_exec_mode)
print("Input token length: ", target_input_token_len)
print("Output token length: ", target_output_token_len)
print("Number of batches: ", num_batches)


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
model_smoothquant.pre_init()

'''
    NOTE(jpyo0803): Warmup
'''
dummy_prompt = (
    "A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the ")
dummy_model_inputs = tokenizer([dummy_prompt], return_tensors='pt').to(
    'cuda:0' if start_gpu else 'cpu')

# dummy = model_smoothquant.generate(
#     **dummy_model_inputs, max_new_tokens=128, do_sample=False)

'''
    NOTE(jpyo0803): Above is warmup
'''

prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
          "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
          "there?")

prompts = [prompt] * num_batches


model_inputs = tokenizer(prompts, return_tensors='pt').to(
    'cuda:0' if start_gpu else 'cpu')


pad_len = target_input_token_len - model_inputs['input_ids'].shape[1]

model_inputs['input_ids'] = pad(
    model_inputs['input_ids'], (0, pad_len), value=77)
model_inputs['attention_mask'] = pad(
    model_inputs['attention_mask'], (0, pad_len), value=1)

# print input token length
print(f"Input token length: {model_inputs['input_ids'].shape[1]}")
assert model_inputs['input_ids'].shape[1] == target_input_token_len



smoothquant.opt.is_prefill = True
# smoothquant.opt.time_stats.on()
my_bmm.Measure_Start = True
timer.enable()

start_time = time.perf_counter_ns()
generated_ids = model_smoothquant.generate(
    **model_inputs, min_length=target_output_token_len, max_length=target_output_token_len, do_sample=False)
end_time = time.perf_counter_ns()
print(f"End-to-end Latency: {(end_time - start_time)/1e9:0.6f} s")
smoothquant.opt.time_stats.print_summary()
# print Output token length
print(f"Output token length: {generated_ids.shape[1]}")
assert generated_ids.shape[1] == target_output_token_len

for i in range(num_batches):
    print(tokenizer.decode(generated_ids[i]))

raw_data = st.SingletonTimer().display_summary(outlier_percent=0.05)


data = []

categories = ['QK BMM, Cast From Int8 To Int32', 'QK BMM, Process Input Tensors Before Offload', 'QK BMM, Generate Decryption Key', 'QK BMM, Host to Device', 'QK BMM, Manage Y', 'QK BMM, Main Computation', 'QK BMM, Device to Host', 'QK BMM, Process Output Tensors After Offload', 'QK BMM, Compute Epilogue', 
               'PV BMM, Cast From Int8 To Int32', 'PV BMM, Process Input Tensors Before Offload', 'PV BMM, Generate Decryption Key', 'PV BMM, Host to Device', 'PV BMM, Manage Y', 'PV BMM, Main Computation', 'PV BMM, Device to Host', 'PV BMM, Process Output Tensors After Offload', 'PV BMM, Compute Epilogue']
#assert len(categories) == 76

for state in ['Prefill', 'Generation']:
    for category in categories:
        key = f'{category} ({state})'
        num_samples, min_time, max_time, avg_time, total_time = raw_data[key]
        sub_data = [key, num_samples, min_time, max_time, avg_time, total_time]
        data.append(sub_data)

        f = open(f'sq_gen_{model_size}_{target_input_token_len}_{target_output_token_len}_mode{smoothquant.opt.my_exec_mode.value}_batch{num_batches}_bmm_only.csv', 'w')
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()

if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode9:
    sgx_lsc.SgxSecureLLM().Destroy()

amt.reset_clock_speed()

if smoothquant.opt.ENABLE_MATMUL_OUTPUT_SUM:
    print("Checksum: ", smoothquant.opt.check_sum)