import torch
from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM
import smoothquant.opt
import os
from torch.nn.functional import pad
import time
import csv

import singleton_timer as st
import accuracy_measure_tools as amt

amt.set_clock_speed()

timer = st.SingletonTimer()

import sgx.sgx_layer_struct as sgx_lsc

timer.disable()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode5
model_size='125m'
target_input_token_len = 1024
target_output_token_len = 2048
num_batches = 1

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
smoothquant.opt.time_stats.on()
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

categories = ['Set Hidden States', 'Copy Residual 1', 'Layer Norm 1', 'Get Hidden States Size', 'Q Proj, Cast From Int8 To Int32', 'Q Proj, Process Input Tensor Before Offload', 'Q Proj, Generate Decryption Key', 'Q Proj, Host to Device', 'Q Proj, GPU Computation', 'Q Proj, Device to Host',
              'Q Proj, Process Output Tensor After Offload', 'Q Proj, Compute Epilogue', 'K Proj, Cast From Int8 To Int32', 'K Proj, Process Input Tensor Before Offload', 'K Proj, Generate Decryption Key',
               'K Proj, Host to Device', 'K Proj, GPU Computation', 'K Proj, Device to Host', 'K Proj, Process Output Tensor After Offload', 'K Proj, Compute Epilogue', 'V Proj, Cast From Int8 To Int32', 'V Proj, Process Input Tensor Before Offload', 'V Proj, Generate Decryption Key',
               'V Proj, Host to Device', 'V Proj, GPU Computation', 'V Proj, Device to Host', 'V Proj, Process Output Tensor After Offload', 'V Proj, Compute Epilogue','Construct KV Cache', 'Get Past KV', 'Reshape Q, K, V', 
               'QK BMM, Cast From Int8 To Int32', 'QK BMM, Process Input Tensors Before Offload', 'QK BMM, Generate Decryption Key', 'QK BMM, Host to Device', 'QK BMM, GPU Computation', 'QK BMM, Device to Host', 'QK BMM, Process Output Tensors After Offload', 'QK BMM, Compute Epilogue', 'Apply Attention Mask', 'Softmax', 'Apply Layer Head Mask', 'Reshape Attention Probabilities', 'Post Softmax Quantization',
              'Transpose V', 'PV BMM, Cast From Int8 To Int32', 'PV BMM, Process Input Tensors Before Offload', 'PV BMM, Generate Decryption Key', 'PV BMM, Host to Device', 'PV BMM, GPU Computation', 'PV BMM, Device to Host', 'PV BMM, Process Output Tensors After Offload', 'PV BMM, Compute Epilogue', 'Reshape Attention Output', 
              'Out Proj, Cast From Int8 To Int32', 'Out Proj, Process Input Tensor Before Offload', 'Out Proj, Generate Decryption Key', 'Out Proj, Host to Device', 'Out Proj, GPU Computation', 'Out Proj, Device to Host', 'Out Proj, Process Output Tensor After Offload', 'Out Proj, Compute Epilogue', 'Add Residual 1', 'Copy Residual 2', 'Layer Norm 2', 
              'FC1, Cast From Int8 To Int32', 'FC1, Process Input Tensor Before Offload', 'FC1, Generate Decryption Key', 'FC1, Host to Device', 'FC1, GPU Computation', 'FC1, Device to Host', 'FC1, Process Output Tensor After Offload', 'FC1, Compute Epilogue', 'ReLU', 
              'FC2, Cast From Int8 To Int32', 'FC2, Process Input Tensor Before Offload', 'FC2, Generate Decryption Key', 'FC2, Host to Device', 'FC2, GPU Computation', 'FC2, Device to Host', 'FC2, Process Output Tensor After Offload', 'FC2, Compute Epilogue', 'Add Residual 2', 'Post Decoder Layer']
#assert len(categories) == 76

for state in ['Prefill', 'Generation']:
    for category in categories:
        key = f'{category} ({state})'
        num_samples, min_time, max_time, avg_time, total_time = raw_data[key]
        sub_data = [key, num_samples, min_time, max_time, avg_time, total_time]
        data.append(sub_data)

        f = open(f'sq_gen_{model_size}_{target_input_token_len}_{target_output_token_len}_mode{smoothquant.opt.my_exec_mode.value}_batch{num_batches}_optane.csv', 'w')
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()

if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
    sgx_lsc.SgxLayerStructC().Destroy()

amt.reset_clock_speed()