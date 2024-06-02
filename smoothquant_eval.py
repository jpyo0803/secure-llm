import smoothquant.opt
from datasets import load_dataset
import torch
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM
import os
import gc
from torch.nn.functional import pad

import singleton_timer as st
import accuracy_measure_tools as amt

import sgx.sgx_layer_struct as sgx_lsc

smoothquant.opt.my_exec_mode = smoothquant.opt.ExecMode.Mode7

amt.set_clock_speed()

timer = st.SingletonTimer()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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



start_gpu = True if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode1 or smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode2 else False


class Evaluator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        for i, batch in enumerate(self.dataset):
            model.pre_init()

            print(f'Processing {i+1}-th Batch')
            if start_gpu:
                input_ids = batch['input_ids'].cuda().unsqueeze(0)
            else:
                input_ids = batch['input_ids'].cpu().unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids)
            end.record()
            torch.cuda.synchronize()
            if i != 0:
                latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        # Latency for 1st batch is ignored
        lantecy = latency / (len(self.dataset) - 1)
        return acc, lantecy


def print_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))


tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-125m')
dataset = load_dataset('lambada', split='validation[:10]')
evaluator = Evaluator(dataset, tokenizer)

print("Execution Mode: ", smoothquant.opt.my_exec_mode)
print("Start Device: ", "CUDA" if start_gpu else "CPU")

'''
    NOTE(jpyo0803): We use torch.float32 instead of torch.float16 
    because CPU cannot handle torch.float16.
'''
model_smoothquant = Int8OPTForCausalLM.from_pretrained(
    'mit-han-lab/opt-125m-smoothquant', torch_dtype=torch.float32, device_map='cuda:0' if start_gpu else 'cpu')



print_model_size(model_smoothquant)
acc_smoothquant, lantecy_smoothquant = evaluator.evaluate(model_smoothquant)
timer.display_summary()
print(
    f'SmoothQuant INT8 accuracy: {acc_smoothquant}, per-sample lantecy: {lantecy_smoothquant:.3f}ms')


if smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode6:
    sgx_lsc.SgxLayerStructC().Destroy()
elif smoothquant.opt.my_exec_mode == smoothquant.opt.ExecMode.Mode8:
    sgx_lsc.SgxLayerStructC().Destroy()

amt.reset_clock_speed()