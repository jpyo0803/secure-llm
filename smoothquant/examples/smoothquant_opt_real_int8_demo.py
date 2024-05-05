import singleton_timer as st
import torch
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM
import smoothquant.opt
import os
import gc
from torch.nn.functional import pad
from datasets import load_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

timer = st.SingletonTimer(False)


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
        iter = 0
        for batch in self.dataset:
            if start_gpu:
                input_ids = batch['input_ids'].cuda().unsqueeze(0)
            else:
                input_ids = batch['input_ids'].cpu().unsqueeze(0)

            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            iter += 1
            print("Iter: ", iter)
            t = timer.start(tag='Test', category='Test', exclude=iter <= 1)
            outputs = model(input_ids)
            timer.end(t)
            end.record()
            torch.cuda.synchronize()
            if iter > 1:
                latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        lantecy = latency / len(self.dataset)
        timer.display_summary()
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


tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-30b')
dataset = load_dataset('lambada', split='validation[:50]')
evaluator = Evaluator(dataset, tokenizer)

Int8OPTForCausalLM.set_exec_mode(smoothquant.opt.ExecutionMode.Mode2)

start_gpu = (smoothquant.opt.my_exec_mode ==
             smoothquant.opt.ExecutionMode.Mode1) or (smoothquant.opt.my_exec_mode == smoothquant.opt.ExecutionMode.Mode2)
print("My Exec Mode: ", smoothquant.opt.my_exec_mode)
print("Start GPU: ", start_gpu)

model_smoothquant = Int8OPTForCausalLM.from_pretrained(
    'mit-han-lab/opt-125m-smoothquant', torch_dtype=torch.float16, device_map='cuda:0' if start_gpu else 'cpu')

print_model_size(model_smoothquant)
acc_smoothquant, lantecy_smoothquant = evaluator.evaluate(model_smoothquant)
print(
    f'SmoothQuant INT8 accuracy: {acc_smoothquant}, per-sample lantecy: {lantecy_smoothquant:.3f}ms')
