import torch
from transformers import OPTForCausalLM, GPT2Tokenizer

device = 'cuda:0'

model = OPTForCausalLM.from_pretrained(
    'facebook/opt-125m', torch_dtype=torch.float32)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-125m')

prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
          "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
          "there?")

input_prompt = ('her pay for the evening was almost double that of the wait staff and although that might not seem like a lot to some people , it was a small fortune to claire . after loading her final tray for a server , claire went to the restroom to freshen up and begin preparations for being loaded into the cake . pam had a couple of young men from college who assisted her into the cake . brian and max were a lot of fun and always made her laugh as they hoisted her up to the top of the cake')

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(
    **model_inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.batch_decode(generated_ids)[0])
