import torch

for i in range(16_777_000, 16_778_000):
    x = torch.tensor([i], dtype=torch.int32)
    x_flt = torch.tensor([i], dtype=torch.float32)
    print(f'{i}-th : {x}, {x_flt}')
    assert x == x_flt.to(torch.int32)

