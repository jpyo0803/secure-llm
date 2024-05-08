import torch
from torch_int.nn.linear import W8A8B8O8LinearReLU, W8A8B8O8Linear, W8A8B32O32Linear, W8A8BFP32OFP32Linear

import cipher.cipher_linear as cl
import random

import singleton_timer as st

import nvtx

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TORCH_USE_CUDA_DSA"] = '1'


def test_linear_s8w_s8a_s8b_s8o():
    timer = st.SingletonTimer(False)

    for i in range(13):
        B = random.randint(10, 10)
        M = 16**random.randint(3, 3)
        N = 16**random.randint(3, 3)

        linear = torch.nn.Linear(M, N, bias=True)

        x = torch.randn(B, M)
        x_scale = x.abs().max() / 127
        qx = (x / x_scale).round().to(torch.int8)
        qx2 = qx.clone().detach()

        y_gt = linear(x)
        y_scale = y_gt.abs().max() / 127

        sq_linear = W8A8B8O8LinearReLU.from_float(linear, x_scale, y_scale).cuda()

        my_linear = cl.Linear_S8W_S8A_S8B_S8O_Relu_Unsecure(sq_linear)

        t = timer.start(tag='CPU to GPU (Mine)', category='CPU to GPU (Mine)', exclude = i < 3)
        # torch.cuda.nvtx.range_push('Mine')
        # torch.cuda.nvtx.range_push('Mine, H to D')
        qx2 = qx2.cuda()
        # torch.cuda.nvtx.range_pop()
        timer.end(t)
        t = timer.start(tag='Computation (Mine)', category='Computation (Mine)', exclude = i < 3)
        # torch.cuda.nvtx.range_push('Mine, Comp')
        my_qy = my_linear(qx2)
        torch.cuda.synchronize()
        # torch.cuda.nvtx.range_pop()
        timer.end(t)
        t = timer.start(tag='GPU to CPU (Mine)', category='GPU to CPU (Mine)', exclude = i < 3)
        # torch.cuda.nvtx.range_push('Mine, D to H')
        my_qy = my_qy.cpu()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_pop()
        timer.end(t)


        # torch.cuda.nvtx.range_push('SmoothQuant')
        t = timer.start(tag='CPU to GPU (SmoothQuant)', category='CPU to GPU (SmoothQuant)', exclude = i < 3)
        # torch.cuda.nvtx.range_push('SmoothQuant, H to D')
        qx = qx.cuda()
        # torch.cuda.nvtx.range_pop()
        timer.end(t)
        t = timer.start(tag='Computation (SmoothQuant)', category='Computation (SmoothQuant)', exclude = i < 3)
        # torch.cuda.nvtx.range_push('SmoothQuant, Comp')
        sq_qy = sq_linear(qx)
        torch.cuda.synchronize()
        # torch.cuda.nvtx.range_pop()
        timer.end(t)
        t = timer.start(tag='GPU to CPU (SmoothQuant)', category='GPU to CPU (SmoothQuant)', exclude = i < 3)
        # torch.cuda.nvtx.range_push('SmoothQuant, D to H')
        sq_qy = sq_qy.cpu()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_pop()
        timer.end(t)
        

        print("sq qy: ", sq_qy)
        print("my qy: ", my_qy)
        # how to write if two torch tensors are equal
        # we say the tensors are same if the difference between two corresponding elements is less or equal to 1
        mask = torch.abs(sq_qy - my_qy) > 1
        not_equal = torch.any(mask)
        # print(f"{i + 1}-th Equal: {equal}", equal)
        # assert equal
        if not_equal:
            # show only the different elements
            print(mask)
            print("sq_qy: ", sq_qy[mask])
            print("my_qy: ", my_qy[mask])

            assert False
        
        
    timer.display_summary()
    timer.reset()


if __name__ == '__main__':
    test_linear_s8w_s8a_s8b_s8o()
