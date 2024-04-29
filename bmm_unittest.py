import unittest
import build.cipher_cpp as cip_cpp
import numpy as np
import random
import singleton_timer as st
import torch
import cipher as cip
import cupy as cp
from torch_int._CUDA import bmm_s8t_s8n_s32t, bmm_s8t_s8n_f32t
import nvtx

a = np.random.randint(0, 10, (5, 5, 5), dtype=np.uint32)
b = np.random.randint(0, 10, (5, 5, 5), dtype=np.uint32)
a2 = np.random.randint(0, 10, (5, 5, 5), dtype=np.int32)
b2 = np.random.randint(0, 10, (5, 5, 5), dtype=np.int32)
a = cp.asarray(a)
b = cp.asarray(b)
a2 = cp.asarray(a2)
b2 = cp.asarray(b2)
c = cp.matmul(a, b)
c2 = cp.matmul(a2, b2)
c = cp.asnumpy(c)
c2 = cp.asnumpy(c2)

a = torch.randint(0, 10, (16, 16, 16), dtype=torch.int8).to(
    torch.device("cuda:0"))
b = torch.randint(0, 10, (16, 16, 16), dtype=torch.int8).to(
    torch.device("cuda:0"))
c = bmm_s8t_s8n_s32t(a, b).to(torch.device("cpu"))


class BmmTest(unittest.TestCase):
    # def test_bmm_unsecure(self):
    #     timer = st.SingletonTimer()

    #     num_test = 100
    #     pass_cnt = 0
    #     B = 8

    #     batch_size = 10
    #     lower = 2**10
    #     upper = 2**10

    #     for i in range(num_test):
    #         M = random.randint(lower, upper)
    #         K = random.randint(lower, upper)
    #         N = random.randint(lower, upper)

    #         x = np.random.randint(-(2**(B-1)), 2**(B-1),
    #                           (batch_size, M, K), dtype=np.int8)
    #         y = np.random.randint(-(2**(B-1)), 2**(B-1),
    #                           (batch_size, K, N), dtype=np.int8)

    #         x2 = torch.tensor(x.copy()).to(torch.float32)
    #         y2 = torch.tensor(y.copy()).to(torch.float32)
    #         # Assuming cip.UnsecureMatmul returns int32; adjust if different

    #         t = timer.start(tag='test_bmm_unsecure',
    #                         category='test_bmm_unsecure')
    #         z = cip.BatchUnsecureMatmul(x, y)
    #         timer.end(t)

    #         t = timer.start(tag='test_bmm_torch',
    #                         category='test_bmm_torch')
    #         x2 = x2.to(torch.device("cuda:0"))
    #         y2 = y2.to(torch.device("cuda:0"))
    #         z2 = torch.bmm(x2, y2)
    #         z2 = z2.to(torch.device("cpu")).to(torch.int32)
    #         timer.end(t)
    #         z = torch.tensor(z)

    #         self.assertTrue(torch.equal(z, z2))
    #         pass_cnt += 1

    #     timer.display_summary()
    #     print(f"'test_bmm_unsecure' passed {pass_cnt} / {num_test}")

    def test_bmm_secure_full(self):
        timer = st.SingletonTimer()

        num_test = 5
        pass_cnt = 0
        B = 8

        batch_size = 10

        for i in range(num_test):
            print("Test #", i + 1)
            # M = 2**random.randint(4, 12)
            # # M = random.randint(1, 1)
            # K = 2**random.randint(4, 12)
            # N = 2**random.randint(4, 12)
            M = 2**(8)
            # M = random.randint(1, 1)
            K = 2**(8)
            N = 2**(8)

            print(M, K, N)
            a = torch.randint(-128, 127, (batch_size, M, K), dtype=torch.int8)
            b = torch.randint(-128, 127, (batch_size, N, K), dtype=torch.int8)
            x = a.numpy()
            y = b.numpy()
            x2 = a.numpy()
            y2 = b.numpy()

            z2 = cip.BatchUnsecureMatmul(x2, y2, i > 0)
            z = cip.BatchSecureMatmul(x, y, i > 0)

            # t = timer.start(tag='test_bmm_torch htod',
            #                 category='test_bmm_torch htod')
            # nvtx.push_range("bmm_s8t_s8n_s32t htod")

            # cuda_dev = torch.device("cuda:0")

            # nvtx.pop_range()
            # timer.end(t)
            # t = timer.start(tag='test_bmm_torch gpu computation',
            #                 category='test_bmm_torch gpu computation')
            # nvtx.push_range("bmm_s8t_s8n_s32t gpu computation")
            # c = bmm_s8t_s8n_s32t(a, b)
            # c = cp.matmul(a, b)
            # nvtx.pop_range()
            # timer.end(t)
            # t = timer.start(tag='test_bmm_torch dtoh',
            #                 category='test_bmm_torch dtoh')
            # nvtx.push_range("bmm_s8t_s8n_s32t dtoh")
            # c = c.to(torch.device("cpu")).to(torch.int32)
            # c = cp.asnumpy(c)
            # nvtx.pop_range()

            # timer.end(t)
            z = torch.tensor(z)
            z2 = torch.tensor(z2)
            self.assertTrue(torch.equal(z, z2))
            pass_cnt += 1

        timer.display_summary()
        print(f"'test_bmm_secure_full' passed {pass_cnt} / {num_test}")


if __name__ == '__main__':
    unittest.main()
