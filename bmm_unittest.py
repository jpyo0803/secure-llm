import unittest
import build.cipher_cpp as cip_cpp
import numpy as np
import random
import singleton_timer as st
import ntl
import torch
import cipher as cip


class BmmTest(unittest.TestCase):
    def test_bmm_unsecure(self):
        timer = st.SingletonTimer()

        num_test = 100
        pass_cnt = 0
        B = 8

        batch_size = 10
        lower = 2**10
        upper = 2**10

        for i in range(num_test):
            M = random.randint(lower, upper)
            K = random.randint(lower, upper)
            N = random.randint(lower, upper)

            x = np.random.randint(-(2**(B-1)), 2**(B-1),
                              (batch_size, M, K), dtype=np.int8)
            y = np.random.randint(-(2**(B-1)), 2**(B-1),
                              (batch_size, K, N), dtype=np.int8)

            x2 = torch.tensor(x.copy()).to(torch.float32)
            y2 = torch.tensor(y.copy()).to(torch.float32)
            # Assuming cip.UnsecureMatmul returns int32; adjust if different

            t = timer.start(tag='test_bmm_unsecure',
                            category='test_bmm_unsecure')
            z = cip.BatchUnsecureMatmul(x, y)
            timer.end(t)
            
            t = timer.start(tag='test_bmm_torch',
                            category='test_bmm_torch')
            x2 = x2.to(torch.device("cuda:0"))
            y2 = y2.to(torch.device("cuda:0"))
            z2 = torch.bmm(x2, y2)
            z2 = z2.to(torch.device("cpu")).to(torch.int32)
            timer.end(t)
            z = torch.tensor(z)

            self.assertTrue(torch.equal(z, z2))
            pass_cnt += 1

        timer.display_summary()
        print(f"'test_bmm_unsecure' passed {pass_cnt} / {num_test}")


if __name__ == '__main__':
    unittest.main()
