import unittest
import build.cipher_cpp as cip_cpp
import numpy as np
import random
import singleton_timer as st
import ntl
import torch
import cipher as cip


class NtlTest(unittest.TestCase):
    # def test_cipher_only(self):
    #     timer = st.SingletonTimer()

    #     num_test = 100
    #     B = 8
    #     for _ in range(num_test):
    #         M = random.randint(1, 100)
    #         K = random.randint(1, 100)
    #         N = random.randint(1, 100)

    #         x = np.random.randint(0, 2**B, (M, K), dtype=np.uint32)
    #         y = np.random.randint(0, 2**B, (K, N), dtype=np.uint32)

    #         row_sum_x = cip_cpp.SumByRow_uint32(x)
    #         col_sum_y = cip_cpp.SumByCol_uint32(y)

    #         x_np = x.copy()
    #         y_np = y.copy()

    #         mod = 2**32
    #         t = timer.start(tag='Select a1, b1, a2, b2',
    #                         category='Select a1, b1, a2, b2')
    #         a1, _ = ntl.GenerateRandomCoprimeLessthanN(mod)
    #         b1 = random.randint(0, mod)
    #         a2, _ = ntl.GenerateRandomCoprimeLessthanN(mod)
    #         b2 = random.randint(0, mod)
    #         timer.end(t)

    #         t = timer.start(tag='Find KeyInvMod', category='Find KeyInvMod')
    #         a_12_inv = ntl.FindKeyInvMod(a1 * a2, mod)
    #         timer.end(t)

    #         t = timer.start(tag='Encryption', category='Encryption')
    #         cip_cpp.EncryptMatrix2D(x, a1, b1, mod)
    #         cip_cpp.EncryptMatrix2D(y, a2, b2, mod)
    #         timer.end(t)

    #         t = timer.start(tag='Matmul', category='Matmul')
    #         z = np.matmul(x, y)
    #         timer.end(t)

    #         t = timer.start(tag='Decryption', category='Decryption')
    #         cip_cpp.DecryptMatrix2D(
    #             z, K, a1, b1, a2, b2, a_12_inv, mod, row_sum_x, col_sum_y)
    #         timer.end(t)

    #         z_np = np.matmul(x_np, y_np)

    #         self.assertTrue(np.array_equal(z, z_np))
    #     timer.display_summary()

    def test_secure_matmul(self):
        timer = st.SingletonTimer()
        
        num_test = 5
        B = 8
        lower = 2**14
        upper = 2**14

        for _ in range(num_test):
            M = random.randint(lower, upper)
            K = random.randint(lower, upper)
            N = random.randint(lower, upper)

            x = torch.randint(-(2**(B-1)), 2**(B-1), (M, K), dtype=torch.int32)
            y = torch.randint(-(2**(B-1)), 2**(B-1), (K, N), dtype=torch.int32)

            x_cp = x.clone().detach().to(torch.float32).to(torch.device("cuda:0"))
            y_cp = y.clone().detach().to(torch.float32).to(torch.device("cuda:0"))

            z = cip.SecureMatmul(x, y, B)

            z_py = torch.matmul(x_cp, y_cp)
            z_py = z_py.to(torch.device("cpu")).to(torch.int32)
            self.assertTrue(torch.equal(z, z_py))
        
        timer.display_log()
        timer.display_summary()


if __name__ == '__main__':
    unittest.main()
