import unittest
import numpy as np
import random
import singleton_timer as st

from cipher import cipher_light as cl
from cipher import cipher_unsecure as cu


class BmmTest(unittest.TestCase):
    def test_bmm_secure(self):
        timer = st.SingletonTimer(False)

        num_test = 1
        num_iter = 10
        pass_cnt = 0
        timer_first = True

        sbmm_light = cl.SBMM_Light()
        sbmm_unsecure = cu.SBMM_Unsecure()
        for i in range(num_test):
            print("Test #", i + 1)
            batch_size = random.randint(1, 1)
            M = 2**random.randint(10, 10)
            K = 2**random.randint(10, 10)
            N = 2**random.randint(10, 10)

            for j in range(num_iter):
                print(batch_size, M, K, N)
                x = np.random.randint(-128, 128,
                                      (batch_size, M, K), dtype=np.int8)
                y = np.random.randint(-128, 128,
                                      (batch_size, K, N), dtype=np.int8)
                x2 = x.copy()
                y2 = y.copy()

                z2 = sbmm_unsecure(x2, y2)
                z = sbmm_light(x, y)

                self.assertTrue(np.array_equal(z, z2))
                if timer_first:
                    sbmm_light.enable_timer()
                    sbmm_unsecure.enable_timer()
                    timer_first = False
                pass_cnt += 1

        timer.display_summary()
        print(f"'test_bmm_secure' passed {pass_cnt} / {num_test * num_iter}")


if __name__ == '__main__':
    unittest.main()
