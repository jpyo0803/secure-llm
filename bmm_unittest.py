import unittest
import numpy as np
import random
import singleton_timer as st

import cipher_light as cl
import cipher_unsecure as cu


class BmmTest(unittest.TestCase):
    def test_bmm_secure(self):
        timer = st.SingletonTimer()

        num_test = 20
        pass_cnt = 0

        sbmm_light = cl.SBMM_Light()
        sbmm_unsecure = cu.SBMM_Unsecure()

        for i in range(num_test):
            print("Test #", i + 1)
            batch_size = random.randint(1, 30)
            M = 2**random.randint(4, 12)
            K = 2**random.randint(4, 12)
            N = 2**random.randint(4, 12)

            print(batch_size, M, K, N)
            x = np.random.randint(-128, 128, (batch_size, M, K), dtype=np.int8)
            y = np.random.randint(-128, 128, (batch_size, K, N), dtype=np.int8)
            x2 = x.copy()
            y2 = y.copy()

            z2 = sbmm_unsecure(x2, y2)
            z = sbmm_light(x, y)

            self.assertTrue(np.array_equal(z, z2))
            if i == 0:
                sbmm_light.enable_timer()
                # sbmm_unsecure.enable_timer()
            pass_cnt += 1

        timer.display_summary()
        print(f"'test_bmm_secure' passed {pass_cnt} / {num_test}")


if __name__ == '__main__':
    unittest.main()
