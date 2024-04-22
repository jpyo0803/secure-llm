import unittest
import build.cipher_cpp as cip_cpp
import singleton_timer as st
import numpy as np
import torch

class NtlTest(unittest.TestCase):
    def test_numpy_rand_gen(self): # Fastest
        timer = st.SingletonTimer()
        num_test = 5
        pass_cnt = 0

        M = N = 2**12
        for _ in range(num_test):
            t = timer.start(tag='numpy_rand_gen', category='numpy_rand_gen')
            rand_arr = np.random.randint(-(2**31), (2**31), (M, N))
            timer.end(t)
            pass_cnt += 1
        print(f"'test_numpy_rand_gen' passed {pass_cnt} / {num_test}")
        timer.display_summary()
 
    def test_torch_rand_gen(self):
        timer = st.SingletonTimer()
        num_test = 5
        pass_cnt = 0

        M = N = 2**12
        for _ in range(num_test):
            t = timer.start(tag='torch_rand_gen', category='torch_rand_gen')
            rand_arr = torch.randint(-(2**31), (2**31), (M, N))
            timer.end(t)
            pass_cnt += 1
        print(f"'test_torch_rand_gen' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_cpp_rand_gen(self):
        timer = st.SingletonTimer()
        num_test = 5
        pass_cnt = 0

        M = N = 2**12
        for _ in range(num_test):
            t = timer.start(tag='cpp_rand_gen', category='cpp_rand_gen')
            rand_arr = cip_cpp.RandInt2D_int32(0, 2**31 - 1, M, N)
            timer.end(t)
            pass_cnt += 1
        print(f"'test_cpp_rand_gen' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    # def test_int64(self):

    # def test_uint32(self):

    # def test_uint64(self):


if __name__ == '__main__':
    unittest.main()
