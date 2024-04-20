import unittest
import build.cipher_cpp as cip_cpp
import numpy as np
import random
import singleton_timer as st


class NtlTest(unittest.TestCase):
    def test_int32(self):
        timer = st.SingletonTimer()
        num_test = 10000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(-2**20, 2**20, size=(M, N))
            t = timer.start(tag='int32', category='int32')
            y = cip_cpp.SumByRow_int32(x)
            timer.end(t)
            y_np = np.sum(x, axis=1)
            self.assertTrue(np.array_equal(y, y_np))
            pass_cnt += 1
        print(f"'test_int32' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_int64(self):
        timer = st.SingletonTimer()
        num_test = 10000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(-2**40, 2**40, size=(M, N))
            t = timer.start(tag='int64', category='int64')
            y = cip_cpp.SumByRow_int64(x)
            timer.end(t)
            y_np = np.sum(x, axis=1)
            self.assertTrue(np.array_equal(y, y_np))
            pass_cnt += 1
        print(f"'test_int64' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_uint32(self):
        timer = st.SingletonTimer()
        num_test = 10000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(0, 2**20, size=(M, N))
            t = timer.start(tag='uint32', category='uint32')
            y = cip_cpp.SumByRow_uint32(x)
            timer.end(t)
            y_np = np.sum(x, axis=1)
            self.assertTrue(np.array_equal(y, y_np))
            pass_cnt += 1
        print(f"'test_uint32' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_uint64(self):
        timer = st.SingletonTimer()
        num_test = 10000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(0, 2**40, size=(M, N))
            t = timer.start(tag='uint64', category='uint64')
            y = cip_cpp.SumByRow_uint64(x)
            timer.end(t)
            y_np = np.sum(x, axis=1)
            self.assertTrue(np.array_equal(y, y_np))
            pass_cnt += 1
        print(f"'test_uint64' passed {pass_cnt} / {num_test}")
        timer.display_summary()


if __name__ == '__main__':
    # Warmup
    x = np.random.randint(-2**20, 2**20, size=(1, 1))
    y = cip_cpp.SumByRow_int32(x)

    unittest.main()
