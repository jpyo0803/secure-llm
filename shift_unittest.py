import unittest
import build.cipher_cpp as cip_cpp
import numpy as np
import random
import singleton_timer as st


class NtlTest(unittest.TestCase):
    def test_int32(self):
        timer = st.SingletonTimer()
        num_test = 1000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(-2**20, 2**20, size=(M, N), dtype=np.int32)
            x_np = x.copy()
            amt = random.randint(-2**20, 2**20)
            t = timer.start(tag='int32', category='int32')
            cip_cpp.Shift_int32(x, amt)
            timer.end(t)
            y_np = np.add(x_np, amt)
            self.assertTrue(np.array_equal(x, y_np))
            pass_cnt += 1
        print(f"'test_int32' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_uint32(self):
        timer = st.SingletonTimer()
        num_test = 1000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(0, 2**20, size=(M, N), dtype=np.uint32)
            x_np = x.copy()
            amt = random.randint(0, 2**20)
            t = timer.start(tag='uint32', category='uint32')
            cip_cpp.Shift_uint32(x, amt)
            timer.end(t)
            y_np = np.add(x_np, amt)
            self.assertTrue(np.array_equal(x, y_np))
            pass_cnt += 1
        print(f"'test_uint32' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_int64(self):
        timer = st.SingletonTimer()
        num_test = 1000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(-2**40, 2**40, size=(M, N), dtype=np.int64)
            x_np = x.copy()
            amt = random.randint(-2**40, 2**40)
            t = timer.start(tag='int64', category='int64')
            cip_cpp.Shift_int64(x, amt)
            timer.end(t)
            y_np = np.add(x_np, amt)
            self.assertTrue(np.array_equal(x, y_np))
            pass_cnt += 1
        print(f"'test_int64' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_uint64(self):
        timer = st.SingletonTimer()
        num_test = 1000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(0, 2**40, size=(M, N), dtype=np.uint64)
            x_np = x.copy()
            amt = random.randint(0, 2**40)
            t = timer.start(tag='uint64', category='uint64')
            cip_cpp.Shift_uint64(x, amt)
            timer.end(t)
            y_np = np.add(x_np, amt)
            self.assertTrue(np.array_equal(x, y_np))
            pass_cnt += 1
        print(f"'test_uint64' passed {pass_cnt} / {num_test}")
        timer.display_summary()


if __name__ == '__main__':
    # Warmup
    x = np.random.randint(-2**20, 2**20, size=(1, 1))
    amt = random.randint(-2**20, 2**20)
    y = cip_cpp.Shift_int32(x, amt)

    unittest.main()
