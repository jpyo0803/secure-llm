import unittest
import build.cipher_cpp as cip_cpp
import numpy as np
import random
import singleton_timer as st


class NtlTest(unittest.TestCase):
    def test_int32(self):
        timer = st.SingletonTimer()

        B = 8
        amt = random.randint(-(2**(B-1)), 2**(B-1))

        num_test = 1000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            K = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(-(2**(B-1)), 2**(B-1),
                                  size=(M, K), dtype=np.int32)
            y = np.random.randint(-(2**(B-1)), 2**(B-1),
                                  size=(K, N), dtype=np.int32)

            x_np = x.copy()
            y_np = y.copy()

            # Generate metadata for
            row_sum_x = cip_cpp.SumByRow_int32(x)
            col_sum_y = cip_cpp.SumByCol_int32(y)

            cip_cpp.Shift_int32(x, amt)
            cip_cpp.Shift_int32(y, amt)

            z = np.matmul(x, y)

            t = timer.start(tag='int32', category='int32')
            cip_cpp.UndoShift_int32(z, amt, K, row_sum_x, col_sum_y)
            timer.end(t)

            z_np = np.matmul(x_np, y_np)

            self.assertTrue(np.array_equal(z, z_np))

            pass_cnt += 1
        print(f"'test_int32' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_int64(self):
        timer = st.SingletonTimer()

        B = 8 * 2
        amt = random.randint(-(2**(B-1)), 2**(B-1))

        num_test = 1000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            K = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(-(2**(B-1)), 2**(B-1),
                                  size=(M, K), dtype=np.int64)
            y = np.random.randint(-(2**(B-1)), 2**(B-1),
                                  size=(K, N), dtype=np.int64)

            x_np = x.copy()
            y_np = y.copy()

            # Generate metadata for
            row_sum_x = cip_cpp.SumByRow_int64(x)
            col_sum_y = cip_cpp.SumByCol_int64(y)

            cip_cpp.Shift_int64(x, amt)
            cip_cpp.Shift_int64(y, amt)

            z = np.matmul(x, y)

            t = timer.start(tag='int64', category='int64')
            cip_cpp.UndoShift_int64(z, amt, K, row_sum_x, col_sum_y)
            timer.end(t)

            z_np = np.matmul(x_np, y_np)

            self.assertTrue(np.array_equal(z, z_np))

            pass_cnt += 1
        print(f"'test_int64' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_uint32(self):
        timer = st.SingletonTimer()

        B = 8
        amt = random.randint(0, 2**B)

        num_test = 1000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            K = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(0, 2**B,
                                  size=(M, K), dtype=np.uint32)
            y = np.random.randint(0, 2**B,
                                  size=(K, N), dtype=np.uint32)

            x_np = x.copy()
            y_np = y.copy()

            # Generate metadata for
            row_sum_x = cip_cpp.SumByRow_uint32(x)
            col_sum_y = cip_cpp.SumByCol_uint32(y)

            cip_cpp.Shift_uint32(x, amt)
            cip_cpp.Shift_uint32(y, amt)

            z = np.matmul(x, y)

            t = timer.start(tag='uint32', category='uint32')
            cip_cpp.UndoShift_uint32(z, amt, K, row_sum_x, col_sum_y)
            timer.end(t)

            z_np = np.matmul(x_np, y_np)

            self.assertTrue(np.array_equal(z, z_np))

            pass_cnt += 1
        print(f"'test_uint32' passed {pass_cnt} / {num_test}")
        timer.display_summary()

    def test_uint64(self):
        timer = st.SingletonTimer()

        B = 8 * 2
        amt = random.randint(0, B)

        num_test = 1000
        pass_cnt = 0
        for _ in range(num_test):
            M = random.randint(1, 100)
            K = random.randint(1, 100)
            N = random.randint(1, 100)
            x = np.random.randint(0, 2**B,
                                  size=(M, K), dtype=np.uint64)
            y = np.random.randint(0, 2**B,
                                  size=(K, N), dtype=np.uint64)

            x_np = x.copy()
            y_np = y.copy()

            # Generate metadata for
            row_sum_x = cip_cpp.SumByRow_uint64(x)
            col_sum_y = cip_cpp.SumByCol_uint64(y)

            cip_cpp.Shift_uint64(x, amt)
            cip_cpp.Shift_uint64(y, amt)

            z = np.matmul(x, y)

            t = timer.start(tag='uint64', category='uint64')
            cip_cpp.UndoShift_uint64(z, amt, K, row_sum_x, col_sum_y)
            timer.end(t)

            z_np = np.matmul(x_np, y_np)

            self.assertTrue(np.array_equal(z, z_np))

            pass_cnt += 1
        print(f"'test_uint64' passed {pass_cnt} / {num_test}")
        timer.display_summary()


if __name__ == '__main__':
    # Warmup
    x = np.random.randint(-2**20, 2**20, size=(1, 1))
    amt = random.randint(-2**20, 2**20)
    y = cip_cpp.Shift_int32(x, amt)

    unittest.main()
