import numpy as np
import singleton_timer as st
import ntl
import cupy as cp

# Secure Batch Matrix Multiplication Light version


class SBMM_Light:
    def __init__(self, timer_on: bool = False):
        self.B = 8
        self.timer_on = timer_on
        self.mod = 2**32
        self.shift_amt = 2**(self.B - 1)

    def __run(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 3 and y.ndim == 3
        assert x.shape[0] == y.shape[0]
        assert x.shape[2] == y.shape[1]
        assert x.dtype == np.int8 and y.dtype == np.int8

        if self.timer_on:
            timer = st.SingletonTimer()

        batch_size, M, K = x.shape
        _, _, N = y.shape

        if self.timer_on:
            t = timer.start(tag='S8 to S32 (L)',
                            category='S8 to S32 (L)')
        x = x.astype(np.int32, copy=False)
        y = y.astype(np.int32, copy=False)
        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='gen. shift metadata (L)',
                            category='gen. shift metadata (L)')
        shift_factor = self.__generate_shift_metedata(
            x, y)
        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='shift inputs (L)',
                            category='shift inputs (L)')
        shifted_x, shifted_y = self.__shift_inputs(x, y)
        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='S32 to U32 (L)',
                            category='S32 to U32 (L)')
        shifted_x = shifted_x.view(np.uint32)
        shifted_y = shifted_y.view(np.uint32)
        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='gen. keys (L)',
                                category='gen. keys (L)')
        b_x = np.random.randint(
            0, self.mod - 1, size=(K), dtype=np.uint32)
        b_y = np.random.randint(
            0, self.mod - 1, size=(K), dtype=np.uint32)
        a_x = np.array(ntl.ntl_cpp.GenerateRandomCoprimeLessthanN_uint64(
            self.mod), dtype=np.uint32)
        a_y = np.array(ntl.ntl_cpp.GenerateRandomCoprimeLessthanN_uint64(
            self.mod), dtype=np.uint32)
        key_inv = np.array(ntl.FindKeyInvModNonPrime(
            a_x * a_y, self.mod), dtype=np.uint32)
        b_factor = np.dot(b_x, b_y)
        # assert (a_x *a_y * key_inv % self.mod) == 1
        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='gen. decryption metadata (L)',
                            category='gen. decryption metadata (L)')
        dec_row_sum_x, dec_col_sum_y = self.__generate_decryption_metadata(
            shifted_x, shifted_y, a_x, a_y, b_x, b_y)
        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='encrypt (L)',
                            category='encrypt (L)')

        enc_x = self.__encrypt_tensor(shifted_x, a_x, b_x, False)
        enc_y = self.__encrypt_tensor(shifted_y, a_y, b_y, True)

        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='HtoD (L)',
                            category='HtoD (L)')

        enc_x_gpu = cp.asarray(enc_x)
        enc_y_gpu = cp.asarray(enc_y)

        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='gpu comp. (L)', category='gpu comp. (L)')
        enc_z_gpu = cp.matmul(enc_x_gpu, enc_y_gpu)
        if self.timer_on:
            timer.end(t)
        if self.timer_on:
            t = timer.start(tag='DtoH (L)', category='DtoH (L)')
        enc_z = cp.asnumpy(enc_z_gpu)
        if self.timer_on:
            timer.end(t)
        if self.timer_on:
            t = timer.start(tag='decrypt (L)',
                            category='decrypt (L)')

        z = self.__decrypt_tensor(
            enc_z, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor)

        if self.timer_on:
            timer.end(t)
        if self.timer_on:
            t = timer.start(tag='U32 to S32 (L)',
                            category='U32 to S32 (L)')
        z = z.view(np.int32)
        if self.timer_on:
            timer.end(t)

        if self.timer_on:
            t = timer.start(tag='undo shift (L)',
                            category='undo shift (L)')
        z -= shift_factor
        if self.timer_on:
            timer.end(t)
        return z

    def __generate_shift_metedata(self, x, y):
        _, _, K = x.shape
        shift_row_sum_x = np.sum(x, axis=2, dtype=np.int32)
        shift_col_sum_y = np.sum(y, axis=1, dtype=np.int32)

        shift_factor = shift_row_sum_x[:, :, np.newaxis] * self.shift_amt + K * \
            self.shift_amt**2 + \
            shift_col_sum_y[:, np.newaxis, :] * self.shift_amt
        return shift_factor

    def __shift_inputs(self, x, y):
        x += self.shift_amt
        y += self.shift_amt
        return x, y

    def __generate_decryption_metadata(self, x, y, a_x, a_y, b_x, b_y):
        # a_x, a_y: (1,)
        # b_x, b_y: (K,)
        # x: (batch_size, M, K)
        # y: (batch_size, K, N)
        dec_row_sum_x = np.sum(x * b_y, axis=2, dtype=np.uint32) * a_x
        dec_col_sum_y = np.sum(
            y * b_x[np.newaxis, :, np.newaxis], axis=1, dtype=np.uint32) * a_y
        return dec_row_sum_x, dec_col_sum_y

    def __encrypt_tensor(self, tensor, a, b, vertical):
        # tensor: (batch_size, M, K), if horizontal
        # tensor: (batch_size, K, N), if vertical
        # a: (1,)
        # b: (K,)

        if vertical:
            return tensor * a + b[np.newaxis, :, np.newaxis]
        return tensor * a + b

    def __decrypt_tensor(self, tensor, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor):
        # tensor: (batch_size, M, N)
        # key_inv: (1, )
        # dec_row_sum_x: (batch_size, M)
        # dec_col_sum_y: (batch_size, N)
        # b_factor: (1,)
        out = tensor - dec_row_sum_x[:, :,
                                     np.newaxis] - dec_col_sum_y[:, np.newaxis, :]
        out -= b_factor
        out *= key_inv
        return out

    def __call__(self, x: np.ndarray, y: np.ndarray):
        return self.__run(x, y)
