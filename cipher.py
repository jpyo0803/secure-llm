import build.cipher_cpp as cip_cpp
import singleton_timer as st
import numpy as np
import cupy as cp
import nvtx


def decrypt_matrix_2d_full(in_matrix, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor):
    adjusted_matrix = in_matrix - \
        (dec_row_sum_x[:, :, np.newaxis] + dec_col_sum_y[:, np.newaxis, :])
    adjusted_matrix -= b_factor[:, np.newaxis, np.newaxis]
    adjusted_matrix *= key_inv
    return adjusted_matrix


def compute_decryption_sums(x, y, b_x, b_y, a_x, a_y):
    # print(b_y.shape)
    # print(x.shape)
    # print(a_x.shape)
    # print("----------")
    batch_size, M, _ = x.shape
    _, _, N = y.shape
    dec_row_sum_x = np.empty((batch_size, M), dtype=np.uint32)
    dec_col_sum_y = np.empty((batch_size, N), dtype=np.uint32)
    for i in range(batch_size):
        dec_row_sum_x[i] = np.einsum(
            'i,ij,j->i', a_x[i], x[i], b_y[i], optimize=True)
        dec_col_sum_y[i] = np.einsum(
            'i,ij,j->i', a_y[i], y[i].T, b_x[i], optimize=True)

    return dec_row_sum_x, dec_col_sum_y


def encrypt_matrix_2d_full(in_matrix, a, b, vertical):
    batch_size, M, N = in_matrix.shape
    # Create a copy of in_matrix to avoid modifying the original array
    result = in_matrix
    # print("a size: ", a.shape)
    # print("a size target: ", 1, N)
    if vertical:
        a = a.reshape(batch_size, 1, N)
        b = b.reshape(batch_size, M, 1)
    else:
        a = a.reshape(batch_size, M, 1)
        b = b.reshape(batch_size, 1, N)

    result *= a
    result += b

    return result


def undo_shift(in_matrix, amt, K, row_sum_x, col_sum_y):
    in_matrix -= amt * (row_sum_x[:, :, np.newaxis] +
                        col_sum_y[:, np.newaxis, :] + K * amt)
    return in_matrix


def BatchSecureMatmulFull(x: np.ndarray, y: np.ndarray):
    assert x.ndim == 3 and y.ndim == 3, "Input tensors must be 3D (batched matrices)"
    assert x.shape[0] == y.shape[0], "Number of batches must be equal"

    batch_size, M, K = x.shape
    _, K, N = y.shape

    results = []

    for i in range(batch_size):
        results.append(SecureMatmulFull(x[i], y[i]))

    z = np.stack(results)
    return z


a_x = None
a_y = None
b_x = None
b_y = None
b_factor = None
key_inv = None


def SecureMatmulFull(x: np.ndarray, y: np.ndarray):
    assert x.ndim == 3 and y.ndim == 3
    assert x.shape[0] == y.shape[0], "Number of batches must be equal"
    assert x.shape[2] == y.shape[1]
    assert x.dtype == np.int8 and y.dtype == np.int8
    timer = st.SingletonTimer()

    B = 8

    batch_size, M, K = x.shape
    _, _, N = y.shape

    # 8 bit -> 32 bit
    t = timer.start(tag='int8 to int32', category='int8 to int32')
    x = x.astype(np.int32, copy=False)
    y = y.astype(np.int32, copy=False)
    timer.end(t)

    mod = 2**32

    t = timer.start(tag='gen. shift metadata', category='gen. shift metadata')
    # Generate metadata for shifting
    shift_row_sum_x = np.sum(x, axis=2, dtype=np.int32)
    shift_col_sum_y = np.sum(y, axis=1, dtype=np.int32)
    timer.end(t)

    amt = 2**(B-1)

    t = timer.start(tag='shift inputs', category='shift inputs')
    # Shift
    x += amt
    y += amt
    timer.end(t)

    # int32 -> uint32
    t = timer.start(tag='int32 to uint32', category='int32 to uint32')
    x = x.astype(np.uint32, copy=False)
    y = y.astype(np.uint32, copy=False)
    timer.end(t)

    # Generate metadata for decryption

    # Do precomputation only once
    # t = timer.start(tag='precomputation', category='precomputation')
    global a_x, a_y, b_x, b_y, b_factor, key_inv
    if a_x is None:
        a_x = np.empty((batch_size, M), dtype=np.uint32)
        a_y = np.empty((batch_size, N), dtype=np.uint32)
        b_x = np.random.randint(
            0, mod - 1, size=(batch_size, K), dtype=np.uint32)
        b_y = np.random.randint(
            0, mod - 1, size=(batch_size, K), dtype=np.uint32)
        b_factor = np.einsum('ij,ij->i', b_x, b_y)
        key_inv = np.empty((batch_size, M, N), dtype=np.uint32)
        for i in range(batch_size):
            a_x[i] = np.asanyarray(
                cip_cpp.GenerateKeySetA(mod, M), dtype=np.uint32)
            a_y[i] = np.asanyarray(
                cip_cpp.GenerateKeySetA(mod, N), dtype=np.uint32)
            key_inv[i] = np.asanyarray(
                cip_cpp.FindKeyInvModFull(a_x[i], a_y[i]), dtype=np.uint32)
    # timer.end(t)
    # b_factor correct
  #  print("x: ", x)
  #  print("y: ", y)
  #  print("ax : ", a_x)
  #  print("ay : ", a_y)
   # print("bx : ", b_x)
  #  print("by : ", b_y)
  #  print("b factor : ", b_factor)
   # print("key inv : ", key_inv)

    t = timer.start(tag='gen. decryption metadata',
                    category='gen. decryption metadata')

    dec_row_sum_x, dec_col_sum_y = compute_decryption_sums(
        x, y, b_x, b_y, a_x, a_y)

    timer.end(t)

    t = timer.start(tag='encryption', category='encryption')

    x = encrypt_matrix_2d_full(x, a_x, b_x, False)
    y = encrypt_matrix_2d_full(y, a_y, b_y, True)

    timer.end(t)

    t = timer.start(tag='host to device', category='host to device')
    nvtx.push_range("host to device")
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    nvtx.pop_range()
    timer.end(t)

    t = timer.start(tag='gpu computation', category='gpu computation')
    nvtx.push_range("gpu computation")
    z_gpu = cp.matmul(x_gpu, y_gpu)
    nvtx.pop_range()
    timer.end(t)
    t = timer.start(tag='device to host', category='device to host')
    nvtx.push_range("device to host")
    z = cp.asnumpy(z_gpu)
    nvtx.pop_range()
    timer.end(t)

    t = timer.start(tag='decryption', category='decryption')
  #  print("z: ", z)
  #  print("key inv: ", key_inv)
   # print("dec row sum x: ", dec_row_sum_x)
  #  print("dec col sum y: ", dec_col_sum_y)
   # print("b_factor: ", b_factor)
    z = decrypt_matrix_2d_full(
        z, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor)
  #  print("dec z: ", z)
  #  assert False
    timer.end(t)

    t = timer.start(tag='uint32 to int32', category='uint32 to int32')
    z = z.astype(np.int32, copy=False)
    timer.end(t)

    t = timer.start(tag='undo shift', category='undo shift')
    # cip_cpp.UndoShift_int32(z, amt, K, shift_row_sum_x, shift_col_sum_y)

    z = undo_shift(z, amt, K, shift_row_sum_x, shift_col_sum_y)

  #  print("unddid:", z)
    timer.end(t)

    return z
