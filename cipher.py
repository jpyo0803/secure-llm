import build.cipher_cpp as cip_cpp
import singleton_timer as st
import numpy as np
import cupy as cp


def decrypt_matrix_2d_full(in_matrix, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor):
    adjusted_matrix = in_matrix - \
        (dec_row_sum_x[:, np.newaxis] + dec_col_sum_y)
    adjusted_matrix -= b_factor
    adjusted_matrix *= key_inv
    return adjusted_matrix


def compute_decryption_sums(x, y, b_x, b_y, a_x, a_y):
    # print(b_y.shape)
    # print(x.shape)
    # print(a_x.shape)
    # print("----------")
    dec_row_sum_x = np.einsum('i,ij,j->i', a_x, x, b_y, optimize=True)
    dec_col_sum_y = np.einsum('i,ij,j->i', a_y, y.T, b_x, optimize=True)

    return dec_row_sum_x, dec_col_sum_y


def encrypt_matrix_2d_full(in_matrix, a, b, vertical):
    M, N = in_matrix.shape
    # Create a copy of in_matrix to avoid modifying the original array
    result = in_matrix
    # print("a size: ", a.shape)
    # print("a size target: ", 1, N)
    if vertical:
        a = a.reshape(1, N)
        b = b.reshape(M, 1)
    else:
        a = a.reshape(M, 1)
        b = b.reshape(1, N)

    result *= a
    result += b

    return result


def undo_shift(in_matrix, amt, K, row_sum_x, col_sum_y):
    in_matrix -= amt * (row_sum_x[:, np.newaxis] + col_sum_y + K * amt)
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


# a_x = None
# a_y = None
# b_x = None
# b_y = None
# b_factor = None
# key_inv = None


def SecureMatmulFull(x: np.ndarray, y: np.ndarray):
    assert x.ndim == 2 and y.ndim == 2
    assert x.shape[1] == y.shape[0]
    assert x.dtype == np.int8 and y.dtype == np.int8
    timer = st.SingletonTimer()

    B = 8

    M, K = x.shape
    _, N = y.shape

    # 8 bit -> 32 bit
    t = timer.start(tag='int8 to int32', category='int8 to int32')
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    timer.end(t)

    mod = 2**32

    t = timer.start(tag='gen. shift metadata', category='gen. shift metadata')
    # Generate metadata for shifting
    shift_row_sum_x = np.sum(x, axis=1, dtype=np.int32)
    shift_col_sum_y = np.sum(y, axis=0, dtype=np.int32)
    timer.end(t)

    amt = 2**(B-1)

    t = timer.start(tag='shift inputs', category='shift inputs')
    # Shift
    x += amt
    y += amt
    timer.end(t)

    # int32 -> uint32
    t = timer.start(tag='int32 to uint32', category='int32 to uint32')
    x = x.astype(np.uint32)
    y = y.astype(np.uint32)
    timer.end(t)

    # Generate metadata for decryption

    # Do precomputation only once
    # t = timer.start(tag='precomputation', category='precomputation')
    # global a_x, a_y, b_x, b_y, b_factor, key_inv
    # if a_x is None:
    a_x = np.asanyarray(cip_cpp.GenerateKeySetA(mod, M), dtype=np.uint32)
    a_y = np.asanyarray(cip_cpp.GenerateKeySetA(mod, N), dtype=np.uint32)
    b_x = np.random.randint(0, mod - 1, (K), dtype=np.uint32)
    b_y = np.random.randint(0, mod - 1, (K), dtype=np.uint32)

    b_factor = np.dot(b_x, b_y)

    key_inv = np.asanyarray(
        cip_cpp.FindKeyInvModFull(a_x, a_y), dtype=np.uint32)
    # timer.end(t)

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
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    timer.end(t)

    z_gpu = cp.matmul(x_gpu, y_gpu)
    t = timer.start(tag='gpu computation', category='gpu computation')
    z_gpu = cp.matmul(x_gpu, y_gpu)
    timer.end(t)
    t = timer.start(tag='device to host', category='device to host')
    z = cp.asnumpy(z_gpu)
    timer.end(t)

    t = timer.start(tag='decryption', category='decryption')
    z = decrypt_matrix_2d_full(
        z, key_inv, dec_row_sum_x, dec_col_sum_y, b_factor)
    timer.end(t)

    t = timer.start(tag='uint32 to int32', category='uint32 to int32')
    z = z.astype(np.int32)
    timer.end(t)

    t = timer.start(tag='undo shift', category='undo shift')
    # cip_cpp.UndoShift_int32(z, amt, K, shift_row_sum_x, shift_col_sum_y)
    z = undo_shift(z, amt, K, shift_row_sum_x, shift_col_sum_y)
    timer.end(t)

    return z
